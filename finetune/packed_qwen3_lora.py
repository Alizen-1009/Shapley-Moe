#!/usr/bin/env python3
"""
Packed-expert adaptive LoRA utilities for Qwen3-MoE.

Qwen3 stores routed expert weights in packed 3D tensors:

    experts.gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
    experts.down_proj:    [num_experts, hidden_size, moe_intermediate_size]

Standard PEFT target_modules cannot assign different ranks to different experts
inside the same packed parameter. This module keeps the expert-wise rank_map by
wrapping each `Qwen3MoeExperts` block with explicit per-expert low-rank deltas.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PACKED_QWEN3_ADAPTER_CONFIG = "packed_qwen3_expert_lora_config.json"
PACKED_QWEN3_ADAPTER_WEIGHTS = "packed_qwen3_expert_lora.bin"
PACKED_QWEN3_ADAPTER_TYPE = "packed_qwen3_expert_lora"


RankMap = Dict[str, Dict[str, int]]


def uses_packed_qwen3_experts(model) -> bool:
    """Detect the packed-expert Qwen3 implementation with 3D expert parameters."""
    try:
        experts = model.model.layers[0].mlp.experts
    except Exception:
        return False

    param_names = {name for name, _ in experts.named_parameters(recurse=False)}
    module_names = set(dict(experts.named_modules()).keys())
    has_packed_params = {"gate_up_proj", "down_proj"}.issubset(param_names)
    has_per_expert_modules = any(
        "gate_proj" in name or "up_proj" in name or "down_proj" in name for name in module_names
    )
    return has_packed_params and not has_per_expert_modules


def is_packed_qwen3_adapter_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, PACKED_QWEN3_ADAPTER_CONFIG))


def _expert_key(expert_idx: int) -> str:
    return f"expert_{expert_idx}"


class ExpertLowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling


class PackedQwen3ExpertAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: Sequence[str],
    ) -> None:
        super().__init__()
        targets = set(target_modules)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.target_modules = tuple(sorted(targets))

        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

        if "gate_proj" in targets:
            self.gate_proj = ExpertLowRankLinear(hidden_size, moe_intermediate_size, rank, alpha, dropout)
        if "up_proj" in targets:
            self.up_proj = ExpertLowRankLinear(hidden_size, moe_intermediate_size, rank, alpha, dropout)
        if "down_proj" in targets:
            self.down_proj = ExpertLowRankLinear(moe_intermediate_size, hidden_size, rank, alpha, dropout)

    def gate_delta_weight(self) -> Optional[torch.Tensor]:
        return None if self.gate_proj is None else self.gate_proj.delta_weight()

    def up_delta_weight(self) -> Optional[torch.Tensor]:
        return None if self.up_proj is None else self.up_proj.delta_weight()

    def down_delta_weight(self) -> Optional[torch.Tensor]:
        return None if self.down_proj is None else self.down_proj.delta_weight()


class PackedQwen3ExpertsLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        layer_idx: int,
        expert_ranks: Mapping[str, int],
        target_modules: Sequence[str],
        alpha_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if alpha_scale <= 0:
            raise ValueError(f"alpha_scale must be positive, got {alpha_scale}")

        self.base_layer = base_layer
        self.layer_idx = int(layer_idx)
        self.alpha_scale = float(alpha_scale)
        self.dropout = float(dropout)
        self.target_modules = tuple(target_modules)
        self.merged = False

        gate_up_shape = tuple(base_layer.gate_up_proj.shape)
        down_shape = tuple(base_layer.down_proj.shape)
        if len(gate_up_shape) != 3 or len(down_shape) != 3:
            raise ValueError(
                f"PackedQwen3ExpertsLoRA expects 3D packed expert tensors, got gate_up={gate_up_shape}, down={down_shape}"
            )

        self.num_experts = int(gate_up_shape[0])
        self.gate_up_out_features = int(gate_up_shape[1])
        self.hidden_size = int(gate_up_shape[2])
        if self.gate_up_out_features % 2 != 0:
            raise ValueError(f"Expected even gate_up out_features, got {self.gate_up_out_features}")
        self.moe_intermediate_size = self.gate_up_out_features // 2

        if int(down_shape[0]) != self.num_experts:
            raise ValueError(
                f"Packed expert tensor expert count mismatch: gate_up={self.num_experts}, down={down_shape[0]}"
            )
        if int(down_shape[1]) != self.hidden_size or int(down_shape[2]) != self.moe_intermediate_size:
            raise ValueError(
                "Packed expert tensor shape mismatch: "
                f"down_proj expected ({self.num_experts}, {self.hidden_size}, {self.moe_intermediate_size}), "
                f"got {down_shape}"
            )

        self.adapters = nn.ModuleDict()
        self.expert_ranks: Dict[str, int] = {}
        self.expert_alphas: Dict[str, int] = {}

        for expert_key, rank in sorted(expert_ranks.items(), key=lambda item: int(item[0])):
            expert_idx = int(expert_key)
            rank = int(rank)
            if rank <= 0:
                continue
            if expert_idx < 0 or expert_idx >= self.num_experts:
                raise ValueError(
                    f"Layer {self.layer_idx} rank_map includes expert {expert_idx}, "
                    f"but this layer has experts in [0, {self.num_experts})"
                )
            alpha = max(1, int(round(rank * self.alpha_scale)))
            key = _expert_key(expert_idx)
            self.adapters[key] = PackedQwen3ExpertAdapter(
                hidden_size=self.hidden_size,
                moe_intermediate_size=self.moe_intermediate_size,
                rank=rank,
                alpha=alpha,
                dropout=self.dropout,
                target_modules=target_modules,
            )
            self.expert_ranks[str(expert_idx)] = rank
            self.expert_alphas[str(expert_idx)] = alpha

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.merged:
            return self.base_layer(hidden_states, top_k_index, top_k_weights)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0].item())
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up = F.linear(current_state, self.base_layer.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)

            adapter_key = _expert_key(expert_idx)
            adapter = self.adapters[adapter_key] if adapter_key in self.adapters else None
            if adapter is not None:
                if adapter.gate_proj is not None:
                    gate = gate + adapter.gate_proj(current_state)
                if adapter.up_proj is not None:
                    up = up + adapter.up_proj(current_state)

            expert_hidden = self.base_layer.act_fn(gate) * up
            current_hidden_states = F.linear(expert_hidden, self.base_layer.down_proj[expert_idx])
            if adapter is not None and adapter.down_proj is not None:
                current_hidden_states = current_hidden_states + adapter.down_proj(expert_hidden)

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def merge(self) -> None:
        if self.merged:
            return

        with torch.no_grad():
            for expert_key, adapter in self.adapters.items():
                expert_idx = int(expert_key.split("_", 1)[1])

                gate_delta = adapter.gate_delta_weight()
                if gate_delta is not None:
                    self.base_layer.gate_up_proj.data[expert_idx, : self.moe_intermediate_size, :] += gate_delta.to(
                        self.base_layer.gate_up_proj.device,
                        self.base_layer.gate_up_proj.dtype,
                    )

                up_delta = adapter.up_delta_weight()
                if up_delta is not None:
                    self.base_layer.gate_up_proj.data[expert_idx, self.moe_intermediate_size :, :] += up_delta.to(
                        self.base_layer.gate_up_proj.device,
                        self.base_layer.gate_up_proj.dtype,
                    )

                down_delta = adapter.down_delta_weight()
                if down_delta is not None:
                    self.base_layer.down_proj.data[expert_idx] += down_delta.to(
                        self.base_layer.down_proj.device,
                        self.base_layer.down_proj.dtype,
                    )

        self.merged = True

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict: Dict[str, torch.Tensor] = {}
        for expert_key, adapter in self.adapters.items():
            adapter_state = adapter.state_dict()
            for name, value in adapter_state.items():
                state_dict[f"adapters.{expert_key}.{name}"] = value.detach().cpu()
        return state_dict

    def load_adapter_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        grouped: Dict[str, Dict[str, torch.Tensor]] = {}
        prefix = "adapters."
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            expert_key, param_name = remainder.split(".", 1)
            grouped.setdefault(expert_key, {})[param_name] = value

        expected = set(self.adapters.keys())
        found = set(grouped.keys())
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing experts: {missing[:5]}")
            if unexpected:
                parts.append(f"unexpected experts: {unexpected[:5]}")
            raise ValueError(
                f"Adapter state for layer {self.layer_idx} does not match current wrapper configuration ({'; '.join(parts)})"
            )

        for expert_key, adapter in self.adapters.items():
            adapter.load_state_dict(grouped[expert_key], strict=True)


@dataclass
class PackedQwen3AdapterSummary:
    wrapped_layers: int
    adapted_experts: int
    trainable_parameters: int


def _iter_layer_wrappers(model) -> Iterable[tuple[str, PackedQwen3ExpertsLoRA]]:
    for module_name, module in model.named_modules():
        if isinstance(module, PackedQwen3ExpertsLoRA):
            yield module_name, module


def apply_packed_qwen3_expert_lora(
    model,
    rank_map: Mapping[str, Mapping[str, int]],
    target_modules: Sequence[str],
    alpha_scale: float,
    dropout: float,
) -> PackedQwen3AdapterSummary:
    if not uses_packed_qwen3_experts(model):
        raise ValueError("The provided model does not appear to use packed Qwen3 experts.")

    unsupported = sorted(set(target_modules) - {"gate_proj", "up_proj", "down_proj"})
    if unsupported:
        raise ValueError(f"Packed Qwen3 expert LoRA only supports gate_proj, up_proj, down_proj; got {unsupported}")

    wrapped_layers = 0
    adapted_experts = 0

    for layer_key, expert_ranks in sorted(rank_map.items(), key=lambda item: int(item[0])):
        layer_idx = int(layer_key)
        if layer_idx < 0 or layer_idx >= len(model.model.layers):
            raise ValueError(
                f"rank_map includes layer {layer_idx}, but model has layers in [0, {len(model.model.layers)})"
            )
        layer = model.model.layers[layer_idx]
        existing = layer.mlp.experts
        if isinstance(existing, PackedQwen3ExpertsLoRA):
            raise ValueError(f"Layer {layer_idx} already has PackedQwen3ExpertsLoRA applied.")

        wrapper = PackedQwen3ExpertsLoRA(
            base_layer=existing,
            layer_idx=layer_idx,
            expert_ranks=expert_ranks,
            target_modules=target_modules,
            alpha_scale=alpha_scale,
            dropout=dropout,
        )
        layer.mlp.experts = wrapper
        wrapped_layers += 1
        adapted_experts += len(wrapper.adapters)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, wrapper in _iter_layer_wrappers(model):
        for parameter in wrapper.adapters.parameters():
            parameter.requires_grad = True

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return PackedQwen3AdapterSummary(
        wrapped_layers=wrapped_layers,
        adapted_experts=adapted_experts,
        trainable_parameters=trainable_parameters,
    )


def get_packed_qwen3_lora_state_dict(model) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for module_name, wrapper in _iter_layer_wrappers(model):
        for key, value in wrapper.adapter_state_dict().items():
            state_dict[f"{module_name}.{key}"] = value
    return state_dict


def save_packed_qwen3_lora(
    model,
    output_dir: str,
    *,
    base_model: str,
    rank_map: Mapping[str, Mapping[str, int]],
    target_modules: Sequence[str],
    alpha_scale: float,
    dropout: float,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    adapter_config = {
        "adapter_type": PACKED_QWEN3_ADAPTER_TYPE,
        "format_version": 1,
        "base_model": base_model,
        "model_type": "qwen3",
        "target_module_suffixes": list(target_modules),
        "lora_alpha_scale": float(alpha_scale),
        "lora_dropout": float(dropout),
        "rank_map": rank_map,
    }
    if extra_metadata:
        adapter_config["metadata"] = dict(extra_metadata)

    config_path = os.path.join(output_dir, PACKED_QWEN3_ADAPTER_CONFIG)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)

    weights_path = os.path.join(output_dir, PACKED_QWEN3_ADAPTER_WEIGHTS)
    torch.save(get_packed_qwen3_lora_state_dict(model), weights_path)
    return weights_path


def load_packed_qwen3_lora_config(adapter_dir: str) -> Dict[str, object]:
    config_path = os.path.join(adapter_dir, PACKED_QWEN3_ADAPTER_CONFIG)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    adapter_type = config.get("adapter_type")
    if adapter_type != PACKED_QWEN3_ADAPTER_TYPE:
        raise ValueError(f"Unsupported packed adapter type {adapter_type!r} in {config_path}")
    return config


def load_packed_qwen3_lora(model, adapter_dir: str) -> Dict[str, object]:
    config = load_packed_qwen3_lora_config(adapter_dir)
    apply_packed_qwen3_expert_lora(
        model,
        rank_map=config["rank_map"],
        target_modules=config["target_module_suffixes"],
        alpha_scale=float(config["lora_alpha_scale"]),
        dropout=float(config.get("lora_dropout", 0.0)),
    )

    weights_path = os.path.join(adapter_dir, PACKED_QWEN3_ADAPTER_WEIGHTS)
    state_dict = torch.load(weights_path, map_location="cpu")

    for module_name, wrapper in _iter_layer_wrappers(model):
        prefix = f"{module_name}."
        local_state = {
            key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        if not local_state:
            raise ValueError(f"No packed LoRA weights found for wrapped module {module_name}")
        wrapper.load_adapter_state_dict(local_state)

    return config


def merge_and_unload_packed_qwen3_lora(model) -> int:
    wrapped = list(_iter_layer_wrappers(model))
    for module_name, wrapper in wrapped:
        wrapper.merge()
        if "." in module_name:
            parent_name, child_name = module_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = module_name
        setattr(parent, child_name, wrapper.base_layer)
    return len(wrapped)
