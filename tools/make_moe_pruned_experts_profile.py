# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert retained-expert selections into a vLLM pruned-experts profile."""

import argparse
import json
from pathlib import Path
from typing import Any


def _load_selected_experts(path: Path) -> tuple[dict[int, set[int]], dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)

    selected_by_layer: dict[int, set[int]] = {}
    metadata = data.get("_metadata", {})
    for layer_key, selected_experts in data.items():
        if layer_key.startswith("_"):
            continue
        if not isinstance(selected_experts, list):
            raise ValueError(f"Layer {layer_key!r} must map to a list of experts.")
        layer_idx = int(layer_key)
        selected_by_layer[layer_idx] = {int(expert) for expert in selected_experts}

    if not selected_by_layer:
        raise ValueError(f"No layer expert selections found in {path}.")
    return selected_by_layer, metadata


def convert_selected_to_pruned_profile(
    selected_by_layer: dict[int, set[int]],
    num_experts: int,
) -> list[list[int]]:
    if num_experts <= 0:
        raise ValueError("--num-experts must be positive.")

    pruned_experts: list[list[int]] = []
    valid_experts = set(range(num_experts))
    for layer_idx in sorted(selected_by_layer):
        selected = selected_by_layer[layer_idx]
        invalid = sorted(expert for expert in selected if expert not in valid_experts)
        if invalid:
            raise ValueError(
                f"Layer {layer_idx} contains out-of-range experts {invalid}; "
                f"expected ids in [0, {num_experts})."
            )
        for expert_idx in sorted(valid_experts - selected):
            pruned_experts.append([layer_idx, expert_idx])
    return pruned_experts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a per-layer retained-expert JSON file, such as SHAPE's "
            "selected_experts output, into a vLLM --moe-pruned-experts-profile."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to selected experts JSON, keyed by layer id.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to write the vLLM pruned-experts profile JSON.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        required=True,
        help="Number of logical routed experts per MoE layer.",
    )
    args = parser.parse_args()

    selected_by_layer, source_metadata = _load_selected_experts(args.input)
    pruned_experts = convert_selected_to_pruned_profile(
        selected_by_layer=selected_by_layer,
        num_experts=args.num_experts,
    )
    total_experts = len(selected_by_layer) * args.num_experts
    total_selected = total_experts - len(pruned_experts)

    profile = {
        "version": 1,
        "pruned_experts": pruned_experts,
        "_metadata": {
            "source": str(args.input),
            "num_experts": args.num_experts,
            "num_layers": len(selected_by_layer),
            "total_experts": total_experts,
            "selected_experts": total_selected,
            "pruned_experts": len(pruned_experts),
            "keep_rate": total_selected / total_experts,
            "source_metadata": source_metadata,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
