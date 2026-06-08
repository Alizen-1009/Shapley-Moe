#!/usr/bin/env python3
"""
DPO fine-tuning for pruned Qwen3-MoE with expert-wise adaptive LoRA ranks.

This is the optional second stage after SFT (see train_adaptive_lora.py). The
recommended recipe is:

    SFT (adaptive LoRA) -> merge_lora.py -> DPO (this script) -> merge_lora.py

so that --model_path here is the *SFT-merged* pruned model. The DPO reference
distribution is then obtained for free by disabling the (new) LoRA adapters,
which recovers the SFT model -- no second model copy is loaded.

Preference data is the {"prompt", "chosen", "rejected"} JSON produced by
data/distill_sft.py --dpo_output (chosen = RFT-correct, rejected = incorrect).

The same rank_map drives LoRA placement, so DPO uses the identical adaptive-rank
structure as SFT (keeping the C/D/E comparison fair).
"""

import argparse
import json
import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from finetune.train_adaptive_lora import (
        build_rank_pattern,
        build_trainer,
        load_rank_map,
        maybe_enable_gradient_checkpointing,
        parse_csv_list,
        prepare_output_dir,
        read_json_or_jsonl,
        resolve_torch_dtype,
        validate_lora_targets_exist,
    )
    from finetune.packed_qwen3_lora import (
        apply_packed_qwen3_expert_lora,
        packed_qwen3_adapters_disabled,
        save_packed_qwen3_lora,
        uses_packed_qwen3_experts,
    )
except ImportError:
    from train_adaptive_lora import (
        build_rank_pattern,
        build_trainer,
        load_rank_map,
        maybe_enable_gradient_checkpointing,
        parse_csv_list,
        prepare_output_dir,
        read_json_or_jsonl,
        resolve_torch_dtype,
        validate_lora_targets_exist,
    )
    from packed_qwen3_lora import (
        apply_packed_qwen3_expert_lora,
        packed_qwen3_adapters_disabled,
        save_packed_qwen3_lora,
        uses_packed_qwen3_experts,
    )


logger = logging.getLogger(__name__)

LABEL_PAD_ID = -100


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _encode_side(
    tokenizer,
    prompt: str,
    answer: str,
    max_seq_length: int,
    add_eos_token: bool,
) -> Optional[Dict[str, List[int]]]:
    """Tokenize prompt+answer into input_ids/labels with the prompt masked out."""
    # Tokenize via the templated string (version-agnostic: in transformers>=5
    # apply_chat_template(tokenize=True) returns a BatchEncoding, not a list).
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if add_eos_token and tokenizer.eos_token_id is not None:
        if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    if not answer_ids:
        return None

    # Keep the whole prompt; truncate the answer tail if the pair is too long.
    if len(prompt_ids) >= max_seq_length:
        return None
    answer_ids = answer_ids[: max_seq_length - len(prompt_ids)]

    input_ids = prompt_ids + answer_ids
    labels = [LABEL_PAD_ID] * len(prompt_ids) + list(answer_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class DPODataset(Dataset):
    """Tokenized {prompt, chosen, rejected} preference pairs."""

    def __init__(self, records: Sequence[Mapping[str, object]], tokenizer, max_seq_length: int, add_eos_token: bool):
        self.features: List[Dict[str, Dict[str, List[int]]]] = []
        for record in records:
            prompt = record.get("prompt")
            chosen = record.get("chosen")
            rejected = record.get("rejected")
            if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
                continue
            chosen_enc = _encode_side(tokenizer, prompt, chosen, max_seq_length, add_eos_token)
            rejected_enc = _encode_side(tokenizer, prompt, rejected, max_seq_length, add_eos_token)
            if chosen_enc is None or rejected_enc is None:
                continue
            self.features.append({"chosen": chosen_enc, "rejected": rejected_enc})

        if not self.features:
            raise ValueError("No usable DPO preference pairs after tokenization.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, Dict[str, List[int]]]:
        return self.features[index]


class DPODataCollator:
    """Pad chosen and rejected sides independently and stack into one batch."""

    def __init__(self, tokenizer):
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        self.pad_token_id = pad_token_id

    def _pad_side(self, side_features: Sequence[Mapping[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in side_features)
        input_ids, attention_mask, labels = [], [], []
        for feature in side_features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [LABEL_PAD_ID] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, features: Sequence[Mapping[str, Dict[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        chosen = self._pad_side([feature["chosen"] for feature in features])
        rejected = self._pad_side([feature["rejected"] for feature in features])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }


# ---------------------------------------------------------------------------
# DPO trainer
# ---------------------------------------------------------------------------

def sequence_logps(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """Sum of log-probabilities of the label (answer) tokens for each sequence."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    labels = labels[:, 1:].to(logits.device)
    mask = labels != LABEL_PAD_ID
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def make_dpo_trainer_cls(trainer_base):
    class DPOTrainer(trainer_base):
        def __init__(self, *args, beta: float = 0.1, reference_cm=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = beta
            self._reference_cm = reference_cm

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            policy_chosen = sequence_logps(
                model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"], inputs["chosen_labels"]
            )
            policy_rejected = sequence_logps(
                model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"], inputs["rejected_labels"]
            )

            with torch.no_grad(), self._reference_cm(model):
                ref_chosen = sequence_logps(
                    model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"], inputs["chosen_labels"]
                )
                ref_rejected = sequence_logps(
                    model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"], inputs["rejected_labels"]
                )

            pi_logratios = policy_chosen - policy_rejected
            ref_logratios = ref_chosen - ref_rejected
            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(self.beta * logits).mean()

            if return_outputs:
                metrics = {
                    "rewards_chosen": (self.beta * (policy_chosen - ref_chosen)).mean().detach(),
                    "rewards_rejected": (self.beta * (policy_rejected - ref_rejected)).mean().detach(),
                    "reward_accuracy": (pi_logratios > ref_logratios).float().mean().detach(),
                }
                return loss, metrics
            return loss

    return DPOTrainer


@contextmanager
def _peft_reference(model):
    """Reference context for standard PEFT models (disable the LoRA adapter)."""
    with model.disable_adapter():
        yield


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise ImportError("train_dpo_lora.py requires transformers.") from exc

    set_seed(args.seed)

    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank_map = load_rank_map(args.rank_map)
    target_module_suffixes = parse_csv_list(args.target_modules)
    rank_pattern, alpha_pattern, lora_target_modules = build_rank_pattern(
        model_type=args.model_type,
        rank_map=rank_map,
        target_modules=target_module_suffixes,
        alpha_scale=args.lora_alpha_scale,
    )

    logger.info("Loading model from %s", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=resolve_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    maybe_enable_gradient_checkpointing(model, args.gradient_checkpointing)

    uses_packed_impl = args.model_type == "qwen3" and uses_packed_qwen3_experts(model)
    if uses_packed_impl:
        summary = apply_packed_qwen3_expert_lora(
            model,
            rank_map=rank_map,
            target_modules=target_module_suffixes,
            alpha_scale=args.lora_alpha_scale,
            dropout=args.lora_dropout,
        )
        logger.info("Applied packed Qwen3 expert LoRA: %d layers, %d experts, %d trainable params",
                    summary.wrapped_layers, summary.adapted_experts, summary.trainable_parameters)
        reference_cm = packed_qwen3_adapters_disabled
    else:
        from peft import LoraConfig, TaskType, get_peft_model

        validate_lora_targets_exist(model, lora_target_modules)
        lora_config = LoraConfig(
            r=args.default_rank,
            lora_alpha=args.default_alpha,
            target_modules=lora_target_modules,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        reference_cm = _peft_reference

    logger.info("Loading DPO preference data from %s", args.dpo_file)
    records = read_json_or_jsonl(args.dpo_file)
    train_dataset = DPODataset(
        records, tokenizer, max_seq_length=args.max_seq_length, add_eos_token=not args.no_add_eos_token
    )
    logger.info("DPO pairs: %d", len(train_dataset))

    data_collator = DPODataCollator(tokenizer)
    prepare_output_dir(args.output_dir, args.overwrite_output_dir, args.resume_from_checkpoint)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
        label_names=["chosen_labels", "rejected_labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        optim=args.optim,
        do_train=True,
    )

    dpo_trainer_cls = make_dpo_trainer_cls(Trainer)
    trainer = build_trainer(
        dpo_trainer_cls,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.beta = args.beta
    trainer._reference_cm = reference_cm

    logger.info("Starting DPO training (beta=%.3f)", args.beta)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving DPO LoRA adapter to %s", args.output_dir)
    if uses_packed_impl:
        save_packed_qwen3_lora(
            model,
            args.output_dir,
            base_model=args.model_path,
            rank_map=rank_map,
            target_modules=target_module_suffixes,
            alpha_scale=args.lora_alpha_scale,
            dropout=args.lora_dropout,
            extra_metadata={"stage": "dpo", "dpo_file": args.dpo_file, "beta": args.beta},
        )
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.model_path,
        "rank_map": args.rank_map,
        "model_type": args.model_type,
        "stage": "dpo",
        "beta": args.beta,
        "dpo_file": args.dpo_file,
        "adapter_backend": "packed_qwen3_expert_lora" if uses_packed_impl else "peft_lora",
    }
    with open(os.path.join(args.output_dir, "dpo_lora_train_info.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DPO fine-tuning for pruned MoE with adaptive LoRA ranks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="SFT-merged pruned model (DPO reference = adapters disabled).")
    parser.add_argument("--rank_map", required=True, help="Path to rank_map JSON (same as SFT).")
    parser.add_argument("--dpo_file", required=True, help="DPO preference pairs JSON (prompt/chosen/rejected).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the DPO LoRA adapter.")
    parser.add_argument("--model_type", default="qwen3", choices=["qwen3", "deepseek"])

    parser.add_argument("--target_modules", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--default_rank", type=int, default=16)
    parser.add_argument("--default_alpha", type=int, default=32)
    parser.add_argument("--lora_alpha_scale", type=float, default=2.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty strength).")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--no_add_eos_token", action="store_true")

    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    args = build_arg_parser().parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled.")
    train(args)


if __name__ == "__main__":
    main()
