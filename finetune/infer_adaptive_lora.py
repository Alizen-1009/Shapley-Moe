#!/usr/bin/env python3
"""
Quick inference script for a pruned model with an optional LoRA adapter.

This is intended for debugging before merging LoRA and running the formal
vLLM/EvalScope evaluation pipeline.
"""

import argparse
import logging
from typing import Optional

import torch

try:
    from finetune.packed_qwen3_lora import is_packed_qwen3_adapter_dir, load_packed_qwen3_lora
except ImportError:
    from packed_qwen3_lora import is_packed_qwen3_adapter_dir, load_packed_qwen3_lora


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def resolve_torch_dtype(dtype: str):
    dtype = dtype.lower()
    if dtype == "auto":
        return "auto"
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        logger.warning("Tokenizer has no apply_chat_template; using raw prompt.")
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(model_path: str, adapter: Optional[str], torch_dtype: str, device_map: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "infer_adaptive_lora.py requires transformers. Install it in the model environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=resolve_torch_dtype(torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
    )

    if adapter:
        if is_packed_qwen3_adapter_dir(adapter):
            logger.info("Loading packed Qwen3 expert LoRA adapter: %s", adapter)
            load_packed_qwen3_lora(model, adapter)
        else:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading a LoRA adapter requires peft. Install it before running this script."
                ) from exc
            logger.info("Loading PEFT LoRA adapter: %s", adapter)
            model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(args: argparse.Namespace) -> str:
    model, tokenizer = load_model(
        model_path=args.model_path,
        adapter=args.adapter,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    prompt = format_prompt(tokenizer, args.prompt, args.use_chat_template)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run quick inference with a pruned model and optional LoRA adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Base or merged model path.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--use_chat_template", action="store_true", help="Wrap prompt with tokenizer chat template.")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--device_map", default="auto", help="Device map for model loading.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    text = generate(args)
    print(text)


if __name__ == "__main__":
    main()
