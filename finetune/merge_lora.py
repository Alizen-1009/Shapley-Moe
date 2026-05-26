#!/usr/bin/env python3
"""
Merge a trained LoRA adapter into a pruned base model.

The merged model can be served by vLLM and evaluated by the existing EvalScope
pipeline as a normal HuggingFace causal language model.
"""

import argparse
import json
import logging
import os
from typing import Optional

import torch


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


def load_tokenizer(tokenizer_path: str, fallback_path: str):
    from transformers import AutoTokenizer

    try:
        logger.info("Loading tokenizer from adapter path: %s", tokenizer_path)
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Failed to load tokenizer from adapter path: %s", exc)
        logger.info("Loading tokenizer from base model path: %s", fallback_path)
        return AutoTokenizer.from_pretrained(fallback_path, trust_remote_code=True)


def merge_lora(
    base_model: str,
    adapter: str,
    output: str,
    *,
    torch_dtype: str = "auto",
    device_map: Optional[str] = "auto",
    max_shard_size: str = "5GB",
    safe_serialization: bool = True,
) -> None:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "merge_lora.py requires peft and transformers. "
            "Install them in the model environment before running this script."
        ) from exc

    logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=resolve_torch_dtype(torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter: %s", adapter)
    model = PeftModel.from_pretrained(model, adapter)

    logger.info("Merging adapter into base model")
    merged_model = model.merge_and_unload()

    os.makedirs(output, exist_ok=True)
    logger.info("Saving merged model to: %s", output)
    merged_model.save_pretrained(
        output,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )

    tokenizer = load_tokenizer(adapter, base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output)

    metadata = {
        "base_model": base_model,
        "adapter": adapter,
        "output": output,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "max_shard_size": max_shard_size,
        "safe_serialization": safe_serialization,
    }
    metadata_path = os.path.join(output, "merged_lora_info.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Saved merge metadata to: %s", metadata_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a pruned base model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model", required=True, help="Path to pruned base model.")
    parser.add_argument("--adapter", required=True, help="Path to trained LoRA adapter.")
    parser.add_argument("--output", required=True, help="Output merged model directory.")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="Dtype used when loading the base model.",
    )
    parser.add_argument("--device_map", default="auto", help="Device map for model loading.")
    parser.add_argument("--max_shard_size", default="5GB", help="Shard size for save_pretrained.")
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="Save as PyTorch binaries instead of safetensors.",
    )
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()

    merge_lora(
        base_model=args.base_model,
        adapter=args.adapter,
        output=args.output,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_shard_size=args.max_shard_size,
        safe_serialization=not args.no_safe_serialization,
    )


if __name__ == "__main__":
    main()
