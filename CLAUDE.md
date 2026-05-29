# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHAPE (SHapley-Aware Pruning of Experts) is a training-free pruning framework for sparse Mixture-of-Experts (MoE) LLMs. It removes redundant experts after pretraining using coalition-aware Shapley value attribution, without retraining or modifying router logic. Accompanies IJCNN 2026 paper.

## Pipeline Commands

The pipeline runs sequentially. Each step depends on the previous step's outputs.

```bash
# 1. Download calibration data
cd data && ./run_download.sh gsm8k 25        # single dataset
cd data && ./run_download.sh --all            # all datasets from config

# 2. Collect routing traces and activation statistics (single pass, all methods)
cd analysis && ./run_collect.sh -m qwen3-30b-a3b --all

# 3. Compute Shapley values from activation traces
cd analysis && ./run_calc_shapley.sh -m qwen3-30b-a3b

# 4. Select experts to retain
cd pruning && ./run_select.sh -m qwen3-30b-a3b -d gsm8k_25 -M shapley -r 0.8

# 5. Export pruned model (zeros out pruned expert weights)
cd pruning && ./run_prune.sh -m qwen3-30b-a3b -d gsm8k_25 -r 0.8

# 6. (Optional) Adaptive LoRA fine-tuning on pruned model
python finetune/build_rank_map.py --shapley_csv results/.../gsm8k_25_shapley.csv \
  --selected_experts results/.../shapley_alpha_per_layer_gsm8k_25_rate0_8.json \
  --output results/.../gsm8k_25_rate0_8_bucket.json --strategy bucket
python finetune/train_adaptive_lora.py --model_path /path/to/pruned --rank_map results/.../rank_map.json \
  --train_file data/calibration/gsm8k_25.json --output_dir adapters/run_name --model_type qwen3 --bf16
python finetune/merge_lora.py --base_model /path/to/pruned --adapter adapters/run_name --output /path/to/merged

# 7. Evaluate with vLLM + EvalScope
cd evaluation && ./vllm-server.sh qwen3-30b-a3b
python evaluation/run_evalscope.py
```

## Architecture

**Data flow:** calibration JSON → activation traces → Shapley CSV → selected experts JSON → pruned HF model

Key modules:
- `analysis/collect_activations.py` — Single inference pass that collects routing traces for all pruning methods (Shapley, Gating, EASYEP, REAP) simultaneously using forward hooks on MoE gate/expert modules.
- `analysis/calc_shapley.py` — Computes coalition-aware Shapley values from expert co-activation counts. Uses a K-expert marginal contribution formula with sub-coalition frequencies.
- `pruning/methods/select_by_shapley.py` — Implements four selection strategies (alpha_per_layer, alpha_global, topk_per_layer, topk_global). `alpha_per_layer` is the primary SHAPE method: bisection search for alpha threshold that achieves target keep rate while preserving per-layer Shapley mass.
- `pruning/save_model.py` — `ModelPruner` class loads model, zeros out unselected expert weights (or adds gate bias), saves as safetensors.
- `finetune/train_adaptive_lora.py` — Expert-wise adaptive LoRA training. Has a special packed implementation for Qwen3 (`packed_qwen3_lora.py`) that handles packed expert weight tensors.
- `finetune/build_rank_map.py` — Generates per-expert LoRA rank assignments using bucket/uniform/random strategies.

## Configuration

- `configs/models.yaml` — Model paths, expert counts, top-k, MoE module patterns. Model names map to local HF checkpoint paths.
- `configs/experiments.yaml` — Datasets, pruning methods, keep rates, Shapley strategies, evaluation settings.

Shell scripts read these configs via inline Python/yaml parsing. The `-m` flag accepts either a config model name or a full filesystem path.

## Key Conventions

- **Keep rate** (`-r`): fraction of experts retained (0.8 = keep 80%, prune 20%). NOT a removal rate.
- **Results layout:** `results/{model_name}/activations/`, `shapley_values/`, `selected_experts/`, `lora_rank_maps/`
- **File naming:** `{dataset}_{samples}` for calibration (e.g. `gsm8k_25`), `shapley_{strategy}_{dataset}_rate{XX}.json` for selected experts, `{dataset}_rate{XX}_{rank_strategy}.json` for rank maps.
- **Supported models:** qwen3-30b-a3b (128 experts, top-8), gpt-oss-20b (32 experts, top-4), deepseekv2-lite-coder (64 experts, top-6).
- Scripts use `--force` / `-f` to recompute existing results; without it they skip files that already exist.

## Dependencies

Python with: torch, transformers, peft, datasets, pandas, tqdm, pyyaml, vllm (for serving), evalscope (for evaluation).
