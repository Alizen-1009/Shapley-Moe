# SHAPE: Coalition-Aware Expert Pruning for Sparse MoE LLMs

**SHAPE** (**SH**apley-**A**ware **P**runing of **E**xperts) is a training-free framework for pruning experts from sparse Mixture-of-Experts (MoE) language models. It estimates each expert's value inside the routed expert coalitions observed on a small calibration set, then preserves the most useful experts at a target keep rate.

This repository accompanies the IJCNN 2026 paper:

> **SHAPE: Coalition-Aware Expert Pruning for Sparse Mixture-of-Experts LLMs**
>
> [Paper](https://arxiv.org/abs/2606.09886) · [Precomputed activations](https://github.com/Alizen-1009/Shapley-Moe/releases/tag/activation-data-v1) · [Detailed project structure](PROJECT_STRUCTURE.md)

<p align="center">
  <img src="assets/shape_overview.png" alt="SHAPE workflow: collect routed expert coalitions, estimate contributions, and select experts" width="95%">
</p>

## Why SHAPE?

Sparse MoE models activate only a few experts per token, but the entire expert pool must normally remain resident in memory. Existing pruning methods often score experts independently using routing frequency, router probability, or activation magnitude. SHAPE instead models the top-k experts selected for each token as a **coalition**.

- **Coalition-aware:** values experts by their marginal contribution to observed routed groups.
- **Training-free:** requires a calibration pass, not gradient updates or expert retraining.
- **Layer-aware:** preserves Shapley contribution mass per layer while matching a global keep rate.
- **Architecture-preserving:** keeps the pretrained router and MoE structure unchanged.
- **Configurable:** supports SHAPE and five pruning baselines across three MoE families.

## Results

The reported experiments use 25 calibration examples per task and evaluate seven tasks: GSM8K, HumanEval, GPQA-Diamond, MATH-500, TruthfulQA, OntoNotes5, and MedMCQA.

A keep rate of `0.8` retains 80% of experts (20% pruning); `0.6` retains 60% (40% pruning).

| Model | Unpruned | Keep 80% | Keep 60% |
| --- | ---: | ---: | ---: |
| Qwen3-30B-A3B | 82.92 | 82.43 | 81.31 |
| GPT-OSS-20B | 82.12 | 82.44 | 79.02 |
| DeepSeek-V2-Lite | 62.08 | 62.44 | 58.81 |

On Qwen3-30B-A3B, SHAPE retains an average score of **81.31** at a 60% keep rate, compared with 79.05 for EASY-EP and 74.22 for the REAP/RAEP-style baseline. See the [paper](https://arxiv.org/abs/2606.09886) for per-task results, ablations, and full baseline comparisons.

<p align="center">
  <img src="assets/shape_memory_vram.png" alt="Peak VRAM at full size and at 20 and 40 percent expert pruning" width="72%">
</p>

## Installation

### Requirements

- Python 3.12 (the provided environment was captured with Python 3.12)
- Linux and a CUDA-capable GPU for model loading, activation collection, export, and evaluation
- Local Hugging Face checkpoints for the models you want to prune

Create an environment and install the dependencies:

```bash
git clone git@github.com:Alizen-1009/Shapley-Moe.git
cd Shapley-Moe

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` and `vllm` wheels are CUDA-specific. If the pinned versions do not match your CUDA installation, install compatible wheels first and then install the remaining dependencies. Use `requirements-lock.txt` when you need the exact captured environment rather than the curated dependency list.

## Quick Start

### Option A: Reproduce selection from published activations

This path skips model inference and downloads the activation snapshot used by the repository. The archive is approximately 34 MB compressed and 228 MB extracted.

```bash
# Download and checksum-verify results/*/activations/.
./data/download_activations.sh

# Recompute Shapley values for one model and dataset.
./analysis/run_calc_shapley.sh \
  --model qwen3-30b-a3b \
  --dataset gsm8k_25

# Select experts at an 80% keep rate.
./pruning/run_select.sh \
  --model qwen3-30b-a3b \
  --dataset gsm8k_25 \
  --method shapley \
  --rate 0.8
```

The downloaded activations are ignored by Git. Compact derived artifacts such as Shapley CSV files and selected-expert JSON files remain versioned in `results/`.

### Option B: Run the full pipeline

#### 1. Configure model paths

Edit [`configs/models.yaml`](configs/models.yaml) so each model points to a local checkpoint. Experiment defaults, datasets, methods, and keep rates are defined in [`configs/experiments.yaml`](configs/experiments.yaml).

```yaml
models:
  qwen3-30b-a3b:
    path: /path/to/Qwen3-30B-A3B
    num_experts: 128
    num_experts_per_tok: 8
    type: qwen3
```

#### 2. Prepare calibration data

The repository already includes the 25-example calibration sets used in the experiments. To regenerate one dataset or all configured datasets:

```bash
./data/run_download.sh gsm8k 25
./data/run_download.sh --all
```

#### 3. Collect routing and activation statistics

```bash
# Run every calibration dataset for a configured model.
./analysis/run_collect.sh --model qwen3-30b-a3b --all

# Or use an explicit checkpoint and one calibration file.
./analysis/run_collect.sh \
  --model /path/to/model \
  --data data/calibration/gsm8k_25.json
```

Output: `results/{model}/activations/`

#### 4. Compute Shapley values

```bash
# One dataset.
./analysis/run_calc_shapley.sh \
  --model qwen3-30b-a3b \
  --dataset gsm8k_25

# Every available activation file for the model.
./analysis/run_calc_shapley.sh --model qwen3-30b-a3b --all
```

Output: `results/{model}/shapley_values/`

#### 5. Select experts

```bash
# Recommended SHAPE strategy.
./pruning/run_select.sh \
  --model qwen3-30b-a3b \
  --dataset gsm8k_25 \
  --method shapley \
  --strategy alpha_per_layer \
  --rate 0.8

# Run every configured dataset, method, and keep rate.
./pruning/run_select.sh \
  --model qwen3-30b-a3b \
  --all-datasets \
  --all-methods \
  --all-rates
```

Output: `results/{model}/selected_experts/`

#### 6. Export a pruned model

```bash
./pruning/run_prune.sh \
  --model qwen3-30b-a3b \
  --dataset gsm8k_25 \
  --method shapley \
  --strategy alpha_per_layer \
  --rate 0.8
```

Use `./pruning/run_prune.sh --help` to choose an output directory, device map, or pruning implementation (`zero_weights`, `gate_bias`, or `both`).

#### 7. Serve and evaluate

```bash
# Start a vLLM server for an exported model selected by its experiment metadata.
bash evaluation/vllm-server.sh \
  --model qwen3-30b-a3b \
  --method shapley \
  --dataset gsm8k_25 \
  --rate 0.8 \
  --port 8801 \
  --tp 8

# In another shell, run the configured EvalScope evaluation.
python evaluation/run_evalscope.py
```

Run any wrapper with `--help` to see all options and defaults.

## Pipeline Artifacts

| Stage | Input | Output |
| --- | --- | --- |
| Calibration | Dataset samples | `data/calibration/{dataset}_25.json` |
| Collection | Model + calibration data | `results/{model}/activations/*.json` |
| Scoring | Activation statistics | `results/{model}/shapley_values/*.csv` |
| Selection | Scores + keep rate | `results/{model}/selected_experts/*.json` |
| Export | Model + selected experts | Pruned model directory |
| Evaluation | Served model | EvalScope results |

Activation statistics, distilled training data, adapters, and exported models are generated artifacts and are excluded from Git.

## Supported Models

| Config name | Experts | Experts per token | Family |
| --- | ---: | ---: | --- |
| `qwen3-30b-a3b` | 128 | 8 | Qwen3 MoE |
| `gpt-oss-20b` | 32 | 4 | GPT-OSS MoE |
| `deepseekv2-lite-coder` | 64 | 6 | DeepSeek-V2 MoE |

The configured calibration datasets cover code, mathematics, reasoning, medical QA, NLP, and truthfulness. See [`configs/experiments.yaml`](configs/experiments.yaml) for the authoritative list.

## Methods and Selection Strategies

### Pruning methods

| Method | Signal |
| --- | --- |
| `shapley` | Coalition-aware Shapley-style contribution (SHAPE) |
| `easyep` | Task-aligned expert selection |
| `reap` | Weighted activation norm |
| `gating` | Mean router softmax score |
| `frequency` | Expert activation count |
| `random` | Uniform random selection |

### SHAPE strategies

| Strategy | Behavior |
| --- | --- |
| `alpha_per_layer` | Preserves an alpha fraction of Shapley mass per layer; recommended |
| `alpha_global` | Applies contribution coverage globally |
| `topk_per_layer` | Keeps a fixed number of experts in every layer |
| `topk_global` | Keeps the globally highest-scoring experts |

## Optional: Adaptive LoRA Recovery

The `finetune/` pipeline allocates different LoRA ranks to retained experts according to their Shapley contribution. It supports distilled SFT, an optional DPO stage, matched uniform/random rank baselines, and adapter merging.

Typical entry points:

```bash
# Distill SFT and DPO data from the original teacher model.
TEACHER_MODEL=/path/to/original-model TP=8 NUM_SAMPLES=4 \
  ./data/download_sft.sh gsm8k

# Run the configured {keep rate} x {rank strategy} SFT matrix.
./finetune/run_experiments.sh

# Optionally run DPO after SFT.
./finetune/run_dpo.sh
```

See [`finetune/PLAN.md`](finetune/PLAN.md) and [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for rank-map generation, individual training commands, merging, and experimental details.

## Repository Layout

```text
Shapley-Moe/
├── configs/       # Model paths and experiment definitions
├── data/          # Calibration data and data download/distillation scripts
├── analysis/      # Activation collection and Shapley scoring
├── pruning/       # Expert selection methods and model export
├── evaluation/    # vLLM serving and EvalScope evaluation
├── finetune/      # Optional adaptive-LoRA recovery pipeline
├── results/       # Compact scores, selections, rank maps, and reports
└── assets/        # README figures
```

For a file-by-file map, see [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

## Conventions

- `--rate` is a **keep rate**, not a removal rate: `0.8` keeps 80% and prunes 20%.
- Dataset names include the calibration size, for example `gsm8k_25`.
- The recommended SHAPE selection strategy is `alpha_per_layer`.
- The default calibration size is 25 examples per task.
- Activation collection requires the original unpruned model; scoring and selection can use the published activation snapshot.

## Citation

If you use SHAPE, please cite:

```bibtex
@inproceedings{zhang2026shape,
  title         = {SHAPE: Coalition-Aware Expert Pruning for Sparse Mixture-of-Experts LLMs},
  author        = {Zhang, Yuhao and Jiang, HongXu and Zhang, YiXiang and Zhang, Zheng},
  booktitle     = {Proceedings of the International Joint Conference on Neural Networks},
  year          = {2026},
  eprint        = {2606.09886},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
