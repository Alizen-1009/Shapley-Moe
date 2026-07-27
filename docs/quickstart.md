# SHAPE Quick Start

This guide covers installation, reproducing expert selection from published activation data, and running the full pruning pipeline. Run all commands from the repository root unless noted otherwise.

## Installation

### Requirements

- Python 3.12 (the provided environment was captured with Python 3.12)
- Linux and a CUDA-capable GPU for model loading, activation collection, export, and evaluation
- Local Hugging Face checkpoints for models you want to prune

```bash
git clone git@github.com:Alizen-1009/Shapley-Moe.git
cd Shapley-Moe

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` and `vllm` wheels are CUDA-specific. If the pinned versions do not match your CUDA installation, install compatible wheels first and then install the remaining dependencies. Use `requirements-lock.txt` when you need the exact captured environment.

## Option A: Use Published Activations

This path skips model inference. It downloads the published activation snapshot (~34 MB compressed, ~228 MB extracted), recomputes Shapley values, and selects experts.

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

Downloaded activations are ignored by Git. Compact derived artifacts such as Shapley CSV files and selected-expert JSON files remain versioned in `results/`.

## Option B: Run the Full Pipeline

### 1. Configure model paths

Edit [`configs/models.yaml`](../configs/models.yaml) so each model points to a local checkpoint. Datasets, methods, keep rates, and evaluation defaults are defined in [`configs/experiments.yaml`](../configs/experiments.yaml).

```yaml
models:
  qwen3-30b-a3b:
    path: /path/to/Qwen3-30B-A3B
    num_experts: 128
    num_experts_per_tok: 8
    type: qwen3
```

### 2. Prepare calibration data

The repository includes the 25-example calibration sets used in the experiments. To regenerate one dataset or all configured datasets:

```bash
./data/run_download.sh gsm8k 25
./data/run_download.sh --all
```

### 3. Collect routing and activation statistics

```bash
# Run every calibration dataset for a configured model.
./analysis/run_collect.sh --model qwen3-30b-a3b --all

# Or use an explicit checkpoint and one calibration file.
./analysis/run_collect.sh \
  --model /path/to/model \
  --data data/calibration/gsm8k_25.json
```

Output: `results/{model}/activations/`

### 4. Compute Shapley values

```bash
# One dataset.
./analysis/run_calc_shapley.sh \
  --model qwen3-30b-a3b \
  --dataset gsm8k_25

# Every available activation file for the model.
./analysis/run_calc_shapley.sh --model qwen3-30b-a3b --all
```

Output: `results/{model}/shapley_values/`

### 5. Select experts

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

### 6. Export a pruned model

```bash
./pruning/run_prune.sh \
  --model qwen3-30b-a3b \
  --dataset gsm8k_25 \
  --method shapley \
  --strategy alpha_per_layer \
  --rate 0.8
```

Use `./pruning/run_prune.sh --help` to choose an output directory, device map, or pruning implementation (`zero_weights`, `gate_bias`, or `both`).

### 7. Serve and evaluate

```bash
# Start a vLLM server for the exported experiment.
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

## Optional: Adaptive LoRA Recovery

The `finetune/` pipeline allocates different LoRA ranks to retained experts according to their Shapley contribution. It supports distilled SFT, an optional DPO stage, matched uniform/random rank baselines, and adapter merging.

```bash
# Distill SFT and DPO data from the original teacher model.
TEACHER_MODEL=/path/to/original-model TP=8 NUM_SAMPLES=4 \
  ./data/download_sft.sh gsm8k

# Run the configured {keep rate} x {rank strategy} SFT matrix.
./finetune/run_experiments.sh

# Optionally run DPO after SFT.
./finetune/run_dpo.sh
```

See [`finetune/PLAN.md`](../finetune/PLAN.md) and [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md) for rank-map generation, individual training commands, merging, and experimental details.

## Conventions

- `--rate` is a **keep rate**, not a removal rate: `0.8` keeps 80% and prunes 20%.
- Dataset names include the calibration size, for example `gsm8k_25`.
- The recommended SHAPE strategy is `alpha_per_layer`.
- The default calibration size is 25 examples per task.
- Activation collection requires the original unpruned model; scoring and selection can use the published activation snapshot.
