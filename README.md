# Shapley-MoE: Expert Pruning Framework for Mixture-of-Experts

## Project Structure

```
shapley-moe/
│
├── data/                               # Datasets
│   ├── calibration/                    # Calibration data (for few-shot)
│   │   ├── arc_easy_25.json
│   │   ├── gsm8k_25.json
│   │   └── ...
│   ├── download_dataset.py
│   └── run_download.sh
│
├── analysis/                           # Analysis scripts
│   ├── collect_activations.py          # Collect activation information (all-in-one)
│   ├── calc_shapley.py                 # Compute Shapley values
│   ├── run_collect.sh                  # Batch activation collection
│   └── run_calc_shapley.sh             # Batch Shapley computation
│
├── pruning/                            # Pruning related
│   ├── methods/                        # Pruning methods
│   │   ├── select_by_shapley.py        # Shapley value pruning (primary method)
│   │   ├── select_by_easyep.py         # EASYEP pruning
│   │   ├── select_by_reap.py          # REAP pruning
│   │   ├── select_by_gating.py         # Gating Score pruning
│   │   ├── select_by_frequency.py      # Activation frequency pruning (baseline)
│   │   └── select_by_random.py         # Random pruning (baseline)
│   ├── save_model.py                   # Save pruned model
│   └── run_select.sh                   # Unified expert selection script
│
├── evaluation/                         # Evaluation scripts
│   ├── run_evalscope.py
│   └── vllm_server.sh
│
├── results/                            # All results (organized by model)
│   ├── {model_name}/                   # e.g. qwen3-30b-a3b/
│   │   ├── activations/                # Activation statistics
│   │   │   ├── {dataset}_shapley.json
│   │   │   ├── {dataset}_gating.json
│   │   │   ├── {dataset}_easyep.json
│   │   │   └── {dataset}_reap.json
│   │   ├── shapley_values/             # Shapley values
│   │   │   └── {dataset}_shapley.csv
│   │   └── selected_experts/           # Selected experts
│   │       ├── shapley_{strategy}_{dataset}_rate{XX}.json
│   │       ├── {method}_{dataset}_rate{XX}.json  # Other methods
│   │       └── ...
│   └── ...
│
├── models/                             # Pruned models
│   └── {model}_{method}_rate{XX}/
│
└── configs/                            # Configuration files
    ├── models.yaml                     # Model configuration
    └── experiments.yaml                # Experiment configuration
```

## Quick Start

### 0. Configuration (recommended to modify first)

All scripts prioritize reading from the `configs/` directory:

```bash
# Modify model paths
vim configs/models.yaml

# Modify experiment parameters (pruning rates, datasets, etc.)
vim configs/experiments.yaml
```

### 1. Data Preparation

```bash
cd data

# Download a single dataset
./run_download.sh gsm8k 25

# Download all datasets from config
./run_download.sh --all
```

### 2. Collect Activation Information

Collect information needed by all pruning methods at once (Shapley/Gating/EASYEP/REAP):

```bash
cd analysis

# Use model name (path auto-read from config)
./run_collect.sh -m qwen3-30b-a3b --all

# Or use full path
./run_collect.sh -m /path/to/model --data ../data/calibration/gsm8k_25.json

# View available models in config
./run_collect.sh --list-models
```

### 3. Compute Shapley Values (optional)

```bash
./run_calc_shapley.sh -m MODEL_NAME
```

### 4. Expert Selection

```bash
cd ../pruning

# Use Shapley pruning, retain 50% (pruning rate read from config)
./run_select.sh -m qwen3-30b-a3b -d gsm8k_25 -M shapley -r 0.5

./run_select.sh -m qwen3-30b-a3b -d gsm8k_25 -M shapley --all-rates

# Batch processing (using all pruning rates from config)
./run_select.sh -m qwen3-30b-a3b --all-datasets -M easyep --all-rates

# Use all methods from config
./run_select.sh -m qwen3-30b-a3b -d gsm8k_25 --all-methods --all-rates
```

### 5. Save Pruned Model

```bash
cd ../pruning

# Specify model, dataset, pruning rate (auto-finds corresponding selection file)
./run_prune.sh -m qwen3-30b-a3b -d gsm8k_25 -r 0.8

# Use other methods
./run_prune.sh -m gpt-oss-20b -d arc_easy_25 -M easyep -r 0.6

# Parameter description
#   -m MODEL     Model name
#   -d DATASET   Dataset name
#   -M METHOD    Pruning method (default: shapley)
#   -s STRATEGY  Shapley strategy (default: alpha_per_layer)
#   -r RATE      Pruning rate (default: 0.8)
```

### 6. Evaluation

```bash
cd ../evaluation

# Start model service (use model name, path auto-read from config)
./vllm-server.sh qwen3-30b-a3b

# Or specify full path
./vllm-server.sh /path/to/model -p 8801

# Run evaluation (datasets read from config)
python run_evalscope.py

# View available datasets
python run_evalscope.py --list-datasets
```

## Pruning Methods

| Method | Description | Formula |
|--------|-------------|---------|
| **Shapley** | Marginal contribution-based pruning (primary method) | Shapley Value |
| **EASYEP** | Considers MoE impact on tokens | `weight × (1 - simibr) × norm` |
| **REAP** | Weighted norm | `weight × norm` |
| **Gating** | Based on router softmax scores | `mean(softmax(gating_logits))` |
| **Frequency** | Based on activation frequency (baseline) | `count(activations)` |
| **Random** | Random selection (baseline) | random |

### Shapley Pruning Strategies

The Shapley method supports four strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `topk_per_layer` | Select top-k experts with highest Shapley values per layer | **Recommended**, simple and straightforward |
| `topk_global` | Select experts with highest Shapley values globally | Some layers can retain more experts |
| `alpha_per_layer` | Cumulative Shapley values reach alpha proportion per layer | Considers contribution distribution |
| `alpha_global` | Cumulative Shapley values reach alpha proportion globally | Global optimization |

**TopK vs Alpha:**
- `topk`: Directly sorted by Shapley value, select top k. Simple, highly interpretable.
- `alpha`: Select minimum experts whose cumulative Shapley value reaches the target proportion of total. Considers contribution distribution.

**Per Layer vs Global:**
- `per_layer`: Independent selection per layer, ensures sufficient experts per layer.
- `global`: Unified global selection, some layers may have fewer experts.

```bash
# Use different strategies
python pruning/methods/select_by_shapley.py \
    --input results/model/shapley_values/gsm8k_25_shapley.csv \
    --output selected.json \
    --pruning_rate 0.5 \
    --strategy topk_per_layer  # or topk_global, alpha_per_layer, alpha_global
```

## Naming Conventions

- **Models**: `qwen3-30b-a3b`, `gpt-oss-20b`, `deepseekv2-lite-coder`
- **Datasets**: `{name}_{samples}` e.g. `gsm8k_25`
- **Pruning methods**: `shapley`, `easyep`, `reap`, `gating`, `frequency`, `random`
- **Shapley strategies**: `alpha_per_layer`, `alpha_global`, `topk_per_layer`, `topk_global`
- **Pruning rate**: `rate0_8`, `rate0_6` (retention ratio, e.g. 0.8 means retain 80%)
- **Selected expert files**:
  - Shapley: `shapley_{strategy}_{dataset}_rate{XX}.json`
  - Other methods: `{method}_{dataset}_rate{XX}.json`

## Configuration

Configuration files are in the `configs/` directory. **All scripts prioritize reading from config files**:

### models.yaml - Model Configuration

```yaml
models:
  qwen3-30b-a3b:
    path: /path/to/qwen3-30b-a3b    # Model path
    num_experts: 128                 # Total number of experts
    num_experts_per_tok: 8           # Number of experts activated per token
    type: qwen3                      # Model type
```

### experiments.yaml - Experiment Configuration

```yaml
# Dataset list
datasets:
  - humaneval_25
  - gsm8k_25
  - ...

# Pruning rates (retention ratio)
pruning_rates:
  - 0.80   # Retain 80%
  - 0.60   # Retain 60%

# Pruning methods
pruning_methods:
  - shapley
  - easyep
  - ...

# Defaults
defaults:
  pruning_rate: 0.5
  pruning_method: shapley
  shapley_strategy: alpha_per_layer
  max_new_tokens: 512
  eval_port: 8801
```

### How Scripts Use Configuration

| Script | Config Read |
|--------|-------------|
| `run_collect.sh` | Model path, max_new_tokens, device |
| `run_select.sh` | Pruning rates, pruning methods, Shapley strategies |
| `run_download.sh` | Dataset list |
| `vllm-server.sh` | Model path, eval_port |
| `run_evalscope.py` | Evaluation datasets, batch_size, timeout |
