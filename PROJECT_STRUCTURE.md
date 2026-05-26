# Shapley-MoE Project Structure

## Refactored Directory Structure

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
│   │   ├── select_by_shapley.py        # Shapley value pruning
│   │   ├── select_by_easyep.py         # EASYEP pruning
│   │   ├── select_by_reap.py           # REAP pruning
│   │   ├── select_by_gating.py         # Gating Score pruning
│   │   ├── select_by_frequency.py      # Activation frequency pruning
│   │   └── select_by_random.py         # Random pruning (baseline)
│   ├── save_model.py                   # Save pruned model
│   ├── run_select.sh                   # Batch expert selection (read from config)
│   └── run_prune.sh                    # Batch model pruning (read from config)
│
├── finetune/                           # Post-pruning adaptive LoRA fine-tuning
│   ├── PLAN.md                         # Adaptive LoRA experiment plan
│   ├── build_rank_map.py               # Build expert-wise LoRA rank maps
│   ├── train_adaptive_lora.py          # Train LoRA on retained experts only
│   ├── merge_lora.py                   # Merge LoRA adapters into pruned models
│   └── infer_adaptive_lora.py          # Quick inference with optional LoRA adapter
│
├── evaluation/                         # Evaluation scripts
│   ├── run_evalscope.py
│   ├── run_eval.sh
│   └── vllm_server.sh
│
├── results/                            # All results (organized by model)
│   ├── {model_name}/                   # e.g. qwen3-30b-a3b/
│   │   ├── activations/                # Activation statistics
│   │   │   ├── {dataset}_shapley.json      # For Shapley
│   │   │   ├── {dataset}_gating.json       # For Gating Score
│   │   │   ├── {dataset}_easyep.json       # For EASYEP
│   │   │   └── {dataset}_reap.json         # For REAP
│   │   ├── shapley_values/             # Shapley values
│   │   │   └── {dataset}_shapley.csv
│   │   ├── selected_experts/           # Selected experts
│   │   │   ├── shapley_{strategy}_{dataset}_rate{XX}.json
│   │   │   ├── {method}_{dataset}_rate{XX}.json  # Other methods
│   │   │   └── ...
│   │   ├── lora_rank_maps/             # Expert-wise LoRA rank assignments
│   │   │   └── {dataset}_rate{XX}_{rank_strategy}.json
│   │   └── eval/                       # Evaluation results
│   │       └── {method}_{dataset}_rate{XX}/
│   └── ...
│
├── models/                             # Pruned models
│   └── {model_name}_{method}_rate{XX}/ # e.g. qwen3-30b-a3b_shapley_rate50/
│
├── configs/                            # Configuration files
│   ├── models.yaml                     # Model path configuration
│   └── experiments.yaml                # Experiment configuration
│
└── README.md                           # Project description
```

## Naming Conventions

### Model Names
- `qwen3-30b-a3b`
- `gpt-oss-20b`
- `deepseekv2-lite-coder`

### Dataset Names
- `arc_easy_25` (25 indicates number of samples)
- `gsm8k_25`
- `humaneval_25`
- ...

### Pruning Methods
- `shapley` - Shapley value pruning
- `easyep` - EASYEP pruning
- `reap` - REAP pruning
- `gating` - Gating Score pruning
- `frequency` - Activation frequency pruning
- `random` - Random pruning (baseline)

### Pruning Rate
- `rate0_8` - Retain 80%
- `rate0_6` - Retain 60%

### File Naming
- Activation statistics: `{dataset}_{type}.json`
- Shapley values: `{dataset}_shapley.csv`
- Selected experts:
  - Shapley: `shapley_{strategy}_{dataset}_rate{XX}.json`
  - Other methods: `{method}_{dataset}_rate{XX}.json`
- LoRA rank maps: `{dataset}_rate{XX}_{rank_strategy}.json`
- Pruned models: `{model}_{method}_rate{XX}/`

### Shapley Strategies
- `alpha_per_layer` - Per-layer Alpha coverage (recommended)
- `alpha_global` - Global Alpha coverage
- `topk_per_layer` - Per-layer Top-K
- `topk_global` - Global Top-K

### LoRA Rank Strategies
- `bucket` - Sort retained experts by per-layer Shapley contribution, then assign ranks by buckets.
- `uniform` - Assign the same rank to every retained expert.
- `random` - Use the same bucket sizes and ranks as `bucket`, but randomly assign them to retained experts.

Default adaptive LoRA bucket setting:

```text
Top 20% retained experts: rank 32
Next 40% retained experts: rank 16
Last 40% retained experts: rank 8
```

The default uniform baseline uses rank 16. This keeps the expected average rank of `bucket`, `random`, and `uniform` approximately equal. Rank maps only contain experts that remain after pruning; pruned experts are omitted and therefore receive no LoRA parameters.

## Workflow

```
1. Data Preparation
   └── python data/download_dataset.py

2. Collect Activation Information
   └── bash analysis/run_collect.sh --model MODEL --all

3. Compute Shapley Values (if needed)
   └── bash analysis/run_calc_shapley.sh --model MODEL

4. Expert Selection
   └── bash pruning/run_select.sh --model MODEL --method {shapley|easyep|reap|...} --rate 0.5

5. Model Pruning
   └── bash pruning/run_prune.sh --model MODEL --selection results/.../selected_experts.json

6. Build LoRA Rank Map (optional post-pruning fine-tuning)
   └── python finetune/build_rank_map.py --shapley_csv results/.../gsm8k_25_shapley.csv --selected_experts results/.../selected.json --output results/.../rank_map.json

7. Train Adaptive LoRA (optional)
   └── python finetune/train_adaptive_lora.py --model_path models/pruned_model --rank_map results/.../rank_map.json --train_file data/calibration/gsm8k_25.json --output_dir adapters/run_name

8. Merge LoRA (optional)
   └── python finetune/merge_lora.py --base_model models/pruned_model --adapter adapters/run_name --output models/merged_model

9. Evaluation
   └── bash evaluation/run_eval.sh --model models/pruned_model/
```

## Current Minimal Fine-Tuning Scope

The current adaptive LoRA work focuses on the smallest reproducible loop:

```text
model: qwen3-30b-a3b
dataset: gsm8k_25
keep rates: rate0_8 and rate0_6
rank maps: bucket, uniform, random
```

Generated files currently kept in the repository workspace:

```text
results/qwen3-30b-a3b/shapley_values/gsm8k_25_shapley.csv
results/qwen3-30b-a3b/lora_rank_maps/gsm8k_25_rate0_8_{bucket,uniform,random}.json
results/qwen3-30b-a3b/lora_rank_maps/gsm8k_25_rate0_6_{bucket,uniform,random}.json
```

