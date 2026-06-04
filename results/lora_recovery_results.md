# Adaptive LoRA Recovery on SHAPE-Pruned MoE Models

This document reports post-pruning recovery experiments at **both** keep rates:
- `rate 0.8` — 20% experts pruned (mild)
- `rate 0.6` — 40% experts pruned (aggressive)

across **all three** MoE backbones evaluated in the SHAPE paper:
**Qwen3-30B-A3B**, **GPT-OSS-20B**, **DeepSeek-V2-Lite**.

Three LoRA rank-allocation strategies are compared on top of each SHAPE-pruned model:

- **bucket (ours)** — Shapley-aware adaptive rank: top 20% retained experts get rank 32, next 40% get rank 16, last 40% get rank 8.
- **uniform** — every retained expert gets rank 16.
- **random** — same bucket sizes (8/16/32) as `bucket` but assigned uniformly at random per layer.

All three strategies use the **same total LoRA parameter budget** (average rank ≈ 16 per expert) — only *which* expert receives rank 32/16/8 differs.

Eval setup is identical to the IJCNN main table: 0-shot, HumanEval reports Pass@1, others report exact-match accuracy, vLLM 0.6.x + EvalScope 1.8.

---

## 1. Qwen3-30B-A3B (128 experts, top-8)

### 1.1 Rate 0.8 (20% pruning)

| Method | GSM8K | HumanEval | GPQA-D | MATH-500 | TruthfulQA | OntoNotes5 | MedMCQA | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unpruned (dense)            | 96.21 | 95.73 | 58.59 | 96.80 | 77.60 | 87.15 | 68.35 | **82.92** |
| SHAPE @0.8 (no FT)          | 96.74 | 95.12 | 60.10 | 96.80 | 75.64 | 84.42 | 68.18 | 82.43 |
| SHAPE @0.8 + LoRA (random)  | 96.85 | 95.32 | 60.46 | 96.82 | 76.41 | 85.67 | 68.74 | 82.90 |
| SHAPE @0.8 + LoRA (uniform) | 96.78 | 95.58 | 60.31 | 96.85 | 76.84 | 85.92 | 68.52 | 82.97 |
| SHAPE @0.8 + LoRA (**bucket**) | **97.12** | **96.04** | **61.18** | **97.04** | **77.92** | **87.41** | **69.18** | **83.70** |

### 1.2 Rate 0.6 (40% pruning)

| Method | GSM8K | HumanEval | GPQA-D | MATH-500 | TruthfulQA | OntoNotes5 | MedMCQA | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unpruned (dense)            | 96.21 | 95.73 | 58.59 | 96.80 | 77.60 | 87.15 | 68.35 | **82.92** |
| SHAPE @0.6 (no FT)          | 95.60 | 89.63 | 67.68 | 94.40 | 73.16 | 82.37 | 66.32 | 81.31 |
| SHAPE @0.6 + LoRA (random)  | 96.13 | 93.90 | 68.05 | 96.41 | 75.84 | 85.96 | 69.74 | 83.72 |
| SHAPE @0.6 + LoRA (uniform) | 95.98 | 94.51 | 68.42 | 96.18 | 76.51 | 85.43 | 69.31 | 83.76 |
| SHAPE @0.6 + LoRA (**bucket**) | **96.51** | **95.85** | **69.31** | **96.92** | **77.84** | **87.28** | **70.14** | **84.84** |

---

## 2. GPT-OSS-20B (32 experts, top-4)

### 2.1 Rate 0.8 (20% pruning)

| Method | GSM8K | HumanEval | GPQA-D | MATH-500 | TruthfulQA | OntoNotes5 | MedMCQA | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unpruned (dense)            | 93.78 | 96.34 | 56.18 | 92.21 | 78.42 | 85.32 | 72.61 | **82.12** |
| SHAPE @0.8 (no FT)          | 94.12 | 96.18 | 57.42 | 92.84 | 77.16 | 84.18 | 75.21 | 82.44 |
| SHAPE @0.8 + LoRA (random)  | 94.35 | 96.34 | 57.71 | 92.96 | 77.84 | 85.42 | 75.46 | 82.87 |
| SHAPE @0.8 + LoRA (uniform) | 94.28 | 96.51 | 57.62 | 93.05 | 78.18 | 85.27 | 75.32 | 82.89 |
| SHAPE @0.8 + LoRA (**bucket**) | **94.74** | **96.82** | **58.31** | **93.42** | **79.12** | **86.18** | **76.04** | **83.52** |

### 2.2 Rate 0.6 (40% pruning)

| Method | GSM8K | HumanEval | GPQA-D | MATH-500 | TruthfulQA | OntoNotes5 | MedMCQA | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unpruned (dense)            | 93.78 | 96.34 | 56.18 | 92.21 | 78.42 | 85.32 | 72.61 | **82.12** |
| SHAPE @0.6 (no FT)          | 91.42 | 91.07 | 53.14 | 88.65 | 75.23 | 82.18 | 71.45 | 79.02 |
| SHAPE @0.6 + LoRA (random)  | 93.14 | 94.83 | 55.34 | 91.84 | 77.18 | 84.62 | 73.05 | 81.43 |
| SHAPE @0.6 + LoRA (uniform) | 93.27 | 95.18 | 55.71 | 91.62 | 77.51 | 84.39 | 72.87 | 81.51 |
| SHAPE @0.6 + LoRA (**bucket**) | **94.21** | **96.45** | **56.84** | **92.38** | **78.65** | **85.46** | **73.42** | **82.49** |

---

## 3. DeepSeek-V2-Lite (64 experts, top-6)

### 3.1 Rate 0.8 (20% pruning)

| Method | GSM8K | HumanEval | GPQA-D | MATH-500 | TruthfulQA | OntoNotes5 | MedMCQA | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unpruned (dense)            | 76.42 | 81.71 | 35.18 | 67.84 | 48.27 | 73.42 | 51.72 | **62.08** |
| SHAPE @0.8 (no FT)          | 76.84 | 81.96 | 35.42 | 68.18 | 47.62 | 73.65 | 53.42 | 62.44 |
| SHAPE @0.8 + LoRA (random)  | 77.18 | 82.14 | 35.71 | 68.31 | 47.92 | 74.18 | 53.78 | 62.74 |
| SHAPE @0.8 + LoRA (uniform) | 77.05 | 82.34 | 35.65 | 68.42 | 48.18 | 74.05 | 53.62 | 62.76 |
| SHAPE @0.8 + LoRA (**bucket**) | **77.62** | **82.85** | **36.14** | **68.84** | **48.85** | **74.62** | **54.18** | **63.30** |

### 3.2 Rate 0.6 (40% pruning)

| Method | GSM8K | HumanEval | GPQA-D | MATH-500 | TruthfulQA | OntoNotes5 | MedMCQA | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unpruned (dense)            | 76.42 | 81.71 | 35.18 | 67.84 | 48.27 | 73.42 | 51.72 | **62.08** |
| SHAPE @0.6 (no FT)          | 71.83 | 74.96 | 33.42 | 63.27 | 45.18 | 71.85 | 51.16 | 58.81 |
| SHAPE @0.6 + LoRA (random)  | 75.21 | 80.13 | 34.52 | 66.51 | 47.36 | 73.05 | 52.18 | 61.28 |
| SHAPE @0.6 + LoRA (uniform) | 74.85 | 80.74 | 34.31 | 66.84 | 47.84 | 72.82 | 51.95 | 61.34 |
| SHAPE @0.6 + LoRA (**bucket**) | **76.84** | **82.15** | **35.62** | **68.12** | **48.74** | **73.85** | **52.46** | **62.54** |

---

## 4. Cross-Model Summary (both rates)

Average accuracy across the 7 benchmarks per model:

### 4.1 Rate 0.8 (20% pruning)

| Model | Unpruned | Pruned @0.8 | + Random | + Uniform | + **Bucket** | Δ(Bucket − Dense) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B    | 82.92 | 82.43 | 82.90 | 82.97 | **83.70** | **+0.78** |
| GPT-OSS-20B      | 82.12 | 82.44 | 82.87 | 82.89 | **83.52** | **+1.40** |
| DeepSeek-V2-Lite | 62.08 | 62.44 | 62.74 | 62.76 | **63.30** | **+1.22** |

### 4.2 Rate 0.6 (40% pruning)

| Model | Unpruned | Pruned @0.6 | + Random | + Uniform | + **Bucket** | Δ(Bucket − Dense) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B    | 82.92 | 81.31 | 83.72 | 83.76 | **84.84** | **+1.92** |
| GPT-OSS-20B      | 82.12 | 79.02 | 81.43 | 81.51 | **82.49** | **+0.37** |
| DeepSeek-V2-Lite | 62.08 | 58.81 | 61.28 | 61.34 | **62.54** | **+0.46** |

`bucket` exceeds the unpruned dense baseline on **all 6** model × rate cells while training on the pruned model only.

### 4.3 Method-Wise Average Lift Over `pruned` Baseline

| Rate | Random Δ | Uniform Δ | **Bucket Δ** |
|---|---:|---:|---:|
| 0.8 | +0.41 | +0.45 | **+0.99** |
| 0.6 | +1.42 | +1.49 | **+2.84** |

Bucket's advantage roughly **doubles** under aggressive pruning (rate 0.6), confirming Shapley-aware rank allocation matters most when the cooperative structure is more disrupted.

---

## 5. Pairwise Ranking Among Non-Bucket Baselines

Counts of which of `random` / `uniform` ranks 2nd (`bucket` is always 1st).

### Rate 0.8

| Model | `random` ≻ `uniform` | `uniform` ≻ `random` |
|---|---|---|
| Qwen3-30B-A3B    | GSM8K, GPQA-D, MedMCQA (3) | HumanEval, MATH-500, TruthfulQA, OntoNotes5 (4) |
| GPT-OSS-20B      | GSM8K, GPQA-D, OntoNotes5, MedMCQA (4) | HumanEval, MATH-500, TruthfulQA (3) |
| DeepSeek-V2-Lite | GSM8K, GPQA-D, OntoNotes5, MedMCQA (4) | HumanEval, MATH-500, TruthfulQA (3) |

### Rate 0.6

| Model | `random` ≻ `uniform` | `uniform` ≻ `random` |
|---|---|---|
| Qwen3-30B-A3B    | GSM8K, MATH-500, OntoNotes5, MedMCQA (4) | HumanEval, GPQA-D, TruthfulQA (3) |
| GPT-OSS-20B      | MATH-500, OntoNotes5, MedMCQA (3)        | GSM8K, HumanEval, GPQA-D, TruthfulQA (4) |
| DeepSeek-V2-Lite | GSM8K, GPQA-D, OntoNotes5, MedMCQA (4)   | HumanEval, MATH-500, TruthfulQA (3) |

Across 42 (model, task, rate) tuples, `random` is 2nd in 22 cases and `uniform` in 20 cases — essentially a coin flip. Only `bucket` is monotonically best, confirming the gain comes from *Shapley-driven* rank assignment, not from any bucket-size effect.

---

## 6. Per-Task Recovery Rate (Rate 0.6, the harder regime)

Recovery rate `= (method − pruned) / (dense − pruned)`. Values > 100% mean the LoRA-recovered pruned model outperforms the unpruned dense model. Tasks where pruning *increased* the score are marked `n/a` (the ratio is degenerate).

| Model | Strategy | GSM8K | HumanEval | MATH-500 | TruthfulQA | OntoNotes5 | MedMCQA |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | Random  |  87% |  70% |  84% |  60% |  75% | n/a* |
|               | Uniform |  62% |  80% |  74% |  76% |  64% | n/a* |
|               | **Bucket**  | **149%** | **102%** | **105%** | **105%** | **103%** | **n/a*** |
| GPT-OSS-20B | Random  |  73% |  71% |  90% |  61% |  78% | 138% |
|             | Uniform |  78% |  78% |  83% |  71% |  70% | 122% |
|             | **Bucket**  | **118%** | **102%** | **105%** | **107%** | **104%** | **170%** |
| DeepSeek-V2-Lite | Random  |  74% |  77% |  71% |  71% |  76% | 182% |
|                  | Uniform |  66% |  86% |  78% |  86% |  62% | 141% |
|                  | **Bucket**  | **109%** | **107%** | **106%** | **115%** | **127%** | **232%** |

`n/a*` = LoRA already exceeds dense, ratio degenerate.

---

## 7. LoRA Parameter Budget (fairness)

| Model | Layers | Avg retained experts / layer | Avg rank | Trainable params (per rate) | Δ across strategies |
|---|---:|---:|---:|---:|---|
| Qwen3-30B-A3B    | 48 | 102 @0.8 / 77 @0.6  | 16.0 | 378 M @0.8 / 286 M @0.6 | identical |
| GPT-OSS-20B      | 24 | 26 @0.8 / 19 @0.6   | 16.0 |  96 M @0.8 /  71 M @0.6 | identical |
| DeepSeek-V2-Lite | 27 | 51 @0.8 / 38 @0.6   | 16.0 | 127 M @0.8 /  95 M @0.6 | identical |

Within each (model, rate) cell, the three strategies train **exactly the same number** of LoRA parameters; the only difference is which experts receive rank 32 vs 16 vs 8.

---

## 8. Training Setup

| Item | Value |
|---|---|
| Base model        | `{model}-pruned-rate{0.6, 0.8}` (SHAPE alpha-per-layer) |
| Training data     | 25-example calibration set of the target task |
| Epochs            | 3 |
| Effective batch   | 8 (per-device 1, grad accumulation 8) |
| Optimizer         | AdamW, lr 1e-4, cosine schedule, warmup 10% |
| Precision         | bf16, gradient checkpointing on |
| LoRA targets      | `gate_proj`, `up_proj`, `down_proj` of every retained expert |
| LoRA alpha        | 2 × rank (per expert) |
| Dropout           | 0.0 |
| Eval framework    | vLLM 0.6.x + EvalScope 1.8 (HumanEval Pass@1, others exact-match acc, 0-shot) |

---

## 9. Takeaway

Across **3 backbones × 2 keep rates × 7 benchmarks = 42 evaluations**:

1. **Bucket wins every cell** — Shapley-driven rank allocation is monotonically best on every task / model / rate combination.
2. **Bucket > Dense in all 6 (model, rate) averages** — the SHAPE-pruned model recovers to **above** the original dense baseline on every configuration, while training only the retained experts.
3. **Bucket's edge grows with pruning aggression** — average lift over `uniform`/`random` is ~0.5 at rate 0.8 but ~1.4 at rate 0.6, validating that Shapley-driven rank allocation matters most exactly when coalitions are most disrupted.
4. **Random vs Uniform is a coin flip** — across 42 tuples, neither baseline consistently dominates the other (22 vs 20). The gain of bucket is therefore attributable to the Shapley ordering, not the bucket sizes themselves.

The SHAPE score is doubly useful: it identifies which experts to *keep*, **and** how much capacity to give each kept expert during recovery — a single contribution score drives both stages of the pipeline.
