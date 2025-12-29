# Shapley-MoE: MoE 专家剪枝框架

## 📁 项目结构

```
shapley-moe/
│
├── data/                               # 数据集
│   ├── calibration/                    # 校准数据（few-shot 用）
│   │   ├── arc_easy_25.json
│   │   ├── gsm8k_25.json
│   │   └── ...
│   ├── download_dataset.py
│   └── run_download.sh
│
├── analysis/                           # 分析脚本
│   ├── collect_activations.py          # 收集激活信息（一体化）
│   ├── calc_shapley.py                 # 计算 Shapley 值
│   ├── run_collect.sh                  # 批量收集激活
│   └── run_calc_shapley.sh             # 批量计算 Shapley
│
├── pruning/                            # 剪枝相关
│   ├── methods/                        # 剪枝方法
│   │   ├── select_by_shapley.py        # Shapley 值剪枝（主方法）
│   │   ├── select_by_easyep.py         # EASYEP 剪枝
│   │   ├── select_by_reap.py           # REAP 剪枝
│   │   ├── select_by_gating.py         # Gating Score 剪枝
│   │   ├── select_by_frequency.py      # 激活频率剪枝（基线）
│   │   └── select_by_random.py         # 随机剪枝（基线）
│   ├── save_model.py                   # 保存剪枝后模型
│   └── run_select.sh                   # 统一的专家选择脚本
│
├── evaluation/                         # 评测脚本
│   ├── run_evalscope.py
│   └── vllm_server.sh
│
├── results/                            # 所有结果（按模型组织）
│   ├── {model_name}/                   # 如 qwen3-30b-a3b/
│   │   ├── activations/                # 激活统计
│   │   │   ├── {dataset}_shapley.json
│   │   │   ├── {dataset}_gating.json
│   │   │   ├── {dataset}_easyep.json
│   │   │   └── {dataset}_reap.json
│   │   ├── shapley_values/             # Shapley 值
│   │   │   └── {dataset}_shapley.csv
│   │   └── selected_experts/           # 选中的专家
│   │       ├── {method}_{dataset}_rate{XX}.json
│   │       └── ...
│   └── ...
│
├── models/                             # 剪枝后的模型
│   └── {model}_{method}_rate{XX}/
│
└── configs/                            # 配置文件
    ├── models.yaml                     # 模型配置
    └── experiments.yaml                # 实验配置
```

## 🚀 快速开始

### 1. 数据准备

```bash
cd data
./run_download.sh
```

### 2. 收集激活信息

一次收集所有剪枝方法需要的信息（Shapley/Gating/EASYEP/REAP）：

```bash
cd analysis

# 处理单个数据集
./run_collect.sh --model /path/to/model --data ../data/calibration/gsm8k_25.json

# 批量处理所有数据集
./run_collect.sh --model /path/to/model --all
```

### 3. 计算 Shapley 值（可选）

```bash
./run_calc_shapley.sh --model MODEL_NAME
```

### 4. 专家选择

```bash
cd ../pruning

# 使用 Shapley 剪枝，保留 50%
./run_select.sh -m qwen3-30b-a3b -d gsm8k_25 -M shapley -r 0.5

# 批量处理
./run_select.sh -m qwen3-30b-a3b --all-datasets -M easyep --all-rates
```

### 5. 保存剪枝模型

```bash
python save_model.py \
    --model /path/to/original/model \
    --selection ../results/{model}/selected_experts/{method}_{dataset}_rate50.json \
    --output ../models/{model}_{method}_rate50
```

### 6. 评测

```bash
cd ../evaluation
./vllm_server.sh  # 启动模型服务
python run_evalscope.py
```

## 📊 剪枝方法

| 方法 | 描述 | 公式 |
|------|------|------|
| **Shapley** | 基于边际贡献的剪枝（主方法） | Shapley Value |
| **EASYEP** | 考虑 MoE 对 token 影响程度 | `weight × (1 - simibr) × norm` |
| **REAP** | 加权范数 | `weight × norm` |
| **Gating** | 基于 router softmax 分数 | `mean(softmax(gating_logits))` |
| **Frequency** | 基于激活频率（基线） | `count(activations)` |
| **Random** | 随机选择（基线） | random |

### Shapley 剪枝策略

Shapley 方法支持四种策略：

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `topk_per_layer` | 每层选择 Shapley 值最高的 top-k 专家 | **推荐**，简单直接 |
| `topk_global` | 全局选择 Shapley 值最高的专家 | 某些层可保留更多专家 |
| `alpha_per_layer` | 每层累积 Shapley 值达到 alpha 比例 | 考虑贡献分布 |
| `alpha_global` | 全局累积 Shapley 值达到 alpha 比例 | 全局优化 |

**TopK vs Alpha:**
- `topk`: 直接按 Shapley 值大小排序，选择前 k 个。简单、可解释性强。
- `alpha`: 选择累积 Shapley 值达到总量 alpha 比例的最少专家。考虑了贡献分布。

**Per Layer vs Global:**
- `per_layer`: 每层独立选择，保证每层都有足够专家。
- `global`: 全局统一选择，某些层可能专家较少。

```bash
# 使用不同策略
python pruning/methods/select_by_shapley.py \
    --input results/model/shapley_values/gsm8k_25_shapley.csv \
    --output selected.json \
    --pruning_rate 0.5 \
    --strategy topk_per_layer  # 或 topk_global, alpha_per_layer, alpha_global
```

## 📝 命名规范

- **模型**: `qwen3-30b-a3b`, `gpt-oss-20b`, `deepseekv2-lite-coder`
- **数据集**: `{name}_{samples}` 如 `gsm8k_25`
- **剪枝方法**: `shapley`, `easyep`, `reap`, `gating`, `frequency`, `random`
- **剪枝率**: `rate25`, `rate50`, `rate75` (保留比例)

## 🔧 配置

配置文件位于 `configs/` 目录：

- `models.yaml`: 模型路径和专家数量
- `experiments.yaml`: 实验配置（数据集、剪枝率等）

