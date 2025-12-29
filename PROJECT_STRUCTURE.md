# Shapley-MoE 项目结构

## 重构后的目录结构

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
│   │   ├── select_by_shapley.py        # Shapley 值剪枝
│   │   ├── select_by_easyep.py         # EASYEP 剪枝
│   │   ├── select_by_reap.py           # REAP 剪枝
│   │   ├── select_by_gating.py         # Gating Score 剪枝
│   │   ├── select_by_frequency.py      # 激活频率剪枝
│   │   └── select_by_random.py         # 随机剪枝（基线）
│   ├── save_model.py                   # 保存剪枝后模型
│   ├── run_select.sh                   # 批量专家选择
│   └── run_prune.sh                    # 批量模型剪枝
│
├── evaluation/                         # 评测脚本
│   ├── run_evalscope.py
│   ├── run_eval.sh
│   └── vllm_server.sh
│
├── results/                            # 所有结果（按模型组织）
│   ├── {model_name}/                   # 如 qwen3-30b-a3b/
│   │   ├── activations/                # 激活统计
│   │   │   ├── {dataset}_shapley.json      # Shapley 用
│   │   │   ├── {dataset}_gating.json       # Gating Score 用
│   │   │   ├── {dataset}_easyep.json       # EASYEP 用
│   │   │   └── {dataset}_reap.json         # REAP 用
│   │   ├── shapley_values/             # Shapley 值
│   │   │   └── {dataset}_shapley.csv
│   │   ├── selected_experts/           # 选中的专家
│   │   │   ├── shapley_{strategy}_{dataset}_rate{XX}.json
│   │   │   ├── {method}_{dataset}_rate{XX}.json  # 其他方法
│   │   │   └── ...
│   │   └── eval/                       # 评测结果
│   │       └── {method}_{dataset}_rate{XX}/
│   └── ...
│
├── models/                             # 剪枝后的模型
│   └── {model_name}_{method}_rate{XX}/ # 如 qwen3-30b-a3b_shapley_rate50/
│
├── configs/                            # 配置文件
│   ├── models.yaml                     # 模型路径配置
│   └── experiments.yaml                # 实验配置
│
└── README.md                           # 项目说明
```

## 命名规范

### 模型名称
- `qwen3-30b-a3b`
- `gpt-oss-20b`
- `deepseekv2-lite-coder`

### 数据集名称
- `arc_easy_25` (25 表示样本数)
- `gsm8k_25`
- `humaneval_25`
- ...

### 剪枝方法
- `shapley` - Shapley 值剪枝
- `easyep` - EASYEP 剪枝
- `reap` - REAP 剪枝
- `gating` - Gating Score 剪枝
- `frequency` - 激活频率剪枝
- `random` - 随机剪枝（基线）

### 剪枝率
- `rate0_8` - 保留 80%
- `rate0_6` - 保留 60%

### 文件命名
- 激活统计: `{dataset}_{type}.json`
- Shapley 值: `{dataset}_shapley.csv`
- 选中专家:
  - Shapley: `shapley_{strategy}_{dataset}_rate{XX}.json`
  - 其他方法: `{method}_{dataset}_rate{XX}.json`
- 剪枝模型: `{model}_{method}_rate{XX}/`

### Shapley 策略
- `alpha_per_layer` - 每层 Alpha 覆盖（推荐）
- `alpha_global` - 全局 Alpha 覆盖
- `topk_per_layer` - 每层 Top-K
- `topk_global` - 全局 Top-K

## 工作流程

```
1. 数据准备
   └── python data/download_dataset.py

2. 收集激活信息
   └── bash analysis/run_collect.sh --model MODEL --all

3. 计算 Shapley 值（如需）
   └── bash analysis/run_calc_shapley.sh --model MODEL

4. 专家选择
   └── bash pruning/run_select.sh --model MODEL --method {shapley|easyep|reap|...} --rate 0.5

5. 模型剪枝
   └── bash pruning/run_prune.sh --model MODEL --selection results/.../selected_experts.json

6. 评测
   └── bash evaluation/run_eval.sh --model models/pruned_model/
```

