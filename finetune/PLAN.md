# 贡献度感知的自适应 LoRA 微调计划

本文档用于规划毕业论文第三部分的实验与代码实现：在 MoE 模型剪枝之后，根据专家贡献度为不同专家分配不同的 LoRA Rank，从而在有限可训练参数量下恢复剪枝后的模型性能。

## 1. 研究目标

前两部分已经完成了专家贡献度评估与专家剪枝。第三部分的目标是进一步利用专家贡献度指导剪枝后微调。

核心假设：

```text
剪枝后的 MoE 模型中，保留下来的专家仍然具有不同的重要性。
在微调参数预算有限的情况下，高贡献专家应该分配更高的 LoRA Rank，
低贡献专家则分配较低的 LoRA Rank。
```

这一部分可以定义为：

```text
面向剪枝 MoE 模型的贡献度感知自适应 LoRA 微调方法
```

需要证明的是：

```text
Shapley 专家贡献度不仅可以用于剪枝阶段的专家选择，
也可以用于剪枝后微调阶段的参数预算分配。
```

## 2. 主要实验范围

主实验模型建议选择：

```text
qwen3-30b-a3b
```

第三部分暂时没有必要覆盖所有模型。对于毕业论文来说，在 Qwen3-MoE 上做完整、可复现、对照充分的实验，比在多个模型上做较浅的实验更有价值。

其他模型，例如：

```text
deepseekv2-lite-coder
gpt-oss-20b
```

可以作为后续扩展实验或未来工作，不作为第三部分的主要实验对象。

## 3. 实验设计

### 3.1 数据集选择

不建议第三部分一开始跑所有数据集。更合适的做法是选择少量具有代表性的任务，覆盖垂直领域能力和通用能力。

推荐核心任务：

```text
垂直领域任务：
- pubmedqa
- med_mcqa
- biomix_qa

通用能力任务：
- gsm8k
- logiqa
- arc_easy
```

如果时间或算力有限，优先选择：

```text
pubmedqa
med_mcqa
gsm8k
logiqa
```

这样既能体现垂直领域任务的重要性，也能观察模型通用推理能力是否被严重影响。

### 3.2 剪枝率设置

建议使用两个剪枝率：

```text
rate0_8：保留 80% 专家
rate0_6：保留 60% 专家
```

其中：

```text
rate0_8 表示轻度剪枝
rate0_6 表示中等程度剪枝
```

第一阶段不建议加入过多剪枝率，否则实验量会迅速膨胀，且论文主线会变得不够集中。

### 3.3 对照方法

主实验建议设置以下对照组：

```text
A. 原始模型
B. Shapley 剪枝模型，不进行微调
C. Shapley 剪枝模型 + 统一 LoRA Rank 微调
D. Shapley 剪枝模型 + 随机 LoRA Rank 分配微调
E. Shapley 剪枝模型 + 基于贡献度的自适应 LoRA Rank 分配微调
```

其中 E 是本文第三部分的核心方法。

需要特别注意公平性：

```text
C、D、E 三组的可训练参数量应该尽量接近。
```

否则实验结果可能被质疑为：

```text
性能提升不是因为 Rank 分配策略更好，而是因为使用了更多 LoRA 参数。
```

### 3.4 LoRA Rank 分配策略

第一版建议采用分桶策略。它实现简单、稳定，并且很适合在论文中解释。

默认分桶策略：

```text
每一层内，根据 Shapley 贡献度对保留专家排序：

Top 20% 专家：        rank 16
20% - 50% 专家：      rank 8
50% - 80% 专家：      rank 4
Bottom 20% 专家：     rank 2
```

也就是说，排序是在每一层内部独立完成的，而不是把所有层的专家混在一起做全局排序。

后续可以增加消融实验：

```text
统一分配：     8 / 8 / 8 / 8
温和自适应：   16 / 8 / 4 / 2
激进自适应：   32 / 16 / 4 / 1
随机分配：     使用同样的 rank 集合，但随机分给保留专家
```

### 3.5 评价指标

任务性能指标：

```text
Accuracy
Exact Match
pass@1
多个任务的平均分
```

资源消耗指标：

```text
可训练参数量
峰值显存占用
训练时间
训练步数
```

性能恢复率指标：

```text
Recovery = (Score_finetuned - Score_pruned) / (Score_original - Score_pruned)
```

这个指标可以直接说明：

```text
微调方法恢复了多少由剪枝造成的性能损失。
```

### 3.6 论文表格设计

主结果表：

```text
方法 | 剪枝率 | 可训练参数量 | pubmedqa | med_mcqa | gsm8k | logiqa | 平均分
```

性能恢复率表：

```text
方法 | rate0_8 恢复率 | rate0_6 恢复率
```

消融实验表：

```text
Rank 分配策略 | 可训练参数量 | 平均性能 | 恢复率
```

## 4. 代码设计

微调相关代码建议放在：

```text
finetune/
```

计划新增文件：

```text
finetune/
├── PLAN.md
├── build_rank_map.py
├── train_adaptive_lora.py
├── merge_lora.py
├── infer_adaptive_lora.py
├── run_build_rank_map.sh
├── run_train.sh
└── run_merge.sh
```

### 4.1 build_rank_map.py

作用：

```text
根据 Shapley 贡献度文件和剪枝后保留专家文件，生成每一层、每个专家对应的 LoRA Rank。
```

输入：

```text
--shapley_csv results/{model}/shapley_values/{dataset}_shapley.csv
--selected_experts results/{model}/selected_experts/{selection_file}.json
--output results/{model}/lora_rank_maps/{rank_map_file}.json
```

输出示例：

```json
{
  "0": {
    "12": 16,
    "45": 8,
    "77": 4,
    "103": 2
  },
  "1": {
    "3": 16,
    "9": 8,
    "21": 4
  },
  "_metadata": {
    "method": "adaptive_lora",
    "rank_strategy": "bucket",
    "rank_buckets": [16, 8, 4, 2],
    "bucket_ratios": [0.2, 0.3, 0.3, 0.2]
  }
}
```

主要流程：

```text
1. 读取 Shapley CSV。
2. 读取 selected_experts JSON。
3. 对每一层，只保留剪枝后仍存在的专家。
4. 按 Shapley_Value 从高到低排序。
5. 根据分桶位置分配 LoRA Rank。
6. 保存 rank_map JSON。
```

### 4.2 train_adaptive_lora.py

作用：

```text
加载剪枝后的 MoE 模型，并根据 rank_map 为不同专家设置不同 LoRA Rank 进行微调。
```

输入：

```text
--model_path models/{pruned_model_dir}
--rank_map results/{model}/lora_rank_maps/{rank_map_file}.json
--train_file data/{train_file}.json
--output_dir models_lora/{adapter_name}
--model_type qwen3
```

主要流程：

```text
1. 加载 tokenizer。
2. 加载剪枝后的模型。
3. 冻结原模型参数。
4. 读取 rank_map。
5. 将 rank_map 转换为 PEFT 使用的 rank_pattern 和 alpha_pattern。
6. 创建 LoraConfig。
7. 使用 get_peft_model 包装模型。
8. 加载并 tokenize 训练数据。
9. 使用 Trainer 或 SFTTrainer 训练。
10. 保存 LoRA adapter 和 tokenizer。
```

第一版只需要支持 Qwen3。

对于 Qwen3，专家模块名预期类似：

```text
model.layers.{layer}.mlp.experts.{expert}.gate_proj
model.layers.{layer}.mlp.experts.{expert}.up_proj
model.layers.{layer}.mlp.experts.{expert}.down_proj
```

代码中应提供函数：

```python
def build_rank_pattern(model_type, rank_map, target_modules):
    ...
```

如果遇到暂不支持的模型类型，第一版可以直接给出明确错误提示。

### 4.3 merge_lora.py

作用：

```text
将训练好的 LoRA adapter 合并回剪枝模型，得到可以被 vLLM 和 EvalScope 直接加载的完整模型。
```

输入：

```text
--base_model models/{pruned_model_dir}
--adapter models_lora/{adapter_name}
--output models/{merged_model_dir}
```

主要流程：

```text
1. 加载剪枝后的基础模型。
2. 使用 PeftModel.from_pretrained 加载 LoRA adapter。
3. 调用 merge_and_unload() 合并参数。
4. 保存合并后的模型和 tokenizer。
```

### 4.4 infer_adaptive_lora.py

作用：

```text
用于正式评测前的小规模推理调试。
```

该脚本不是正式评测入口，而是用于确认：

```text
1. 剪枝模型可以正常加载。
2. LoRA adapter 可以正常加载。
3. 模型可以正常生成。
4. 自适应 LoRA 对输出产生了实际影响。
```

正式评测仍然使用现有流程：

```text
evaluation/vllm-server.sh
evaluation/run_evalscope.py
```

前提是先通过 `merge_lora.py` 得到合并后的模型目录。

## 5. 配置文件设计

后续可以在 `configs/experiments.yaml` 中增加 LoRA 相关配置：

```yaml
lora:
  rank_strategy: bucket
  rank_buckets: [16, 8, 4, 2]
  bucket_ratios: [0.2, 0.3, 0.3, 0.2]
  uniform_rank: 8
  random_seed: 42
  lora_dropout: 0.05
  lora_alpha_scale: 2
  target_modules:
    - gate_proj
    - up_proj
    - down_proj
```

实现时应允许命令行参数覆盖配置文件中的默认值。

## 6. 最小闭环目标

第一阶段不追求覆盖所有任务，而是先跑通一个完整流程：

```text
模型：qwen3-30b-a3b
数据集：gsm8k_25
剪枝率：rate0_8
剪枝方法：shapley + alpha_per_layer
LoRA 策略：基于贡献度的分桶自适应 Rank
```

完整路径：

```text
Shapley CSV
-> selected_experts JSON
-> rank_map JSON
-> adaptive LoRA 训练
-> LoRA 合并
-> vLLM 启动服务
-> EvalScope 评测
```

当这个闭环跑通之后，再扩展到：

```text
1. rate0_6
2. pubmedqa / med_mcqa / biomix_qa / logiqa / arc_easy
3. 统一 LoRA baseline
4. 随机 Rank LoRA baseline
5. Rank 分配策略消融实验
```

## 7. 实现顺序

推荐实现顺序：

```text
1. 实现 build_rank_map.py。
2. 使用一个 Shapley CSV 和 selected_experts JSON 手动检查生成的 rank_map。
3. 实现 Qwen3 版本的 train_adaptive_lora.py。
4. 先用很小的数据跑通一次训练，确认 PEFT 的 rank_pattern 能正常工作。
5. 在同一个训练脚本中支持统一 LoRA baseline。
6. 在 build_rank_map.py 中支持随机 Rank baseline。
7. 实现 merge_lora.py。
8. 将合并后的模型接入现有 vLLM 和 EvalScope 流程。
9. 添加可复现实验 shell 脚本。
10. 执行完整实验并整理结果。
```

## 8. 论文写作思路

第三部分应作为前两部分的自然延伸：

```text
1. Shapley 贡献度可以识别重要专家。
2. 重要专家在剪枝阶段被保留。
3. 剪枝后，保留专家之间仍存在贡献度差异。
4. 因此，微调阶段也应根据贡献度差异分配参数预算。
```

核心对比：

```text
统一 LoRA：对所有保留专家一视同仁。
自适应 LoRA：将更多可训练参数分配给高贡献专家。
```

预期结论：

```text
在相近可训练参数量下，基于专家贡献度的自适应 LoRA
相比统一 Rank 或随机 Rank 分配，可以恢复更多剪枝造成的性能损失。
```

