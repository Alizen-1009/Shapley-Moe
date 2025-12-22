# 数据集下载工具

用于下载 HuggingFace 数据集并提取样本，用于 few-shot 专家剪枝。

## 🚀 快速开始

### 方式 1：使用 Shell 脚本（推荐）

```bash
# 下载 GSM8K 前 25 条（默认）
bash download.sh

# 下载 GSM8K 前 50 条
bash download.sh gsm8k 50

# 下载 HellaSwag 前 100 条
bash download.sh hellaswag 100

# 下载 GSM8K 前 30 条（包含答案）
bash download.sh gsm8k 30 --with-answers

# 列出所有可用数据集
bash download.sh --list
```

### 方式 2：使用 Python 脚本

```bash
# 基础用法
python download_dataset.py --dataset gsm8k --num_samples 25

# 包含答案
python download_dataset.py --dataset gsm8k --num_samples 50 --with_answers

# 自定义输出文件名
python download_dataset.py --dataset hellaswag --num_samples 100 --output my_data.json

# 列出可用数据集
python download_dataset.py --list
```

---

## 📊 支持的数据集

| 数据集 | 名称 | 说明 | 样本数 |
|--------|------|------|--------|
| GSM8K | `gsm8k` | 小学数学应用题 | ~7.5K |
| HellaSwag | `hellaswag` | 常识推理 | ~39K |
| ARC Challenge | `arc_challenge` | AI2 推理挑战 | ~1.1K |
| MMLU | `mmlu` | 多学科知识理解 | ~14K |
| TruthfulQA | `truthfulqa` | 真实性问答 | ~817 |

---

## 🔧 参数说明

### Shell 脚本参数

```bash
bash download.sh [数据集名称] [样本数量] [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 数据集名称 | 要下载的数据集 | `gsm8k` |
| 样本数量 | 提取的样本数 | `25` |
| `--with-answers` | 包含答案 | - |
| `--list` | 列出可用数据集 | - |
| `--help` | 显示帮助 | - |

### Python 脚本参数

```bash
python download_dataset.py [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | 必需 | 数据集名称 |
| `--num_samples` | int | 25 | 样本数量 |
| `--output` | str | 自动 | 输出文件名 |
| `--with_answers` | flag | False | 包含答案 |
| `--dataset_path` | str | - | 自定义 HF 路径 |
| `--dataset_config` | str | - | 数据集配置 |
| `--split` | str | train | 数据集分割 |
| `--text_field` | str | - | 文本字段名 |
| `--answer_field` | str | - | 答案字段名 |
| `--list` | flag | - | 列出数据集 |

---

## 💡 使用示例

### 示例 1：下载不同数量的样本

```bash
# 10 条样本（快速测试）
bash download.sh gsm8k 10

# 25 条样本（默认，用于 few-shot）
bash download.sh gsm8k 25

# 100 条样本（大规模剪枝）
bash download.sh gsm8k 100
```

### 示例 2：下载不同数据集

```bash
# 数学推理
bash download.sh gsm8k 25

# 常识推理
bash download.sh hellaswag 50

# 知识问答
bash download.sh mmlu 30

# 多个数据集
bash download.sh gsm8k 25
bash download.sh hellaswag 25
bash download.sh arc_challenge 25
```

### 示例 3：包含答案（用于 few-shot 学习）

```bash
# 下载问题和答案
bash download.sh gsm8k 25 --with-answers

# 生成的文件会包含完整的问题+答案格式
```

### 示例 4：自定义数据集

```bash
# 使用自定义 HuggingFace 数据集
python download_dataset.py \
  --dataset my_dataset \
  --dataset_path "username/dataset-name" \
  --num_samples 50 \
  --text_field "input" \
  --answer_field "output"
```

---

## 📁 输出格式

### 仅问题格式

```json
[
    {
        "text": "Natalia sold clips to 48 of her friends in April..."
    },
    {
        "text": "Weng earns $12 an hour for babysitting..."
    }
]
```

### 问题+答案格式（`--with-answers`）

```json
[
    {
        "text": "Natalia sold clips...\nNatalia sold 48/2 = <<48/2=24>>24...",
        "question": "Natalia sold clips...",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24..."
    }
]
```

---

## 🌐 网络问题解决

如果遇到网络连接问题，脚本已自动配置使用国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或手动设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash download.sh gsm8k 25
```

---

## 🔍 查看可用数据集

```bash
# 列出所有预定义数据集
bash download.sh --list

# 或
python download_dataset.py --list
```

输出示例：
```
可用的预定义数据集:
======================================================================

gsm8k:
  路径: openai/gsm8k
  配置: main
  分割: train
  文本字段: question

hellaswag:
  路径: Rowan/hellaswag
  配置: None
  分割: train
  文本字段: ctx
...
```

---

## 📝 文件说明

| 文件 | 说明 |
|------|------|
| `download_dataset.py` | Python 下载脚本（通用） |
| `download.sh` | Shell 便捷脚本 |
| `download_gsm8k.py` | 旧版 GSM8K 专用脚本（已废弃） |
| `download_gsm8k.sh` | 旧版 Shell 脚本（已废弃） |
| `gsm8k_25_samples.json` | GSM8K 25 条样本（仅问题） |
| `test.json` | 测试文件 |

---

## ⚙️ 添加新数据集

在 `download_dataset.py` 中添加配置：

```python
DATASET_CONFIGS = {
    "your_dataset": {
        "path": "username/dataset-name",
        "config": "config_name",  # 可选
        "split": "train",
        "text_field": "question",
        "answer_field": "answer",
    },
    # ...
}
```

---

## 🎯 用于 few-shot 剪枝

下载的数据集可以直接用于专家剪枝：

```bash
# 1. 下载数据集
bash download.sh gsm8k 25

# 2. 使用数据集进行剪枝
python ../analysis/analyze_experts_hook.py \
  --model_path /path/to/model \
  --data_path dateset/gsm8k_25.json \
  --num_samples 25
```

