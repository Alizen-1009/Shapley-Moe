#!/usr/bin/env python3
"""
通用数据集下载工具 - 支持下载任意 HuggingFace 数据集并提取样本用于 few-shot 剪枝
"""

import json
import os
import argparse
from datasets import load_dataset


# 预定义的数据集配置
DATASET_CONFIGS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "text_field": "question",
        "answer_field": "answer",
    },
    "humaneval": {
        "path": "openai_humaneval",
        "config": None,
        "split": "test",
        "text_field": "prompt",
        "answer_field": "canonical_solution",
    },


    "truthful_qa": {
        "path": "truthfulqa/truthful_qa",
        "config": "generation",
        "split": "validation",
        "text_field": "question",
        "answer_field": "best_answer",
    },
    # 新增数据集
    "math_500": {
        "path": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "text_field": "problem",
        "answer_field": "answer",
    },
    "med_mcqa": {
        "path": "openlifescienceai/medmcqa",
        "config": None,
        "split": "train",
        "text_field": "question",
        "answer_field": "cop",  # correct option (1-4)
    },
    "gpqa_diamond": {
        "path": "hendrydong/gpqa_diamond",  # 公开版本，无需授权
        "config": None,
        "split": "test",
        "text_field": "problem",
        "answer_field": "solution",
    },
    "ontonotes5": {
        "path": "SpeedOfMagic/ontonotes_english",  # 可用的公开版本
        "config": None,
        "split": "train",
        "text_field": "tokens",  # 返回 token 列表
        "answer_field": "ner_tags",
    },
    # 新增数据集 - 第二批
    "logiqa": {
        "path": "dmayhem93/agieval-logiqa-en",
        "config": None,
        "split": "test",
        "text_field": "query",
        "answer_field": "gold",
    },
    "aime24": {
        "path": "Maxwell-Jia/AIME_2024",
        "config": None,
        "split": "train",
        "text_field": "Problem",
        "answer_field": "Answer",
    },
    "aime25": {
        "path": "opencompass/AIME2025",
        "config": "AIME2025-I",  # 可选: AIME2025-I 或 AIME2025-II
        "split": "test",
        "text_field": "question",
        "answer_field": "answer",
    },
    "biomix_qa": {
        "path": "kg-rag/BiomixQA",
        "config": "mcq",  # 可选: mcq 或 true_false
        "split": "train",
        "text_field": "text",
        "answer_field": "correct_answer",
    },
    "pubmedqa": {
        "path": "qiaojin/PubMedQA",
        "config": "pqa_labeled",
        "split": "train",
        "text_field": "question",
        "answer_field": "final_decision",
    },
    # ARC 数据集 (AI2 Reasoning Challenge) - 评测使用 test 划分
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "text_field": "question",
        "answer_field": "answerKey",
    },
    "arc_easy": {
        "path": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
        "text_field": "question",
        "answer_field": "answerKey",
    },
   
}


def download_and_extract(
    dataset_name: str,
    num_samples: int = 25,
    output_file: str = None,
    with_answers: bool = False,
    dataset_path: str = None,
    dataset_config: str = None,
    split: str = "train",
    text_field: str = None,
    answer_field: str = None,
):
    """
    下载数据集并提取样本

    Args:
        dataset_name: 数据集名称（预定义）或自定义路径
        num_samples: 要提取的样本数量
        output_file: 输出文件名（默认自动生成）
        with_answers: 是否包含答案
        dataset_path: 自定义数据集路径（覆盖预定义）
        dataset_config: 数据集配置名称
        split: 数据集分割（train/test/validation）
        text_field: 文本字段名称
        answer_field: 答案字段名称
    """

    # 如果是预定义的数据集，使用配置
    if dataset_name in DATASET_CONFIGS and not dataset_path:
        config = DATASET_CONFIGS[dataset_name]
        dataset_path = config["path"]
        dataset_config = config.get("config")
        split = config["split"]
        text_field = text_field or config["text_field"]
        answer_field = answer_field or config.get("answer_field")
    else:
        # 自定义数据集
        dataset_path = dataset_path or dataset_name
        if not text_field:
            text_field = "text"  # 默认文本字段

    print(f"正在下载数据集: {dataset_path}")
    print(f"  配置: {dataset_config}")
    print(f"  分割: {split}")
    print(f"  样本数: {num_samples}")

    # 下载数据集
    try:
        if dataset_config:
            dataset = load_dataset(dataset_path, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print(f"\n提示: 如果网络问题，请设置环境变量:")
        print(f"  export HF_ENDPOINT=https://hf-mirror.com")
        return None

    print(f"✓ 数据集下载完成！共 {len(dataset)} 条数据")
    print(f"正在提取前 {num_samples} 条数据...")

    # 提取样本
    samples = []
    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]

        # 获取文本
        if text_field in item:
            text = item[text_field]
        else:
            # 如果找不到指定字段，尝试找第一个字符串字段
            for key, value in item.items():
                if isinstance(value, str):
                    text = value
                    text_field = key
                    print(f"⚠️  使用字段 '{key}' 作为文本")
                    break
            else:
                text = str(item)  # 最后的fallback

        # 如果 text 是列表（如 tokens），拼接成字符串
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        # 构建样本
        sample = {"text": text}

        # 如果需要包含答案
        if with_answers and answer_field and answer_field in item:
            answer = item[answer_field]
            # 根据答案类型构建完整文本
            if isinstance(answer, str):
                sample["text"] = f"{text}\n{answer}"
                sample["question"] = text
                sample["answer"] = answer
            elif isinstance(answer, list):
                sample["text"] = f"{text}\n选项: {', '.join(map(str, answer))}"
                sample["question"] = text
                sample["choices"] = answer
            else:
                sample["text"] = f"{text}\n{str(answer)}"

        samples.append(sample)

    # 生成输出文件名
    if not output_file:
        suffix = "_with_answers" if with_answers else ""
        output_file = f"{dataset_name}_{num_samples}{suffix}.json"

    # 保存到 results 文件夹
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    print(f"✓ 成功保存 {len(samples)} 条数据到: {output_path}")

    # 打印示例
    print("\n前3条数据示例:")
    print("=" * 70)
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n示例 {i}:")
        text_preview = sample["text"][:150].replace("\n", " ")
        print(f"  {text_preview}...")
    print("=" * 70)

    return output_path


def list_available_datasets():
    """列出所有预定义的数据集"""
    print("\n可用的预定义数据集:")
    print("=" * 70)
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  路径: {config['path']}")
        print(f"  配置: {config.get('config', 'None')}")
        print(f"  分割: {config['split']}")
        print(f"  文本字段: {config['text_field']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="通用数据集下载工具 - 下载任意 HuggingFace 数据集并提取样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 下载 GSM8K 前 25 条数据
  python download_dataset.py --dataset gsm8k --num_samples 25

  # 下载 GSM8K 前 50 条数据（包含答案）
  python download_dataset.py --dataset gsm8k --num_samples 50 --with_answers

  # 下载 HellaSwag 前 100 条数据
  python download_dataset.py --dataset hellaswag --num_samples 100

  # 下载自定义数据集
  python download_dataset.py --dataset custom --dataset_path "username/dataset-name" --num_samples 30

  # 列出所有可用的预定义数据集
  python download_dataset.py --list
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="数据集名称（gsm8k/truthful_qa/math_500/gpqa_diamond 等）或自定义名称",
    )
    parser.add_argument(
        "--num_samples", type=int, default=25, help="提取的样本数量（默认: 25）"
    )
    parser.add_argument(
        "--output", type=str, help="输出文件名（默认: {dataset}_{num_samples}.json）"
    )
    parser.add_argument(
        "--with_answers", action="store_true", help="是否包含答案（用于 few-shot 学习）"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="自定义数据集 HuggingFace 路径（例如: openai/gsm8k）",
    )
    parser.add_argument(
        "--dataset_config", type=str, help="数据集配置名称（例如: main）"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集分割（train/test/validation，默认: train）",
    )
    parser.add_argument(
        "--text_field", type=str, help="文本字段名称（例如: question, text）"
    )
    parser.add_argument(
        "--answer_field", type=str, help="答案字段名称（例如: answer, choices）"
    )
    parser.add_argument(
        "--list", action="store_true", help="列出所有可用的预定义数据集"
    )

    args = parser.parse_args()

    # 列出数据集
    if args.list:
        list_available_datasets()
        return

    # 检查必需参数
    if not args.dataset:
        parser.error("请指定 --dataset 参数，或使用 --list 查看可用数据集")

    print("=" * 70)
    print("通用数据集下载工具")
    print("=" * 70)

    # 下载和提取
    download_and_extract(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_file=args.output,
        with_answers=args.with_answers,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        answer_field=args.answer_field,
    )

    print("\n✓ 全部完成！")


if __name__ == "__main__":
    main()
