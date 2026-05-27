#!/usr/bin/env python3
"""
Universal dataset download tool - Download any HuggingFace dataset and extract samples for few-shot pruning
"""

import json
import os
import argparse
from datasets import load_dataset


# Predefined dataset configurations
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
    # Additional datasets
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
        "path": "hendrydong/gpqa_diamond",  # public version, no authorization required
        "config": None,
        "split": "test",
        "text_field": "problem",
        "answer_field": "solution",
    },
    "ontonotes5": {
        "path": "SpeedOfMagic/ontonotes_english",  # available public version
        "config": None,
        "split": "train",
        "text_field": "tokens",  # returns token list
        "answer_field": "ner_tags",
    },
    # Additional datasets - second batch
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
        "config": "AIME2025-I",  # optional: AIME2025-I or AIME2025-II
        "split": "test",
        "text_field": "question",
        "answer_field": "answer",
    },
    "biomix_qa": {
        "path": "kg-rag/BiomixQA",
        "config": "mcq",  # optional: mcq or true_false
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
    # ARC dataset (AI2 Reasoning Challenge) - evaluation uses test split
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
    all_samples: bool = False,
    dataset_path: str = None,
    dataset_config: str = None,
    split: str = "train",
    text_field: str = None,
    answer_field: str = None,
):
    """
    Download dataset and extract samples

    Args:
        dataset_name: Dataset name (predefined) or custom path
        num_samples: Number of samples to extract
        output_file: Output filename (default: auto-generated)
        with_answers: Whether to include answers
        all_samples: Whether to export the full split
        dataset_path: Custom dataset path (overrides predefined)
        dataset_config: Dataset config name
        split: Dataset split (train/test/validation)
        text_field: Text field name
        answer_field: Answer field name
    """

    # If it's a predefined dataset, use config
    if dataset_name in DATASET_CONFIGS and not dataset_path:
        config = DATASET_CONFIGS[dataset_name]
        dataset_path = config["path"]
        dataset_config = config.get("config")
        split = config["split"]
        text_field = text_field or config["text_field"]
        answer_field = answer_field or config.get("answer_field")
    else:
        # Custom dataset
        dataset_path = dataset_path or dataset_name
        if not text_field:
            text_field = "text"  # default text field

    print(f"Downloading dataset: {dataset_path}")
    print(f"  Config: {dataset_config}")
    print(f"  Split: {split}")
    if all_samples or num_samples <= 0:
        print("  Number of samples: all")
    else:
        print(f"  Number of samples: {num_samples}")

    # Download dataset
    try:
        if dataset_config:
            dataset = load_dataset(dataset_path, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"\nHint: If you have network issues, set the environment variable:")
        print(f"  export HF_ENDPOINT=https://hf-mirror.com")
        return None

    print(f"✓ Dataset download complete! Total {len(dataset)} entries")
    sample_count = len(dataset) if all_samples or num_samples <= 0 else min(num_samples, len(dataset))
    print(f"Extracting {sample_count} entries...")

    # Extract samples
    samples = []
    for i in range(sample_count):
        item = dataset[i]

        # Get text
        if text_field in item:
            text = item[text_field]
        else:
            # If specified field not found, try to find the first string field
            for key, value in item.items():
                if isinstance(value, str):
                    text = value
                    text_field = key
                    print(f"⚠️  Using field '{key}' as text")
                    break
            else:
                text = str(item)  # last resort fallback

        # If text is a list (e.g. tokens), join into string
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        # Build sample
        sample = {"text": text}

        # If answers should be included
        if with_answers and answer_field and answer_field in item:
            answer = item[answer_field]
            # Build complete text based on answer type
            if isinstance(answer, str):
                sample["text"] = f"{text}\n{answer}"
                sample["question"] = text
                sample["answer"] = answer
            elif isinstance(answer, list):
                sample["text"] = f"{text}\nChoices: {', '.join(map(str, answer))}"
                sample["question"] = text
                sample["choices"] = answer
            else:
                sample["text"] = f"{text}\n{str(answer)}"

        samples.append(sample)

    # Generate output filename
    if not output_file:
        suffix = "_with_answers" if with_answers else ""
        count_label = "all" if all_samples or num_samples <= 0 else str(num_samples)
        output_file = f"{dataset_name}_{count_label}{suffix}.json"

    # Save to results folder by default, but honor absolute/relative custom output paths.
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, "results")
    if os.path.isabs(output_file):
        output_path = output_file
    else:
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, output_file)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    print(f"✓ Successfully saved {len(samples)} entries to: {output_path}")

    # Print examples
    print("\nFirst 3 data examples:")
    print("=" * 70)
    for i, sample in enumerate(samples[:3], 1):
        print(f"\nExample {i}:")
        text_preview = sample["text"][:150].replace("\n", " ")
        print(f"  {text_preview}...")
    print("=" * 70)

    return output_path


def list_available_datasets():
    """List all available predefined datasets"""
    print("\nAvailable predefined datasets:")
    print("=" * 70)
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Path: {config['path']}")
        print(f"  Config: {config.get('config', 'None')}")
        print(f"  Split: {config['split']}")
        print(f"  Text field: {config['text_field']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Universal dataset download tool - Download any HuggingFace dataset and extract samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Download first 25 entries from GSM8K
  python download_dataset.py --dataset gsm8k --num_samples 25

  # Download first 50 entries from GSM8K (with answers)
  python download_dataset.py --dataset gsm8k --num_samples 50 --with_answers

  # Download the full GSM8K train split with answers
  python download_dataset.py --dataset gsm8k --all_samples --with_answers

  # Download first 100 entries from HellaSwag
  python download_dataset.py --dataset hellaswag --num_samples 100

  # Download custom dataset
  python download_dataset.py --dataset custom --dataset_path "username/dataset-name" --num_samples 30

  # List all available predefined datasets
  python download_dataset.py --list
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (gsm8k/truthful_qa/math_500/gpqa_diamond etc.) or custom name",
    )
    parser.add_argument(
        "--num_samples", type=int, default=25, help="Number of samples to extract (default: 25)"
    )
    parser.add_argument(
        "--output", type=str, help="Output filename (default: {dataset}_{num_samples}.json)"
    )
    parser.add_argument(
        "--with_answers", action="store_true", help="Whether to include answers (for few-shot learning)"
    )
    parser.add_argument(
        "--all_samples",
        action="store_true",
        help="Export the full split instead of truncating to num_samples",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Custom dataset HuggingFace path (e.g.: openai/gsm8k)",
    )
    parser.add_argument(
        "--dataset_config", type=str, help="Dataset config name (e.g.: main)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train/test/validation, default: train)",
    )
    parser.add_argument(
        "--text_field", type=str, help="Text field name (e.g.: question, text)"
    )
    parser.add_argument(
        "--answer_field", type=str, help="Answer field name (e.g.: answer, choices)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available predefined datasets"
    )

    args = parser.parse_args()

    # List datasets
    if args.list:
        list_available_datasets()
        return

    # Check required arguments
    if not args.dataset:
        parser.error("Please specify --dataset, or use --list to view available datasets")

    print("=" * 70)
    print("Universal Dataset Download Tool")
    print("=" * 70)

    # Download and extract
    download_and_extract(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_file=args.output,
        with_answers=args.with_answers,
        all_samples=args.all_samples,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        answer_field=args.answer_field,
    )

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
