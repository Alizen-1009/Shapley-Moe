#!/usr/bin/env python3
"""
Distill SFT data from the original (unpruned) teacher model with RFT filtering.

The goal of the third stage of SHAPE is to *recover* the capability lost by
expert pruning. The cleanest training signal for that is the original model's
own behavior, so this script uses the unpruned model as the teacher:

    1. Read a question pool (question + gold answer) produced by
       download_dataset.py --all_samples --with_answers.
    2. Sample N completions per question from the teacher via vLLM (offline).
    3. Rejection-sample (RFT): keep only completions whose final answer matches
       the gold answer, up to --keep_per_question per question.
    4. Write chat-format SFT records that train_adaptive_lora.py can consume
       directly (it reads the "messages" field first).

Output record format:

    {
      "messages": [
        {"role": "user", "content": "<question>"},
        {"role": "assistant", "content": "<teacher completion>"}
      ],
      "question": "<question>",
      "answer": "<teacher completion>"
    }

For tasks without a registered answer checker, RFT is skipped and the first
completion is kept (a warning is logged) unless --require_checker is set.
"""

import argparse
import json
import logging
import os
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer extraction / checkers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(raw: str) -> Optional[float]:
    """Parse a number that may contain thousands separators or a trailing dot."""
    cleaned = raw.replace(",", "").rstrip(".")
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _last_number(text: str) -> Optional[float]:
    matches = _NUMBER_RE.findall(text)
    for raw in reversed(matches):
        value = _normalize_number(raw)
        if value is not None:
            return value
    return None


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _gold_number(gold: str) -> Optional[float]:
    """GSM8K gold answers end with the reliable '#### N' marker."""
    marker = gold.rfind("####")
    if marker != -1:
        match = _NUMBER_RE.search(gold[marker + 4 :])
        if match:
            value = _normalize_number(match.group(0))
            if value is not None:
                return value
    return _last_number(gold)


def _pred_number(text: str) -> Optional[float]:
    """Final answer from a model completion.

    Do NOT use '####' here: models emit it as a markdown H4 header, which would
    grab an intermediate number. Prefer \\boxed{...}, else the last number.
    """
    boxed = _BOXED_RE.findall(text)
    for raw in reversed(boxed):
        match = _NUMBER_RE.search(raw)
        if match:
            value = _normalize_number(match.group(0))
            if value is not None:
                return value
    return _last_number(text)


def check_gsm8k(prediction: str, gold: str, tol: float = 1e-4) -> bool:
    gold_value = _gold_number(gold)
    pred_value = _pred_number(prediction)
    if gold_value is None or pred_value is None:
        return False
    return abs(pred_value - gold_value) <= tol


# Registry of dataset -> correctness checker. Extend as more tasks are added.
ANSWER_CHECKERS: Dict[str, Callable[[str, str], bool]] = {
    "gsm8k": check_gsm8k,
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_pool(path: str, max_questions: int = 0) -> List[Dict[str, str]]:
    """Load (question, gold answer) pairs from a download_dataset.py output file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")

    pool: List[Dict[str, str]] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        question = record.get("question") or record.get("text")
        answer = record.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        pool.append({"question": question.strip(), "answer": answer if isinstance(answer, str) else ""})
        if max_questions and len(pool) >= max_questions:
            break

    if not pool:
        raise ValueError(f"No usable (question, answer) pairs found in {path}")

    return pool


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")


def clean_completion(raw: str, strip_thinking: bool) -> str:
    """Normalize a decoded completion: drop chat special tokens and optionally thinking.

    The teacher decodes from token_ids (some vLLM builds leave CompletionOutput.text
    empty), so raw may contain trailing <|im_end|>/<|endoftext|> and a <think> block.
    """
    text = _SPECIAL_TOKEN_RE.sub("", raw)
    if not strip_thinking:
        return text.strip()

    without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if without_think:
        return without_think
    # Model kept everything inside (or never closed) the think block: drop the tags
    # but keep the reasoning so a correct final answer is not lost.
    return re.sub(r"</?think>", "", text).strip()


def build_prompts(questions: Sequence[str], tokenizer, enable_thinking: bool) -> List[str]:
    prompts: List[str] = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Tokenizers that do not accept enable_thinking.
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    return prompts


def make_record(question: str, answer: str) -> Dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "question": question,
        "answer": answer,
    }


def make_dpo_record(question: str, chosen: str, rejected: str) -> Dict[str, object]:
    """Preference pair for DPO: a correct (chosen) vs incorrect (rejected) answer."""
    return {
        "prompt": question,
        "chosen": chosen,
        "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------

def distill(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "distill_sft.py requires transformers and vllm. Install them in the "
            "inference environment before running this script."
        ) from exc

    checker = ANSWER_CHECKERS.get(args.dataset)
    if checker is None:
        if args.require_checker:
            raise ValueError(
                f"No answer checker registered for dataset '{args.dataset}'. "
                "Register one in ANSWER_CHECKERS or drop --require_checker."
            )
        logger.warning(
            "No answer checker for '%s'; RFT filtering disabled, keeping first completion per question.",
            args.dataset,
        )

    pool = load_pool(args.pool, max_questions=args.max_questions)
    logger.info("Loaded %d questions from %s", len(pool), args.pool)

    logger.info("Loading tokenizer from %s", args.teacher_model)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    questions = [item["question"] for item in pool]
    prompts = build_prompts(questions, tokenizer, enable_thinking=args.enable_thinking)

    logger.info("Loading teacher model (tp=%d) from %s", args.tp, args.teacher_model)
    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    logger.info("Generating %d sample(s) per question ...", args.num_samples)
    outputs = llm.generate(prompts, sampling)

    strip_thinking = not args.keep_thinking
    want_dpo = bool(args.dpo_output)
    records: List[Dict[str, object]] = []
    dpo_records: List[Dict[str, object]] = []
    solved = 0
    dpo_solved = 0
    for idx, (item, output) in enumerate(zip(pool, outputs)):
        gold = item["answer"]
        # Decode from token_ids: some vLLM builds leave CompletionOutput.text empty.
        raw_completions = [
            tokenizer.decode(candidate.token_ids, skip_special_tokens=False)
            for candidate in output.outputs
        ]
        completions = [clean_completion(text, strip_thinking) for text in raw_completions]
        completions = [text for text in completions if text]

        # Split completions by correctness (requires a checker + gold answer).
        graded = checker is not None and bool(gold)
        correct: List[str] = []
        incorrect: List[str] = []
        if graded:
            for text in completions:
                (correct if checker(text, gold) else incorrect).append(text)

        if idx < args.debug_n:
            if idx == 0:
                logger.info("[debug prompt] %r", prompts[0][-300:])
            logger.info("[debug q%d] gold=%r correct=%d incorrect=%d", idx, gold[-60:], len(correct), len(incorrect))

        # SFT: keep correct completions (or first completions when ungraded).
        if graded:
            kept = correct[: args.keep_per_question]
        else:
            kept = completions[: args.keep_per_question]
        if kept:
            solved += 1
            for text in kept:
                records.append(make_record(item["question"], text))

        # DPO: pair each correct (chosen) with an incorrect (rejected) from the same prompt.
        if want_dpo and correct and incorrect:
            dpo_solved += 1
            for chosen, rejected in zip(correct[: args.dpo_max_pairs], incorrect[: args.dpo_max_pairs]):
                dpo_records.append(make_dpo_record(item["question"], chosen, rejected))

    if not records:
        raise RuntimeError(
            "No SFT records produced. The teacher solved 0 questions under the "
            "current checker/sampling settings. Try raising --num_samples or "
            "--temperature, or inspect the answer checker."
        )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if want_dpo:
        if not dpo_records:
            logger.warning(
                "No DPO pairs produced: every solved question lacked an incorrect "
                "sample. Raise --num_samples or --temperature to surface negatives."
            )
        else:
            dpo_dir = os.path.dirname(os.path.abspath(args.dpo_output))
            os.makedirs(dpo_dir, exist_ok=True)
            with open(args.dpo_output, "w", encoding="utf-8") as f:
                json.dump(dpo_records, f, indent=2, ensure_ascii=False)

    solve_rate = 100.0 * solved / len(pool)
    print("=" * 70)
    print("Distillation complete")
    print("=" * 70)
    print(f"Dataset:            {args.dataset}")
    print(f"Teacher:            {args.teacher_model}")
    print(f"Questions:          {len(pool)}")
    print(f"Solved (>=1 kept):  {solved} ({solve_rate:.1f}%)")
    print(f"SFT records:        {len(records)}")
    print(f"RFT filtering:      {'on' if (checker and any(item['answer'] for item in pool)) else 'off'}")
    print(f"Output (SFT):       {args.output}")
    if want_dpo:
        print(f"DPO pairs:          {len(dpo_records)} (from {dpo_solved} questions)")
        print(f"Output (DPO):       {args.dpo_output if dpo_records else '(none written)'}")
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill RFT-filtered SFT data from the original teacher model via vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--teacher_model", required=True, help="Path to the original (unpruned) teacher model.")
    parser.add_argument("--pool", required=True, help="Question pool JSON from download_dataset.py.")
    parser.add_argument("--output", required=True, help="Output SFT JSON path.")
    parser.add_argument("--dpo_output", default=None, help="Optional DPO preference-pair JSON path (chosen/rejected).")
    parser.add_argument("--dataset", required=True, help="Dataset name; selects the RFT answer checker.")

    parser.add_argument("--num_samples", type=int, default=4, help="Completions sampled per question.")
    parser.add_argument("--keep_per_question", type=int, default=1, help="Max correct completions kept per question (SFT).")
    parser.add_argument("--dpo_max_pairs", type=int, default=1, help="Max (chosen, rejected) pairs per question (DPO).")
    parser.add_argument("--max_questions", type=int, default=0, help="Limit questions for a smoke test (0 = all).")
    parser.add_argument("--require_checker", action="store_true", help="Fail if no answer checker is registered.")
    parser.add_argument("--debug_n", type=int, default=0, help="Log raw/cleaned completions for the first N questions.")

    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top_p.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens per completion.")
    parser.add_argument("--enable_thinking", action="store_true", help="Request Qwen3 thinking mode in the chat template.")
    parser.add_argument("--keep_thinking", action="store_true", help="Keep <think> blocks in the stored SFT target (default: strip them).")

    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size.")
    parser.add_argument("--dtype", default="bfloat16", help="vLLM model dtype.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max model length.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args = build_arg_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.keep_per_question <= 0:
        raise ValueError("--keep_per_question must be positive.")
    distill(args)


if __name__ == "__main__":
    main()
