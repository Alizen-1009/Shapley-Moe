#!/bin/bash
# =============================================================================
# Distillation-based SFT data generation.
#
# The third stage of SHAPE recovers capability lost by expert pruning, so the
# teacher is the ORIGINAL (unpruned) model. For each dataset this script:
#   1. Downloads the full train-split question pool (with gold answers) for RFT.
#   2. Distills answers from the teacher via vLLM and keeps only completions
#      whose final answer matches gold (rejection sampling / RFT).
#
# Output: data/sft/{dataset}_distill.json  (chat-format SFT data consumed by
#         finetune/train_adaptive_lora.py).
#
# Usage:
#   ./download_sft.sh                       # gsm8k, defaults
#   ./download_sft.sh gsm8k logiqa          # multiple datasets
#   TEACHER_MODEL=/root/models/Qwen3-30B TP=8 NUM_SAMPLES=4 ./download_sft.sh gsm8k
#   MAX_QUESTIONS=50 ./download_sft.sh gsm8k   # quick smoke test
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Teacher = original unpruned model (see configs/models.yaml).
TEACHER_MODEL="${TEACHER_MODEL:-/root/models/Qwen3-30B}"

# Distillation knobs (override via environment).
TP="${TP:-8}"                       # vLLM tensor-parallel size
NUM_SAMPLES="${NUM_SAMPLES:-4}"     # completions sampled per question
KEEP="${KEEP:-1}"                   # correct completions kept per question (SFT)
MAX_QUESTIONS="${MAX_QUESTIONS:-0}" # 0 = all questions
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
DPO="${DPO:-1}"                     # 1 = also emit DPO preference pairs
DPO_MAX_PAIRS="${DPO_MAX_PAIRS:-1}" # (chosen, rejected) pairs per question

DATASETS="${@:-gsm8k}"

POOL_DIR="${SCRIPT_DIR}/sft/pool"
OUT_DIR="${SCRIPT_DIR}/sft"
mkdir -p "$POOL_DIR" "$OUT_DIR"

for dataset in $DATASETS; do
    pool="${POOL_DIR}/${dataset}.json"
    output="${OUT_DIR}/${dataset}_distill.json"

    echo "============================================================"
    echo "Dataset: ${dataset}"
    echo "============================================================"

    if [ ! -f "$pool" ]; then
        echo "[1/2] Downloading question pool -> ${pool}"
        python3 "${SCRIPT_DIR}/download_dataset.py" \
            --dataset "$dataset" \
            --all_samples \
            --with_answers \
            --output "$pool"
    else
        echo "[1/2] Question pool exists, skipping download: ${pool}"
    fi

    echo "[2/2] Distilling from teacher -> ${output}"
    EXTRA_ARGS=()
    if [ "$MAX_QUESTIONS" -gt 0 ]; then
        EXTRA_ARGS+=(--max_questions "$MAX_QUESTIONS")
    fi
    if [ "$DPO" = "1" ]; then
        EXTRA_ARGS+=(--dpo_output "${OUT_DIR}/${dataset}_dpo.json" --dpo_max_pairs "$DPO_MAX_PAIRS")
    fi

    python3 "${SCRIPT_DIR}/distill_sft.py" \
        --teacher_model "$TEACHER_MODEL" \
        --pool "$pool" \
        --output "$output" \
        --dataset "$dataset" \
        --num_samples "$NUM_SAMPLES" \
        --keep_per_question "$KEEP" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS" \
        --tp "$TP" \
        "${EXTRA_ARGS[@]}"

    echo "Done: ${output}"
    echo ""
done
