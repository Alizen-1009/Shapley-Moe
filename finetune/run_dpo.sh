#!/bin/bash
# =============================================================================
# DPO stage orchestration for Qwen3-30B-A3B (optional second stage after SFT).
#
# Recipe:  SFT (run_experiments.sh) -> SFT-merged model -> DPO (this script)
#          -> DPO-merged model -> evaluation.
#
# For each {keep rate} x {rank strategy} this script:
#   1. Trains a DPO LoRA on top of the SFT-merged model (reference = adapters off).
#   2. Merges the DPO adapter back into a servable model.
#
# Prerequisites:
#   - SFT-merged models from run_experiments.sh at ${MERGED_BASE}/..._merged
#   - Rank maps in results/.../lora_rank_maps/
#   - DPO data from data/download_sft.sh  (data/sft/{dataset}_dpo.json)
#
# Usage:
#   ./run_dpo.sh                 # train + merge, all rate x strategy
#   ./run_dpo.sh --dry-run
#   ./run_dpo.sh --step train    # only DPO training
#   ./run_dpo.sh --step merge    # only merge DPO adapters
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_NAME="qwen3-30b-a3b"
DATASET="gsm8k_25"
KEEP_RATES="0_8 0_6"
RANK_STRATEGIES="bucket uniform random"

RESULTS_DIR="${PROJECT_DIR}/results/${MODEL_NAME}"
DPO_FILE="${PROJECT_DIR}/data/sft/gsm8k_dpo.json"

PRUNED_BASE="${PRUNED_BASE:-/root/autodl-tmp}"
ADAPTER_BASE="${ADAPTER_BASE:-/root/autodl-tmp/lora_outputs}"
MERGED_BASE="${MERGED_BASE:-/root/autodl-tmp/merged_models}"

# DPO hyperparameters
BETA="${BETA:-0.1}"
MAX_SEQ_LENGTH=1024
BATCH_SIZE=1
GRAD_ACCUM=8
LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-1}"
SAVE_STEPS=100
LOGGING_STEPS=5

DRY_RUN=false
STEP="all"  # all, train, merge

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --step) STEP="$2"; shift 2 ;;
        --dpo-file) DPO_FILE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--step all|train|merge] [--dpo-file PATH]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

run_cmd() {
    if [ "$DRY_RUN" = true ]; then echo "[DRY-RUN] $*"; else "$@"; fi
}

echo "============================================================"
echo "DPO orchestration  model=${MODEL_NAME} dataset=${DATASET}"
echo "  rates=${KEEP_RATES}  strategies=${RANK_STRATEGIES}  beta=${BETA}"
echo "  dpo_file=${DPO_FILE}  step=${STEP}"
echo "============================================================"

if [ "$DRY_RUN" = false ] && [ ! -f "$DPO_FILE" ]; then
    echo "[ERROR] DPO data not found: ${DPO_FILE}"
    echo "        Run: cd data && DPO=1 ./download_sft.sh gsm8k"
    exit 1
fi

# ---- Step 1: DPO training ----
if [ "$STEP" = "all" ] || [ "$STEP" = "train" ]; then
    for rate in $KEEP_RATES; do
        for strategy in $RANK_STRATEGIES; do
            SFT_MERGED="${MERGED_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}_merged"
            RANK_MAP="${RESULTS_DIR}/lora_rank_maps/${DATASET}_rate${rate}_${strategy}.json"
            OUTPUT_DIR="${ADAPTER_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}_dpo"

            if [ "$DRY_RUN" = false ] && [ ! -d "$SFT_MERGED" ]; then
                echo "[WARN] SFT-merged model missing, skipping: ${SFT_MERGED}"; continue
            fi
            if [ "$DRY_RUN" = false ] && [ ! -f "$RANK_MAP" ]; then
                echo "[WARN] rank map missing, skipping: ${RANK_MAP}"; continue
            fi
            if [ "$DRY_RUN" = false ] && [ -f "${OUTPUT_DIR}/dpo_lora_train_info.json" ]; then
                echo "[INFO] DPO adapter exists, skipping: ${OUTPUT_DIR}"; continue
            fi

            echo "[INFO] DPO train: rate${rate} ${strategy}"
            run_cmd python3 "${SCRIPT_DIR}/train_dpo_lora.py" \
                --model_path "$SFT_MERGED" \
                --rank_map "$RANK_MAP" \
                --dpo_file "$DPO_FILE" \
                --output_dir "$OUTPUT_DIR" \
                --model_type qwen3 \
                --beta "$BETA" \
                --torch_dtype bf16 --bf16 --gradient_checkpointing \
                --max_seq_length "$MAX_SEQ_LENGTH" \
                --per_device_train_batch_size "$BATCH_SIZE" \
                --gradient_accumulation_steps "$GRAD_ACCUM" \
                --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
                --logging_steps "$LOGGING_STEPS" --save_steps "$SAVE_STEPS" \
                --save_total_limit 2 --report_to none --overwrite_output_dir
        done
    done
fi

# ---- Step 2: merge DPO adapters ----
if [ "$STEP" = "all" ] || [ "$STEP" = "merge" ]; then
    for rate in $KEEP_RATES; do
        for strategy in $RANK_STRATEGIES; do
            SFT_MERGED="${MERGED_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}_merged"
            ADAPTER_DIR="${ADAPTER_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}_dpo"
            DPO_MERGED="${MERGED_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}_dpo_merged"

            if [ "$DRY_RUN" = false ] && [ ! -d "$ADAPTER_DIR" ]; then
                echo "[WARN] DPO adapter missing, skipping: ${ADAPTER_DIR}"; continue
            fi
            if [ "$DRY_RUN" = false ] && [ -f "${DPO_MERGED}/merged_lora_info.json" ]; then
                echo "[INFO] DPO-merged exists, skipping: ${DPO_MERGED}"; continue
            fi

            echo "[INFO] DPO merge: rate${rate} ${strategy}"
            run_cmd python3 "${SCRIPT_DIR}/merge_lora.py" \
                --base_model "$SFT_MERGED" \
                --adapter "$ADAPTER_DIR" \
                --output "$DPO_MERGED" \
                --torch_dtype bf16
        done
    done
fi

echo ""
echo "DPO stage complete. Merged models: ${MERGED_BASE}/..._dpo_merged"
echo "Next: evaluation/vllm-server.sh <dpo_merged_path> && python evaluation/run_evalscope.py"
