#!/bin/bash
# =============================================================================
# Batch finetune experiment orchestration for Qwen3-30B-A3B
# =============================================================================
#
# Runs the full experiment matrix:
#   - Dataset: gsm8k (default)
#   - Keep rates: rate0_8, rate0_6
#   - Rank strategies: bucket, uniform, random
#
# Prerequisites:
#   - Pruned models exist at ${PRUNED_DIR}
#   - Shapley CSVs and selected expert JSONs exist in results/
#   - SFT training data exists (run data/download_sft.sh first)
#
# Usage:
#   ./run_experiments.sh                    # Run all experiments
#   ./run_experiments.sh --dry-run          # Print commands without running
#   ./run_experiments.sh --step train       # Only run training step
#   ./run_experiments.sh --step merge       # Only merge adapters
#   ./run_experiments.sh --step rank_map    # Only build rank maps
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Configuration - Edit these paths for your environment
# =============================================================================

MODEL_NAME="qwen3-30b-a3b"
DATASET="gsm8k_25"
KEEP_RATES="0_8 0_6"
RANK_STRATEGIES="bucket uniform random"

# Paths
RESULTS_DIR="${PROJECT_DIR}/results/${MODEL_NAME}"
SHAPLEY_CSV="${RESULTS_DIR}/shapley_values/${DATASET}_shapley.csv"
TRAIN_FILE="${PROJECT_DIR}/data/sft/gsm8k_sft.json"

# Pruned model directory pattern: {PRUNED_BASE}/{MODEL_NAME}_rate{RATE}_pruned
PRUNED_BASE="${PRUNED_BASE:-/root/autodl-tmp}"
# Output adapters base directory
ADAPTER_BASE="${ADAPTER_BASE:-/root/autodl-tmp/lora_outputs}"
# Merged models base directory
MERGED_BASE="${MERGED_BASE:-/root/autodl-tmp/merged_models}"

# Training hyperparameters
MAX_SEQ_LENGTH=1024
BATCH_SIZE=1
GRAD_ACCUM=8
LR="2e-4"
EPOCHS=1
SAVE_STEPS=50
LOGGING_STEPS=5

# =============================================================================
# Parse arguments
# =============================================================================

DRY_RUN=false
STEP="all"  # all, rank_map, train, merge

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --step) STEP="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --pruned-base) PRUNED_BASE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--step all|rank_map|train|merge] [--train-file PATH] [--pruned-base PATH]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

# =============================================================================
# Validation
# =============================================================================

echo "============================================================================="
log_info "Finetune Experiment Orchestration"
echo "============================================================================="
log_info "Model:           ${MODEL_NAME}"
log_info "Dataset:         ${DATASET}"
log_info "Keep rates:      ${KEEP_RATES}"
log_info "Rank strategies: ${RANK_STRATEGIES}"
log_info "Train file:      ${TRAIN_FILE}"
log_info "Step:            ${STEP}"
if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - no commands will be executed"
fi
echo "============================================================================="

if [ "$DRY_RUN" = false ]; then
    if [ ! -f "$SHAPLEY_CSV" ]; then
        log_error "Shapley CSV not found: ${SHAPLEY_CSV}"
        log_error "Run analysis/run_calc_shapley.sh first."
        exit 1
    fi

    if [ ! -f "$TRAIN_FILE" ]; then
        log_error "Training file not found: ${TRAIN_FILE}"
        log_error "Run data/download_sft.sh first."
        exit 1
    fi
fi

# =============================================================================
# Step 1: Build rank maps
# =============================================================================

if [ "$STEP" = "all" ] || [ "$STEP" = "rank_map" ]; then
    echo ""
    log_info "=== Step 1: Building rank maps ==="
    echo ""

    for rate in $KEEP_RATES; do
        SELECTED="${RESULTS_DIR}/selected_experts/shapley_alpha_per_layer_${DATASET}_rate${rate}.json"
        if [ "$DRY_RUN" = false ] && [ ! -f "$SELECTED" ]; then
            log_warn "Selected experts not found, skipping rate ${rate}: ${SELECTED}"
            continue
        fi

        for strategy in $RANK_STRATEGIES; do
            OUTPUT="${RESULTS_DIR}/lora_rank_maps/${DATASET}_rate${rate}_${strategy}.json"

            if [ "$DRY_RUN" = false ] && [ -f "$OUTPUT" ]; then
                log_info "Rank map exists, skipping: ${OUTPUT}"
                continue
            fi

            log_info "Building rank map: rate${rate} ${strategy}"
            run_cmd python3 "${SCRIPT_DIR}/build_rank_map.py" \
                --shapley_csv "$SHAPLEY_CSV" \
                --selected_experts "$SELECTED" \
                --output "$OUTPUT" \
                --strategy "$strategy"
        done
    done
fi

# =============================================================================
# Step 2: Train LoRA adapters
# =============================================================================

if [ "$STEP" = "all" ] || [ "$STEP" = "train" ]; then
    echo ""
    log_info "=== Step 2: Training LoRA adapters ==="
    echo ""

    for rate in $KEEP_RATES; do
        MODEL_PATH="${PRUNED_BASE}/${MODEL_NAME}_rate${rate}_pruned"
        if [ "$DRY_RUN" = false ] && [ ! -d "$MODEL_PATH" ]; then
            log_warn "Pruned model not found, skipping: ${MODEL_PATH}"
            continue
        fi

        for strategy in $RANK_STRATEGIES; do
            RANK_MAP="${RESULTS_DIR}/lora_rank_maps/${DATASET}_rate${rate}_${strategy}.json"
            OUTPUT_DIR="${ADAPTER_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}"

            if [ "$DRY_RUN" = false ] && [ ! -f "$RANK_MAP" ]; then
                log_warn "Rank map not found, skipping: ${RANK_MAP}"
                continue
            fi

            if [ "$DRY_RUN" = false ] && [ -d "$OUTPUT_DIR" ] && [ -f "${OUTPUT_DIR}/adaptive_lora_train_info.json" ]; then
                log_info "Adapter exists, skipping: ${OUTPUT_DIR}"
                continue
            fi

            log_info "Training: rate${rate} ${strategy}"
            run_cmd python3 "${SCRIPT_DIR}/train_adaptive_lora.py" \
                --model_path "$MODEL_PATH" \
                --rank_map "$RANK_MAP" \
                --train_file "$TRAIN_FILE" \
                --output_dir "$OUTPUT_DIR" \
                --model_type qwen3 \
                --torch_dtype bf16 \
                --bf16 \
                --gradient_checkpointing \
                --max_seq_length "$MAX_SEQ_LENGTH" \
                --per_device_train_batch_size "$BATCH_SIZE" \
                --gradient_accumulation_steps "$GRAD_ACCUM" \
                --learning_rate "$LR" \
                --num_train_epochs "$EPOCHS" \
                --logging_steps "$LOGGING_STEPS" \
                --save_steps "$SAVE_STEPS" \
                --save_total_limit 2 \
                --report_to none \
                --overwrite_output_dir

            if [ $? -eq 0 ]; then
                log_success "Training done: ${OUTPUT_DIR}"
            else
                log_error "Training failed: rate${rate} ${strategy}"
            fi
        done
    done
fi

# =============================================================================
# Step 3: Merge adapters
# =============================================================================

if [ "$STEP" = "all" ] || [ "$STEP" = "merge" ]; then
    echo ""
    log_info "=== Step 3: Merging LoRA adapters ==="
    echo ""

    for rate in $KEEP_RATES; do
        MODEL_PATH="${PRUNED_BASE}/${MODEL_NAME}_rate${rate}_pruned"
        if [ "$DRY_RUN" = false ] && [ ! -d "$MODEL_PATH" ]; then
            log_warn "Pruned model not found, skipping: ${MODEL_PATH}"
            continue
        fi

        for strategy in $RANK_STRATEGIES; do
            ADAPTER_DIR="${ADAPTER_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}"
            MERGED_DIR="${MERGED_BASE}/${MODEL_NAME}_${DATASET}_rate${rate}_${strategy}_merged"

            if [ "$DRY_RUN" = false ] && [ ! -d "$ADAPTER_DIR" ]; then
                log_warn "Adapter not found, skipping: ${ADAPTER_DIR}"
                continue
            fi

            if [ "$DRY_RUN" = false ] && [ -d "$MERGED_DIR" ] && [ -f "${MERGED_DIR}/merged_lora_info.json" ]; then
                log_info "Merged model exists, skipping: ${MERGED_DIR}"
                continue
            fi

            log_info "Merging: rate${rate} ${strategy}"
            run_cmd python3 "${SCRIPT_DIR}/merge_lora.py" \
                --base_model "$MODEL_PATH" \
                --adapter "$ADAPTER_DIR" \
                --output "$MERGED_DIR" \
                --torch_dtype bf16

            if [ $? -eq 0 ]; then
                log_success "Merge done: ${MERGED_DIR}"
            else
                log_error "Merge failed: rate${rate} ${strategy}"
            fi
        done
    done
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================================="
log_info "Experiment run complete."
echo "============================================================================="
echo ""
log_info "Adapters:      ${ADAPTER_BASE}/"
log_info "Merged models: ${MERGED_BASE}/"
echo ""
log_info "Next steps:"
log_info "  1. Start vLLM: evaluation/vllm-server.sh <merged_model_path>"
log_info "  2. Evaluate:   python evaluation/run_evalscope.py"
echo ""
