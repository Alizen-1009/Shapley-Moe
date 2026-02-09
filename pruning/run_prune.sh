#!/bin/bash
# =============================================================================
# Model Pruning Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_CONFIG="${PROJECT_DIR}/configs/models.yaml"

# Read model path from config
get_model_path() {
    python3 -c "
import yaml
with open('$MODELS_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('models', {}).get('$1', {}).get('path', ''))
" 2>/dev/null
}

# Default parameters
MODEL=""
DATASET=""
METHOD="shapley"
STRATEGY="alpha_per_layer"
RATE="0.8"
DEVICE_MAP="auto"
PRUNE_STRATEGY="auto"  # Pruning strategy: auto (auto-select based on method), zero_weights, gate_bias, both
OUTPUT_DIR=""  # Custom output directory (optional)

show_help() {
    echo "Usage: $0 -m MODEL -d DATASET [-M METHOD] [-s STRATEGY] [-r RATE] [-p PRUNE_STRATEGY]"
    echo ""
    echo "Options:"
    echo "  -m MODEL           Model name (e.g. qwen3-30b-a3b)"
    echo "  -d DATASET         Dataset name (e.g. gsm8k_25)"
    echo "  -M METHOD          Pruning method (default: shapley)"
    echo "  -s STRATEGY        Shapley strategy (default: alpha_per_layer)"
    echo "  -r RATE            Retention rate (default: 0.8)"
    echo "  -p PRUNE_STRATEGY  Pruning strategy (default: auto)"
    echo "                     - auto: Auto-select based on pruning method (default)"
    echo "                     - zero_weights: Zero out expert weights"
    echo "                     - gate_bias: Modify gate so pruned experts won't be selected"
    echo "                     - both: Use both strategies simultaneously"
    echo "  -o OUTPUT_DIR      Custom output directory (optional, default saves alongside original model)"
    echo "  --device MAP       Device mapping (default: auto)"
    echo ""
    echo "Examples:"
    echo "  $0 -m qwen3-30b-a3b -d gsm8k_25 -r 0.8"
    echo "  $0 -m gpt-oss-20b -d arc_easy_25 -M easyep -r 0.6 -p both"
    echo ""
    echo "Output directory example:"
    echo "  Original model: /root/hf_models/deepseekv2-lite-coder"
    echo "  Pruned model:   /root/hf_models/deepseekv2-lite-coder_shapley_alpha_per_layer_gsm8k_25_rate0_6"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL="$2"; shift 2 ;;
        -d) DATASET="$2"; shift 2 ;;
        -M) METHOD="$2"; shift 2 ;;
        -s) STRATEGY="$2"; shift 2 ;;
        -r) RATE="$2"; shift 2 ;;
        -p) PRUNE_STRATEGY="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE_MAP="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Error: Must specify model (-m) and dataset (-d)"
    show_help
    exit 1
fi

# Get model path
MODEL_PATH=$(get_model_path "$MODEL")
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model '$MODEL' not found in config"
    exit 1
fi

# Build selection file path
RATE_STR=$(echo "$RATE" | sed 's/\./_/g')
if [ "$METHOD" = "shapley" ]; then
    SELECTION_FILE="${PROJECT_DIR}/results/${MODEL}/selected_experts/${METHOD}_${STRATEGY}_${DATASET}_rate${RATE_STR}.json"
else
    SELECTION_FILE="${PROJECT_DIR}/results/${MODEL}/selected_experts/${METHOD}_${DATASET}_rate${RATE_STR}.json"
fi

# Check selection file
if [ ! -f "$SELECTION_FILE" ]; then
    echo "Error: Selection file does not exist: $SELECTION_FILE"
    exit 1
fi

# Build output directory name
OUTPUT_SUFFIX=""
if [ "$METHOD" = "shapley" ]; then
    OUTPUT_SUFFIX="${METHOD}_${STRATEGY}_${DATASET}_rate${RATE_STR}"
else
    OUTPUT_SUFFIX="${METHOD}_${DATASET}_rate${RATE_STR}"
fi

# If no output directory specified, default saves alongside the original model
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_PARENT_DIR=$(dirname "$MODEL_PATH")
    MODEL_BASENAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="${MODEL_PARENT_DIR}/${MODEL_BASENAME}_${OUTPUT_SUFFIX}"
fi

echo "============================================"
echo "Model: $MODEL"
echo "Model path: $MODEL_PATH"
echo "Selection file: $SELECTION_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Pruning strategy: $PRUNE_STRATEGY"
echo "============================================"

# Run pruning
python3 "${SCRIPT_DIR}/save_model.py" \
    --model_path "$MODEL_PATH" \
    --selection_file "$SELECTION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device_map "$DEVICE_MAP" \
    --strategy "$PRUNE_STRATEGY"

echo "✓ Done: $OUTPUT_DIR"
