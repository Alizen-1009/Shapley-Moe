#!/bin/bash

# =============================================================================
# vLLM Server Startup Script
# Usage:
#   ./vllm-server.sh                              # Use default parameters
#   ./vllm-server.sh --model qwen3-30b-a3b --rate 0.5
#   ./vllm-server.sh --method random --dataset arc_easy_25
# =============================================================================

# Default parameters
MODEL="deepseekv2-lite-coder"      # Model name
METHOD="shapley"                    # Pruning method: shapley, random, frequency, gating, easyep, reap
STRATEGY="alpha_per_layer"          # Shapley strategy: alpha_per_layer, alpha_global, topk_per_layer, topk_global
DATASET="gsm8k_25"                  # Dataset name
RATE="0.8"                          # Pruning ratio (0.1-0.9)
PORT="8801"                         # Service port
TP_SIZE="8"                         # tensor-parallel-size
MODEL_BASE="/root/yuhao/Shapley-Moe/models"  # Model base directory

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)     MODEL="$2";     shift 2 ;;
        --method)       METHOD="$2";    shift 2 ;;
        --strategy|-s)  STRATEGY="$2";  shift 2 ;;
        --dataset|-d)   DATASET="$2";   shift 2 ;;
        --rate|-r)      RATE="$2";      shift 2 ;;
        --port|-p)      PORT="$2";      shift 2 ;;
        --tp)           TP_SIZE="$2";   shift 2 ;;
        --base)         MODEL_BASE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model, -m     Model name (default: $MODEL)"
            echo "  --method        Pruning method: shapley|random|frequency|gating|easyep|reap (default: $METHOD)"
            echo "  --strategy, -s  Shapley strategy: alpha_per_layer|alpha_global|topk_per_layer|topk_global (default: $STRATEGY)"
            echo "  --dataset, -d   Dataset name (default: $DATASET)"
            echo "  --rate, -r      Pruning ratio 0.1-0.9 (default: $RATE)"
            echo "  --port, -p      Service port (default: $PORT)"
            echo "  --tp            tensor-parallel-size (default: $TP_SIZE)"
            echo "  --base          Model base directory (default: $MODEL_BASE)"
            echo ""
            echo "Examples:"
            echo "  $0 --model qwen3-30b-a3b --rate 0.5"
            echo "  $0 --method random --dataset arc_easy_25"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Format pruning ratio (0.8 -> 0_8)
RATE_FORMATTED=$(echo "$RATE" | tr '.' '_')

# Build model directory name
if [[ "$METHOD" == "shapley" ]]; then
    # Shapley method includes strategy
    MODEL_DIR="${MODEL}_${METHOD}_${STRATEGY}_${DATASET}_rate${RATE_FORMATTED}"
else
    # Other methods don't need strategy
    MODEL_DIR="${MODEL}_${METHOD}_${DATASET}_rate${RATE_FORMATTED}"
fi

MODEL_PATH="${MODEL_BASE}/${MODEL_DIR}"

# Show configuration
echo "=============================================="
echo "vLLM Server Configuration"
echo "=============================================="
echo "Model:          $MODEL"
echo "Pruning method: $METHOD"
[[ "$METHOD" == "shapley" ]] && echo "Strategy:       $STRATEGY"
echo "Dataset:        $DATASET"
echo "Pruning ratio:  $RATE"
echo "Port:           $PORT"
echo "TP Size:        $TP_SIZE"
echo "----------------------------------------------"
echo "Model path:     $MODEL_PATH"
echo "=============================================="

# Check if model exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo ""
    echo "⚠️  Warning: Model directory does not exist!"
    echo "   Please check path or run pruning to generate model first"
    echo ""
    # List available models
    echo "Available models:"
    ls -1 "$MODEL_BASE" 2>/dev/null | head -10
    exit 1
fi

# Start vLLM service
echo ""
echo "🚀 Starting vLLM service..."

vllm serve \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --served-model-name "${MODEL}_pruned" \
    --trust-remote-code \
    --port "$PORT"
