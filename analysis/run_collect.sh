#!/bin/bash
# =============================================================================
# Unified Few-Shot collection of all pruning information - batch script
# =============================================================================
#
# Features:
#   Collects information needed by Shapley/Gating Score/EASYEP/REAP four pruning methods
#   in a single inference pass. Results are organized and saved by model.
#   Preferentially reads configuration from configs/
#
# Output directory structure:
#   results/{model_name}/activations/
#   ├── {dataset}_shapley.json   # Shapley value calculation
#   ├── {dataset}_gating.json    # Gating Score pruning
#   ├── {dataset}_easyep.json    # EASYEP pruning
#   └── {dataset}_reap.json      # REAP pruning
#
# Usage:
#   ./run_collect.sh [options]
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/experiments.yaml"
MODELS_CONFIG="${PROJECT_DIR}/configs/models.yaml"

# =============================================================================
# Configuration reading functions
# =============================================================================

# Generic config reading function
read_config() {
    local key="$1"
    local default="$2"
    local config_file="${3:-$CONFIG_FILE}"
    
    if [ ! -f "$config_file" ]; then
        echo "$default"
        return
    fi
    
    local value=$(python3 -c "
import yaml
try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    keys = '$key'.split('.')
    result = config
    for k in keys:
        result = result.get(k, None)
        if result is None:
            break
    if result is not None:
        if isinstance(result, list):
            print(' '.join(str(x) for x in result))
        else:
            print(result)
except:
    pass
" 2>/dev/null)
    
    if [ -n "$value" ]; then
        echo "$value"
    else
        echo "$default"
    fi
}

# Read model path from models.yaml
get_model_path() {
    local model_name="$1"
    
    if [ ! -f "$MODELS_CONFIG" ]; then
        echo ""
        return
    fi
    
    python3 -c "
import yaml
try:
    with open('$MODELS_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    models = config.get('models', {})
    if '$model_name' in models:
        print(models['$model_name'].get('path', ''))
except:
    pass
" 2>/dev/null
}

# List all available models
list_available_models() {
    if [ ! -f "$MODELS_CONFIG" ]; then
        echo ""
        return
    fi
    
    python3 -c "
import yaml
try:
    with open('$MODELS_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    models = config.get('models', {})
    for name in models.keys():
        print(name)
except:
    pass
" 2>/dev/null
}

# Read defaults from config
DEFAULT_MAX_TOKENS=$(read_config "defaults.max_new_tokens" "512")
DEFAULT_DEVICE=$(read_config "defaults.device" "auto")

# Default parameters
MODEL_PATH=""
MODEL_NAME=""
DATA_FILE=""
DATA_DIR="${PROJECT_DIR}/data/calibration"
OUTPUT_DIR="${PROJECT_DIR}/results"
MAX_NEW_TOKENS="$DEFAULT_MAX_TOKENS"
DEVICE="$DEFAULT_DEVICE"
RUN_ALL=false
FORCE=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Show help
show_help() {
    echo "============================================================================="
    echo "Unified Few-Shot collection of all pruning information"
    echo "============================================================================="
    echo ""
    echo "Usage:"
    echo "  $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model NAME|PATH   Model name (reads path from config) or full path"
    echo "  --data FILE             Single data file"
    echo "  --data-dir DIR          Data file directory (default: $DATA_DIR)"
    echo "  --all                   Iterate over all data files (*.json)"
    echo "  --output DIR            Output directory (default: $OUTPUT_DIR)"
    echo "  --max-tokens NUM        Maximum number of generated tokens (default: $DEFAULT_MAX_TOKENS)"
    echo "  --device DEVICE         Device (default: $DEFAULT_DEVICE)"
    echo "  -f, --force             Force recomputation (overwrite existing results)"
    echo "  --list-models           List all models in config"
    echo "  --help                  Show this help"
    echo ""
    echo "Config files:"
    echo "  Model config: $MODELS_CONFIG"
    echo "  Experiment config: $CONFIG_FILE"
    echo ""
    echo "Examples:"
    echo "  # Use model name (automatically reads path from config)"
    echo "  $0 -m qwen3-30b-a3b --all"
    echo ""
    echo "  # Use full model path"
    echo "  $0 -m /path/to/model --data ${DATA_DIR}/gsm8k_25.json"
    echo ""
    echo "  # List available models"
    echo "  $0 --list-models"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) 
            MODEL_INPUT="$2"
            # Check if it's a path or model name
            if [[ "$MODEL_INPUT" == /* ]] || [[ "$MODEL_INPUT" == ./* ]]; then
                # It's a path
                MODEL_PATH="$MODEL_INPUT"
                MODEL_NAME=$(basename "$MODEL_PATH")
            else
                # It's a model name, read path from config
                MODEL_NAME="$MODEL_INPUT"
                MODEL_PATH=$(get_model_path "$MODEL_NAME")
                if [ -z "$MODEL_PATH" ]; then
                    log_warning "Model '$MODEL_NAME' path not found in config, using name as path"
                    MODEL_PATH="$MODEL_NAME"
                fi
            fi
            shift 2 
            ;;
        --data) DATA_FILE="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --max-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        --list-models)
            echo "Available models in config:"
            list_available_models | while read model; do
                path=$(get_model_path "$model")
                echo "  - $model: $path"
            done
            exit 0
            ;;
        --help|-h) show_help; exit 0 ;;
        *) log_error "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

# Validate model argument
if [ -z "$MODEL_PATH" ]; then
    log_error "Must specify model (-m MODEL)"
    show_help
    exit 1
fi

cd "$SCRIPT_DIR"

# Function to run analysis on a single data file
run_single_analysis() {
    local data_file="$1"
    local dataset_name=$(basename "$data_file" .json)
    
    echo ""
    echo "--------------------------------"
    log_info "Processing dataset: $dataset_name"
    log_info "Model: $MODEL_NAME"
    echo "--------------------------------"
    
    # Check if data file exists
    if [ ! -f "$data_file" ]; then
        log_error "Data file does not exist: $data_file"
        return 1
    fi
    
    # Build command
    local cmd="python3 collect_activations.py"
    cmd="$cmd --model \"$MODEL_PATH\""
    cmd="$cmd --data \"$data_file\""
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    cmd="$cmd --max_new_tokens $MAX_NEW_TOKENS"
    cmd="$cmd --device $DEVICE"
    
    if [ "$FORCE" = true ]; then
        cmd="$cmd --force"
    fi
    
    # Run analysis
    if eval $cmd; then
        log_success "$dataset_name analysis completed"
        return 0
    else
        log_error "$dataset_name analysis failed"
        return 1
    fi
}

# =============================================================================
# Main logic
# =============================================================================

echo ""
echo "============================================================================="
log_info "Unified Few-Shot collection of all pruning information"
echo "============================================================================="
log_info "Model name: $MODEL_NAME"
log_info "Model path: $MODEL_PATH"
log_info "Output directory: $OUTPUT_DIR"
log_info "Generation: $MAX_NEW_TOKENS tokens"
log_info "Device: $DEVICE"
if [ "$FORCE" = true ]; then
    log_info "Mode: Force recomputation"
else
    log_info "Mode: Skip existing results"
fi
echo "============================================================================="

if [ "$RUN_ALL" = true ]; then
    # Iterate over all data files
    log_info "Data directory: $DATA_DIR"
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        log_error "Data directory does not exist: $DATA_DIR"
        exit 1
    fi
    
    # Find all JSON files
    data_files=("$DATA_DIR"/*.json)
    
    if [ ! -e "${data_files[0]}" ]; then
        log_error "No JSON files found in $DATA_DIR"
        exit 1
    fi
    
    total=${#data_files[@]}
    success=0
    failed=0
    failed_files=()
    
    log_info "Found $total data files"
    echo ""
    
    # Process each file
    for data_file in "${data_files[@]}"; do
        if run_single_analysis "$data_file"; then
            ((success++))
        else
            ((failed++))
            failed_files+=("$data_file")
        fi
    done
    
    # Output summary
    echo ""
    echo "============================================================================="
    log_info "Batch processing completed"
    echo "============================================================================="
    log_info "Total: $total datasets"
    log_success "Succeeded: $success"
    
    if [ $failed -gt 0 ]; then
        log_error "Failed: $failed"
        echo ""
        echo "Failed files:"
        for file in "${failed_files[@]}"; do
            echo "  - $file"
        done
    fi
    
    echo ""
    log_info "Results directory: ${OUTPUT_DIR}/${MODEL_NAME}/activations/"
    echo ""
    
elif [ -n "$DATA_FILE" ]; then
    # Single file processing
    log_info "Data: $DATA_FILE"
    echo ""
    
    if run_single_analysis "$DATA_FILE"; then
        echo ""
        echo "============================================================================="
        log_success "Analysis completed!"
        echo "============================================================================="
        log_info "Results directory: ${OUTPUT_DIR}/${MODEL_NAME}/activations/"
        echo ""
    else
        exit 1
    fi
else
    log_error "Please specify --data or --all"
    show_help
    exit 1
fi
