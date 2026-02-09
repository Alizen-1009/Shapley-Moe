#!/bin/bash
# =============================================================================
# Unified Expert Selection Script
# =============================================================================
#
# Features:
#   Perform expert selection based on specified method, model, dataset, and pruning rate
#   Reads configuration preferentially from configs/experiments.yaml
#
# Usage:
#   ./run_select.sh [options]
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/experiments.yaml"
MODELS_CONFIG="${PROJECT_DIR}/configs/models.yaml"

# =============================================================================
# Read default values from config file
# =============================================================================

# General config reading function
read_config() {
    local key="$1"
    local default="$2"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$default"
        return
    fi
    
    local value=$(python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    # Support nested keys, e.g. 'defaults.pruning_rate'
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
except Exception as e:
    pass
" 2>/dev/null)
    
    if [ -n "$value" ]; then
        echo "$value"
    else
        echo "$default"
    fi
}

# Read list config
read_config_list() {
    local key="$1"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo ""
        return
    fi
    
    python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    keys = '$key'.split('.')
    result = config
    for k in keys:
        result = result.get(k, None)
        if result is None:
            break
    if result is not None and isinstance(result, list):
        print(' '.join(str(x) for x in result))
except:
    pass
" 2>/dev/null
}

# Read default values from config file
DEFAULT_RATE=$(read_config "defaults.pruning_rate" "0.5")
DEFAULT_METHOD=$(read_config "defaults.pruning_method" "shapley")
DEFAULT_STRATEGY=$(read_config "defaults.shapley_strategy" "alpha_per_layer")

# Default parameters
MODEL=""
DATASET=""
METHOD="$DEFAULT_METHOD"
RATE="$DEFAULT_RATE"
STRATEGY="$DEFAULT_STRATEGY"
OUTPUT_DIR=""

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

show_help() {
    echo "============================================================================="
    echo "Unified Expert Selection Script"
    echo "============================================================================="
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL       Model name (e.g. qwen3-30b-a3b, gpt-oss-20b)"
    echo "  -d, --dataset DATASET   Dataset name (e.g. gsm8k_25, arc_easy_25)"
    echo "  -M, --method METHOD     Pruning method (default: $DEFAULT_METHOD)"
    echo "                          Options: shapley|easyep|reap|gating|frequency|random"
    echo "  -r, --rate RATE         Retention rate (default: $DEFAULT_RATE)"
    echo "  -s, --strategy STRATEGY Shapley strategy (default: $DEFAULT_STRATEGY)"
    echo "                          alpha_per_layer - Per-layer Alpha coverage (recommended)"
    echo "                          alpha_global    - Global Alpha coverage"
    echo "                          topk_per_layer  - Per-layer Top-K"
    echo "                          topk_global     - Global Top-K"
    echo "  -o, --output DIR        Output directory (default: results/{model}/selected_experts/)"
    echo "  --all-datasets          Process all datasets"
    echo "  --all-rates             Use all pruning rates from config file"
    echo "  --all-methods           Use all pruning methods from config file"
    echo "  -h, --help              Show help"
    echo ""
    echo "Config file: $CONFIG_FILE"
    echo ""
    echo "Examples:"
    echo "  $0 -m qwen3-30b-a3b -d gsm8k_25 -M shapley -r 0.5"
    echo "  $0 -m gpt-oss-20b --all-datasets -M easyep --all-rates"
    echo "  $0 -m deepseekv2-lite-coder -d arc_easy_25 --all-methods -r 0.5"
    echo ""
}

# Parse arguments
ALL_DATASETS=false
ALL_RATES=false
ALL_METHODS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2 ;;
        -d|--dataset) DATASET="$2"; shift 2 ;;
        -M|--method) METHOD="$2"; shift 2 ;;
        -r|--rate) RATE="$2"; shift 2 ;;
        -s|--strategy) STRATEGY="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --all-datasets) ALL_DATASETS=true; shift ;;
        --all-rates) ALL_RATES=true; shift ;;
        --all-methods) ALL_METHODS=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Validate parameters
if [ -z "$MODEL" ]; then
    log_error "Must specify model name (-m MODEL)"
    show_help
    exit 1
fi

# Set default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${PROJECT_DIR}/results/${MODEL}/selected_experts"
fi
mkdir -p "$OUTPUT_DIR"

# Get dataset list
if [ "$ALL_DATASETS" = true ]; then
    # Get dataset list from activations directory (new format: {dataset}_shapley.json)
    DATASETS=($(ls ${PROJECT_DIR}/results/${MODEL}/activations/*_shapley.json 2>/dev/null | xargs -n1 basename | sed 's/_shapley.json//' || echo ""))
    if [ ${#DATASETS[@]} -eq 0 ]; then
        log_error "No activation data found for model ${MODEL}"
        exit 1
    fi
else
    if [ -z "$DATASET" ]; then
        log_error "Must specify dataset (-d DATASET) or use --all-datasets"
        exit 1
    fi
    DATASETS=("$DATASET")
fi

# Get pruning rate list (from config file)
if [ "$ALL_RATES" = true ]; then
    CONFIG_RATES=$(read_config_list "pruning_rates")
    if [ -n "$CONFIG_RATES" ]; then
        RATES=($CONFIG_RATES)
        log_info "Read pruning rates from config file: ${RATES[*]}"
    else
        RATES=("0.80" "0.60")
        log_warning "No pruning rates found in config file, using defaults: ${RATES[*]}"
    fi
else
    RATES=("$RATE")
fi

# Get method list (from config file)
if [ "$ALL_METHODS" = true ]; then
    CONFIG_METHODS=$(read_config_list "pruning_methods")
    if [ -n "$CONFIG_METHODS" ]; then
        METHODS=($CONFIG_METHODS)
        log_info "Read pruning methods from config file: ${METHODS[*]}"
    else
    METHODS=("shapley" "easyep" "reap" "gating" "frequency" "random")
        log_warning "No pruning methods found in config file, using defaults"
    fi
else
    METHODS=("$METHOD")
fi

# Execute expert selection
log_info "============================================================================="
log_info "Expert Selection"
log_info "============================================================================="
log_info "Model: $MODEL"
log_info "Datasets: ${DATASETS[*]}"
log_info "Methods: ${METHODS[*]}"
log_info "Pruning rates: ${RATES[*]}"
log_info "Shapley strategy: $STRATEGY"
log_info "Output directory: $OUTPUT_DIR"
log_info "============================================================================="

total=0
success=0
failed=0

for ds in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for rate in "${RATES[@]}"; do
            total=$((total + 1))
            
            # Determine input file (new directory structure)
            case $method in
                shapley)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/shapley_values/${ds}_shapley.csv"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_shapley.py"
                    ;;
                easyep)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_easyep.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_easyep.py"
                    ;;
                reap)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_reap.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_reap.py"
                    ;;
                gating)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_gating.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_gating.py"
                    ;;
                frequency)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_shapley.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_frequency.py"
                    ;;
                random)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_shapley.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_random.py"
                    ;;
            esac
            
            # Check input file
            if [ ! -f "$INPUT_FILE" ]; then
                log_error "Input file does not exist: $INPUT_FILE"
                failed=$((failed + 1))
                continue
            fi
            
            # Generate output filename (including strategy info)
            RATE_STR=$(echo "$rate" | sed 's/\./_/g')
            # Shapley method needs to distinguish strategy, others use per_layer
            if [ "$method" = "shapley" ]; then
                OUTPUT_FILE="${OUTPUT_DIR}/${method}_${STRATEGY}_${ds}_rate${RATE_STR}.json"
            else
                OUTPUT_FILE="${OUTPUT_DIR}/${method}_${ds}_rate${RATE_STR}.json"
            fi
            
            log_info "Processing: ${method} / ${ds} / rate=${rate}"
            
            # Run script
            if python3 "$SCRIPT" \
                --input "$INPUT_FILE" \
                --output "$OUTPUT_FILE" \
                --pruning_rate "$rate" \
                --strategy "$STRATEGY" 2>&1 | grep -E "^(✓|Selection|Retained)" ; then
                log_success "  ✓ Done: $OUTPUT_FILE"
                success=$((success + 1))
            else
                log_error "  ✗ Failed"
                failed=$((failed + 1))
            fi
        done
    done
done

log_info "============================================================================="
log_info "Done! Success: $success / $total, Failed: $failed"
log_info "============================================================================="
