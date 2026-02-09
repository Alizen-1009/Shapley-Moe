#!/bin/bash
# =============================================================================
# Batch Shapley value calculation
# =============================================================================
#
# Features:
#   Calculates Shapley values for each expert from activation statistics JSON files
#
# Usage:
#   ./run_calc_shapley.sh [options]
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
MODEL=""
DATASET=""
NUM_EXPERTS=""
FORCE=false
RUN_ALL=false

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
    echo "Batch Shapley value calculation"
    echo "============================================================================="
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL       Model name (required)"
    echo "  -d, --dataset DATASET   Dataset name (optional, processes all by default)"
    echo "  -n, --num-experts NUM   Number of experts per layer (optional, auto-detected)"
    echo "  -f, --force             Force recomputation (overwrite existing files)"
    echo "  --all                   Process all datasets"
    echo "  -h, --help              Show help information"
    echo ""
    echo "Examples:"
    echo "  $0 -m gpt-oss-20b                     # Calculate for all datasets"
    echo "  $0 -m qwen3-30b-a3b -d gsm8k_25       # Calculate for a single dataset"
    echo "  $0 -m deepseekv2-lite-coder -f       # Force recomputation"
    echo ""
}

# Get number of experts from config file
get_num_experts_from_config() {
    local model_name="$1"
    
    local config_file="${PROJECT_DIR}/configs/models.yaml"
    if [ ! -f "$config_file" ]; then
        echo ""
        return
    fi
    
    # Use python to parse YAML
    local num=$(python3 -c "
import yaml
try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    models = config.get('models', {})
    if '$model_name' in models:
        print(models['$model_name'].get('num_experts', ''))
except:
    pass
" 2>/dev/null)
    
    echo "$num"
}

# Auto-detect number of experts (from JSON file)
detect_num_experts_from_json() {
    local json_file="$1"
    
    # Find the maximum expert ID from JSON file
    local max_expert=$(python3 -c "
import json
import ast
with open('$json_file', 'r') as f:
    data = json.load(f)
max_id = 0
for layer_data in data.get('layers', {}).values():
    for combo_str in layer_data.keys():
        try:
            combo = ast.literal_eval(combo_str)
            max_id = max(max_id, max(combo))
        except:
            pass
print(max_id + 1)
" 2>/dev/null)
    
    echo "$max_expert"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2 ;;
        -d|--dataset) DATASET="$2"; shift 2 ;;
        -n|--num-experts) NUM_EXPERTS="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        --all) RUN_ALL=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Validate arguments
if [ -z "$MODEL" ]; then
    log_error "Must specify model name (-m MODEL)"
    show_help
    exit 1
fi

# Set directories
INPUT_DIR="${PROJECT_DIR}/results/${MODEL}/activations"
OUTPUT_DIR="${PROJECT_DIR}/results/${MODEL}/shapley_values"

# Check input directory
if [ ! -d "$INPUT_DIR" ]; then
    log_error "Activation data directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================================="
log_info "Batch Shapley value calculation"
echo "============================================================================="
log_info "Model: $MODEL"
log_info "Input directory: $INPUT_DIR"
log_info "Output directory: $OUTPUT_DIR"
if [ "$FORCE" = true ]; then
    log_warning "Mode: Force recomputation"
else
    log_info "Mode: Skip existing results"
fi
echo "============================================================================="

# Get dataset list
if [ -n "$DATASET" ]; then
    # Single dataset specified
    DATASETS=("$DATASET")
else
    # Process all datasets
    DATASETS=($(ls ${INPUT_DIR}/*_shapley.json 2>/dev/null | xargs -n1 basename | sed 's/_shapley.json//' || echo ""))
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
    log_error "No activation data files found"
    exit 1
fi

log_info "Datasets: ${DATASETS[*]}"
echo ""

# Statistics
total=0
success=0
skipped=0
failed=0

# Process each dataset
for ds in "${DATASETS[@]}"; do
    total=$((total + 1))
    
    input_file="${INPUT_DIR}/${ds}_shapley.json"
    output_file="${OUTPUT_DIR}/${ds}_shapley.csv"
    
    echo "----------------------------------------"
    log_info "Processing: $ds"
    
    # Check input file
    if [ ! -f "$input_file" ]; then
        log_error "Input file does not exist: $input_file"
        failed=$((failed + 1))
        continue
    fi
    
    # Check if output file already exists
    if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
        log_success "Already exists, skipping"
        skipped=$((skipped + 1))
        continue
    fi
    
    # Get number of experts
    if [ -z "$NUM_EXPERTS" ]; then
        # 1. First try to get from config file
        num_exp=$(get_num_experts_from_config "$MODEL")
        if [ -n "$num_exp" ] && [ "$num_exp" -gt 0 ] 2>/dev/null; then
            log_info "Got number of experts from config: $num_exp"
        else
            # 2. Detect from JSON file
            detected=$(detect_num_experts_from_json "$input_file")
            if [ -n "$detected" ] && [ "$detected" -gt 0 ] && [ "$detected" -lt 1000 ]; then
                num_exp="$detected"
                log_info "Detected number of experts: $num_exp"
            else
                num_exp=64
                log_warning "Cannot determine number of experts, using default: $num_exp"
            fi
        fi
    else
        num_exp="$NUM_EXPERTS"
        log_info "Using specified number of experts: $num_exp"
    fi
    
    # Run calculation
    log_info "Starting calculation..."
    if python3 "${SCRIPT_DIR}/calc_shapley.py" \
        --input_file "$input_file" \
        --output_csv "$output_file" \
        --num_experts "$num_exp" 2>&1 | grep -E "^(✓|Processing|Saving|Layer)" | tail -5; then
        
        if [ -f "$output_file" ]; then
            log_success "Done: $output_file"
            success=$((success + 1))
        else
            log_error "Output file not generated"
            failed=$((failed + 1))
        fi
    else
        log_error "Calculation failed"
        failed=$((failed + 1))
    fi
done

echo ""
echo "============================================================================="
log_info "Statistics"
echo "============================================================================="
log_info "Total: $total"
log_success "Succeeded: $success"
if [ $skipped -gt 0 ]; then
    log_info "Skipped: $skipped"
fi
if [ $failed -gt 0 ]; then
    log_error "Failed: $failed"
fi
echo ""

# List all generated files
log_info "Shapley value files:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

echo ""
echo "============================================================================="
log_success "Done!"
echo "============================================================================="
