#!/bin/bash
# =============================================================================
# Universal Dataset Download Script
# =============================================================================
#
# Features:
#   Download datasets, can read dataset list from config file
#
# Usage:
#   ./run_download.sh [dataset_name] [num_samples] [options]
#
# =============================================================================

# Set up mirror (if needed)
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/experiments.yaml"

cd "$SCRIPT_DIR"

# Activate conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate lm-evaluation 2>/dev/null || conda activate lighteval 2>/dev/null || true
fi

# =============================================================================
# Functions for reading from config file
# =============================================================================

# Read dataset list from config
read_datasets_from_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo ""
        return
    fi
    
    python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    datasets = config.get('datasets', [])
    for ds in datasets:
        # Remove suffix to get dataset name
        name = ds.rsplit('_', 1)[0] if '_' in ds else ds
        print(name)
except:
    pass
" 2>/dev/null | sort -u
}

# =============================================================================
# Help information
# =============================================================================

show_help() {
    echo "============================================================================="
    echo "Universal Dataset Download Tool"
    echo "============================================================================="
    echo ""
    echo "Usage:"
    echo "  $0 [dataset_name] [num_samples] [options]"
    echo ""
    echo "Parameters:"
    echo "  dataset_name    Dataset name (default: gsm8k)"
    echo "  num_samples     Number of samples to extract (default: 25)"
    echo ""
    echo "Options:"
    echo "  --with-answers    Include answers (for few-shot)"
    echo "  --all             Download all datasets from config file"
    echo "  --list            List all available datasets"
    echo "  --list-config     List datasets from config file"
    echo "  --help            Show this help info"
    echo ""
    echo "Config file: $CONFIG_FILE"
    echo ""
    echo "Examples:"
    echo "  $0                              # Download gsm8k 25 entries"
    echo "  $0 gsm8k 50                     # Download gsm8k 50 entries"
    echo "  $0 hellaswag 100                # Download hellaswag 100 entries"
    echo "  $0 gsm8k 30 --with-answers      # Download gsm8k 30 entries (with answers)"
    echo "  $0 --all                        # Download all datasets from config"
    echo "  $0 --list                       # List available datasets"
    echo ""
}

# =============================================================================
# Parse arguments
# =============================================================================

DATASET=""
NUM_SAMPLES=25
WITH_ANSWERS=""
DOWNLOAD_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --list)
            python3 download_dataset.py --list
            exit 0
            ;;
        --list-config)
            echo "Datasets from config file:"
            read_datasets_from_config | while read ds; do
                echo "  - $ds"
            done
            exit 0
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --with-answers)
            WITH_ANSWERS="--with_answers"
            shift
            ;;
        [0-9]*)
            NUM_SAMPLES=$1
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            DATASET=$1
            shift
            ;;
    esac
done

# =============================================================================
# Download logic
# =============================================================================

if [ "$DOWNLOAD_ALL" = true ]; then
    # Download all datasets from config
    echo "============================================================================="
    echo "Batch download all datasets from config"
    echo "============================================================================="
    echo "Number of samples: $NUM_SAMPLES"
    if [ -n "$WITH_ANSWERS" ]; then
        echo "Include answers: yes"
    fi
    echo "============================================================================="
    echo ""
    
    CONFIG_DATASETS=$(read_datasets_from_config)
    
    if [ -z "$CONFIG_DATASETS" ]; then
        echo "Error: No dataset list found in config file"
        exit 1
    fi
    
    total=0
    success=0
    failed=0
    
    for ds in $CONFIG_DATASETS; do
        total=$((total + 1))
        echo ""
        echo "----------------------------------------"
        echo "Downloading: $ds ($NUM_SAMPLES entries)"
        echo "----------------------------------------"
        
        if python3 download_dataset.py \
            --dataset "$ds" \
            --num_samples "$NUM_SAMPLES" \
            $WITH_ANSWERS; then
            success=$((success + 1))
            echo "✓ $ds download complete"
        else
            failed=$((failed + 1))
            echo "✗ $ds download failed"
        fi
    done
    
    echo ""
    echo "============================================================================="
    echo "Batch download complete"
    echo "============================================================================="
    echo "Total: $total, Success: $success, Failed: $failed"
    echo "============================================================================="
    
else
    # Download single dataset
    if [ -z "$DATASET" ]; then
        DATASET="gsm8k"
    fi
    
    echo "============================================================================="
echo "Downloading dataset: $DATASET"
echo "Number of samples: $NUM_SAMPLES"
if [ -n "$WITH_ANSWERS" ]; then
    echo "Include answers: yes"
fi
    echo "============================================================================="
echo ""

# Run download
python3 download_dataset.py \
    --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" \
    $WITH_ANSWERS

echo ""
    echo "============================================================================="
echo "✓ Done!"
    echo "============================================================================="
fi
