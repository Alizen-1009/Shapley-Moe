#!/bin/bash
# Download datasets with answers for SFT fine-tuning.
# Output goes to data/sft/{dataset}_sft.json with question/answer fields.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="${SCRIPT_DIR}/sft"
mkdir -p "$SFT_DIR"

DATASETS="${@:-gsm8k}"

for dataset in $DATASETS; do
    echo "Downloading ${dataset} with answers..."
    python3 "${SCRIPT_DIR}/download_dataset.py" \
        --dataset "$dataset" \
        --all_samples \
        --with_answers \
        --output "${SFT_DIR}/${dataset}_sft.json"
    echo "Done: ${SFT_DIR}/${dataset}_sft.json"
    echo ""
done
