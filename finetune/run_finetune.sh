#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${PRUNED_BASE:-/root/autodl-tmp}/qwen3-30b-a3b_rate0_8_pruned"
RANK_MAP="${PROJECT_DIR}/results/qwen3-30b-a3b/lora_rank_maps/gsm8k_25_rate0_8_bucket.json"

# SFT training data with answers (download with: data/download_sft.sh gsm8k)
TRAIN_FILE="${PROJECT_DIR}/data/sft/gsm8k_sft.json"

# Save LoRA outputs on the large data disk.
OUTPUT_DIR="${ADAPTER_BASE:-/root/autodl-tmp/lora_outputs}/qwen3_gsm8k_rate0_8_bucket"

mkdir -p "$(dirname "$OUTPUT_DIR")"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Pruned model directory not found: $MODEL_PATH"
  exit 1
fi

if [ ! -f "$RANK_MAP" ]; then
  echo "Rank map not found: $RANK_MAP"
  exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Training file not found: $TRAIN_FILE"
  echo ""
  echo "Prepare a supervised training file first, for example:"
  echo "  /root/autodl-tmp/data/gsm8k_sft.jsonl"
  echo ""
  echo "Supported record formats include:"
  echo '  {"question": "...", "answer": "..."}'
  echo '  {"prompt": "...", "response": "..."}'
  echo '  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
  echo ""
  echo "If you only want a smoke test, you can temporarily point TRAIN_FILE to:"
  echo "  ${PROJECT_DIR}/data/calibration/gsm8k_25.json"
  echo "but that file contains only questions and is not suitable for real SFT."
  exit 1
fi

cd "$PROJECT_DIR"

python finetune/train_adaptive_lora.py \
  --model_path "$MODEL_PATH" \
  --rank_map "$RANK_MAP" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --model_type qwen3 \
  --torch_dtype bf16 \
  --bf16 \
  --gradient_checkpointing \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 2 \
  --report_to none \
  --overwrite_output_dir
