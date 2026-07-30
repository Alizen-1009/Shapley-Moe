#!/usr/bin/env bash
# Run one task's dense and matched post-top-k faithfulness evaluations.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_faithfulness.sh DATASET GPU PORT [--force]

DATASET: gsm8k_25 | humaneval_25 | med_mcqa_25
GPU:     CUDA device index reserved for this run
PORT:    local OpenAI-compatible API port
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

DATASET=$1
GPU=$2
PORT=$3
FORCE=${4:-}
if [[ -n "$FORCE" && "$FORCE" != "--force" ]]; then
  usage >&2
  exit 2
fi

case "$DATASET" in
  gsm8k_25) EVAL_DATASET=gsm8k ;;
  humaneval_25) EVAL_DATASET=humaneval ;;
  med_mcqa_25) EVAL_DATASET=med_mcqa ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

ROOT=${SHAPE_ROOT:-/root/workspace/chuanwu/Shapley-Moe}
RIY_ROOT=${RIY_ROOT:-/root/workspace/chuanwu/vllm-moe_pruning}
MODEL=${MODEL_PATH:-/root/workspace/chuanwu/models/Qwen3-30B-A3B-Instruct-2507}
VLLM_PY=${VLLM_PY:-/root/workspace/chuanwu/venvs/shape-vllm/bin}
EVAL_PY=${EVAL_PY:-/root/workspace/chuanwu/venvs/shape-eval/bin}
PROFILE_DIR="$ROOT/results/qwen3-30b-a3b/faithfulness_profiles/$DATASET"
OUT_ROOT="$ROOT/results/qwen3-30b-a3b/faithfulness_eval/$DATASET"
MODEL_REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
VARIANTS=(dense remove_low random_seed42 random_seed43 random_seed44 remove_high)
SERVER_PID=

mkdir -p "$OUT_ROOT"

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap stop_server EXIT INT TERM

wait_for_server() {
  local log=$1
  for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited before becoming ready; tail of $log:" >&2
      tail -100 "$log" >&2
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for vLLM on port $PORT" >&2
  tail -100 "$log" >&2
  return 1
}

for variant in "${VARIANTS[@]}"; do
  variant_dir="$OUT_ROOT/$variant"
  report="$variant_dir/eval/reports/shape-qwen3/$EVAL_DATASET.json"
  if [[ -f "$report" && "$FORCE" != "--force" ]]; then
    echo "[$DATASET/$variant] report exists; skipping"
    continue
  fi

  mkdir -p "$variant_dir"
  server_log="$variant_dir/server.log"
  eval_log="$variant_dir/eval.log"
  profile_args=()
  if [[ "$variant" != dense ]]; then
    profile="$PROFILE_DIR/$variant.json"
    [[ -f "$profile" ]] || { echo "Missing profile: $profile" >&2; exit 1; }
    profile_args=(--riy-expert-profile "$profile")
  fi

  cat > "$variant_dir/manifest.txt" <<EOF
started_at=$(date --iso-8601=seconds)
dataset=$DATASET
eval_dataset=$EVAL_DATASET
variant=$variant
gpu=$GPU
port=$PORT
model=$MODEL
model_revision=$MODEL_REVISION
riy_source_commit=$(git -C "$RIY_ROOT" rev-parse HEAD)
shape_commit=$(git -C "$ROOT" rev-parse HEAD)
torch_version=$($VLLM_PY/python -c 'import torch; print(torch.__version__)' 2>/dev/null)
vllm_version=$($VLLM_PY/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
evalscope_version=$($EVAL_PY/python -c 'import evalscope; print(evalscope.__version__)' 2>/dev/null)
moe_backend=triton
dtype=bfloat16
tensor_parallel_size=1
max_model_len=24576
max_tokens=20480
temperature=0
seed=0
eval_batch_size=128
EOF

  echo "[$DATASET/$variant] starting vLLM on GPU $GPU port $PORT"
  CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PY/vllm" serve "$MODEL" \
    --served-model-name shape-qwen3 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --moe-backend triton \
    --dtype bfloat16 \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 128 \
    --enforce-eager \
    "${profile_args[@]}" >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$server_log"

  echo "[$DATASET/$variant] running EvalScope"
  set +e
  "$EVAL_PY/evalscope" eval \
    --model shape-qwen3 \
    --api-url "http://127.0.0.1:$PORT/v1" \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets "$EVAL_DATASET" \
    --generation-config '{"max_tokens":20480,"temperature":0,"seed":0,"timeout":600}' \
    --eval-batch-size 128 \
    --seed 0 \
    --work-dir "$variant_dir/eval" \
    --no-timestamp \
    --no-collect-perf >"$eval_log" 2>&1
  eval_status=$?
  set -e
  stop_server

  if [[ $eval_status -ne 0 || ! -f "$report" ]]; then
    echo "[$DATASET/$variant] evaluation failed (status=$eval_status)" >&2
    tail -100 "$eval_log" >&2
    exit 1
  fi
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "$variant_dir/manifest.txt"
  echo "[$DATASET/$variant] complete: $report"
done

echo "[$DATASET] all faithfulness variants complete"
