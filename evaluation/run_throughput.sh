#!/usr/bin/env bash
# Benchmark dense and SHAPE post-top-k models sequentially on one GPU.
set -euo pipefail

GPU=${1:-0}
PORT=${2:-8801}
ROOT=${SHAPE_ROOT:-/root/workspace/chuanwu/Shapley-Moe}
RIY_ROOT=${RIY_ROOT:-/root/workspace/chuanwu/vllm-moe_pruning}
MODEL=${MODEL_PATH:-/root/workspace/chuanwu/models/Qwen3-30B-A3B-Instruct-2507}
VLLM_BIN=${VLLM_BIN:-/root/workspace/chuanwu/venvs/shape-vllm/bin}
PROFILE_DIR="$ROOT/results/qwen3-30b-a3b/throughput_profiles"
OUT_ROOT=${THROUGHPUT_OUT_ROOT:-$ROOT/results/qwen3-30b-a3b/throughput_eval/fixed_512in_128out_c128}
NUM_PROMPTS=${THROUGHPUT_NUM_PROMPTS:-1000}
REPEATS=${THROUGHPUT_REPEATS:-3}
MODEL_REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
VARIANTS=(dense keep_0_8 keep_0_6)
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
      echo "vLLM exited before becoming ready" >&2
      tail -100 "$log" >&2
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for vLLM" >&2
  tail -100 "$log" >&2
  return 1
}

run_benchmark() {
  local num_prompts=$1
  local seed=$2
  local result_dir=$3
  local result_filename=$4
  "$VLLM_BIN/vllm" bench serve \
    --backend vllm \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/completions \
    --model shape-qwen3 \
    --dataset-name random \
    --num-prompts "$num_prompts" \
    --random-input-len 512 \
    --random-output-len 128 \
    --random-range-ratio 0 \
    --request-rate inf \
    --max-concurrency 128 \
    --ignore-eos \
    --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --result-dir "$result_dir" \
    --result-filename "$result_filename"
}

for variant in "${VARIANTS[@]}"; do
  variant_dir="$OUT_ROOT/$variant"
  mkdir -p "$variant_dir"
  complete=true
  for repeat in $(seq 1 "$REPEATS"); do
    [[ -f "$variant_dir/repeat_${repeat}.json" ]] || complete=false
  done
  if [[ "$complete" == true ]]; then
    echo "[$variant] all repeats exist; skipping"
    continue
  fi

  profile_args=()
  if [[ "$variant" != dense ]]; then
    profile="$PROFILE_DIR/$variant.json"
    [[ -f "$profile" ]] || { echo "Missing profile: $profile" >&2; exit 1; }
    profile_args=(--riy-expert-profile "$profile")
  fi

  cat > "$variant_dir/manifest.txt" <<EOF
started_at=$(date --iso-8601=seconds)
variant=$variant
gpu=$GPU
model=$MODEL
model_revision=$MODEL_REVISION
riy_source_commit=$(git -C "$RIY_ROOT" rev-parse HEAD)
shape_commit=$(git -C "$ROOT" rev-parse HEAD)
vllm_version=$($VLLM_BIN/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
torch_version=$($VLLM_BIN/python -c 'import torch; print(torch.__version__)' 2>/dev/null)
compiled_wheel_commit=e222c33f2
moe_backend=triton
dtype=bfloat16
tensor_parallel_size=1
max_model_len=4096
gpu_memory_utilization=0.8
num_prompts=$NUM_PROMPTS
input_tokens=512
output_tokens=128
request_rate=inf
max_concurrency=128
ignore_eos=true
repeats=$REPEATS
seed=42
EOF

  server_log="$variant_dir/server.log"
  echo "[$variant] starting server on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN/vllm" serve "$MODEL" \
    --served-model-name shape-qwen3 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --moe-backend triton \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 256 \
    --no-enable-prefix-caching \
    "${profile_args[@]}" >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$server_log"

  echo "[$variant] warmup"
  run_benchmark 64 41 "$variant_dir" warmup.json >"$variant_dir/warmup.log" 2>&1
  rm -f "$variant_dir/warmup.json"

  for repeat in $(seq 1 "$REPEATS"); do
    result="$variant_dir/repeat_${repeat}.json"
    if [[ -f "$result" ]]; then
      echo "[$variant] repeat $repeat exists; skipping"
      continue
    fi
    echo "[$variant] measured repeat $repeat/$REPEATS"
    run_benchmark "$NUM_PROMPTS" 42 "$variant_dir" "repeat_${repeat}.json" \
      >"$variant_dir/repeat_${repeat}.log" 2>&1
    [[ -f "$result" ]] || { echo "Missing benchmark result: $result" >&2; exit 1; }
  done

  stop_server
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "$variant_dir/manifest.txt"
  echo "[$variant] complete"
done

echo "All throughput variants complete"
