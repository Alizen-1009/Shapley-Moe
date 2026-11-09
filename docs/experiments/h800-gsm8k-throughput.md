# H800: real GSM8K serving throughput

## Protocol

Compare the original Qwen3-30B-A3B-Instruct-2507 with the existing SHAPE keep-0.8 and keep-0.6 selections. Use one H800 (index 1, rechecked for occupancy), BF16, TP=1, Triton MoE and Triton attention. Do not modify or duplicate checkpoint weights.

- Dataset: all 1,319 GSM8K **test** questions, original order. Source revision and SHA256: `data/gsm8k-throughput-source.json`.
- Zero-shot chat: user question followed by `Please reason step by step, and put your final numeric answer after '#### '.` Model's local chat template is applied by the server.
- Sampling: temperature 0, top-p 1, seed 42, maximum 2,048 generated tokens. EOS is respected; do not force a fixed output length.
- Concurrency: 128 simultaneous client requests and 128 maximum server sequences; unlimited request arrival rate within that concurrency limit.
- Server: max model length 4,096, max batched tokens 8,192, GPU memory utilization 0.85, prefix caching off, FlashInfer sampler and RIY monitoring off.
- Formal run uses torch.compile and CUDA graphs, capture size capped at 128. An eager smoke does not establish graph-mode readiness; a separate 8-question pilot checks all three variants first.
- Warmup: first 256 questions, with the same generation settings, excluded from measurement. Then 3 measured full-test-set passes per variant. Variants run sequentially: original, keep-0.8, keep-0.6.

Primary metric is **sum of server-reported successful completion tokens / whole measured batch wall time**. Also save request throughput, actual output lengths, length-capped counts, failures and p50/p95 non-streaming request latency. Request latency excludes client semaphore queue time; batch wall time includes draining all requests. These are not TTFT/ITL measurements.

Repeated runs are deterministic in settings, but numerical execution may still vary. Report the arithmetic mean and sample standard deviation of the 3 per-run throughput values. Speedup is the ratio of throughput means, not a ratio of request counts.

## Fairness and limits

- Every variant uses the same GPU, request set, template, limits and client concurrency. Server logs are retained to identify kernel/config fallbacks.
- Each pruned variant must log `post_topk_drop` and correct per-layer retained-expert counts for all 48 layers before measurement.
- Failures are retained, not retried or silently dropped. A failed measured batch invalidates the experiment and stops further runs.
- GPU occupancy is checked before and after measured batches. A foreign GPU process invalidates the batch. Other GPUs and the host may still have other users' jobs; this is not an isolated-host benchmark, and transient contention between checks cannot be excluded.
- Using equal GPU memory utilization gives the smaller models more KV-cache capacity. This is an equal-deployment-budget test, not a fixed-KV-capacity microbenchmark.
- Real generated lengths can change after pruning. Report token lengths and truncation alongside throughput; do not infer speedup from shorter answers or claim this is a fixed-token workload.
- This run does not grade GSM8K accuracy. The 910C accuracy experiment is separate.
- Existing actual keep fractions are approximately 79.93% and 59.78%; do not change selections to round them.

## Runner

Use the isolated environment, not generic project requirements:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shape-vllm
export TMPDIR=$HOME/shape-experiments/tmp

# ROOT must contain the runner, runtime config and existing profile/selection files.
ROOT=$HOME/shape-experiments/shape
DATA=$HOME/shape-experiments/datasets/gsm8k/test.jsonl
MODEL=$HOME/model-dir

# Separate graph-mode pilot; never report its 8-question throughput as the final result.
python "$ROOT/evaluation/run_gsm8k_throughput.py" \
  --root "$ROOT" --dataset "$DATA" --model "$MODEL" --gpu 1 \
  --output "$HOME/shape-experiments/runs/gsm8k_graph_pilot" \
  --limit 8 --warmup 8 --repeats 1

# Formal experiment; default is all 1,319 questions, 256 warmup, 3 repeats.
python "$ROOT/evaluation/run_gsm8k_throughput.py" \
  --root "$ROOT" --dataset "$DATA" --model "$MODEL" --gpu 1 \
  --output "$HOME/shape-experiments/runs/gsm8k_c128_max2048_r3_graph"
```

Always use a **new output directory**; the runner refuses to overwrite an existing manifest. On startup/measurement failure, inspect `failure.json`, the launcher log and the active variant's `server.log`. Do not mix eager and graph-mode variants into one comparison.

## Artifacts

Each run saves its manifest, exact runner source, server commands/logs, warmup summaries, profile verification, per-repeat metrics and gzip-compressed per-request completions. Final `summary.json` is written only after every requested variant completes. `summary.partial.json` is explicitly incomplete.

After completion, independently recompute accounting from all 11,871 archived request records:

```bash
python "$ROOT/tools/verify_gsm8k_throughput.py" \
  "$HOME/shape-experiments/runs/gsm8k_c128_max2048_r3_graph_run2"
```

The verifier checks 9 complete passes, unique question indices, identical per-question input token lengths, matching server settings, source snapshot hash, graph-capture evidence, retained-expert logs, raw token totals, wall-time throughput and aggregation. It writes `verification.json`; this is not answer-accuracy or numerical routing-equivalence verification.

Preserve the full remote run. Copy small metric summaries and manifests into `results/qwen3-30b-a3b/` for Git; archive large responses/logs outside Git (for example `.outputs/h800/`). The reference dataset is regenerable from its pinned source and hash.

No exported pruned checkpoints are needed. The runner stops only the server process group it created, and leaves original weights and other users' jobs unchanged.

## Execution record (2026-11-09)

- Three graph-mode pilots (8 measured questions, 8 warmup, 1 repeat each) completed with zero failed or capped requests. Small pilot artifacts are in `results/qwen3-30b-a3b/h800_gsm8k_graph_pilot/`; these are **not full-test-set results**.
- The first formal launcher was stopped before model startup by a port-availability check that treated TCP TIME_WAIT as an occupied listener. A local TCP reproducer showed plain bind failing while `SO_REUSEADDR` succeeded; the precheck now uses the same reuse policy as the server. A regression test verifies both TIME_WAIT acceptance and rejection of a real listener. No measured data was generated by that attempt.
- Formal run directory: `/home/zhangyh/shape-experiments/runs/gsm8k_c128_max2048_r3_graph_run2`.
- Launcher log: `/home/zhangyh/shape-experiments/logs/gsm8k-formal-run2.log`.
- Exact launch command/PID: `/home/zhangyh/shape-experiments/artifacts/gsm8k-formal-process.json`.
- Metric/profile and port-precheck tests pass locally and in the H800 environment.

The formal run completed and its accounting was independently verified on H800 and locally. All 11,871 requests succeeded. Mean output throughput (sample SD, 3 repeats): original **2,854.8 ± 2.4 tok/s**, keep-0.8 **2,826.6 ± 20.4 tok/s** (-0.99%), keep-0.6 **3,266.1 ± 7.8 tok/s** (+14.41%). Output lengths and truncation differ; see the full interpretation and caveats in `results/qwen3-30b-a3b/h800_gsm8k_throughput/README.md`.

Small result artifacts are stored in that results directory. The complete archive, including compressed completions and server logs, is retained remotely and at `.outputs/h800/gsm8k_c128_max2048_r3_graph_run2/` locally. The runner's servers have stopped and GPU 1 was verified free. No full pruned checkpoint copies were needed.
