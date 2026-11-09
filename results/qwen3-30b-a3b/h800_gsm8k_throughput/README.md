# H800 GSM8K throughput results

Completed and independently verified on 2026-11-09. Single NVIDIA H800 PCIe, Qwen3-30B-A3B-Instruct-2507 BF16, TP=1, concurrency 128, Triton attention/MoE, torch.compile + CUDA graphs. Each variant: 256 warmup prompts excluded, then all 1,319 GSM8K test questions repeated 3 times. Temperature 0, seed 42, max output 2,048 tokens, EOS respected, prefix caching disabled.

| Variant | Output tokens/s, mean ± sample SD | Relative to original | Mean output tokens/request | Requests/s, mean | Capped requests / 3,957 |
|---|---:|---:|---:|---:|---:|
| Original | 2,854.8 ± 2.4 | 1.0000× | 281.03 | 10.158 | 60 (1.52%) |
| Keep 0.8 (prune ~20%) | 2,826.6 ± 20.4 | 0.9901× (-0.99%) | 270.66 | 10.443 | 35 (0.88%) |
| Keep 0.6 (prune ~40%) | 3,266.1 ± 7.8 | 1.1441× (+14.41%) | 266.90 | 12.237 | 24 (0.61%) |

## Interpretation

Under this specific real-prompt serving workload, keep-0.8 did not improve output-token throughput; keep-0.6 improved it by approximately 14.4%. Do not claim every pruning rate speeds up serving. Actual retained-expert fractions are 0.79931640625 and 0.5978190104166666, using the unmodified existing expert selections.

Output lengths differ, and some generations reach the token cap. This is **not a fixed-input/fixed-output kernel microbenchmark**, and these measurements do not establish GSM8K accuracy. Request throughput improves by a different percentage because answer lengths change. Both kernels and tail draining can affect aggregate output-token throughput. All three variants used the same input questions, per-question input token counts, prompt template, generation limits and server budgets.

The server reported missing device-specific tuned Triton MoE configs and used generic configurations, including for the original E=128 case. No tuning or source changes were made to improve a particular result. Equal GPU memory utilization allows the pruned models larger KV caches; this is an equal-deployment-budget comparison, not equal KV capacity. Other GPUs on the host had unrelated jobs; the selected GPU was checked for foreign processes before/after measured batches, but host isolation or transient contention is not proven.

## Verification and artifacts

- 9 complete measured passes; **11,871 successful requests, zero failures**.
- Per-request archived token usage, wall times, throughput, percentiles and final aggregation recomputed successfully on H800 and again locally.
- All per-question input token counts match across the 9 passes.
- Logs confirm CUDA-graph capture, torch.compile, and correct `post_topk_drop` expert counts in all 48 layers for both pruned variants.
- No server errors or preemption messages were found in the saved server logs.
- Accuracy and numerical equivalence against a zero-weight reference were **not** checked.

Files here: `summary.json`, `manifest.json`, `verification.json`, and per-variant/per-repeat metrics and server commands. Larger logs, exact runner snapshot and compressed per-request completions are preserved at:

- Remote: `/home/zhangyh/shape-experiments/runs/gsm8k_c128_max2048_r3_graph_run2/`
- Local ignored archive: `.outputs/h800/gsm8k_c128_max2048_r3_graph_run2/`

Recompute checks with:

```bash
python3 tools/verify_gsm8k_throughput.py .outputs/h800/gsm8k_c128_max2048_r3_graph_run2
```

Protocol and reproduction: `docs/experiments/h800-gsm8k-throughput.md`. Environment references: `configs/runtime/h800-vllm.json`, `requirements/h800-vllm.lock.txt`.

The benchmark's own servers were stopped and GPU 1 was verified at 0 MiB used after completion. No full pruned checkpoints were created or deleted; original weights and other users' processes were unchanged.
