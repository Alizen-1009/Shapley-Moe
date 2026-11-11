# GSM8K expert-routing comparison

## Status and purpose

**Completed and independently verified remotely and locally.** All 1,319 questions and three variants finished. Results and figures: `results/qwen3-30b-a3b/expert_routing_gsm8k/README.md`. This is a separate mechanism-analysis experiment, not a replacement or rerun of the completed H800 throughput benchmark. It tests whether expert removal reduces routed contributions proportionally to the stored expert count, and whether downstream routing changes.

Keep-0.8 removed 20.07% of stored experts but only 1.51% of selected contribution slots (prompt and reference-answer tokens combined). This supports a low-routed-contribution-reduction mechanism, **not a measurement of executed work**. Routing measurements do not by themselves causally explain the observed -0.99% output-token throughput change. Kernel execution, attention/router overhead, scheduling and different generated lengths remain possible factors.

## Fixed scope

- Model: Qwen3-30B-A3B-Instruct-2507 only; pin and record checkpoint/tokenizer revisions.
- Variants: original, SHAPE keep-0.8, SHAPE keep-0.6; no LoRA or new expert selection.
- Reuse unmodified `results/qwen3-30b-a3b/selected_experts/shapley_alpha_per_layer_gsm8k_25_rate0_8.json` and `shapley_alpha_per_layer_gsm8k_25_rate0_6.json`. Record checksums and actual retained fractions.
- Use all 1,319 GSM8K test questions and their reference solutions, with provenance matched to `data/gsm8k-throughput-source.json`. Never estimate new selections using these test examples.
- Prefer H800 for relevance to its completed throughput result; recheck availability before execution. Do not interrupt foreign jobs. This experiment does not require freeing the shared 910C machine.

## Controlled token protocol

Apply the same chat prompt as the completed H800 experiment with `add_generation_prompt=True`, then append the reference solution's token IDs (`add_special_tokens=False`). No assistant EOS is appended. Save the exact tokenized sequence manifest/checksum. Teacher-force the **same token IDs** through all three variants. Do not generate different answers for the primary routing comparison.

Record prompt and answer-token routing separately. Use the identical token-position masks for all variants; exclude padding and explicitly document treatment of template/BOS/EOS tokens. Do not silently truncate: establish a sufficient sequence budget, or report exclusions/truncation before a formal run. Run in evaluation mode with no gradients, a fixed dtype and backend, and consistent batching.

Answer-position statistics describe processing each gold token, not the preceding position predicting it: the final gold token is included, while the assistant-prefix position predicting the first answer token is excluded. This is a controlled reference-token workload, not a measurement of routing in the original free-generation throughput run. Teacher-forced answer-token routing is also not a decode-time kernel timing measurement.

## Semantics and validation gates

Distinguish original top-k selection, surviving selected contributions, and backend expert execution:

1. Compute top-k using the unchanged router and its native weighting/normalization rules.
2. Drop selected experts absent from the existing retention set, with no replacement and **no additional survivor renormalization**.
3. Retain original expert IDs; do not confuse compacted local IDs with original IDs.

A zero-weight implementation is suitable for measuring contribution semantics, but may still execute zero-weight expert kernels. Do not label surviving contributions as measured executed work or infer kernel speedup from this implementation.

Before collection, validate selections (all layers, unique in-range IDs), unchanged router weights, and original-top-k-before-drop behavior on a small example, including a token for which selected experts are dropped. A no-drop control must preserve the original output within a declared numerical tolerance. On a fixed hidden-state probe, pre-drop top-k must remain identical; in a full pruned-model pass, downstream hidden states and therefore routes may change legitimately.

## Required statistics

For each variant, layer and prompt/answer partition, save:

- Number of valid token positions and pre-drop top-k expert-selection counts.
- Pre-drop selected routing-weight sums by expert.
- Surviving selected counts and routing-weight sums by expert.
- Dropped selection fraction and dropped routing-weight fraction, with explicit denominators.
- Mean surviving selected experts per token and distribution from 0 through top-k.
- Aligned-token top-k overlap with the original model, computed before dropping contributions; state the overlap definition.

Use original-model routes with each retention mask as an additional **counterfactual mask-only** calculation. Contrast this with full pruned-model routes to separate the direct mask effect from downstream route redistribution. This calculation is not itself an executed pruned-model experiment.

Store sufficient aligned route records (or independently verifiable aggregates and overlap counts) to reproduce reported statistics. Count conservation must hold: selected = surviving + dropped, and each layer's total selected count = valid token count × top-k. Check finite weights and nonnegative counts.

## Figures

1. Layer × original-expert-ID heatmaps of selection frequency for all three variants, using a common scale; mark the retention masks rather than erasing pre-drop selections for removed experts.
2. Separate effective-contribution heatmaps and signed pre-drop frequency-difference heatmaps; use a common diverging scale for differences.
3. Per-layer surviving selections per token and dropped routing-weight fractions for keep-0.8 and keep-0.6, separating prompt and answer positions.
4. Per-layer aligned-token top-k overlap to show routing redistribution.

Normalize by the corresponding valid token counts, label selected versus surviving explicitly, and publish numeric tables alongside figures. No figures should be presented as measured until collection and validation finish.

## Interpretation and optional follow-up

Report what the measurements show even if they refute the low-work-reduction hypothesis. A 20% reduction in stored experts need not imply a 20% reduction in selected contributions, executed FLOPs, or wall time. Routing weight is a contribution-mass proxy, not a compute-cost estimate.

A stronger causal throughput analysis would require separately scoped instrumentation of the pinned H800 serving backend (actual dispatched work and MoE/attention/router timing, with instrumentation overhead controlled). Do not time the hooked routing collector and compare it to uninstrumented serving throughput. Any such profiling must preserve the completed benchmark and be reported separately.

## Implementation and reproduction

`analysis/compare_expert_routing.py` is independent of the old free-generation collector and does not change the throughput runner. The experiment uses Transformers 5.22.1 native Qwen3 router outputs and `grouped_mm` experts with BF16/SDPA. It loads one checkpoint, returns masked weights from router hooks, and never mutates checkpoint weights. Original and both pruned passes are interleaved per question, with no KV cache, and all hook handles are removed between passes. Thus comparisons include downstream routing redistribution, not just an original-route mask overlay. The collector still executes removed experts; it is not a speed benchmark or a measurement of skipped GEMMs.

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 PYTHONHASHSEED=0 python analysis/compare_expert_routing.py \
  --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --dataset /path/to/gsm8k/test.jsonl \
  --selection-08 results/qwen3-30b-a3b/selected_experts/shapley_alpha_per_layer_gsm8k_25_rate0_8.json \
  --selection-06 results/qwen3-30b-a3b/selected_experts/shapley_alpha_per_layer_gsm8k_25_rate0_6.json \
  --output /path/to/new-routing-run

python tools/verify_expert_routing.py /path/to/new-routing-run \
  --dataset /path/to/gsm8k/test.jsonl --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --output /path/to/new-routing-run/verification.json
python tools/plot_expert_routing.py /path/to/new-routing-run --output /path/to/new-figures
```

Omitting `--limit` selects the full dataset; every limited run is labeled a pilot. Recheck the GPU rather than assuming index 3 is available. Plotting requires numpy and matplotlib, but not a GPU. The verifier independently replays archived routes, selection masks, counterfactuals and all summary fields; supplying dataset and model also rechecks split coverage and every input token/partition.

### Execution history

- `pilot1` stopped before routing collection because Transformers 5.22.1 defaults `apply_chat_template(tokenize=True)` to a `BatchEncoding`. Fixed by explicitly setting `return_dict=False`, with a regression test. Retained the failed log/output directory.
- `pilot2` completed two questions; exact BF16 no-hook/no-drop control equality, fixed-hidden dropped-weight probes, and independent raw-route plus source-tokenization verification all passed.
- Full run: `h800:~/shape-experiments/routing/formal1/`, GPU index 3, UUID `GPU-11ccfb08-6fe7-8b08-f8ff-121a3153efe5`. Dataset preflight: 1,319 sequences, 276,281 tokens including template tokens, maximum 543 tokens; no truncation. Full collection and source-tokenization/raw-statistics verification passed. Local re-verification of the transferred raw archive and source split also passed. GPU 3 had no compute processes after completion. See the result report for exact statistics and archived evidence.

## Artifacts and completion criteria

Use a new output directory separate from `h800_gsm8k_throughput`. Preserve model/runtime versions, hardware, exact command/source snapshot, token and selection checksums, masks, validation outcomes, raw/aggregate statistics, and plotting inputs. The experiment is complete only after all three variants pass validation and the statistics/figures can be regenerated. No original weights, existing selections, or completed benchmark artifacts may be overwritten.
