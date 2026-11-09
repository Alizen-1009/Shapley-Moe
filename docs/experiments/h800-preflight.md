# H800 throughput experiment: preflight

Checked: 2026-11-05, approximately 20:05 +08:00. Resource availability is a snapshot, not a reservation.

## Hardware

- SSH host: `h800`; hostname: `server-205`.
- 8 × NVIDIA H800 PCIe; each reports 81,559 MiB total memory.
- NVIDIA driver: `580.95.05`.
- GPU indices 1, 2, 3 reported no allocated GPU memory and no compute processes.
- Tentative single-GPU target: GPU 1, UUID `GPU-13e1f305-ea69-0daf-ea9a-c40fee289704`.
- GPUs 0, 4, 5, 6, 7 have existing jobs. No processes were stopped.
- Recheck occupancy immediately before installation smoke tests and benchmarks; use the same GPU for all variants.

## Model

Path: `/home/zhangyh/model-dir`.

README identifies the checkpoint as `Qwen/Qwen3-30B-A3B-Instruct-2507`. Configuration is consistent with Qwen3 MoE: 48 layers, 128 experts per layer, top-k 8, BF16, 262,144 maximum positions.

Structural validation passed:

- All 16 indexed safetensors shards exist.
- 18,867 tensors, all BF16.
- Tensor names agree with the shard index.
- Tensor shapes, byte counts, contiguous offsets and shard file sizes are consistent.
- Total tensor payload: 61,064,245,248 bytes (56.87 GiB), matching index metadata.
- Chat-template SHA-256: `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326`.

Limit: this checks structure and model identity clues, not tensor contents against publisher hashes. Exact upstream revision and weight-content checksums remain to be recorded for reproducibility.

## Storage

| Path | Available, rounded | Writable by SSH user | Notes |
|---|---:|---|---|
| `/` | 35 GB | Not tested | Avoid large experiment files here |
| `/home/zhangyh` | 181 GB | Yes | Data disk, not root filesystem |
| `/disk-123` | 181 GB | Yes | Same filesystem as `/home`; not additional capacity |
| `/data-123` | 265 GB | No at mount root | Do not use another user's directory without authorization |

Selected future workspace: `/home/zhangyh/shape-experiments/` (not created during preflight).

Reuse existing weights in place. Put the environment, package caches, temporary build files and results on the home data disk. Available space is reasonable for an isolated serving environment and profile-based experiments, but source-build requirements must be checked before installation; do not duplicate model weights or retain large build caches unnecessarily.

## Basic tools

- OS: Linux x86_64, kernel `5.15.0-139-generic`, glibc 2.29.
- System Python: 3.8.10; do not assume it is suitable for the pinned vLLM.
- Existing Conda base: `/home/zhangyh/miniconda3`; no additional Conda environments were listed in the earlier inspection.
- `nvcc`, `gcc`, `git`, `docker` are on PATH; their compatibility and Docker access have not yet been verified.

## Scope

The initial preflight was read-only. Environment compatibility and real model loading belong to the next step.

## Follow-up: dedicated Conda environment

With user approval, created `shape-vllm` under the existing user's Miniconda, without modifying base:

```bash
/home/zhangyh/miniconda3/bin/conda create --name shape-vllm \
  --override-channels --channel conda-forge python=3.12 pip --yes
```

- Environment: `/home/zhangyh/miniconda3/envs/shape-vllm`.
- Verified Python: 3.12.16; pip: 26.2.2.
- Python and SSL imports succeed; `python -m pip check` reports no broken requirements.
- Activate with `source /home/zhangyh/miniconda3/etc/profile.d/conda.sh && conda activate shape-vllm`.
- At environment creation time, PyTorch and vLLM had not been installed. Installation is tracked separately below.
- No model downloads, model deletion, service startup or GPU process termination was performed.

## Follow-up: pinned serving stack preparation

Machine-specific source references are recorded in `configs/runtime/h800-vllm.json`. Do not use the repository's generic `requirements.txt` to install this environment: it pins vLLM 0.22.0 / Torch 2.11.0, whereas the selected custom source and upstream precompiled wheel require Torch 2.13.0.

- Source archive: custom commit `4986caaa5044cf023e4118f9cc15aa30bc2af9dd`.
- Archive SHA-256 verified after copying to H800: `46053401a69b2d62fdcdb2e348c98df0707b711ed9240c11a36a9799b6638a4f`.
- Source extraction: `/home/zhangyh/shape-experiments/src/vllm-4986caaa`.
- Upstream precompiled commit: `e222c33f2f568b54b41d2ebfc44f6c85ea072737`, wheel version `0.23.1rc1.dev1458+ge222c33f2`, CUDA 13.0.
- System CUDA compiler: 12.8.61; GCC: 9.4.0. Prefer precompiled extensions over local compilation.
- Initial Git transfer was too slow. Retrieved the commit-addressed archive via the local machine instead, verified its hash on H800, and stopped only our own stalled Git-transfer processes.
- Source and wheel commits are different, following the previously recorded experiment. Native sources are not identical; imports alone do not establish compatibility. Actual Qwen3 BF16 Triton serving tests are required.
- Installation and eager serving were subsequently verified; see the results below.

Existing version-2 `post_topk_drop` profiles were checked against the retained-expert selections for all 48 layers. Both exactly match the complements, with no duplicate or out-of-range experts:

| Profile | Pruned experts, across all layers | Actual keep fraction | SHA-256 |
|---|---:|---:|---|
| `throughput_profiles/keep_0_8.json` | 1,233 | 0.79931640625 | `f73bec65d9acedea15ca67a684e5faf99aae41185c66f2885849995d8f37e5df` |
| `throughput_profiles/keep_0_6.json` | 2,471 | 0.5978190104166666 | `34749c73e32ad0361b57aa1beeace86159117ca02dac69fddceedcf5d933c945` |

Profile paths are relative to `results/qwen3-30b-a3b/`. Nominal keep rates differ slightly from actual rates due to the existing expert selection budgets; do not silently change the selections to round the totals.

## Installation and eager serving verification (2026-11-09)

Installed in `shape-vllm`: Torch `2.13.0+cu130`, custom vLLM `0.23.1rc1.dev1458+g4986caaa.precompiled`, Transformers `5.22.1`. `pip check` passed. Native extensions `_C_stable_libtorch` and `_moe_C_stable_libtorch` load, and a tiny BF16 CUDA matrix multiply passed on GPU 1. This vLLM uses stable-libtorch extension names rather than `vllm._C`.

PyPI downloads initially ran at only 30–40 KB/s. Switching to hash-verified mirror URLs raised observed downloads to roughly 4.7–4.9 MB/s. Three missing mirror files (`cuda-bindings`, `pydantic`, `pydantic_core`) were transferred separately with the original PyPI hashes; two other URLs use Aliyun. No dependency version changes were needed to work around missing files.

All three variants started on one H800 and returned nonempty answers to `What is 2 + 2? Answer briefly.`:

| Variant | Logged model memory (GiB) | Logged kept experts | Response |
|---|---:|---:|---|
| Original | 56.93 | Full model | `2 + 2 = 4` |
| Keep 0.8 | 46.13 | 4,911 / 6,144 | `2 + 2 = 4` |
| Keep 0.6 | 35.24 | 3,673 / 6,144 | `2 + 2 = 4.` |

The logs for both pruned variants report `routing_mode=post_topk_drop` for all 48 layers. Each layer's retained-expert count was checked against the profile. Model memory here is the loader's model-allocation measurement, **not total GPU usage or a throughput result**.

Settings: BF16, TP=1, Triton attention and MoE, `--enforce-eager`, max model length 4096, max sequences 128, GPU memory utilization 0.85, prefix caching disabled, FlashInfer sampler disabled. Full commands and response usage counts are saved in `results/qwen3-30b-a3b/h800_environment_smoke/summary.json`. Remote detailed logs: `/home/zhangyh/shape-experiments/logs/smoke-*-server.log`.

All smoke servers were stopped and GPU 1 memory released. No original weights were changed, and no full pruned model copies were generated.

### Reproduction files

- `configs/runtime/h800-vllm.json`: source/wheel URLs, immutable commits and SHA-256 hashes.
- `requirements/h800-vllm.lock.txt`: 197 resolved wheel distributions with SHA-256 hashes; Linux x86_64 / Python 3.12 specific.
- `requirements/h800-vllm-build.txt`: exact editable-install build tools.
- `requirements/h800-vllm-installed.txt`: observed post-install pip snapshot (includes machine-local editable source path; not a portable install command).
- `requirements/h800-conda-explicit.txt`: explicit Conda package snapshot for Linux x86_64.
- `tools/install_h800_vllm.sh`: replays dependency installation and the pinned precompiled source overlay in an existing Python 3.12 environment. Use `bash tools/install_h800_vllm.sh`; override `SHAPE_VLLM_ENV` and `SHAPE_WORKSPACE` as needed. Do not run it inside the generic project environment or use the generic requirements alongside it.

The replay script has syntax/config checks but has not been rerun end-to-end in a fresh environment. CUDA graphs, torch.compile and full GSM8K throughput were subsequently verified: see `docs/experiments/h800-gsm8k-throughput.md` and `results/qwen3-30b-a3b/h800_gsm8k_throughput/README.md`. A zero-weight-reference equivalence test and accuracy evaluation remain unperformed. The small eager smoke alone is not an accuracy evaluation.
