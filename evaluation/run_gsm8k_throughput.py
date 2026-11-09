#!/usr/bin/env python3
"""Sequential, reproducible single-GPU GSM8K serving benchmark.

Uses the same test questions, chat template, sampling settings and server limits
for every variant. Measures actual output tokens / full batch wall time; output
lengths may differ between variants. This is not an accuracy evaluator.
"""
import argparse
import asyncio
import contextlib
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

VARIANTS = ("dense", "keep_0_8", "keep_0_6")
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
DATA_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
INSTRUCTION = "Please reason step by step, and put your final numeric answer after '#### '."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_questions(path):
    if digest(path) != DATA_SHA256:
        raise ValueError("GSM8K file does not match the pinned test-set SHA256")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != 1319 or any(not r.get("question") for r in rows):
        raise ValueError("Expected all 1,319 GSM8K test questions")
    return rows


def validate_profile(path, selection):
    profile = json.loads(Path(path).read_text())
    raw = json.loads(Path(selection).read_text())
    kept = {k: v for k, v in raw.items() if k != "_metadata"}
    if profile.get("version") != 2 or profile.get("routing_mode") != "post_topk_drop":
        raise ValueError("Only SHAPE version-2 post_topk_drop profiles are accepted")
    if set(kept) != {str(i) for i in range(48)}:
        raise ValueError("Selection must cover all 48 layers")
    for experts in kept.values():
        if (not experts or len(set(experts)) != len(experts)
                or any(type(e) is not int or not 0 <= e < 128 for e in experts)):
            raise ValueError("Invalid retained expert list")
    expected = {(i, e) for i in range(48) for e in set(range(128)) - set(kept[str(i)])}
    pairs = profile["pruned_experts"]
    if len(pairs) != len(expected) or {tuple(p) for p in pairs} != expected:
        raise ValueError("Profile differs from original expert selection")
    return {"sha256": digest(path), "pruned_experts": len(expected),
            "actual_keep_rate": (6144 - len(expected)) / 6144}


def summarize(records, elapsed):
    if elapsed <= 0 or not records:
        raise ValueError("Nonempty records and positive wall time required")
    good = [r for r in records if r.get("success")]
    output = sum(r["completion_tokens"] for r in good)
    inputs = sum(r["prompt_tokens"] for r in good)
    lengths = [r["completion_tokens"] for r in good]
    latencies = sorted(r["latency_seconds"] for r in good)
    def percentile(p):
        if not latencies:
            return None
        index = (len(latencies) - 1) * p
        a, b = math.floor(index), math.ceil(index)
        return latencies[a] + (latencies[b] - latencies[a]) * (index - a)
    return {"num_requests": len(records), "successful_requests": len(good),
            "failed_requests": len(records) - len(good), "wall_seconds": elapsed,
            "prompt_tokens": inputs, "completion_tokens": output,
            "output_tokens_per_second": output / elapsed,
            "requests_per_second": len(good) / elapsed,
            "mean_output_tokens": statistics.mean(lengths) if lengths else None,
            "max_output_tokens": max(lengths) if lengths else None,
            "length_capped_requests": sum(r.get("finish_reason") == "length" for r in good),
            "latency_p50_seconds": percentile(0.5),
            "latency_p95_seconds": percentile(0.95)}


async def request_batch(rows, base_url, concurrency, max_tokens, timeout, seed):
    import aiohttp  # Installed in the serving environment, not needed for pure unit tests.
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async def one(index, row):
            async with semaphore:
                started = time.perf_counter()
                try:
                    body = {"model": "shape-qwen3", "messages": [{"role": "user",
                            "content": row["question"] + "\n\n" + INSTRUCTION}],
                            "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
                            "seed": seed, "stream": False}
                    async with session.post(base_url + "/v1/chat/completions", json=body) as response:
                        data = await response.json()
                        if response.status != 200:
                            raise RuntimeError(f"HTTP {response.status}: {str(data)[:500]}")
                    usage = data["usage"]
                    choice = data["choices"][0]
                    if choice["finish_reason"] not in ("stop", "length"):
                        raise ValueError(f"Unexpected finish reason: {choice['finish_reason']}")
                    if (type(usage["completion_tokens"]) is not int or usage["completion_tokens"] <= 0
                            or type(usage["prompt_tokens"]) is not int or usage["prompt_tokens"] <= 0):
                        raise ValueError("Missing or invalid server token usage")
                    return {"index": index, "success": True,
                            "latency_seconds": time.perf_counter() - started,
                            "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"],
                            "finish_reason": choice["finish_reason"],
                            "content": choice["message"].get("content"),
                            "system_fingerprint": data.get("system_fingerprint")}
                except Exception as exc:
                    return {"index": index, "success": False,
                            "latency_seconds": time.perf_counter() - started,
                            "error": f"{type(exc).__name__}: {exc}"}
        started = time.perf_counter()
        records = await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
        elapsed = time.perf_counter() - started
    return records, summarize(records, elapsed)


def gpu_info(gpu):
    text = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
    fields = [s.strip() for s in text.strip().split(",")]
    return {"index": fields[0], "uuid": fields[1], "name": fields[2], "driver": fields[3],
            "memory_total_mib": int(fields[4]), "memory_used_mib": int(fields[5]),
            "utilization_percent": int(fields[6])}


def compute_pids(gpu_uuid):
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(parts[1].strip()) for line in text.splitlines()
            if len(parts := line.split(",")) == 2 and parts[0].strip() == gpu_uuid}


def assert_exclusive_gpu(gpu, server_pid):
    info = gpu_info(gpu)
    foreign = []
    for pid in compute_pids(info["uuid"]):
        try:
            if os.getpgid(pid) != server_pid:
                foreign.append(pid)
        except ProcessLookupError:
            pass
    if foreign:
        raise RuntimeError(f"Foreign GPU processes appeared: {foreign}; measurement invalid")
    return info


def verify_server_profile(log_path, profile_path=None):
    text = Path(log_path).read_text()
    rows = re.findall(r"RIY layer (\d+): routing_mode=(\w+), kept=(\d+)/128 experts", text)
    if profile_path is None:
        if rows:
            raise ValueError("Baseline unexpectedly loaded a pruning profile")
        return {"routing_mode": "original", "verified_layers": 0}
    profile = json.loads(Path(profile_path).read_text())
    expected = {layer: 128 for layer in range(48)}
    for layer, expert in profile["pruned_experts"]:
        expected[layer] -= 1
    if len(rows) != 48 or {int(row[0]) for row in rows} != set(range(48)):
        raise ValueError("Server did not confirm all 48 pruned layers")
    for layer, mode, count in rows:
        if mode != "post_topk_drop" or int(count) != expected[int(layer)]:
            raise ValueError("Server routing mode or retained-expert count differs from profile")
    return {"routing_mode": "post_topk_drop", "verified_layers": 48,
            "kept_experts": sum(expected.values())}


def stop_server(process):
    # Only signal the process group created by this runner; never pkill GPU jobs.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_ready(process, base_url, deadline_seconds):
    end = time.monotonic() + deadline_seconds
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}; inspect server.log")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("Server startup deadline exceeded; inspect server.log")


def aggregate(results):
    out = {}
    for variant, runs in results.items():
        values = [r["output_tokens_per_second"] for r in runs]
        out[variant] = {"repeats": len(runs), "output_tokens_per_second_mean": statistics.mean(values),
                        "output_tokens_per_second_stdev": statistics.stdev(values) if len(values) > 1 else 0,
                        "mean_output_tokens": statistics.mean(r["mean_output_tokens"] for r in runs),
                        "failed_requests": sum(r["failed_requests"] for r in runs),
                        "length_capped_requests": sum(r["length_capped_requests"] for r in runs)}
    if "dense" in out:
        baseline = out["dense"]["output_tokens_per_second_mean"]
        for row in out.values():
            row["speedup_vs_dense"] = row["output_tokens_per_second_mean"] / baseline
    return out


def ensure_port_available(port):
    # Match the server's address-reuse policy: TIME_WAIT is not a listener.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def run(args):
    root = Path(args.root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Output directory already contains a run; use a fresh directory")
    data = load_questions(args.dataset)
    rows = data[:args.limit] if args.limit else data
    info = gpu_info(args.gpu)
    if info["memory_used_mib"] != 0 or compute_pids(info["uuid"]):
        raise RuntimeError(f"GPU is occupied: {info}")
    ensure_port_available(args.port)
    base_url = f"http://127.0.0.1:{args.port}"
    profiles = {}
    for variant in args.variants:
        if variant != "dense":
            rate = variant.removeprefix("keep_")
            profiles[variant] = validate_profile(
                root / f"results/qwen3-30b-a3b/throughput_profiles/{variant}.json",
                root / f"results/qwen3-30b-a3b/selected_experts/shapley_alpha_per_layer_gsm8k_25_rate{rate}.json")
    import importlib.metadata
    manifest = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "arguments": vars(args),
                "gpu": info, "runner_sha256": digest(__file__), "data_sha256": DATA_SHA256,
                "data_commit": DATA_COMMIT, "dataset_split": "test", "num_questions": len(rows),
                "question_order": "original file order", "instruction": INSTRUCTION,
                "generation": {"temperature": 0, "top_p": 1, "max_tokens": args.max_tokens,
                               "seed": args.seed, "stream": False},
                "profiles": profiles,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "aiohttp")},
                "runtime_config": json.loads((root / "configs/runtime/h800-vllm.json").read_text()),
                "scope": "Real GSM8K chat workload; variable actual output lengths, same limits. Not an accuracy evaluation."}
    write_json(out / "manifest.json", manifest)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    all_results = {}
    for variant in args.variants:
        variant_dir = out / variant
        variant_dir.mkdir()
        if gpu_info(args.gpu)["memory_used_mib"] != 0:
            raise RuntimeError("GPU no longer free before next variant")
        cmd = [str(Path(sys.executable).with_name("vllm")), "serve", args.model,
               "--served-model-name", "shape-qwen3", "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--seed", str(args.seed),
               "--max-model-len", "4096", "--max-num-seqs", str(args.concurrency),
               "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.85",
               "--moe-backend", "triton", "--attention-backend", "TRITON_ATTN",
               "--no-enable-prefix-caching", "--max-cudagraph-capture-size", str(args.concurrency)]
        if args.eager:
            cmd.append("--enforce-eager")
        if variant != "dense":
            cmd += ["--riy-expert-profile", str(root / f"results/qwen3-30b-a3b/throughput_profiles/{variant}.json")]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_RIY_MONITOR="0")
        write_json(variant_dir / "server-command.json", {"command": cmd,
                   "environment_overrides": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_RIY_MONITOR")}})
        print(f"[{variant}] starting server", flush=True)
        with (variant_dir / "server.log").open("w") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                wait_ready(process, base_url, args.startup_timeout)
                profile_path = (root / f"results/qwen3-30b-a3b/throughput_profiles/{variant}.json"
                                if variant != "dense" else None)
                profile_check = verify_server_profile(variant_dir / "server.log", profile_path)
                write_json(variant_dir / "profile-verification.json", profile_check)
                assert_exclusive_gpu(args.gpu, process.pid)
                print(f"[{variant}] warmup {args.warmup} prompts", flush=True)
                _, warmup = asyncio.run(request_batch(data[:args.warmup], base_url,
                    args.concurrency, args.max_tokens, args.request_timeout, args.seed))
                write_json(variant_dir / "warmup.json", warmup)
                if warmup["failed_requests"]:
                    raise RuntimeError("Warmup requests failed")
                runs = []
                for repeat in range(1, args.repeats + 1):
                    assert_exclusive_gpu(args.gpu, process.pid)
                    print(f"[{variant}] measured repeat {repeat}/{args.repeats}, {len(rows)} prompts", flush=True)
                    records, metrics = asyncio.run(request_batch(rows, base_url, args.concurrency,
                        args.max_tokens, args.request_timeout, args.seed))
                    with gzip.open(variant_dir / f"repeat_{repeat}_responses.jsonl.gz", "wt", encoding="utf-8") as f:
                        for record in records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics["gpu_after"] = assert_exclusive_gpu(args.gpu, process.pid)
                    write_json(variant_dir / f"repeat_{repeat}.json", metrics)
                    print(f"[{variant}] repeat {repeat}: {metrics['output_tokens_per_second']:.2f} output tok/s, "
                          f"{metrics['failed_requests']} failed, {metrics['length_capped_requests']} capped", flush=True)
                    if metrics["failed_requests"]:
                        raise RuntimeError("Measured requests failed; do not publish incomplete throughput")
                    runs.append(metrics)
                all_results[variant] = runs
                write_json(out / "summary.partial.json", aggregate(all_results))
            finally:
                stop_server(process)
                for _ in range(30):
                    if gpu_info(args.gpu)["memory_used_mib"] == 0:
                        break
                    time.sleep(1)
        print(f"[{variant}] finished; server stopped", flush=True)
    write_json(out / "summary.json", {"status": "complete", "results": aggregate(all_results),
               "repeats": all_results, "manifest": "manifest.json"})
    print("All variants completed. Summary:", out / "summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="SHAPE checkout containing configs and profiles")
    parser.add_argument("--dataset", required=True, help="Pinned GSM8K test.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=18801)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--limit", type=int, help="Pilot only: use first N test questions")
    parser.add_argument("--eager", action="store_true", help="Disable compile/graphs uniformly; record as eager results")
    args = parser.parse_args()
    if (Path(args.output) / "manifest.json").exists():
        parser.error("Output already contains a run; choose a new directory")
    if (args.concurrency < 1 or args.repeats < 1 or not 1 <= args.warmup <= 1319
            or not 1 <= args.max_tokens <= 3072 or (args.limit is not None and not 1 <= args.limit <= 1319)
            or len(set(args.variants)) != len(args.variants)):
        parser.error("Invalid benchmark limits or duplicate variants")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except BaseException as exc:
        path = Path(args.output)
        if path.exists():
            write_json(path / "failure.json", {"error": f"{type(exc).__name__}: {exc}",
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        raise


if __name__ == "__main__":
    main()
