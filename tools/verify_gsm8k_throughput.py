#!/usr/bin/env python3
"""Recompute full benchmark metrics from archived per-request records (no GPU)."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "evaluation/run_gsm8k_throughput.py"
spec = importlib.util.spec_from_file_location("gsm8k_bench", RUNNER)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    final = json.loads((output / "summary.json").read_text())
    args = manifest["arguments"]
    assert final["status"] == "complete"
    assert manifest["num_questions"] == 1319 and args["limit"] is None
    assert args["repeats"] == 3 and set(args["variants"]) == set(bench.VARIANTS)
    assert args["concurrency"] == 128 and args["max_tokens"] == 2048 and args["warmup"] == 256
    assert args["eager"] is False
    assert manifest["data_sha256"] == bench.DATA_SHA256
    assert hashlib.sha256((output / "runner.py").read_bytes()).hexdigest() == manifest["runner_sha256"]
    results = {}
    prompt_lengths = None
    base_command = None
    for variant in bench.VARIANTS:
        directory = output / variant
        profile = json.loads((directory / "profile-verification.json").read_text())
        log = (directory / "server.log").read_text()
        assert "Graph capturing finished" in log and "torch.compile took" in log
        profile_path = None
        if variant != "dense":
            profile_path = RUNNER.parents[1] / f"results/qwen3-30b-a3b/throughput_profiles/{variant}.json"
            assert bench.digest(profile_path) == manifest["profiles"][variant]["sha256"]
        assert bench.verify_server_profile(directory / "server.log", profile_path) == profile
        if variant == "dense":
            assert profile["routing_mode"] == "original"
        else:
            assert profile["routing_mode"] == "post_topk_drop" and profile["verified_layers"] == 48
        command = json.loads((directory / "server-command.json").read_text())["command"]
        normalized = list(command)
        if "--riy-expert-profile" in normalized:
            index = normalized.index("--riy-expert-profile")
            del normalized[index:index + 2]
        if base_command is None:
            base_command = normalized
        assert normalized == base_command, "Server settings differ across variants"
        results[variant] = []
        for repeat in range(1, 4):
            metrics = json.loads((directory / f"repeat_{repeat}.json").read_text())
            with gzip.open(directory / f"repeat_{repeat}_responses.jsonl.gz", "rt") as source:
                rows = [json.loads(line) for line in source]
            assert len(rows) == 1319
            assert [row["index"] for row in rows] == list(range(1319)), "Missing/duplicate/reordered question indices"
            assert all(row["success"] for row in rows)
            assert all(0 < row["completion_tokens"] <= 2048 for row in rows)
            assert all(math.isfinite(row["latency_seconds"]) and row["latency_seconds"] > 0 for row in rows)
            current_prompts = [row["prompt_tokens"] for row in rows]
            if prompt_lengths is None:
                prompt_lengths = current_prompts
            assert current_prompts == prompt_lengths, "Input token counts differ"
            recomputed = bench.summarize(rows, metrics["wall_seconds"])
            for key, value in recomputed.items():
                assert metrics[key] == value, f"Metric mismatch: {variant}, repeat {repeat}, {key}"
            assert metrics["failed_requests"] == 0
            assert metrics["gpu_after"]["uuid"] == manifest["gpu"]["uuid"]
            results[variant].append(metrics)
        assert results[variant] == final["repeats"][variant]
    assert bench.aggregate(results) == final["results"]
    report = {"status": "verified", "verified_full_passes": 9,
              "verified_requests": 1319 * 9, "same_per_question_prompt_lengths": True,
              "all_metrics_recomputed_from_response_records": True,
              "note": "Verifies accounting/config consistency, not answer accuracy or zero-weight-reference equivalence."}
    bench.write_json(output / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Complete formal run directory, including compressed response records")
    print(json.dumps(verify(parser.parse_args().output), indent=2))
