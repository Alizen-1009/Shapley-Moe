import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_gsm8k", ROOT / "tools/verify_gsm8k_throughput.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
bench = verifier.bench


class VerifyTests(unittest.TestCase):
    def test_recomputes_all_requests_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "runner.py").write_bytes(b"test archived runner")
            manifest = {"arguments": {"limit": None, "repeats": 3, "variants": list(bench.VARIANTS),
                        "concurrency": 128, "max_tokens": 2048, "warmup": 256, "eager": False},
                        "num_questions": 1319, "data_sha256": bench.DATA_SHA256,
                        "runner_sha256": hashlib.sha256(b"test archived runner").hexdigest(),
                        "profiles": {}, "gpu": {"uuid": "test-gpu"}}
            all_results = {}
            for variant in bench.VARIANTS:
                directory = out / variant
                directory.mkdir()
                profile_path = None
                command = ["vllm", "serve", "test-model"]
                log = "Graph capturing finished\ntorch.compile took 1 s\n"
                if variant != "dense":
                    profile_path = ROOT / f"results/qwen3-30b-a3b/throughput_profiles/{variant}.json"
                    manifest["profiles"][variant] = {"sha256": bench.digest(profile_path)}
                    command += ["--riy-expert-profile", str(profile_path)]
                    counts = {i: 128 for i in range(48)}
                    for layer, expert in json.loads(profile_path.read_text())["pruned_experts"]:
                        counts[layer] -= 1
                    log += "\n".join(f"RIY layer {i}: routing_mode=post_topk_drop, kept={count}/128 experts"
                                     for i, count in counts.items())
                (directory / "server.log").write_text(log)
                bench.write_json(directory / "profile-verification.json",
                                 bench.verify_server_profile(directory / "server.log", profile_path))
                bench.write_json(directory / "server-command.json", {"command": command})
                records = [{"index": i, "success": True, "completion_tokens": 1,
                            "prompt_tokens": 2, "latency_seconds": 0.1, "finish_reason": "stop"}
                           for i in range(1319)]
                runs = []
                for repeat in range(1, 4):
                    with gzip.open(directory / f"repeat_{repeat}_responses.jsonl.gz", "wt") as f:
                        for row in records:
                            f.write(json.dumps(row) + "\n")
                    metrics = bench.summarize(records, 5)
                    metrics["gpu_after"] = {"uuid": "test-gpu"}
                    bench.write_json(directory / f"repeat_{repeat}.json", metrics)
                    runs.append(metrics)
                all_results[variant] = runs
            bench.write_json(out / "manifest.json", manifest)
            bench.write_json(out / "summary.json", {"status": "complete", "repeats": all_results,
                                                   "results": bench.aggregate(all_results)})
            result = verifier.verify(out)
            self.assertEqual(result["verified_requests"], 11871)
            path = out / "keep_0_6/repeat_3.json"
            metrics = json.loads(path.read_text())
            metrics["output_tokens_per_second"] += 1
            bench.write_json(path, metrics)
            with self.assertRaisesRegex(AssertionError, "Metric mismatch"):
                verifier.verify(out)


if __name__ == "__main__":
    unittest.main()
