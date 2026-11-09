import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "evaluation/run_gsm8k_throughput.py"
spec = importlib.util.spec_from_file_location("gsm8k_throughput", PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ThroughputTests(unittest.TestCase):
    def test_metrics_use_full_wall_time_and_actual_tokens(self):
        rows = [
            {"success": True, "completion_tokens": 100, "prompt_tokens": 20,
             "latency_seconds": 3, "finish_reason": "stop"},
            {"success": True, "completion_tokens": 300, "prompt_tokens": 30,
             "latency_seconds": 4, "finish_reason": "length"},
            {"success": False, "error": "timeout", "latency_seconds": 10},
        ]
        result = bench.summarize(rows, 10)
        self.assertEqual(result["output_tokens_per_second"], 40)
        self.assertEqual(result["requests_per_second"], 0.2)
        self.assertEqual(result["mean_output_tokens"], 200)
        self.assertEqual(result["failed_requests"], 1)
        self.assertEqual(result["length_capped_requests"], 1)
        self.assertEqual(result["prompt_tokens"], 50)
        self.assertEqual(result["latency_p50_seconds"], 3.5)

    def test_all_failures_are_visible(self):
        result = bench.summarize([{"success": False}], 5)
        self.assertEqual(result["failed_requests"], 1)
        self.assertEqual(result["output_tokens_per_second"], 0)
        self.assertIsNone(result["mean_output_tokens"])

    def test_invalid_measurements_rejected(self):
        for rows, elapsed in [([], 1), ([{"success": False}], 0)]:
            with self.assertRaises(ValueError):
                bench.summarize(rows, elapsed)

    def test_aggregate_sample_standard_deviation_and_speedup(self):
        def row(value):
            return {"output_tokens_per_second": value, "mean_output_tokens": 200,
                    "failed_requests": 0, "length_capped_requests": 1}
        result = bench.aggregate({"dense": [row(90), row(100), row(110)],
                                  "keep_0_8": [row(110), row(120), row(130)]})
        self.assertEqual(result["dense"]["output_tokens_per_second_mean"], 100)
        self.assertEqual(result["dense"]["output_tokens_per_second_stdev"], 10)
        self.assertEqual(result["keep_0_8"]["speedup_vs_dense"], 1.2)
        self.assertEqual(result["keep_0_8"]["length_capped_requests"], 3)

    def test_real_profiles_match_existing_selections(self):
        root = PATH.parents[1]
        for rate in ["0_8", "0_6"]:
            result = bench.validate_profile(
                root / f"results/qwen3-30b-a3b/throughput_profiles/keep_{rate}.json",
                root / f"results/qwen3-30b-a3b/selected_experts/shapley_alpha_per_layer_gsm8k_25_rate{rate}.json")
            self.assertGreater(result["pruned_experts"], 0)

    def test_wrong_routing_mode_rejected(self):
        root = PATH.parents[1]
        original = root / "results/qwen3-30b-a3b/throughput_profiles/keep_0_8.json"
        profile = json.loads(original.read_text())
        profile["routing_mode"] = "pre_topk_mask"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):
                bench.validate_profile(path,
                    root / "results/qwen3-30b-a3b/selected_experts/shapley_alpha_per_layer_gsm8k_25_rate0_8.json")

    def test_server_log_confirms_each_layer_and_routing_mode(self):
        root = PATH.parents[1]
        profile_path = root / "results/qwen3-30b-a3b/throughput_profiles/keep_0_8.json"
        counts = {layer: 128 for layer in range(48)}
        for layer, expert in json.loads(profile_path.read_text())["pruned_experts"]:
            counts[layer] -= 1
        text = "\n".join(f"RIY layer {layer}: routing_mode=post_topk_drop, kept={count}/128 experts"
                         for layer, count in counts.items())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(text)
            result = bench.verify_server_profile(path, profile_path)
            self.assertEqual(result["verified_layers"], 48)
            self.assertEqual(result["kept_experts"], 4911)
            with self.assertRaises(ValueError):
                bench.verify_server_profile(path)
            path.write_text(text.replace("post_topk_drop", "pre_topk_mask"))
            with self.assertRaises(ValueError):
                bench.verify_server_profile(path, profile_path)
            path.write_text("")
            with self.assertRaises(ValueError):
                bench.verify_server_profile(path, profile_path)
            self.assertEqual(bench.verify_server_profile(path)["routing_mode"], "original")

    def test_port_check_allows_time_wait_but_not_live_listener(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen()
            with self.assertRaises(OSError):
                bench.ensure_port_available(port)
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = server.accept()
                accepted.close()  # Active close leaves the server port in TIME_WAIT.
                self.assertEqual(client.recv(1), b"")
        bench.ensure_port_available(port)

    def test_dataset_checksum_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            path.write_text('{"question":"not the official test set"}\n')
            with self.assertRaises(ValueError):
                bench.load_questions(path)


if __name__ == "__main__":
    unittest.main()
