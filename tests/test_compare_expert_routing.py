import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from analysis.compare_expert_routing import (
    finalize, gate_hook, load_selection, merge, save_json, statistics, token_sequence,
)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=True):
        text = "<user>" + messages[0]["content"] + "</user><assistant>"
        if not tokenize:
            return text
        ids = self.encode(text)
        return {"input_ids": ids} if return_dict else ids

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(i, i+1) for i in range(len(text))])


class ProtocolTests(unittest.TestCase):
    def test_selection_metadata_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            for raw in ({"0": [0, 2], "1": [1], "_metadata": {"target_rate": .8}},):
                path.write_text(json.dumps(raw))
                masks, meta = load_selection(path, 2, 3)
                self.assertEqual(masks, [[True, False, True], [False, True, False]])
                self.assertEqual(meta["target_rate"], .8)
            for raw in ({"0": [0]}, {"0": [0, 0], "1": [1]},
                        {"0": [3], "1": [1]}, {"0": [], "1": [1]},
                        {"0": [True], "1": [1]}, {"0": [0], "1": [1], "extra": []}):
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    load_selection(path, 2, 3)

    def test_tokens_explicit_spans_and_no_eos(self):
        ids, partition = token_sequence(CharacterTokenizer(), "Q", "#### 42", 8192)
        self.assertEqual(ids[-7:], list(map(ord, "#### 42")))
        self.assertEqual(partition[-7:], [1]*7)
        self.assertEqual(partition[:6], [-1]*6)
        self.assertEqual(partition[6], 0)
        self.assertIn(-1, partition[7:-7])
        with self.assertRaises(ValueError):
            token_sequence(CharacterTokenizer(), "Q", "A", 2)
        with self.assertRaises(ValueError):
            token_sequence(CharacterTokenizer(), "Q", "", 8192)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                save_json(path, {})

    @unittest.skipUnless(HAS_NUMPY, "numpy unavailable")
    def test_statistics_counterfactual_and_overlap(self):
        import numpy as np
        ids = np.array([[0, 1], [1, 2], [0, 2]])
        w = np.array([[.75, .25], [.5, .5], [.2, .8]])
        original = {0: (ids, w)}
        masks = [[True, False, True]]
        stats = statistics(original, [0, 1, -1], masks, original, 3, 2)
        self.assertEqual(stats["0/prompt"]["selected_counts"], [1, 1, 0])
        self.assertEqual(stats["0/prompt"]["surviving_counts"], [1, 0, 0])
        self.assertEqual(stats["0/answer"]["surviving_histogram"], [0, 1, 0])
        total = {}
        merge(total, stats)
        merge(total, statistics(original, [0, 1, -1], masks, original, 3, 2))
        finalize(total, 2)
        self.assertEqual(total["0/prompt"]["tokens"], 2)
        self.assertEqual(total["0/prompt"]["dropped_weight_fraction"], .25)
        self.assertEqual(total["0/prompt"]["overlap_fraction"], 1)
        changed = {0: (np.array([[0, 2], [0, 2], [0, 2]]), w)}
        self.assertEqual(statistics(changed, [0, 1, -1], masks, original, 3, 2)["0/prompt"]["overlap_count"], 1)

    @unittest.skipUnless(HAS_TORCH and HAS_NUMPY, "torch/numpy unavailable")
    def test_tiny_experts_mask_vs_zero_down_and_no_drop(self):
        import torch
        torch.manual_seed(7)
        # Tiny real expert module: cloned down projections are only a few bytes.
        class TinyExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.up = torch.nn.Parameter(torch.randn(3, 4, 2))
                self.down = torch.nn.Parameter(torch.randn(3, 2, 4))

            def forward(self, hidden, ids, weights):
                value = torch.relu(torch.einsum("tkhi,ti->tkh", self.up[ids], hidden))
                value = torch.einsum("tkih,tkh->tki", self.down[ids], value)
                return (value * weights[..., None]).sum(1)

        experts = TinyExperts()
        hidden = torch.randn(2, 2)
        logits = torch.tensor([[3., 2., 1.], [1., 3., 2.]])
        weights, ids = logits.softmax(-1).topk(2)
        weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
        native = (logits, weights, ids)
        records = {}
        masked = gate_hook([True, False, True], [0, 1], records, 0, 3, 2)(None, (), native)
        self.assertIs(masked[0], logits)
        self.assertIs(masked[2], ids)
        self.assertTrue(torch.equal(masked[1], weights * (ids != 1)))
        self.assertLess(masked[1][0].sum(), weights[0].sum())  # no renormalization
        import copy
        zeroed = copy.deepcopy(experts)
        with torch.no_grad():
            zeroed.down[1].zero_()
        torch.testing.assert_close(experts(hidden, ids, masked[1]), zeroed(hidden, ids, weights))
        control = gate_hook([True]*3, [0, 1], {}, 0, 3, 2)(None, (), native)
        torch.testing.assert_close(experts(hidden, ids, control[1]), experts(hidden, ids, weights), rtol=0, atol=0)
        self.assertEqual(records[0][0].tolist(), ids.tolist())
        with self.assertRaises(ValueError):
            gate_hook(None, [0, 1], {}, 0, 3, 2)(None, (), (logits, weights*float("nan"), ids))


if __name__ == "__main__":
    unittest.main()
