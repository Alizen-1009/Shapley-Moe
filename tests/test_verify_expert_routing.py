"""Synthetic raw-archive regression tests; no model or torch required."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_NUMPY = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_NUMPY, 'numpy unavailable')
class RoutingVerificationTests(unittest.TestCase):
    def make_archive(self, root):
        from tools.verify_expert_routing import sha
        masks = {name: [[i < retained for i in range(128)] for _ in range(48)]
                 for name, retained in [('keep_08', 6), ('keep_06', 4)]}
        def write(name, data):
            (root / name).write_text(json.dumps(data))
        write('manifest.json', dict(selections=masks,
              actual_retained_fractions={n: sum(m[0])/128 for n, m in masks.items()},
              arguments=dict(limit=1)))
        (root / 'tokens.jsonl').write_text(json.dumps(dict(index=0, dataset_line=1,
            input_ids=[11, 12, 13], partition=[-1, 0, 1])) + '\n')
        path = root / 'question_00000.jsonl.gz'
        with gzip.open(path, 'wt') as stream:
            for variant in ['original', 'keep_08', 'keep_06']:
                for layer in range(48):
                    stream.write(json.dumps(dict(variant=variant, layer=layer,
                        ids=[list(range(8))]*3, weights=[[.125]*8]*3))+'\n')
        (root / 'progress.jsonl').write_text(json.dumps(dict(index=0, tokens=3,
            routes_sha256=sha(path)))+'\n')
        variants = {}
        for name, kept in [('original', 8), ('keep_08', 6), ('keep_06', 4),
                           ('counterfactual_08', 6), ('counterfactual_06', 4)]:
            stats = {}
            for layer in range(48):
                for part in ['prompt', 'answer']:
                    stats[f'{layer}/{part}'] = dict(tokens=1,
                        selected_counts=[1]*8+[0]*120,
                        selected_weights=[.125]*8+[0.]*120,
                        surviving_counts=[1]*kept+[0]*(128-kept),
                        surviving_weights=[.125]*kept+[0.]*(128-kept),
                        surviving_histogram=[int(k == kept) for k in range(9)],
                        overlap_count=8, dropped_count=8-kept,
                        dropped_weight=1-kept/8, selection_denominator=8,
                        weight_denominator=1., dropped_selection_fraction=1-kept/8,
                        dropped_weight_fraction=1-kept/8, mean_survivors=float(kept),
                        overlap_fraction=1.)
            variants[name] = stats
        write('validation.json', dict(synthetic=True))
        write('summary.json', dict(questions=1, pilot=True, router_unchanged=True,
            validation=dict(synthetic=True), tokens_sha256=sha(root/'tokens.jsonl'),
            variants=variants))

    def test_recompute_and_reject_summary_corruption(self):
        from tools.verify_expert_routing import verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_archive(root)
            result = verify(root)
            self.assertTrue(result['verified'])
            self.assertEqual(result['route_rows'], 144)
            self.assertEqual(result['included_tokens'], 2)
            summary = json.loads((root/'summary.json').read_text())
            summary['variants']['keep_08']['2/answer']['surviving_weights'][0] += .01
            (root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'float differs'):
                verify(root)

    def test_checksum_and_pilot_label(self):
        from tools.verify_expert_routing import verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_archive(root)
            with (root/'question_00000.jsonl.gz').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'routes checksum'):
                verify(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_archive(root)
            summary = json.loads((root/'summary.json').read_text())
            summary['pilot'] = False
            (root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'pilot mislabeled'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
