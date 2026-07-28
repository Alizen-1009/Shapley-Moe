import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_faithfulness_profiles import build_profiles


class BuildFaithfulnessProfilesTest(unittest.TestCase):
    def test_preserves_layer_budgets_and_orders_high_low_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.csv"
            selection_path = root / "selection.json"

            with scores_path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Layer", "Expert_ID", "Total_Activations", "Shapley_Value"])
                for layer in range(2):
                    for expert in range(5):
                        writer.writerow([layer, expert, 1, expert + layer * 10])

            selection_path.write_text(
                json.dumps(
                    {
                        "0": [2, 3, 4],
                        "1": [2, 3, 4],
                        "_metadata": {"target_rate": 0.6},
                    }
                )
            )

            profiles = build_profiles(
                shapley_csv=scores_path,
                reference_selection=selection_path,
                model="test-model",
                model_revision="test-revision",
                dataset="test-data",
                random_seeds=[42, 43],
            )

            self.assertEqual(profiles["remove_low"]["pruned_experts"], [[0, 0], [0, 1], [1, 0], [1, 1]])
            self.assertEqual(profiles["remove_high"]["pruned_experts"], [[0, 3], [0, 4], [1, 3], [1, 4]])

            for name, profile in profiles.items():
                self.assertEqual(profile["version"], 2, name)
                self.assertEqual(profile["routing_mode"], "post_topk_drop", name)
                self.assertEqual(profile["model_revision"], "test-revision", name)
                self.assertEqual(profile["metadata"]["selected_experts"], 6, name)
                self.assertEqual(profile["metadata"]["pruned_experts"], 4, name)
                self.assertEqual(profile["metadata"]["per_layer_kept"], {"0": 3, "1": 3}, name)

            self.assertEqual(profiles["random_seed42"], build_profiles(
                shapley_csv=scores_path,
                reference_selection=selection_path,
                model="test-model",
                model_revision="test-revision",
                dataset="test-data",
                random_seeds=[42],
            )["random_seed42"])
            self.assertNotEqual(
                profiles["random_seed42"]["pruned_experts"],
                profiles["random_seed43"]["pruned_experts"],
            )

    def test_rejects_missing_expert_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.csv"
            selection_path = root / "selection.json"
            scores_path.write_text(
                "Layer,Expert_ID,Total_Activations,Shapley_Value\n"
                "0,0,1,1.0\n"
                "0,1,1,2.0\n"
            )
            selection_path.write_text(json.dumps({"0": [0], "1": [0]}))

            with self.assertRaisesRegex(ValueError, "missing Shapley scores"):
                build_profiles(
                    shapley_csv=scores_path,
                    reference_selection=selection_path,
                    model="test-model",
                    model_revision="test-revision",
                    dataset="test-data",
                    random_seeds=[],
                )


if __name__ == "__main__":
    unittest.main()
