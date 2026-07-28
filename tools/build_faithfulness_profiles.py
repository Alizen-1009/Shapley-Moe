#!/usr/bin/env python3
"""Build matched expert-removal profiles for Shapley faithfulness tests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ScoresByLayer = Dict[int, Dict[int, float]]
SelectionByLayer = Dict[int, List[int]]


def _load_scores(path: Path) -> ScoresByLayer:
    scores: ScoresByLayer = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"Layer", "Expert_ID", "Shapley_Value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            layer = int(row["Layer"])
            expert = int(row["Expert_ID"])
            layer_scores = scores.setdefault(layer, {})
            if expert in layer_scores:
                raise ValueError(f"duplicate score for layer {layer}, expert {expert}")
            layer_scores[expert] = float(row["Shapley_Value"])
    if not scores:
        raise ValueError(f"no Shapley scores found in {path}")
    return scores


def _load_selection(path: Path) -> SelectionByLayer:
    raw = json.loads(path.read_text())
    selection: SelectionByLayer = {}
    for layer_key, experts in raw.items():
        if layer_key.startswith("_"):
            continue
        if not isinstance(experts, list):
            raise ValueError(f"selection for layer {layer_key} must be a list")
        layer = int(layer_key)
        selected = sorted({int(expert) for expert in experts})
        if len(selected) != len(experts):
            raise ValueError(f"selection for layer {layer} contains duplicate experts")
        selection[layer] = selected
    if not selection:
        raise ValueError(f"no layer selections found in {path}")
    return selection


def _validate(scores: ScoresByLayer, selection: SelectionByLayer) -> None:
    if set(scores) != set(selection):
        missing = sorted(set(selection) - set(scores))
        extra = sorted(set(scores) - set(selection))
        raise ValueError(
            "missing Shapley scores for selection layers "
            f"{missing}; score-only layers: {extra}"
        )

    for layer, selected in selection.items():
        score_ids = set(scores[layer])
        selected_ids = set(selected)
        if not selected_ids.issubset(score_ids):
            missing = sorted(selected_ids - score_ids)
            raise ValueError(
                f"missing Shapley scores for layer {layer}, experts {missing}"
            )
        if len(selected) < 1 or len(selected) > len(score_ids):
            raise ValueError(
                f"invalid keep count {len(selected)} for layer {layer} "
                f"with {len(score_ids)} experts"
            )


def _profile(
    *,
    selection: Mapping[int, Sequence[int]],
    all_experts: Mapping[int, Iterable[int]],
    model: str,
    model_revision: str,
    dataset: str,
    variant: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    pruned = [
        [layer, expert]
        for layer in sorted(selection)
        for expert in sorted(set(all_experts[layer]) - set(selection[layer]))
    ]
    per_layer_kept = {
        str(layer): len(selection[layer]) for layer in sorted(selection)
    }
    total_experts = sum(len(set(all_experts[layer])) for layer in selection)
    selected_experts = sum(per_layer_kept.values())
    metadata: Dict[str, Any] = {
        "variant": variant,
        "num_layers": len(selection),
        "total_experts": total_experts,
        "selected_experts": selected_experts,
        "pruned_experts": len(pruned),
        "keep_rate": selected_experts / total_experts,
        "per_layer_kept": per_layer_kept,
    }
    if seed is not None:
        metadata["seed"] = seed
    return {
        "version": 2,
        "routing_mode": "post_topk_drop",
        "model": model,
        "model_revision": model_revision,
        "dataset": dataset,
        "pruned_experts": pruned,
        "metadata": metadata,
    }


def build_profiles(
    *,
    shapley_csv: Path,
    reference_selection: Path,
    model: str,
    model_revision: str,
    dataset: str,
    random_seeds: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    """Build profiles with identical per-layer keep counts.

    ``remove_low`` reuses the reference SHAPE selection. ``remove_high`` keeps
    the lowest-scoring experts, thereby removing the highest-scoring experts.
    Random variants preserve the same number of experts in every layer.
    """
    scores = _load_scores(shapley_csv)
    reference = _load_selection(reference_selection)
    _validate(scores, reference)
    all_experts = {layer: layer_scores.keys() for layer, layer_scores in scores.items()}

    remove_high: SelectionByLayer = {}
    for layer, selected in reference.items():
        keep_count = len(selected)
        ranked_ascending = sorted(
            scores[layer], key=lambda expert: (scores[layer][expert], expert)
        )
        remove_high[layer] = sorted(ranked_ascending[:keep_count])

    profiles = {
        "remove_low": _profile(
            selection=reference,
            all_experts=all_experts,
            model=model,
            model_revision=model_revision,
            dataset=dataset,
            variant="remove_low",
        ),
        "remove_high": _profile(
            selection=remove_high,
            all_experts=all_experts,
            model=model,
            model_revision=model_revision,
            dataset=dataset,
            variant="remove_high",
        ),
    }

    for seed in random_seeds:
        rng = random.Random(seed)
        random_selection: SelectionByLayer = {}
        for layer, selected in sorted(reference.items()):
            random_selection[layer] = sorted(
                rng.sample(sorted(scores[layer]), len(selected))
            )
        name = f"random_seed{seed}"
        profiles[name] = _profile(
            selection=random_selection,
            all_experts=all_experts,
            model=model,
            model_revision=model_revision,
            dataset=dataset,
            variant="random",
            seed=seed,
        )
    return profiles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapley-csv", type=Path, required=True)
    parser.add_argument("--reference-selection", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, nargs="*", default=[42, 43, 44])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profiles = build_profiles(
        shapley_csv=args.shapley_csv,
        reference_selection=args.reference_selection,
        model=args.model,
        model_revision=args.model_revision,
        dataset=args.dataset,
        random_seeds=args.random_seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, profile in profiles.items():
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(profile, indent=2) + "\n")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
