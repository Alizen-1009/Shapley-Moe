#!/usr/bin/env python3
"""
Random Pruning Expert Selection Tool (Baseline Experiment)

Supports two strategies:
1. Global pruning rate: Retain X% of experts across all layers (randomly selected)
2. Per-layer pruning rate: Retain X% of experts per layer (randomly selected, recommended)

Unlike Shapley value based selection, this tool selects experts completely at random, for use as a baseline experiment.
"""

import pandas as pd
import argparse
import os
import json
import ast
import numpy as np
import random
from typing import Dict, List, Tuple, Optional


def parse_shapley_json(json_file: str) -> pd.DataFrame:
    """
    Parse expert activation counts from shapley JSON file
    
    Args:
        json_file: Shapley JSON file path
        
    Returns:
        DataFrame with Layer, Expert_ID, Total_Activations columns
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Count activations per expert per layer
    from collections import defaultdict
    layer_expert_counts = defaultdict(lambda: defaultdict(int))
    
    for layer_id, combinations in data.get('layers', {}).items():
        for combo_str, count in combinations.items():
            try:
                combo = ast.literal_eval(combo_str)
                for expert_id in combo:
                    layer_expert_counts[int(layer_id)][int(expert_id)] += count
            except (ValueError, SyntaxError):
                continue
    
    # Convert to DataFrame
    results = []
    for layer_id in sorted(layer_expert_counts.keys()):
        for expert_id, count in layer_expert_counts[layer_id].items():
            results.append({
                'Layer': layer_id,
                'Expert_ID': expert_id,
                'Total_Activations': count
            })
    
    df = pd.DataFrame(results)
    
    if df.empty:
        raise ValueError(f"Unable to parse any expert activation data from {json_file}")
    
    return df


def _infer_num_experts(df: pd.DataFrame) -> int:
    """Infer total number of experts from data (default max(Expert_ID)+1)."""
    if "Expert_ID" not in df.columns or df["Expert_ID"].empty:
        raise ValueError("Input data missing Expert_ID, cannot infer num_experts.")
    return int(df["Expert_ID"].max()) + 1


def _complete_layers_with_missing_experts(
    df: pd.DataFrame,
    num_experts: int,
    strict_missing: bool = False,
    missing_sample_limit: int = 20,
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    Force each layer to have a complete set of Expert_IDs [0..num_experts-1].
    - Missing rows are filled with Shapley_Value=0, Total_Activations=0 (if column exists)
    - Returns the completed df and a list of missing expert_ids per layer
    """
    if "Layer" not in df.columns or "Expert_ID" not in df.columns:
        raise ValueError("Input CSV must contain Layer and Expert_ID columns.")

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError("num_experts must be a positive integer.")

    missing_by_layer: Dict[int, List[int]] = {}
    layers = sorted(df["Layer"].unique().tolist())

    # Standardize columns
    df = df.copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Expert_ID"] = df["Expert_ID"].astype(int)

    has_total_acts = "Total_Activations" in df.columns
    if "Shapley_Value" not in df.columns:
        # If no Shapley_Value column, create an all-zero column (random pruning doesn't need it)
        df["Shapley_Value"] = 0.0

    completed_parts = []
    full_index = list(range(num_experts))

    for layer_id in layers:
        layer_df = df[df["Layer"] == layer_id].copy()
        present = set(layer_df["Expert_ID"].tolist())
        missing = [i for i in full_index if i not in present]
        if missing:
            missing_by_layer[int(layer_id)] = missing

        if strict_missing and missing:
            sample = missing[:missing_sample_limit]
            raise ValueError(
                f"Layer {layer_id} missing {len(missing)}/{num_experts} experts (sample: {sample})."
            )

        # Fill with reindex
        layer_df = layer_df.set_index("Expert_ID").reindex(full_index)
        layer_df.index.name = "Expert_ID"
        layer_df = layer_df.reset_index()
        layer_df["Layer"] = int(layer_id)

        # Fill missing values with 0
        layer_df["Shapley_Value"] = layer_df["Shapley_Value"].fillna(0.0)
        if has_total_acts:
            layer_df["Total_Activations"] = layer_df["Total_Activations"].fillna(0).astype(int)

        completed_parts.append(layer_df)

    completed_df = pd.concat(completed_parts, ignore_index=True)
    return completed_df, missing_by_layer


def select_random_by_global_pruning_rate(
    df: pd.DataFrame,
    target_pruning_rate: float,
    seed: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Global random pruning: Retain X% of experts across all layers (randomly selected)

    Args:
        df: Expert data (must contain Layer and Expert_ID)
        target_pruning_rate: Target pruning rate (retained expert ratio, e.g. 0.4 means keep 40%)
        seed: Random seed (for reproducibility)

    Returns:
        selection_results: {layer_id: [expert_ids]}
        actual_rate: Actual pruning rate
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate total experts
    total_experts = len(df)
    target_count = int(total_experts * target_pruning_rate)

    print(f"Total experts: {total_experts}")
    print(f"Target retention: {target_count} experts ({target_pruning_rate:.1%})")
    print(f"Using random selection...")

    # Get all expert (layer_id, expert_id) pairs
    all_experts = [(int(row["Layer"]), int(row["Expert_ID"])) for _, row in df.iterrows()]

    # Randomly select target number of experts
    selected_pairs = random.sample(all_experts, min(target_count, len(all_experts)))

    # Group by layer
    selection_results = {}
    for layer_id, expert_id in selected_pairs:
        if layer_id not in selection_results:
            selection_results[layer_id] = []
        selection_results[layer_id].append(expert_id)

    # Sort expert IDs per layer (for readability)
    for layer_id in selection_results:
        selection_results[layer_id].sort()

    actual_count = len(selected_pairs)
    actual_rate = actual_count / total_experts

    print(f"✓ Actual retention: {actual_count}/{total_experts} ({actual_rate:.1%})")

    return selection_results, actual_rate


def select_random_by_per_layer_pruning_rate(
    df: pd.DataFrame,
    target_pruning_rate: float,
    seed: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Per-layer random pruning: Retain X% of experts per layer (randomly selected)

    Args:
        df: Expert data (must contain Layer and Expert_ID)
        target_pruning_rate: Target pruning rate (retained expert ratio per layer)
        seed: Random seed (for reproducibility)

    Returns:
        selection_results: {layer_id: [expert_ids]}
        avg_rate: Average per-layer pruning rate
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_layers = df["Layer"].nunique()
    experts_per_layer = int(df.groupby("Layer").size().max())

    print(f"Number of layers: {num_layers}")
    print(f"Experts per layer: {experts_per_layer}")
    print(
        f"Target per-layer retention: {int(experts_per_layer * target_pruning_rate)} experts ({target_pruning_rate:.1%})"
    )
    print(f"Using random selection...")

    selection_results = {}
    total_selected = 0

    for layer_id, group in df.groupby("Layer"):
        layer_id = int(layer_id)
        all_expert_ids = group["Expert_ID"].astype(int).tolist()
        target_count = int(len(all_expert_ids) * target_pruning_rate)

        # Randomly select target number of experts
        selected_experts = random.sample(all_expert_ids, min(target_count, len(all_expert_ids)))
        selected_experts.sort()  # Sort for readability

        selection_results[layer_id] = selected_experts
        total_selected += len(selected_experts)

    avg_rate = total_selected / num_layers / experts_per_layer

    print(f"✓ Average per-layer retention: {total_selected / num_layers:.1f}/{experts_per_layer} ({avg_rate:.1%})")

    return selection_results, avg_rate


def save_results(
    selection_results: Dict[int, List[int]],
    pruning_rate: float,
    strategy: str,
    df: pd.DataFrame,
    missing_by_layer: Optional[Dict[int, List[int]]],
    expected_num_experts: Optional[int],
    output_dir: str,
    seed: Optional[int] = None,
):
    """Save selection results and statistics"""

    os.makedirs(output_dir, exist_ok=True)

    # Generate filenames
    rate_str = f"{int(pruning_rate * 100)}"
    output_json = os.path.join(
        output_dir, f"selected_experts_random_{strategy}_rate{rate_str}.json"
    )
    output_csv = os.path.join(
        output_dir, f"selection_stats_random_{strategy}_rate{rate_str}.csv"
    )

    # Save selection results (convert to string keys to match existing format)
    selection_results_str_keys = {str(k): v for k, v in selection_results.items()}
    with open(output_json, "w") as f:
        json.dump(selection_results_str_keys, f, indent=4)

    # Generate statistics
    stats_data = []
    for layer_id, selected_experts in selection_results.items():
        layer_df = df[df["Layer"] == layer_id]
        total_experts = len(layer_df)

        # Calculate selected experts' Shapley value (if exists)
        if "Shapley_Value" in layer_df.columns:
            total_shapley = layer_df["Shapley_Value"].sum()
            selected_shapley = layer_df[layer_df["Expert_ID"].isin(selected_experts)][
                "Shapley_Value"
            ].sum()
            shapley_ratio = (
                selected_shapley / total_shapley if total_shapley > 0 else 0
            )
        else:
            total_shapley = 0
            selected_shapley = 0
            shapley_ratio = 0

        stats_data.append(
            {
                "Layer": layer_id,
                "Total_Experts": total_experts,
                "Selected_Experts": len(selected_experts),
                "Pruning_Rate": len(selected_experts) / total_experts,
                "Total_Shapley": total_shapley,
                "Selected_Shapley": selected_shapley,
                "Shapley_Ratio": shapley_ratio,
                "Expected_Num_Experts": expected_num_experts if expected_num_experts is not None else "",
                "Missing_Experts_Count": (
                    len(missing_by_layer.get(int(layer_id), [])) if missing_by_layer else 0
                ),
                "Missing_Experts_Sample": (
                    ",".join(map(str, missing_by_layer.get(int(layer_id), [])[:20]))
                    if missing_by_layer and missing_by_layer.get(int(layer_id))
                    else ""
                ),
            }
        )

    # Save statistics
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_csv, index=False)

    # Calculate overall statistics
    total_experts = len(df)
    total_selected = sum(len(experts) for experts in selection_results.values())
    actual_global_rate = total_selected / total_experts

    print(f"\n{'='*70}")
    print(f"Random pruning completed!")
    print(f"{'='*70}")
    print(f"Strategy: {strategy} (random)")
    print(f"Target pruning rate: {pruning_rate:.1%}")
    print(f"Actual pruning rate: {actual_global_rate:.1%}")
    if seed is not None:
        print(f"Random seed: {seed}")
    print(f"Total experts: {total_experts}")
    print(f"Retained experts: {total_selected}")
    print(f"Pruned experts: {total_experts - total_selected}")
    print(f"\nResults saved:")
    print(f"  - {output_json}")
    print(f"  - {output_csv}")
    print(f"{'='*70}")

    # Print per-layer summary
    print("\nPer-layer summary:")
    print(
        stats_df[
            ["Layer", "Total_Experts", "Selected_Experts", "Pruning_Rate"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Random Pruning Expert Selection Tool (Baseline Experiment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy description:

  global    - Global pruning rate: Retain X%% of experts across all layers (randomly selected)
  per_layer - Per-layer pruning rate: Retain X%% of experts per layer (randomly selected, recommended)

Example usage:

  # Per-layer, randomly retain 50%% experts (recommended)
  python select_experts_random.py \\
      --input ../calc_shapley/results/expert_shapley_values_all_layers.csv \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # Global, randomly retain 40%% experts
  python select_experts_random.py \\
      --input ../calc_shapley/results/expert_shapley_values_all_layers.csv \\
      --pruning_rate 0.4 \\
      --strategy global

  # Per-layer, randomly retain 30%% experts with random seed
  python select_experts_random.py \\
      --input shapley_values.csv \\
      --pruning_rate 0.3 \\
      --strategy per_layer \\
      --seed 42 \\
      --output ./my_results
        """,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Input file path (supports CSV or JSON format)"
    )
    parser.add_argument(
        "--pruning_rate",
        type=float,
        required=True,
        help="Retention rate (ratio of experts to retain, 0-1, e.g. 0.4 means keep 40%%, prune 60%%)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["global", "per_layer"],
        default="per_layer",
        help="Pruning strategy (global: global pruning rate, per_layer: per-layer pruning rate, default: per_layer)",
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory (default: results)"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="(Optional) Total experts per layer. If provided or inferable, missing experts will be filled to avoid silent bugs.",
    )
    parser.add_argument(
        "--strict_missing_experts",
        action="store_true",
        help="If missing expert rows are found in a layer, exit with error (default: off, fills with 0 instead).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for reproducibility, default: None, results differ each run)",
    )

    args = parser.parse_args()

    # Validate pruning rate
    if not 0 < args.pruning_rate <= 1:
        parser.error("Pruning rate must be in the range (0, 1]")

    # Check input file
    if not os.path.exists(args.input):
        parser.error(f"Input file does not exist: {args.input}")

    print("=" * 70)
    print("Random Pruning Expert Selection (Baseline Experiment)")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Target pruning rate: {args.pruning_rate:.1%}")
    print(f"Strategy: {args.strategy} (random)")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Read data
    print(f"\nReading data...")
    if args.input.endswith('.json'):
        df = parse_shapley_json(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"✓ Read complete: {len(df)} records")

    # Fill missing experts (strongly recommended)
    inferred = _infer_num_experts(df) if args.num_experts is None else int(args.num_experts)
    df_completed, missing_by_layer = _complete_layers_with_missing_experts(
        df,
        num_experts=inferred,
        strict_missing=args.strict_missing_experts,
    )
    if missing_by_layer:
        total_layers = df_completed["Layer"].nunique()
        affected = len(missing_by_layer)
        worst_layer = max(missing_by_layer.items(), key=lambda kv: len(kv[1]))
        print(
            f"⚠️ Missing expert rows detected: {affected}/{total_layers} layers have missing entries. Worst layer: Layer {worst_layer[0]} missing {len(worst_layer[1])}/{inferred}."
        )
    else:
        print("✓ All layers have complete expert rows, no missing entries.")

    df = df_completed

    # Select experts randomly based on strategy
    print(f"\nUsing {args.strategy} strategy for random selection...\n")

    if args.strategy == "global":
        # Global pruning rate strategy
        selection_results, actual_rate = select_random_by_global_pruning_rate(
            df, args.pruning_rate, args.seed
        )
    else:
        # Per-layer pruning rate strategy
        selection_results, actual_rate = select_random_by_per_layer_pruning_rate(
            df, args.pruning_rate, args.seed
        )

    # Save results
    save_results(
        selection_results,
        args.pruning_rate,
        args.strategy,
        df,
        missing_by_layer,
        inferred,
        args.output,
        args.seed,
    )


if __name__ == "__main__":
    main()
