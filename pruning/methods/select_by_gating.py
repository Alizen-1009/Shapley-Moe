#!/usr/bin/env python3
"""
Gating Score Based Expert Selection Tool (Baseline Experiment)

Read expert gating scores from gating_scores JSON files in the few-shot/results directory, then prune based on gating score.

Supports two strategies:
1. Global pruning rate: Retain X% of experts across all layers (sorted by average gating score)
2. Per-layer pruning rate: Retain X% of experts per layer (sorted by average gating score, recommended)
"""

import pandas as pd
import argparse
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


def parse_gating_scores_json(json_file: str) -> pd.DataFrame:
    """
    Parse expert gating scores from gating_scores JSON file
    
    Args:
        json_file: gating_scores JSON file path
        
    Returns:
        DataFrame with Layer, Expert_ID, Avg_Gating_Score, Sum_Gating_Score columns
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for layer_id, expert_data in data['layers'].items():
        for expert_id, stats in expert_data.items():
            results.append({
                'Layer': int(layer_id),
                'Expert_ID': int(expert_id),
                'Avg_Gating_Score': stats.get('avg_gating_score', 0.0),
                'Sum_Gating_Score': stats.get('sum_gating_score', 0.0),
                'Count': stats.get('count', 0)
            })
    
    df = pd.DataFrame(results)
    
    if df.empty:
        raise ValueError(f"Unable to parse any expert gating score data from {json_file}")
    
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
    - Missing rows are filled with Avg_Gating_Score=0, Sum_Gating_Score=0
    - Returns the completed df and a list of missing expert_ids per layer
    """
    if "Layer" not in df.columns or "Expert_ID" not in df.columns:
        raise ValueError("Input data must contain Layer and Expert_ID columns.")

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError("num_experts must be a positive integer.")

    missing_by_layer: Dict[int, List[int]] = {}
    layers = sorted(df["Layer"].unique().tolist())

    # Standardize columns
    df = df.copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Expert_ID"] = df["Expert_ID"].astype(int)

    if "Avg_Gating_Score" not in df.columns:
        raise ValueError("Input data must contain Avg_Gating_Score column.")
    
    # Add Shapley_Value column (for statistics, not actually used here)
    if "Shapley_Value" not in df.columns:
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
        layer_df["Avg_Gating_Score"] = layer_df["Avg_Gating_Score"].fillna(0.0)
        if "Sum_Gating_Score" in layer_df.columns:
            layer_df["Sum_Gating_Score"] = layer_df["Sum_Gating_Score"].fillna(0.0)
        if "Count" in layer_df.columns:
            layer_df["Count"] = layer_df["Count"].fillna(0).astype(int)
        layer_df["Shapley_Value"] = layer_df["Shapley_Value"].fillna(0.0)

        completed_parts.append(layer_df)

    completed_df = pd.concat(completed_parts, ignore_index=True)
    return completed_df, missing_by_layer


def select_by_gating_score_alpha(df: pd.DataFrame, alpha: float) -> Tuple[Dict[int, List[int]], int]:
    """
    Select experts based on alpha factor and gating score

    Args:
        df: Data containing Avg_Gating_Score
        alpha: Cumulative gating score ratio threshold

    Returns:
        selection_results: {layer_id: [expert_ids]}
        total_selected: Total number of selected experts
    """
    selection_results = {}
    total_selected = 0

    for layer_id, group in df.groupby("Layer"):
        # Sort by average gating score in descending order
        group_sorted = group.sort_values("Avg_Gating_Score", ascending=False)
        total_gating_score = group_sorted["Avg_Gating_Score"].sum()
        target_gating_score = total_gating_score * alpha

        cumulative_gating_score = 0
        selected_experts = []

        for _, row in group_sorted.iterrows():
            cumulative_gating_score += row["Avg_Gating_Score"]
            selected_experts.append(int(row["Expert_ID"]))

            if cumulative_gating_score >= target_gating_score:
                break

        selection_results[int(layer_id)] = selected_experts
        total_selected += len(selected_experts)

    return selection_results, total_selected


def find_alpha_for_global_pruning_rate(
    df: pd.DataFrame,
    target_pruning_rate: float,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> Tuple[float, Dict[int, List[int]], float]:
    """
    Use binary search to find alpha such that global pruning rate is close to target (based on gating score)

    Args:
        df: Gating score data
        target_pruning_rate: Target pruning rate (retained expert ratio, e.g. 0.4 means keep 40%)
        tolerance: Tolerance
        max_iterations: Maximum iterations

    Returns:
        best_alpha: Best alpha found
        selection_results: Expert selection results
        actual_rate: Actual pruning rate
    """
    total_experts = len(df)
    target_count = int(total_experts * target_pruning_rate)

    print(f"Total experts: {total_experts}")
    print(f"Target retention: {target_count} experts ({target_pruning_rate:.1%})")
    print(f"Starting binary search for alpha (based on gating score)...")

    # Binary search
    left, right = 0.0, 1.0
    best_alpha = 0.5
    best_selection = None
    best_diff = float("inf")

    for iteration in range(max_iterations):
        mid_alpha = (left + right) / 2
        selection, selected_count = select_by_gating_score_alpha(df, mid_alpha)

        diff = selected_count - target_count
        abs_diff = abs(diff)

        print(
            f"  Iteration {iteration+1}: alpha={mid_alpha:.4f}, selected={selected_count}, target={target_count}, diff={diff}"
        )

        # Update best result
        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        # Check tolerance
        actual_rate = selected_count / total_experts
        if abs(actual_rate - target_pruning_rate) < tolerance:
            print(f"✓ Found suitable alpha={mid_alpha:.4f}")
            break

        # Adjust search range
        if selected_count < target_count:
            left = mid_alpha
        else:
            right = mid_alpha

    actual_rate = (
        sum(len(experts) for experts in best_selection.values()) / total_experts
    )

    print(f"\n✓ Best alpha = {best_alpha:.4f}")
    print(
        f"  Actual retention: {sum(len(experts) for experts in best_selection.values())}/{total_experts} ({actual_rate:.1%})"
    )

    return best_alpha, best_selection, actual_rate


def find_alpha_for_per_layer_pruning_rate(
    df: pd.DataFrame,
    target_pruning_rate: float,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> Tuple[float, Dict[int, List[int]], float]:
    """
    Use binary search to find alpha such that average per-layer pruning rate is close to target (based on gating score)

    Args:
        df: Gating score data
        target_pruning_rate: Target pruning rate (retained expert ratio per layer)
        tolerance: Tolerance
        max_iterations: Maximum iterations

    Returns:
        best_alpha: Best alpha found
        selection_results: Expert selection results
        avg_rate: Average per-layer pruning rate
    """
    num_layers = df["Layer"].nunique()
    experts_per_layer = int(df.groupby("Layer").size().max())

    print(f"Number of layers: {num_layers}")
    print(f"Experts per layer: {experts_per_layer}")
    print(
        f"Target per-layer retention: {int(experts_per_layer * target_pruning_rate)} experts ({target_pruning_rate:.1%})"
    )
    print(f"Starting binary search for alpha (based on gating score)...")

    # Binary search
    left, right = 0.0, 1.0
    best_alpha = 0.5
    best_selection = None
    best_diff = float("inf")

    for iteration in range(max_iterations):
        mid_alpha = (left + right) / 2
        selection, total_selected = select_by_gating_score_alpha(df, mid_alpha)

        # Calculate average experts retained per layer
        avg_selected_per_layer = total_selected / num_layers
        target_per_layer = experts_per_layer * target_pruning_rate

        diff = avg_selected_per_layer - target_per_layer
        abs_diff = abs(diff)

        print(
            f"  Iteration {iteration+1}: alpha={mid_alpha:.4f}, avg per layer={avg_selected_per_layer:.1f}, target={target_per_layer:.1f}, diff={diff:.1f}"
        )

        # Update best result
        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        # Check tolerance
        actual_rate = avg_selected_per_layer / experts_per_layer
        if abs(actual_rate - target_pruning_rate) < tolerance:
            print(f"✓ Found suitable alpha={mid_alpha:.4f}")
            break

        # Adjust search range
        if avg_selected_per_layer < target_per_layer:
            left = mid_alpha
        else:
            right = mid_alpha

    avg_rate = (
        sum(len(experts) for experts in best_selection.values())
        / num_layers
        / experts_per_layer
    )

    print(f"\n✓ Best alpha = {best_alpha:.4f}")
    print(
        f"  Average per-layer retention: {sum(len(experts) for experts in best_selection.values()) / num_layers:.1f}/{experts_per_layer} ({avg_rate:.1%})"
    )

    return best_alpha, best_selection, avg_rate


def save_results(
    selection_results: Dict[int, List[int]],
    alpha: float,
    pruning_rate: float,
    strategy: str,
    df: pd.DataFrame,
    missing_by_layer: Optional[Dict[int, List[int]]],
    expected_num_experts: Optional[int],
    output_dir: str,
):
    """Save selection results and statistics"""

    os.makedirs(output_dir, exist_ok=True)

    # Generate filenames
    rate_str = f"{int(pruning_rate * 100)}"
    output_json = os.path.join(
        output_dir, f"selected_experts_gating_{strategy}_rate{rate_str}.json"
    )
    output_csv = os.path.join(
        output_dir, f"selection_stats_gating_{strategy}_rate{rate_str}.csv"
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
        total_gating_score = layer_df["Avg_Gating_Score"].sum()
        selected_gating_score = layer_df[layer_df["Expert_ID"].isin(selected_experts)][
            "Avg_Gating_Score"
        ].sum()
        gating_score_ratio = (
            selected_gating_score / total_gating_score if total_gating_score > 0 else 0
        )

        # Shapley value (if exists, usually 0 here)
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
                "Total_Gating_Score": total_gating_score,
                "Selected_Gating_Score": selected_gating_score,
                "Gating_Score_Ratio": gating_score_ratio,
                "Total_Shapley": total_shapley,
                "Selected_Shapley": selected_shapley,
                "Shapley_Ratio": shapley_ratio,
                "Alpha": alpha,
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
    print(f"Gating Score based selection completed!")
    print(f"{'='*70}")
    print(f"Strategy: {strategy} (gating score)")
    print(f"Best alpha: {alpha:.4f}")
    print(f"Target pruning rate: {pruning_rate:.1%}")
    print(f"Actual pruning rate: {actual_global_rate:.1%}")
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
            ["Layer", "Total_Experts", "Selected_Experts", "Pruning_Rate", "Gating_Score_Ratio"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gating Score Based Expert Selection Tool (Baseline Experiment) - Reads gating scores from gating_scores JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy description:

  global    - Global pruning rate: Retain X%% of experts across all layers (sorted by average gating score)
  per_layer - Per-layer pruning rate: Retain X%% of experts per layer (sorted by average gating score, recommended)

Example usage:

  # Per-layer, retain 50%% experts by gating score (recommended)
  python select_experts_by_gating_score.py \\
      --input ../few-shot/results/gpt-oss-20b_arc_easy_25_gating_scores.json \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # Global, retain 40%% experts by gating score
  python select_experts_by_gating_score.py \\
      --input ../few-shot/results/gpt-oss-20b_arc_easy_25_gating_scores.json \\
      --pruning_rate 0.4 \\
      --strategy global
        """,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="gating_scores JSON file path (files in few-shot/results directory)"
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
        "--tolerance", type=float, default=0.01, help="Binary search tolerance (default: 0.01)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Binary search max iterations (default: 50)",
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

    args = parser.parse_args()

    # Validate pruning rate
    if not 0 < args.pruning_rate <= 1:
        parser.error("Pruning rate must be in the range (0, 1]")

    # Check input file
    if not os.path.exists(args.input):
        parser.error(f"Input file does not exist: {args.input}")

    print("=" * 70)
    print("Gating Score Based Expert Selection (Baseline Experiment)")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Target pruning rate: {args.pruning_rate:.1%}")
    print(f"Strategy: {args.strategy} (gating score)")
    print("=" * 70)

    # Parse gating_scores JSON file
    print(f"\nParsing gating_scores JSON file...")
    df = parse_gating_scores_json(args.input)
    print(f"✓ Parsing complete: {len(df)} records")
    print(f"  Layers: {df['Layer'].nunique()}")
    print(f"  Expert ID range: {df['Expert_ID'].min()} - {df['Expert_ID'].max()}")
    print(f"  Average gating score range: {df['Avg_Gating_Score'].min():.6f} - {df['Avg_Gating_Score'].max():.6f}")

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

    # Select experts based on strategy
    print(f"\nUsing {args.strategy} strategy for binary search (based on gating score)...\n")

    if args.strategy == "global":
        # Global pruning rate strategy
        best_alpha, selection_results, actual_rate = find_alpha_for_global_pruning_rate(
            df, args.pruning_rate, args.tolerance, args.max_iterations
        )
    else:
        # Per-layer pruning rate strategy
        best_alpha, selection_results, actual_rate = (
            find_alpha_for_per_layer_pruning_rate(
                df, args.pruning_rate, args.tolerance, args.max_iterations
            )
        )

    # Save results
    save_results(
        selection_results,
        best_alpha,
        args.pruning_rate,
        args.strategy,
        df,
        missing_by_layer,
        inferred,
        args.output,
    )


if __name__ == "__main__":
    main()
