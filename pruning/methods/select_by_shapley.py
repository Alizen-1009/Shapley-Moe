#!/usr/bin/env python3
"""
Shapley Value Based Expert Selection Tool

Supports four pruning strategies:

1. topk_per_layer  - Select top-k experts with highest Shapley values per layer (recommended, simple and direct)
2. topk_global     - Select experts with highest Shapley values globally
3. alpha_per_layer - Use alpha factor, select fewest experts whose cumulative Shapley values reach alpha ratio per layer
4. alpha_global    - Use alpha factor, cumulate globally

Strategy comparison:
- topk: Directly sort by Shapley value magnitude, select top k. Simple, highly interpretable.
- alpha: Select the fewest experts whose cumulative Shapley values reach alpha ratio of total. Considers contribution distribution.

- per_layer: Select independently per layer, ensures sufficient experts in each layer.
- global: Select uniformly across all layers, some layers may have fewer experts.
"""

import pandas as pd
import argparse
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


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
    Missing rows are filled with Shapley_Value=0
    """
    if "Layer" not in df.columns or "Expert_ID" not in df.columns:
        raise ValueError("Input CSV must contain Layer and Expert_ID columns.")

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError("num_experts must be a positive integer.")

    missing_by_layer: Dict[int, List[int]] = {}
    layers = sorted(df["Layer"].unique().tolist())

    df = df.copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Expert_ID"] = df["Expert_ID"].astype(int)

    has_total_acts = "Total_Activations" in df.columns
    if "Shapley_Value" not in df.columns:
        raise ValueError("Input CSV must contain Shapley_Value column.")

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

        layer_df = layer_df.set_index("Expert_ID").reindex(full_index)
        layer_df.index.name = "Expert_ID"
        layer_df = layer_df.reset_index()
        layer_df["Layer"] = int(layer_id)

        layer_df["Shapley_Value"] = layer_df["Shapley_Value"].fillna(0.0)
        if has_total_acts:
            layer_df["Total_Activations"] = layer_df["Total_Activations"].fillna(0).astype(int)

        completed_parts.append(layer_df)

    completed_df = pd.concat(completed_parts, ignore_index=True)
    return completed_df, missing_by_layer


# =============================================================================
# Strategy 1: TopK Per Layer - Select top-k experts with highest Shapley values per layer
# =============================================================================

def select_topk_per_layer(
    df: pd.DataFrame,
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    Select top-k experts with highest Shapley values per layer
    
    Args:
        df: Shapley value data
        pruning_rate: Retention rate (0.0-1.0)
        num_experts: Total experts per layer
        
    Returns:
        {layer_id: [selected_expert_ids]}
    """
    keep_count = max(1, int(num_experts * pruning_rate))
    
    print(f"TopK Per Layer strategy:")
    print(f"  Keeping {keep_count}/{num_experts} experts per layer (retention rate: {pruning_rate:.1%})")
    
    selection_results = {}
    
    for layer_id, group in df.groupby("Layer"):
        # Sort by Shapley value in descending order
        sorted_experts = group.sort_values("Shapley_Value", ascending=False)
        # Select top k
        selected = sorted_experts.head(keep_count)["Expert_ID"].astype(int).tolist()
        selection_results[int(layer_id)] = sorted(selected)
    
    total_selected = sum(len(v) for v in selection_results.values())
    print(f"  Total retained: {total_selected} experts")
    
    return selection_results


# =============================================================================
# Strategy 2: TopK Global - Select experts with highest Shapley values globally
# =============================================================================

def select_topk_global(
    df: pd.DataFrame,
    pruning_rate: float
) -> Dict[int, List[int]]:
    """
    Select experts with highest Shapley values globally
    
    Args:
        df: Shapley value data
        pruning_rate: Retention rate (0.0-1.0)
        
    Returns:
        {layer_id: [selected_expert_ids]}
    """
    total_experts = len(df)
    keep_count = max(df["Layer"].nunique(), int(total_experts * pruning_rate))
    
    print(f"TopK Global strategy:")
    print(f"  Keeping {keep_count}/{total_experts} experts globally (retention rate: {pruning_rate:.1%})")
    
    # Sort globally by Shapley value
    sorted_df = df.sort_values("Shapley_Value", ascending=False)
    
    # Select top-k
    selected_df = sorted_df.head(keep_count)
    
    # Group by layer
    selection_results = defaultdict(list)
    for _, row in selected_df.iterrows():
        selection_results[int(row["Layer"])].append(int(row["Expert_ID"]))
    
    # Ensure at least one expert per layer
    for layer_id in df["Layer"].unique():
        layer_id = int(layer_id)
        if layer_id not in selection_results or not selection_results[layer_id]:
            # Select the expert with highest Shapley value in this layer
            layer_df = df[df["Layer"] == layer_id]
            best_expert = layer_df.loc[layer_df["Shapley_Value"].idxmax(), "Expert_ID"]
            selection_results[layer_id].append(int(best_expert))
    
    # Sort
    result = {k: sorted(v) for k, v in selection_results.items()}
    
    total_selected = sum(len(v) for v in result.values())
    print(f"  Actual retained: {total_selected} experts")
    
    return dict(result)


# =============================================================================
# Strategy 3: Alpha Per Layer - Cumulative Shapley values reaching alpha ratio per layer
# =============================================================================

def select_by_alpha(df: pd.DataFrame, alpha: float) -> Tuple[Dict[int, List[int]], int]:
    """
    Select experts based on alpha factor (independently per layer)
    Select the fewest experts whose cumulative Shapley values reach alpha ratio of layer total
    """
    selection_results = {}
    total_selected = 0

    for layer_id, group in df.groupby("Layer"):
        group_sorted = group.sort_values("Shapley_Value", ascending=False)
        total_shapley = group_sorted["Shapley_Value"].sum()
        target_shapley = total_shapley * alpha

        cumulative_shapley = 0
        selected_experts = []

        for _, row in group_sorted.iterrows():
            cumulative_shapley += row["Shapley_Value"]
            selected_experts.append(int(row["Expert_ID"]))

            if cumulative_shapley >= target_shapley:
                break

        selection_results[int(layer_id)] = selected_experts
        total_selected += len(selected_experts)

    return selection_results, total_selected


def select_alpha_per_layer(
    df: pd.DataFrame,
    pruning_rate: float,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Use binary search to find alpha such that average retention rate per layer is close to target
    """
    num_layers = df["Layer"].nunique()
    experts_per_layer = int(df.groupby("Layer").size().max())

    print(f"Alpha Per Layer strategy:")
    print(f"  Target per-layer retention: {int(experts_per_layer * pruning_rate)}/{experts_per_layer} (retention rate: {pruning_rate:.1%})")
    print(f"  Starting binary search for alpha...")

    left, right = 0.0, 1.0
    best_alpha = 0.5
    best_selection = None
    best_diff = float("inf")

    for iteration in range(max_iterations):
        mid_alpha = (left + right) / 2
        selection, total_selected = select_by_alpha(df, mid_alpha)

        avg_selected_per_layer = total_selected / num_layers
        target_per_layer = experts_per_layer * pruning_rate

        diff = avg_selected_per_layer - target_per_layer
        abs_diff = abs(diff)

        if iteration < 5 or iteration % 10 == 0:
            print(f"    Iteration {iteration+1}: alpha={mid_alpha:.4f}, avg per layer={avg_selected_per_layer:.1f}")

        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        actual_rate = avg_selected_per_layer / experts_per_layer
        if abs(actual_rate - pruning_rate) < tolerance:
            print(f"  ✓ Found alpha={mid_alpha:.4f}")
            break

        if avg_selected_per_layer < target_per_layer:
            left = mid_alpha
        else:
            right = mid_alpha

    total_selected = sum(len(v) for v in best_selection.values())
    print(f"  Best alpha = {best_alpha:.4f}")
    print(f"  Total retained: {total_selected} experts")

    return best_selection, best_alpha


# =============================================================================
# Strategy 4: Alpha Global - Cumulative Shapley values reaching alpha ratio globally
# =============================================================================

def select_alpha_global(
    df: pd.DataFrame,
    pruning_rate: float,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Use binary search to find alpha such that global retention rate is close to target
    """
    total_experts = len(df)
    target_count = int(total_experts * pruning_rate)

    print(f"Alpha Global strategy:")
    print(f"  Target retention: {target_count}/{total_experts} (retention rate: {pruning_rate:.1%})")
    print(f"  Starting binary search for alpha...")

    left, right = 0.0, 1.0
    best_alpha = 0.5
    best_selection = None
    best_diff = float("inf")

    for iteration in range(max_iterations):
        mid_alpha = (left + right) / 2
        selection, selected_count = select_by_alpha(df, mid_alpha)

        diff = selected_count - target_count
        abs_diff = abs(diff)

        if iteration < 5 or iteration % 10 == 0:
            print(f"    Iteration {iteration+1}: alpha={mid_alpha:.4f}, selected={selected_count}")

        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        actual_rate = selected_count / total_experts
        if abs(actual_rate - pruning_rate) < tolerance:
            print(f"  ✓ Found alpha={mid_alpha:.4f}")
            break

        if selected_count < target_count:
            left = mid_alpha
        else:
            right = mid_alpha

    total_selected = sum(len(v) for v in best_selection.values())
    print(f"  Best alpha = {best_alpha:.4f}")
    print(f"  Total retained: {total_selected} experts")

    return best_selection, best_alpha


# =============================================================================
# Save Results
# =============================================================================

def save_results(
    selection_results: Dict[int, List[int]],
    strategy: str,
    pruning_rate: float,
    df: pd.DataFrame,
    output_file: str,
    alpha: Optional[float] = None,
):
    """Save selection results"""
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # Calculate statistics
    total_experts = len(df)
    total_selected = sum(len(v) for v in selection_results.values())
    actual_rate = total_selected / total_experts

    # Save results (convert to string keys)
    output_data = {str(k): sorted(v) for k, v in selection_results.items()}
    
    # Add metadata
    output_data["_metadata"] = {
        "method": "shapley",
        "strategy": strategy,
        "target_rate": pruning_rate,
        "actual_rate": actual_rate,
        "total_experts": total_experts,
        "selected_experts": total_selected,
    }
    if alpha is not None:
        output_data["_metadata"]["alpha"] = alpha

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Selection completed!")
    print(f"{'='*70}")
    print(f"Strategy: {strategy}")
    print(f"Target retention rate: {pruning_rate:.1%}")
    print(f"Actual retention rate: {actual_rate:.1%}")
    print(f"Retained experts: {total_selected}/{total_experts}")
    if alpha is not None:
        print(f"Alpha: {alpha:.4f}")
    print(f"\nResults saved: {output_file}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Shapley Value Based Expert Selection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Four pruning strategies:

  topk_per_layer  - Select top-k experts with highest Shapley values per layer (recommended)
  topk_global     - Select experts with highest Shapley values globally
  alpha_per_layer - Cumulative Shapley values reaching alpha ratio per layer
  alpha_global    - Cumulative Shapley values reaching alpha ratio globally

Example usage:

  # Per-layer TopK (recommended, simple and direct)
  python select_by_shapley.py \\
      --input gsm8k_25_shapley.csv \\
      --output selected_experts.json \\
      --pruning_rate 0.5 \\
      --strategy topk_per_layer

  # Global TopK
  python select_by_shapley.py \\
      --input gsm8k_25_shapley.csv \\
      --output selected_experts.json \\
      --pruning_rate 0.5 \\
      --strategy topk_global

  # Alpha per layer (considers contribution distribution)
  python select_by_shapley.py \\
      --input gsm8k_25_shapley.csv \\
      --output selected_experts.json \\
      --pruning_rate 0.5 \\
      --strategy alpha_per_layer
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Shapley value CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument(
        "--pruning_rate",
        type=float,
        required=True,
        help="Retention rate (0.0-1.0), e.g. 0.5 means keep 50%%",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["alpha_per_layer", "alpha_global", "topk_per_layer", "topk_global",
                 "per_layer", "global"],  # Backward compatible
        default="alpha_per_layer",
        help="Pruning strategy (default: alpha_per_layer)",
    )
    parser.add_argument("--num_experts", type=int, default=None, help="Total experts per layer")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Tolerance for Alpha strategy")
    parser.add_argument("--max_iterations", type=int, default=50, help="Max iterations for Alpha strategy")

    args = parser.parse_args()

    # Backward compatible parameter names
    if args.strategy == "per_layer":
        args.strategy = "alpha_per_layer"
        print("Note: 'per_layer' has been renamed to 'alpha_per_layer'")
    elif args.strategy == "global":
        args.strategy = "alpha_global"
        print("Note: 'global' has been renamed to 'alpha_global'")

    if not 0 < args.pruning_rate <= 1:
        parser.error("pruning_rate must be in the range (0, 1]")

    if not os.path.exists(args.input):
        parser.error(f"Input file does not exist: {args.input}")

    print("=" * 70)
    print("Shapley Value Expert Selection")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Target retention rate: {args.pruning_rate:.1%}")
    print(f"Strategy: {args.strategy}")
    print("=" * 70)

    # Read data
    print(f"\nReading data...")
    df = pd.read_csv(args.input)
    print(f"✓ Read complete: {len(df)} records")

    # Fill missing experts
    inferred = _infer_num_experts(df) if args.num_experts is None else int(args.num_experts)
    df_completed, missing_by_layer = _complete_layers_with_missing_experts(df, num_experts=inferred)
    
    if missing_by_layer:
        affected = len(missing_by_layer)
        print(f"⚠️ {affected} layers have missing experts, filled with Shapley=0")
    else:
        print("✓ All layers have complete experts")

    df = df_completed
    print(f"Total experts: {len(df)} ({df['Layer'].nunique()} layers × {inferred} experts/layer)")
    print()

    # Execute selection
    alpha = None
    
    if args.strategy == "topk_per_layer":
        selection_results = select_topk_per_layer(df, args.pruning_rate, inferred)
        
    elif args.strategy == "topk_global":
        selection_results = select_topk_global(df, args.pruning_rate)
        
    elif args.strategy == "alpha_per_layer":
        selection_results, alpha = select_alpha_per_layer(
            df, args.pruning_rate, args.tolerance, args.max_iterations
        )
        
    elif args.strategy == "alpha_global":
        selection_results, alpha = select_alpha_global(
            df, args.pruning_rate, args.tolerance, args.max_iterations
        )

    # Save results
    save_results(
        selection_results,
        args.strategy,
        args.pruning_rate,
        df,
        args.output,
        alpha,
    )


if __name__ == "__main__":
    main()
