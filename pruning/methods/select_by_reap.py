#!/usr/bin/env python3
"""
REAP (Router-weighted Expert Activation Pruning) Expert Selection Script

Based on the REAP paper's approach, using router_weight × expert_activation_norm as expert importance metric.
REAP score = mean(router_weight × expert_activation_norm)

Input format:
    *_reap.json files from analyze_all_in_one.py
    JSON structure:
    {
        "model": "...",
        "dataset": "...",
        "total_samples": N,
        "total_layers": M,
        "layers": {
            "0": {
                "0": {"reap_sum": ..., "reap_mean": ..., "count": ...},
                "1": {...},
                ...
            },
            ...
        }
    }

Output format:
    {
        "0": [0, 1, 3, 5, ...],  // Layer 0 retained expert IDs
        "1": [2, 4, 6, 8, ...],  // Layer 1 retained expert IDs
        ...
    }

Usage examples:
    # Select by pruning rate
    python select_experts_by_reap.py \\
        --input reap_scores.json \\
        --output selected_experts.json \\
        --pruning_rate 0.5 \\
        --strategy per_layer
        
    # Select by target number
    python select_experts_by_reap.py \\
        --input reap_scores.json \\
        --output selected_experts.json \\
        --target_number 128
"""

import json
import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_reap_json(input_file: str) -> Tuple[Dict[int, Dict[int, float]], int]:
    """
    Parse REAP JSON file
    
    Args:
        input_file: REAP JSON file path
        
    Returns:
        expert_scores: {layer_id: {expert_id: reap_mean_score}}
        num_experts: Number of experts per layer (inferred)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    expert_scores = defaultdict(dict)
    max_expert_id = 0
    
    layers_data = data.get("layers", {})
    
    for layer_str, layer_data in layers_data.items():
        layer_idx = int(layer_str)
        
        for expert_str, expert_info in layer_data.items():
            expert_id = int(expert_str)
            
            # Use reap_mean as primary score
            if isinstance(expert_info, dict):
                score = expert_info.get("reap_mean", 0.0)
            else:
                score = float(expert_info)
            
            expert_scores[layer_idx][expert_id] = score
            max_expert_id = max(max_expert_id, expert_id)
    
    num_experts = max_expert_id + 1
    
    logger.info(f"Parsed REAP data: {len(expert_scores)} layers, {num_experts} experts/layer")
    
    return dict(expert_scores), num_experts


def _complete_layers_with_all_experts(
    expert_scores: Dict[int, Dict[int, float]],
    num_experts: int
) -> Dict[int, Dict[int, float]]:
    """
    Ensure each layer has scores for all experts (missing ones set to 0)
    """
    completed = {}
    for layer_id in sorted(expert_scores.keys()):
        layer_scores = expert_scores[layer_id].copy()
        for exp_id in range(num_experts):
            if exp_id not in layer_scores:
                layer_scores[exp_id] = 0.0
        completed[layer_id] = layer_scores
    return completed


def select_by_target_number(
    expert_scores: Dict[int, Dict[int, float]],
    target_number: int
) -> Dict[int, List[int]]:
    """
    Select experts with highest REAP scores per layer by target number
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        target_number: Number of experts to retain per layer
        
    Returns:
        selected: {layer_id: [expert_ids]}
    """
    selected = {}
    
    for layer_id in sorted(expert_scores.keys()):
        layer_scores = expert_scores[layer_id]
        
        # Sort by score in descending order
        sorted_experts = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top target_number
        selected_experts = [exp_id for exp_id, _ in sorted_experts[:target_number]]
        selected[layer_id] = sorted(selected_experts)
    
    return selected


def select_by_pruning_rate_per_layer(
    expert_scores: Dict[int, Dict[int, float]],
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    Per-layer pruning: select experts based on per-layer REAP scores
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        pruning_rate: Retention rate (0.0-1.0)
        num_experts: Number of experts per layer
        
    Returns:
        selected: {layer_id: [expert_ids]}
    """
    # Calculate number of experts to retain per layer
    keep_count = int(num_experts * pruning_rate)
    keep_count = max(1, keep_count)  # Keep at least 1 expert
    
    logger.info(f"Per-layer pruning: retention rate={pruning_rate:.2%}, keeping {keep_count}/{num_experts} experts per layer")
    
    return select_by_target_number(expert_scores, keep_count)


def select_by_pruning_rate_global(
    expert_scores: Dict[int, Dict[int, float]],
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    Global pruning: select experts based on global REAP score distribution
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        pruning_rate: Retention rate (0.0-1.0)
        num_experts: Number of experts per layer
        
    Returns:
        selected: {layer_id: [expert_ids]}
    """
    # Collect all expert scores
    all_scores = []
    for layer_id, layer_scores in expert_scores.items():
        for exp_id, score in layer_scores.items():
            all_scores.append((layer_id, exp_id, score))
    
    # Sort by score
    all_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate retention count
    total_experts = len(all_scores)
    keep_count = int(total_experts * pruning_rate)
    keep_count = max(len(expert_scores), keep_count)  # Keep at least 1 per layer
    
    # Select highest scoring experts
    selected_set = set()
    for layer_id, exp_id, _ in all_scores[:keep_count]:
        selected_set.add((layer_id, exp_id))
    
    # Ensure at least one expert per layer
    for layer_id in expert_scores.keys():
        layer_experts = [(l, e) for l, e in selected_set if l == layer_id]
        if not layer_experts:
            # Select the expert with highest score in this layer
            best_expert = max(expert_scores[layer_id].items(), key=lambda x: x[1])
            selected_set.add((layer_id, best_expert[0]))
    
    # Organize results
    selected = defaultdict(list)
    for layer_id, exp_id in selected_set:
        selected[layer_id].append(exp_id)
    
    # Sort
    result = {layer_id: sorted(experts) for layer_id, experts in selected.items()}
    
    # Calculate actual retention rate
    total_kept = sum(len(v) for v in result.values())
    actual_rate = total_kept / total_experts
    
    logger.info(f"Global pruning: target retention rate={pruning_rate:.2%}, actual retention rate={actual_rate:.2%}")
    logger.info(f"Retained {total_kept}/{total_experts} experts")
    
    return dict(result)


def save_results(
    selected: Dict[int, List[int]],
    output_file: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save selection results
    """
    # Convert keys to strings (JSON requirement)
    output_data = {str(k): v for k, v in selected.items()}
    
    if metadata:
        output_data["_metadata"] = metadata
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Results saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="REAP Expert Selection - Based on router_weight × expert_activation_norm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Select by pruning rate (per-layer strategy, recommended)
  python select_experts_by_reap.py \\
      --input results/model_dataset_reap.json \\
      --output selected_50pct.json \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # Select by pruning rate (global strategy)
  python select_experts_by_reap.py \\
      --input results/model_dataset_reap.json \\
      --output selected_global.json \\
      --pruning_rate 0.5 \\
      --strategy global

  # Select by target number
  python select_experts_by_reap.py \\
      --input results/model_dataset_reap.json \\
      --output selected_128experts.json \\
      --target_number 128
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="REAP JSON file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--pruning_rate",
        type=float,
        default=None,
        help="Retention rate (0.0-1.0), mutually exclusive with --target_number"
    )
    parser.add_argument(
        "--target_number",
        type=int,
        default=None,
        help="Number of experts to retain per layer, mutually exclusive with --pruning_rate"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["global", "per_layer"],
        default="per_layer",
        help="Pruning strategy: global or per_layer (recommended)"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Total experts per layer (optional, auto-inferred)"
    )
    
    args = parser.parse_args()
    
    # Parameter validation
    if args.pruning_rate is None and args.target_number is None:
        parser.error("Must specify either --pruning_rate or --target_number")
    
    if args.pruning_rate is not None and args.target_number is not None:
        parser.error("--pruning_rate and --target_number are mutually exclusive")
    
    if args.pruning_rate is not None and not (0.0 < args.pruning_rate <= 1.0):
        parser.error("--pruning_rate must be in the range (0.0, 1.0]")
    
    # Check input file
    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        return
    
    # Parse input
    logger.info("=" * 70)
    logger.info("REAP Expert Selection")
    logger.info("=" * 70)
    logger.info(f"Input file: {args.input}")
    
    expert_scores, inferred_num = parse_reap_json(args.input)
    
    if not expert_scores:
        logger.error("Failed to parse expert score data")
        return
    
    # Determine number of experts
    num_experts = args.num_experts if args.num_experts else inferred_num
    logger.info(f"Number of experts: {num_experts}")
    
    # Fill missing experts
    expert_scores = _complete_layers_with_all_experts(expert_scores, num_experts)
    
    # Execute selection
    if args.target_number is not None:
        logger.info(f"Selection strategy: by target number ({args.target_number}/layer)")
        selected = select_by_target_number(expert_scores, args.target_number)
        metadata = {
            "method": "reap",
            "selection_type": "target_number",
            "target_number": args.target_number,
            "num_experts": num_experts
        }
    else:
        logger.info(f"Selection strategy: {args.strategy} (retention rate {args.pruning_rate:.2%})")
        if args.strategy == "per_layer":
            selected = select_by_pruning_rate_per_layer(
                expert_scores, args.pruning_rate, num_experts
            )
        else:
            selected = select_by_pruning_rate_global(
                expert_scores, args.pruning_rate, num_experts
            )
        metadata = {
            "method": "reap",
            "selection_type": "pruning_rate",
            "pruning_rate": args.pruning_rate,
            "strategy": args.strategy,
            "num_experts": num_experts
        }
    
    # Print statistics
    total_selected = sum(len(v) for v in selected.values())
    total_possible = len(selected) * num_experts
    actual_rate = total_selected / total_possible if total_possible > 0 else 0
    
    logger.info(f"\nSelection result statistics:")
    logger.info(f"  Total layers: {len(selected)}")
    logger.info(f"  Retained experts: {total_selected} / {total_possible}")
    logger.info(f"  Actual retention rate: {actual_rate:.2%}")
    
    # Display per-layer statistics
    logger.info(f"\nExperts retained per layer:")
    for layer_id in sorted(selected.keys())[:10]:  # Show only first 10 layers
        logger.info(f"  Layer {layer_id}: {len(selected[layer_id])} experts")
    if len(selected) > 10:
        logger.info(f"  ... ({len(selected)} layers total)")
    
    # Save results
    save_results(selected, args.output, metadata)
    
    logger.info("=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
