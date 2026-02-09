#!/usr/bin/env python3
"""
EASYEP Expert Selection Script

Expert selection based on the original formula from the EASYEP paper:
    score = Σ(weight × (1 - simibr) × norm)

Where:
- weight: router softmax weight
- simibr: cos_sim(x_before_moe, x_after_rmoe) - cosine similarity between MoE input and routed output
- norm: L2 norm of expert output

(1 - simibr) being larger means MoE has greater impact on this token

Input format:
    *_easyep.json files from analyze_all_in_one.py
    JSON structure:
    {
        "model": "...",
        "dataset": "...",
        "description": "EASYEP score: weight × (1 - simibr) × norm",
        "layers": {
            "0": {
                "0": {"easyep_sum": ..., "easyep_mean": ..., "activation_count": ...},
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
    # Select by pruning rate (per layer)
    python select_experts_by_easyep.py \\
        --input easyep_scores.json \\
        --output selected_experts.json \\
        --pruning_rate 0.5 \\
        --strategy per_layer
        
    # Select by target number
    python select_experts_by_easyep.py \\
        --input easyep_scores.json \\
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


def parse_easyep_json(input_file: str) -> Tuple[Dict[int, Dict[int, float]], int]:
    """
    Parse EASYEP JSON file
    
    Args:
        input_file: EASYEP JSON file path
        
    Returns:
        expert_scores: {layer_id: {expert_id: score}}
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
            
            # Prefer easyep_sum, fall back to total_weight (backward compatible)
            if isinstance(expert_info, dict):
                score = expert_info.get("easyep_sum", 0.0)
                if score == 0.0:
                    score = expert_info.get("total_weight", 0.0)
            else:
                score = float(expert_info)
            
            expert_scores[layer_idx][expert_id] = score
            max_expert_id = max(max_expert_id, expert_id)
    
    num_experts = max_expert_id + 1
    
    logger.info(f"Parsed EASYEP data: {len(expert_scores)} layers, {num_experts} experts/layer")
    
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
    Select experts with highest scores per layer by target number
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
    Per-layer pruning: select experts based on per-layer scores
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        pruning_rate: Retention rate (0.0-1.0)
        num_experts: Number of experts per layer
        
    Returns:
        selected: {layer_id: [expert_ids]}
    """
    keep_count = int(num_experts * pruning_rate)
    keep_count = max(1, keep_count)
    
    logger.info(f"Per-layer pruning: retention rate={pruning_rate:.2%}, keeping {keep_count}/{num_experts} experts per layer")
    
    return select_by_target_number(expert_scores, keep_count)


def select_by_pruning_rate_global(
    expert_scores: Dict[int, Dict[int, float]],
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    Global pruning: select experts based on global score distribution
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
    keep_count = max(len(expert_scores), keep_count)
    
    # Select highest scoring experts
    selected_set = set()
    for layer_id, exp_id, _ in all_scores[:keep_count]:
        selected_set.add((layer_id, exp_id))
    
    # Ensure at least one expert per layer
    for layer_id in expert_scores.keys():
        layer_experts = [(l, e) for l, e in selected_set if l == layer_id]
        if not layer_experts:
            best_expert = max(expert_scores[layer_id].items(), key=lambda x: x[1])
            selected_set.add((layer_id, best_expert[0]))
    
    # Organize results
    selected = defaultdict(list)
    for layer_id, exp_id in selected_set:
        selected[layer_id].append(exp_id)
    
    result = {layer_id: sorted(experts) for layer_id, experts in selected.items()}
    
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
    output_data = {str(k): v for k, v in selected.items()}
    
    if metadata:
        output_data["_metadata"] = metadata
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Results saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="EASYEP Expert Selection - Based on weight × (1 - simibr) × norm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EASYEP formula:
    score = Σ(weight × max(1 - cos_sim(x_before, x_after_rmoe), 0) × expert_norm)

Example usage:

  # Select by retention rate (per-layer strategy, recommended)
  python select_experts_by_easyep.py \\
      --input results/model_dataset_easyep.json \\
      --output selected_50pct.json \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # Select by target number
  python select_experts_by_easyep.py \\
      --input results/model_dataset_easyep.json \\
      --output selected_128experts.json \\
      --target_number 128
        """
    )
    
    parser.add_argument("--input", type=str, required=True, help="EASYEP JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--pruning_rate", type=float, default=None, help="Retention rate (0.0-1.0)")
    parser.add_argument("--target_number", type=int, default=None, help="Number of experts to retain per layer")
    parser.add_argument("--strategy", type=str, choices=["global", "per_layer"], default="per_layer",
                       help="Pruning strategy: global or per_layer (recommended)")
    parser.add_argument("--num_experts", type=int, default=None, help="Total experts per layer")
    
    args = parser.parse_args()
    
    # Parameter validation
    if args.pruning_rate is None and args.target_number is None:
        parser.error("Must specify either --pruning_rate or --target_number")
    
    if args.pruning_rate is not None and args.target_number is not None:
        parser.error("--pruning_rate and --target_number are mutually exclusive")
    
    if args.pruning_rate is not None and not (0.0 < args.pruning_rate <= 1.0):
        parser.error("--pruning_rate must be in the range (0.0, 1.0]")
    
    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        return
    
    logger.info("=" * 70)
    logger.info("EASYEP Expert Selection")
    logger.info("=" * 70)
    logger.info(f"Input file: {args.input}")
    logger.info("Formula: score = weight × (1 - simibr) × norm")
    
    expert_scores, inferred_num = parse_easyep_json(args.input)
    
    if not expert_scores:
        logger.error("Failed to parse expert score data")
        return
    
    num_experts = args.num_experts if args.num_experts else inferred_num
    logger.info(f"Number of experts: {num_experts}")
    
    expert_scores = _complete_layers_with_all_experts(expert_scores, num_experts)
    
    # Execute selection
    if args.target_number is not None:
        logger.info(f"Selection strategy: by target number ({args.target_number}/layer)")
        selected = select_by_target_number(expert_scores, args.target_number)
        metadata = {
            "method": "easyep",
            "formula": "weight × (1 - simibr) × norm",
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
            "method": "easyep",
            "formula": "weight × (1 - simibr) × norm",
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
    
    logger.info(f"\nExperts retained per layer:")
    for layer_id in sorted(selected.keys())[:10]:
        logger.info(f"  Layer {layer_id}: {len(selected[layer_id])} experts")
    if len(selected) > 10:
        logger.info(f"  ... ({len(selected)} layers total)")
    
    save_results(selected, args.output, metadata)
    
    logger.info("=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
