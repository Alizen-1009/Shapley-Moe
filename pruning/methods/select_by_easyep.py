#!/usr/bin/env python3
"""
EASYEP 专家选择脚本

基于 EASYEP 论文的原始公式进行专家选择：
    score = Σ(weight × (1 - simibr) × norm)

其中：
- weight: router softmax 权重
- simibr: cos_sim(x_before_moe, x_after_rmoe) - MoE输入与routed输出的余弦相似度
- norm: 专家输出的 L2 范数

(1 - simibr) 越大，说明 MoE 对这个 token 的影响越大

输入格式：
    来自 analyze_all_in_one.py 的 *_easyep.json 文件
    JSON 结构:
    {
        "model": "...",
        "dataset": "...",
        "description": "EASYEP 得分：weight × (1 - simibr) × norm",
        "layers": {
            "0": {
                "0": {"easyep_sum": ..., "easyep_mean": ..., "activation_count": ...},
                ...
            },
            ...
        }
    }

输出格式：
    {
        "0": [0, 1, 3, 5, ...],  // Layer 0 保留的专家 ID
        "1": [2, 4, 6, 8, ...],  // Layer 1 保留的专家 ID
        ...
    }

使用示例：
    # 按剪枝率选择（每层）
    python select_experts_by_easyep.py \\
        --input easyep_scores.json \\
        --output selected_experts.json \\
        --pruning_rate 0.5 \\
        --strategy per_layer
        
    # 按目标数量选择
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
    解析 EASYEP JSON 文件
    
    Args:
        input_file: EASYEP JSON 文件路径
        
    Returns:
        expert_scores: {layer_id: {expert_id: score}}
        num_experts: 每层专家数量（推断）
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
            
            # 优先使用 easyep_sum，其次是 total_weight（兼容旧格式）
            if isinstance(expert_info, dict):
                score = expert_info.get("easyep_sum", 0.0)
                if score == 0.0:
                    score = expert_info.get("total_weight", 0.0)
            else:
                score = float(expert_info)
            
            expert_scores[layer_idx][expert_id] = score
            max_expert_id = max(max_expert_id, expert_id)
    
    num_experts = max_expert_id + 1
    
    logger.info(f"解析 EASYEP 数据: {len(expert_scores)} 层, {num_experts} 专家/层")
    
    return dict(expert_scores), num_experts


def _complete_layers_with_all_experts(
    expert_scores: Dict[int, Dict[int, float]],
    num_experts: int
) -> Dict[int, Dict[int, float]]:
    """
    确保每层都有所有专家的得分（缺失的设为0）
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
    按目标数量选择每层得分最高的专家
    """
    selected = {}
    
    for layer_id in sorted(expert_scores.keys()):
        layer_scores = expert_scores[layer_id]
        
        # 按得分降序排序
        sorted_experts = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前 target_number 个
        selected_experts = [exp_id for exp_id, _ in sorted_experts[:target_number]]
        selected[layer_id] = sorted(selected_experts)
    
    return selected


def select_by_pruning_rate_per_layer(
    expert_scores: Dict[int, Dict[int, float]],
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    每层剪枝：根据每层得分选择专家
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        pruning_rate: 保留比例 (0.0-1.0)
        num_experts: 每层专家数量
        
    Returns:
        selected: {layer_id: [expert_ids]}
    """
    keep_count = int(num_experts * pruning_rate)
    keep_count = max(1, keep_count)
    
    logger.info(f"每层剪枝: 保留率={pruning_rate:.2%}, 每层保留 {keep_count}/{num_experts} 个专家")
    
    return select_by_target_number(expert_scores, keep_count)


def select_by_pruning_rate_global(
    expert_scores: Dict[int, Dict[int, float]],
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    全局剪枝：根据全局得分分布选择专家
    """
    # 收集所有专家的得分
    all_scores = []
    for layer_id, layer_scores in expert_scores.items():
        for exp_id, score in layer_scores.items():
            all_scores.append((layer_id, exp_id, score))
    
    # 按得分排序
    all_scores.sort(key=lambda x: x[2], reverse=True)
    
    # 计算保留数量
    total_experts = len(all_scores)
    keep_count = int(total_experts * pruning_rate)
    keep_count = max(len(expert_scores), keep_count)
    
    # 选择得分最高的专家
    selected_set = set()
    for layer_id, exp_id, _ in all_scores[:keep_count]:
        selected_set.add((layer_id, exp_id))
    
    # 确保每层至少有一个专家
    for layer_id in expert_scores.keys():
        layer_experts = [(l, e) for l, e in selected_set if l == layer_id]
        if not layer_experts:
            best_expert = max(expert_scores[layer_id].items(), key=lambda x: x[1])
            selected_set.add((layer_id, best_expert[0]))
    
    # 整理结果
    selected = defaultdict(list)
    for layer_id, exp_id in selected_set:
        selected[layer_id].append(exp_id)
    
    result = {layer_id: sorted(experts) for layer_id, experts in selected.items()}
    
    total_kept = sum(len(v) for v in result.values())
    actual_rate = total_kept / total_experts
    
    logger.info(f"全局剪枝: 目标保留率={pruning_rate:.2%}, 实际保留率={actual_rate:.2%}")
    logger.info(f"保留 {total_kept}/{total_experts} 个专家")
    
    return dict(result)


def save_results(
    selected: Dict[int, List[int]],
    output_file: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    保存选择结果
    """
    output_data = {str(k): v for k, v in selected.items()}
    
    if metadata:
        output_data["_metadata"] = metadata
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 结果已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="EASYEP 专家选择 - 基于 weight × (1 - simibr) × norm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EASYEP 公式:
    score = Σ(weight × max(1 - cos_sim(x_before, x_after_rmoe), 0) × expert_norm)

示例用法:

  # 按保留率选择（每层策略，推荐）
  python select_experts_by_easyep.py \\
      --input results/model_dataset_easyep.json \\
      --output selected_50pct.json \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # 按目标数量选择
  python select_experts_by_easyep.py \\
      --input results/model_dataset_easyep.json \\
      --output selected_128experts.json \\
      --target_number 128
        """
    )
    
    parser.add_argument("--input", type=str, required=True, help="EASYEP JSON 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--pruning_rate", type=float, default=None, help="保留率 (0.0-1.0)")
    parser.add_argument("--target_number", type=int, default=None, help="每层保留的专家数量")
    parser.add_argument("--strategy", type=str, choices=["global", "per_layer"], default="per_layer",
                       help="剪枝策略: global 或 per_layer（推荐）")
    parser.add_argument("--num_experts", type=int, default=None, help="每层专家总数")
    
    args = parser.parse_args()
    
    # 参数验证
    if args.pruning_rate is None and args.target_number is None:
        parser.error("必须指定 --pruning_rate 或 --target_number 之一")
    
    if args.pruning_rate is not None and args.target_number is not None:
        parser.error("--pruning_rate 和 --target_number 只能指定其一")
    
    if args.pruning_rate is not None and not (0.0 < args.pruning_rate <= 1.0):
        parser.error("--pruning_rate 必须在 (0.0, 1.0] 范围内")
    
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    logger.info("=" * 70)
    logger.info("EASYEP 专家选择")
    logger.info("=" * 70)
    logger.info(f"输入文件: {args.input}")
    logger.info("公式: score = weight × (1 - simibr) × norm")
    
    expert_scores, inferred_num = parse_easyep_json(args.input)
    
    if not expert_scores:
        logger.error("未能解析专家得分数据")
        return
    
    num_experts = args.num_experts if args.num_experts else inferred_num
    logger.info(f"专家数量: {num_experts}")
    
    expert_scores = _complete_layers_with_all_experts(expert_scores, num_experts)
    
    # 执行选择
    if args.target_number is not None:
        logger.info(f"选择策略: 按目标数量 ({args.target_number}/层)")
        selected = select_by_target_number(expert_scores, args.target_number)
        metadata = {
            "method": "easyep",
            "formula": "weight × (1 - simibr) × norm",
            "selection_type": "target_number",
            "target_number": args.target_number,
            "num_experts": num_experts
        }
    else:
        logger.info(f"选择策略: {args.strategy} (保留率 {args.pruning_rate:.2%})")
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
    
    # 打印统计
    total_selected = sum(len(v) for v in selected.values())
    total_possible = len(selected) * num_experts
    actual_rate = total_selected / total_possible if total_possible > 0 else 0
    
    logger.info(f"\n选择结果统计:")
    logger.info(f"  总层数: {len(selected)}")
    logger.info(f"  保留专家: {total_selected} / {total_possible}")
    logger.info(f"  实际保留率: {actual_rate:.2%}")
    
    logger.info(f"\n每层保留专家数:")
    for layer_id in sorted(selected.keys())[:10]:
        logger.info(f"  Layer {layer_id}: {len(selected[layer_id])} 个专家")
    if len(selected) > 10:
        logger.info(f"  ... (共 {len(selected)} 层)")
    
    save_results(selected, args.output, metadata)
    
    logger.info("=" * 70)
    logger.info("完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
