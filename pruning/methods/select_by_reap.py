#!/usr/bin/env python3
"""
REAP (Router-weighted Expert Activation Pruning) 专家选择脚本

基于 REAP 论文的思想，使用 router_weight × expert_activation_norm 作为专家重要性度量。
REAP score = mean(router_weight × expert_activation_norm)

输入格式：
    来自 analyze_all_in_one.py 的 *_reap.json 文件
    JSON 结构:
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

输出格式：
    {
        "0": [0, 1, 3, 5, ...],  // Layer 0 保留的专家 ID
        "1": [2, 4, 6, 8, ...],  // Layer 1 保留的专家 ID
        ...
    }

使用示例：
    # 按剪枝率选择
    python select_experts_by_reap.py \\
        --input reap_scores.json \\
        --output selected_experts.json \\
        --pruning_rate 0.5 \\
        --strategy per_layer
        
    # 按目标数量选择
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
    解析 REAP JSON 文件
    
    Args:
        input_file: REAP JSON 文件路径
        
    Returns:
        expert_scores: {layer_id: {expert_id: reap_mean_score}}
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
            
            # 使用 reap_mean 作为主要得分
            if isinstance(expert_info, dict):
                score = expert_info.get("reap_mean", 0.0)
            else:
                score = float(expert_info)
            
            expert_scores[layer_idx][expert_id] = score
            max_expert_id = max(max_expert_id, expert_id)
    
    num_experts = max_expert_id + 1
    
    logger.info(f"解析 REAP 数据: {len(expert_scores)} 层, {num_experts} 专家/层")
    
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
    按目标数量选择每层 REAP 得分最高的专家
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        target_number: 每层保留的专家数量
        
    Returns:
        selected: {layer_id: [expert_ids]}
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
    每层剪枝：根据每层 REAP 得分选择专家
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        pruning_rate: 剪枝比例 (0.0-1.0)，表示保留的比例
        num_experts: 每层专家数量
        
    Returns:
        selected: {layer_id: [expert_ids]}
    """
    # 计算每层保留的专家数量
    keep_count = int(num_experts * pruning_rate)
    keep_count = max(1, keep_count)  # 至少保留1个专家
    
    logger.info(f"每层剪枝: 保留率={pruning_rate:.2%}, 每层保留 {keep_count}/{num_experts} 个专家")
    
    return select_by_target_number(expert_scores, keep_count)


def select_by_pruning_rate_global(
    expert_scores: Dict[int, Dict[int, float]],
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    全局剪枝：根据全局 REAP 得分分布选择专家
    
    Args:
        expert_scores: {layer_id: {expert_id: score}}
        pruning_rate: 剪枝比例 (0.0-1.0)，表示保留的比例
        num_experts: 每层专家数量
        
    Returns:
        selected: {layer_id: [expert_ids]}
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
    keep_count = max(len(expert_scores), keep_count)  # 每层至少保留1个
    
    # 选择得分最高的专家
    selected_set = set()
    for layer_id, exp_id, _ in all_scores[:keep_count]:
        selected_set.add((layer_id, exp_id))
    
    # 确保每层至少有一个专家
    for layer_id in expert_scores.keys():
        layer_experts = [(l, e) for l, e in selected_set if l == layer_id]
        if not layer_experts:
            # 选择该层得分最高的专家
            best_expert = max(expert_scores[layer_id].items(), key=lambda x: x[1])
            selected_set.add((layer_id, best_expert[0]))
    
    # 整理结果
    selected = defaultdict(list)
    for layer_id, exp_id in selected_set:
        selected[layer_id].append(exp_id)
    
    # 排序
    result = {layer_id: sorted(experts) for layer_id, experts in selected.items()}
    
    # 计算实际保留率
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
    # 将 key 转换为字符串（JSON 要求）
    output_data = {str(k): v for k, v in selected.items()}
    
    if metadata:
        output_data["_metadata"] = metadata
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 结果已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="REAP 专家选择 - 基于 router_weight × expert_activation_norm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 按剪枝率选择（每层策略，推荐）
  python select_experts_by_reap.py \\
      --input results/model_dataset_reap.json \\
      --output selected_50pct.json \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # 按剪枝率选择（全局策略）
  python select_experts_by_reap.py \\
      --input results/model_dataset_reap.json \\
      --output selected_global.json \\
      --pruning_rate 0.5 \\
      --strategy global

  # 按目标数量选择
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
        help="REAP JSON 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--pruning_rate",
        type=float,
        default=None,
        help="保留率 (0.0-1.0)，与 --target_number 二选一"
    )
    parser.add_argument(
        "--target_number",
        type=int,
        default=None,
        help="每层保留的专家数量，与 --pruning_rate 二选一"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["global", "per_layer"],
        default="per_layer",
        help="剪枝策略: global (全局) 或 per_layer (每层，推荐)"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="每层专家总数（可选，自动推断）"
    )
    
    args = parser.parse_args()
    
    # 参数验证
    if args.pruning_rate is None and args.target_number is None:
        parser.error("必须指定 --pruning_rate 或 --target_number 之一")
    
    if args.pruning_rate is not None and args.target_number is not None:
        parser.error("--pruning_rate 和 --target_number 只能指定其一")
    
    if args.pruning_rate is not None and not (0.0 < args.pruning_rate <= 1.0):
        parser.error("--pruning_rate 必须在 (0.0, 1.0] 范围内")
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 解析输入
    logger.info("=" * 70)
    logger.info("REAP 专家选择")
    logger.info("=" * 70)
    logger.info(f"输入文件: {args.input}")
    
    expert_scores, inferred_num = parse_reap_json(args.input)
    
    if not expert_scores:
        logger.error("未能解析专家得分数据")
        return
    
    # 确定专家数量
    num_experts = args.num_experts if args.num_experts else inferred_num
    logger.info(f"专家数量: {num_experts}")
    
    # 补全缺失的专家
    expert_scores = _complete_layers_with_all_experts(expert_scores, num_experts)
    
    # 执行选择
    if args.target_number is not None:
        logger.info(f"选择策略: 按目标数量 ({args.target_number}/层)")
        selected = select_by_target_number(expert_scores, args.target_number)
        metadata = {
            "method": "reap",
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
            "method": "reap",
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
    
    # 显示每层统计
    logger.info(f"\n每层保留专家数:")
    for layer_id in sorted(selected.keys())[:10]:  # 只显示前10层
        logger.info(f"  Layer {layer_id}: {len(selected[layer_id])} 个专家")
    if len(selected) > 10:
        logger.info(f"  ... (共 {len(selected)} 层)")
    
    # 保存结果
    save_results(selected, args.output, metadata)
    
    logger.info("=" * 70)
    logger.info("完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

