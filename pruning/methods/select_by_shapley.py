#!/usr/bin/env python3
"""
基于 Shapley 值的专家选择工具

支持四种剪枝策略：

1. topk_per_layer  - 每层选择 Shapley 值最高的 top-k 专家（推荐，简单直接）
2. topk_global     - 全局选择 Shapley 值最高的专家
3. alpha_per_layer - 使用 alpha 因子，每层累积 Shapley 值达到 alpha 比例
4. alpha_global    - 使用 alpha 因子，全局累积

策略对比：
- topk: 直接按 Shapley 值大小排序，选择前 k 个。简单、可解释性强。
- alpha: 选择累积 Shapley 值达到总量 alpha 比例的最少专家。考虑了贡献分布。

- per_layer: 每层独立选择，保证每层都有足够专家。
- global: 全局统一选择，某些层可能专家较少。
"""

import pandas as pd
import argparse
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def _infer_num_experts(df: pd.DataFrame) -> int:
    """从数据中推断专家总数（默认 max(Expert_ID)+1）。"""
    if "Expert_ID" not in df.columns or df["Expert_ID"].empty:
        raise ValueError("输入数据缺少 Expert_ID，无法推断 num_experts。")
    return int(df["Expert_ID"].max()) + 1


def _complete_layers_with_missing_experts(
    df: pd.DataFrame,
    num_experts: int,
    strict_missing: bool = False,
    missing_sample_limit: int = 20,
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    强制把每层补齐为 [0..num_experts-1] 的完整 Expert_ID 集合。
    缺失行补 Shapley_Value=0
    """
    if "Layer" not in df.columns or "Expert_ID" not in df.columns:
        raise ValueError("输入 CSV 必须包含 Layer 与 Expert_ID 两列。")

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError("num_experts 必须为正整数。")

    missing_by_layer: Dict[int, List[int]] = {}
    layers = sorted(df["Layer"].unique().tolist())

    df = df.copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Expert_ID"] = df["Expert_ID"].astype(int)

    has_total_acts = "Total_Activations" in df.columns
    if "Shapley_Value" not in df.columns:
        raise ValueError("输入 CSV 必须包含 Shapley_Value 列。")

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
                f"Layer {layer_id} 缺失 {len(missing)}/{num_experts} 个专家（样例: {sample}）。"
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
# 策略 1: TopK Per Layer - 每层选择 Shapley 值最高的 top-k 专家
# =============================================================================

def select_topk_per_layer(
    df: pd.DataFrame,
    pruning_rate: float,
    num_experts: int
) -> Dict[int, List[int]]:
    """
    每层选择 Shapley 值最高的 top-k 专家
    
    Args:
        df: Shapley 值数据
        pruning_rate: 保留率 (0.0-1.0)
        num_experts: 每层专家总数
        
    Returns:
        {layer_id: [selected_expert_ids]}
    """
    keep_count = max(1, int(num_experts * pruning_rate))
    
    print(f"TopK Per Layer 策略:")
    print(f"  每层保留 {keep_count}/{num_experts} 个专家 (保留率: {pruning_rate:.1%})")
    
    selection_results = {}
    
    for layer_id, group in df.groupby("Layer"):
        # 按 Shapley 值降序排序
        sorted_experts = group.sort_values("Shapley_Value", ascending=False)
        # 选择前 k 个
        selected = sorted_experts.head(keep_count)["Expert_ID"].astype(int).tolist()
        selection_results[int(layer_id)] = sorted(selected)
    
    total_selected = sum(len(v) for v in selection_results.values())
    print(f"  总保留: {total_selected} 个专家")
    
    return selection_results


# =============================================================================
# 策略 2: TopK Global - 全局选择 Shapley 值最高的专家
# =============================================================================

def select_topk_global(
    df: pd.DataFrame,
    pruning_rate: float
) -> Dict[int, List[int]]:
    """
    全局选择 Shapley 值最高的专家
    
    Args:
        df: Shapley 值数据
        pruning_rate: 保留率 (0.0-1.0)
        
    Returns:
        {layer_id: [selected_expert_ids]}
    """
    total_experts = len(df)
    keep_count = max(df["Layer"].nunique(), int(total_experts * pruning_rate))
    
    print(f"TopK Global 策略:")
    print(f"  全局保留 {keep_count}/{total_experts} 个专家 (保留率: {pruning_rate:.1%})")
    
    # 全局按 Shapley 值排序
    sorted_df = df.sort_values("Shapley_Value", ascending=False)
    
    # 选择 top-k
    selected_df = sorted_df.head(keep_count)
    
    # 按层分组
    selection_results = defaultdict(list)
    for _, row in selected_df.iterrows():
        selection_results[int(row["Layer"])].append(int(row["Expert_ID"]))
    
    # 确保每层至少有一个专家
    for layer_id in df["Layer"].unique():
        layer_id = int(layer_id)
        if layer_id not in selection_results or not selection_results[layer_id]:
            # 选择该层 Shapley 值最高的专家
            layer_df = df[df["Layer"] == layer_id]
            best_expert = layer_df.loc[layer_df["Shapley_Value"].idxmax(), "Expert_ID"]
            selection_results[layer_id].append(int(best_expert))
    
    # 排序
    result = {k: sorted(v) for k, v in selection_results.items()}
    
    total_selected = sum(len(v) for v in result.values())
    print(f"  实际保留: {total_selected} 个专家")
    
    return dict(result)


# =============================================================================
# 策略 3: Alpha Per Layer - 每层累积 Shapley 值达到 alpha 比例
# =============================================================================

def select_by_alpha(df: pd.DataFrame, alpha: float) -> Tuple[Dict[int, List[int]], int]:
    """
    根据 alpha 因子选择专家（每层独立）
    选择累积 Shapley 值达到该层总量 alpha 比例的最少专家
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
    使用二分查找找到 alpha，使得每层平均保留率接近目标
    """
    num_layers = df["Layer"].nunique()
    experts_per_layer = int(df.groupby("Layer").size().max())

    print(f"Alpha Per Layer 策略:")
    print(f"  目标每层保留: {int(experts_per_layer * pruning_rate)}/{experts_per_layer} (保留率: {pruning_rate:.1%})")
    print(f"  开始二分查找 alpha...")

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
            print(f"    迭代 {iteration+1}: alpha={mid_alpha:.4f}, 平均每层={avg_selected_per_layer:.1f}")

        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        actual_rate = avg_selected_per_layer / experts_per_layer
        if abs(actual_rate - pruning_rate) < tolerance:
            print(f"  ✓ 找到 alpha={mid_alpha:.4f}")
            break

        if avg_selected_per_layer < target_per_layer:
            left = mid_alpha
        else:
            right = mid_alpha

    total_selected = sum(len(v) for v in best_selection.values())
    print(f"  最佳 alpha = {best_alpha:.4f}")
    print(f"  总保留: {total_selected} 个专家")

    return best_selection, best_alpha


# =============================================================================
# 策略 4: Alpha Global - 全局累积 Shapley 值达到 alpha 比例
# =============================================================================

def select_alpha_global(
    df: pd.DataFrame,
    pruning_rate: float,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> Tuple[Dict[int, List[int]], float]:
    """
    使用二分查找找到 alpha，使得全局保留率接近目标
    """
    total_experts = len(df)
    target_count = int(total_experts * pruning_rate)

    print(f"Alpha Global 策略:")
    print(f"  目标保留: {target_count}/{total_experts} (保留率: {pruning_rate:.1%})")
    print(f"  开始二分查找 alpha...")

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
            print(f"    迭代 {iteration+1}: alpha={mid_alpha:.4f}, 选中={selected_count}")

        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        actual_rate = selected_count / total_experts
        if abs(actual_rate - pruning_rate) < tolerance:
            print(f"  ✓ 找到 alpha={mid_alpha:.4f}")
            break

        if selected_count < target_count:
            left = mid_alpha
        else:
            right = mid_alpha

    total_selected = sum(len(v) for v in best_selection.values())
    print(f"  最佳 alpha = {best_alpha:.4f}")
    print(f"  总保留: {total_selected} 个专家")

    return best_selection, best_alpha


# =============================================================================
# 保存结果
# =============================================================================

def save_results(
    selection_results: Dict[int, List[int]],
    strategy: str,
    pruning_rate: float,
    df: pd.DataFrame,
    output_file: str,
    alpha: Optional[float] = None,
):
    """保存选择结果"""
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # 计算统计信息
    total_experts = len(df)
    total_selected = sum(len(v) for v in selection_results.values())
    actual_rate = total_selected / total_experts

    # 保存结果（转换为字符串键）
    output_data = {str(k): sorted(v) for k, v in selection_results.items()}
    
    # 添加元数据
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
    print(f"选择完成！")
    print(f"{'='*70}")
    print(f"策略: {strategy}")
    print(f"目标保留率: {pruning_rate:.1%}")
    print(f"实际保留率: {actual_rate:.1%}")
    print(f"保留专家: {total_selected}/{total_experts}")
    if alpha is not None:
        print(f"Alpha: {alpha:.4f}")
    print(f"\n结果已保存: {output_file}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="基于 Shapley 值的专家选择工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
四种剪枝策略：

  topk_per_layer  - 每层选择 Shapley 值最高的 top-k 专家（推荐）
  topk_global     - 全局选择 Shapley 值最高的专家
  alpha_per_layer - 每层累积 Shapley 值达到 alpha 比例
  alpha_global    - 全局累积 Shapley 值达到 alpha 比例

示例用法：

  # 每层 TopK（推荐，简单直接）
  python select_by_shapley.py \\
      --input gsm8k_25_shapley.csv \\
      --output selected_experts.json \\
      --pruning_rate 0.5 \\
      --strategy topk_per_layer

  # 全局 TopK
  python select_by_shapley.py \\
      --input gsm8k_25_shapley.csv \\
      --output selected_experts.json \\
      --pruning_rate 0.5 \\
      --strategy topk_global

  # Alpha 每层（考虑贡献分布）
  python select_by_shapley.py \\
      --input gsm8k_25_shapley.csv \\
      --output selected_experts.json \\
      --pruning_rate 0.5 \\
      --strategy alpha_per_layer
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Shapley 值 CSV 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument(
        "--pruning_rate",
        type=float,
        required=True,
        help="保留率 (0.0-1.0)，例如 0.5 表示保留 50%%",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["alpha_per_layer", "alpha_global", "topk_per_layer", "topk_global",
                 "per_layer", "global"],  # 兼容旧参数
        default="alpha_per_layer",
        help="剪枝策略（默认: alpha_per_layer）",
    )
    parser.add_argument("--num_experts", type=int, default=None, help="每层专家总数")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Alpha 策略的容差")
    parser.add_argument("--max_iterations", type=int, default=50, help="Alpha 策略的最大迭代次数")

    args = parser.parse_args()

    # 兼容旧参数名
    if args.strategy == "per_layer":
        args.strategy = "alpha_per_layer"
        print("注意: 'per_layer' 已重命名为 'alpha_per_layer'")
    elif args.strategy == "global":
        args.strategy = "alpha_global"
        print("注意: 'global' 已重命名为 'alpha_global'")

    if not 0 < args.pruning_rate <= 1:
        parser.error("pruning_rate 必须在 (0, 1] 范围内")

    if not os.path.exists(args.input):
        parser.error(f"输入文件不存在: {args.input}")

    print("=" * 70)
    print("Shapley 值专家选择")
    print("=" * 70)
    print(f"输入文件: {args.input}")
    print(f"目标保留率: {args.pruning_rate:.1%}")
    print(f"策略: {args.strategy}")
    print("=" * 70)

    # 读取数据
    print(f"\n读取数据...")
    df = pd.read_csv(args.input)
    print(f"✓ 读取完成: {len(df)} 条记录")

    # 补齐缺失专家
    inferred = _infer_num_experts(df) if args.num_experts is None else int(args.num_experts)
    df_completed, missing_by_layer = _complete_layers_with_missing_experts(df, num_experts=inferred)
    
    if missing_by_layer:
        affected = len(missing_by_layer)
        print(f"⚠️ {affected} 层存在缺失专家，已补齐为 Shapley=0")
    else:
        print("✓ 每层专家完整")

    df = df_completed
    print(f"专家总数: {len(df)} ({df['Layer'].nunique()} 层 × {inferred} 专家/层)")
    print()

    # 执行选择
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

    # 保存结果
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
