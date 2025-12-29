#!/usr/bin/env python3
"""
基于激活频率的专家选择工具（对照实验）

从 few-shot/results 目录中的 aggregated JSON 文件读取专家激活次数，然后基于激活频率进行剪枝。

支持两种策略：
1. 全局剪枝率：在所有层中总共保留 X% 的专家（按激活频率排序）
2. 每层剪枝率：每层都保留 X% 的专家（按激活频率排序，推荐）
"""

import pandas as pd
import argparse
import os
import json
import numpy as np
import ast
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_aggregated_json(json_file: str) -> pd.DataFrame:
    """
    从 aggregated JSON 文件中解析专家激活次数
    
    Args:
        json_file: aggregated JSON 文件路径
        
    Returns:
        DataFrame 包含 Layer, Expert_ID, Total_Activations 列
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 统计每层每个专家的激活次数
    layer_expert_counts = defaultdict(lambda: defaultdict(int))
    
    for layer_id, combinations in data['layers'].items():
        for combo_str, count in combinations.items():
            # 解析元组字符串，例如 '(5, 13, 16, 22)'
            try:
                combo = ast.literal_eval(combo_str)
                for expert_id in combo:
                    layer_expert_counts[int(layer_id)][int(expert_id)] += count
            except (ValueError, SyntaxError) as e:
                # 如果解析失败，跳过这个组合
                continue
    
    # 转换为 DataFrame
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
        raise ValueError(f"无法从 {json_file} 中解析出任何专家激活数据")
    
    return df


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
    - 缺失行补 Total_Activations=0
    - 返回补齐后的 df，以及每层缺失的 expert_id 列表
    """
    if "Layer" not in df.columns or "Expert_ID" not in df.columns:
        raise ValueError("输入数据必须包含 Layer 与 Expert_ID 两列。")

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError("num_experts 必须为正整数。")

    missing_by_layer: Dict[int, List[int]] = {}
    layers = sorted(df["Layer"].unique().tolist())

    # 标准化列
    df = df.copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Expert_ID"] = df["Expert_ID"].astype(int)

    if "Total_Activations" not in df.columns:
        raise ValueError("输入数据必须包含 Total_Activations 列（激活频率）。")
    
    # 添加 Shapley_Value 列（用于统计，但这里不实际使用）
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
                f"Layer {layer_id} 缺失 {len(missing)}/{num_experts} 个专家（样例: {sample}）。"
            )

        # 用 reindex 补齐
        layer_df = layer_df.set_index("Expert_ID").reindex(full_index)
        layer_df.index.name = "Expert_ID"
        layer_df = layer_df.reset_index()
        layer_df["Layer"] = int(layer_id)

        # 缺失值补 0
        layer_df["Total_Activations"] = layer_df["Total_Activations"].fillna(0).astype(int)
        layer_df["Shapley_Value"] = layer_df["Shapley_Value"].fillna(0.0)

        completed_parts.append(layer_df)

    completed_df = pd.concat(completed_parts, ignore_index=True)
    return completed_df, missing_by_layer


def select_by_activation_alpha(df: pd.DataFrame, alpha: float) -> Tuple[Dict[int, List[int]], int]:
    """
    根据 alpha 因子和激活频率选择专家

    Args:
        df: 包含 Total_Activations 的数据
        alpha: 累积激活频率比例阈值

    Returns:
        selection_results: {layer_id: [expert_ids]}
        total_selected: 总共选中的专家数
    """
    selection_results = {}
    total_selected = 0

    for layer_id, group in df.groupby("Layer"):
        # 按激活频率降序排序
        group_sorted = group.sort_values("Total_Activations", ascending=False)
        total_activations = group_sorted["Total_Activations"].sum()
        target_activations = total_activations * alpha

        cumulative_activations = 0
        selected_experts = []

        for _, row in group_sorted.iterrows():
            cumulative_activations += row["Total_Activations"]
            selected_experts.append(int(row["Expert_ID"]))

            if cumulative_activations >= target_activations:
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
    使用二分查找找到合适的 alpha，使得总剪枝率接近目标值（基于激活频率）

    Args:
        df: 激活频率数据
        target_pruning_rate: 目标剪枝率（保留的专家比例，例如 0.4 表示保留40%）
        tolerance: 容差
        max_iterations: 最大迭代次数

    Returns:
        best_alpha: 找到的最佳 alpha
        selection_results: 专家选择结果
        actual_rate: 实际剪枝率
    """
    # 计算总专家数（补齐后应为 num_layers * num_experts）
    total_experts = len(df)
    target_count = int(total_experts * target_pruning_rate)

    print(f"总专家数: {total_experts}")
    print(f"目标保留: {target_count} 个专家 ({target_pruning_rate:.1%})")
    print(f"开始二分查找 alpha（基于激活频率）...")

    # 二分查找
    left, right = 0.0, 1.0
    best_alpha = 0.5
    best_selection = None
    best_diff = float("inf")

    for iteration in range(max_iterations):
        mid_alpha = (left + right) / 2
        selection, selected_count = select_by_activation_alpha(df, mid_alpha)

        diff = selected_count - target_count
        abs_diff = abs(diff)

        print(
            f"  迭代 {iteration+1}: alpha={mid_alpha:.4f}, 选中={selected_count}, 目标={target_count}, 差={diff}"
        )

        # 更新最佳结果
        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        # 检查是否达到容差
        actual_rate = selected_count / total_experts
        if abs(actual_rate - target_pruning_rate) < tolerance:
            print(f"✓ 找到合适的 alpha={mid_alpha:.4f}")
            break

        # 调整搜索区间
        if selected_count < target_count:
            # 选中的太少，需要增加 alpha（降低阈值）
            left = mid_alpha
        else:
            # 选中的太多，需要减少 alpha（提高阈值）
            right = mid_alpha

    actual_rate = (
        sum(len(experts) for experts in best_selection.values()) / total_experts
    )

    print(f"\n✓ 最佳 alpha = {best_alpha:.4f}")
    print(
        f"  实际保留: {sum(len(experts) for experts in best_selection.values())}/{total_experts} ({actual_rate:.1%})"
    )

    return best_alpha, best_selection, actual_rate


def find_alpha_for_per_layer_pruning_rate(
    df: pd.DataFrame,
    target_pruning_rate: float,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> Tuple[float, Dict[int, List[int]], float]:
    """
    使用二分查找找到合适的 alpha，使得每层的平均剪枝率接近目标值（基于激活频率）

    Args:
        df: 激活频率数据
        target_pruning_rate: 目标剪枝率（每层保留的专家比例）
        tolerance: 容差
        max_iterations: 最大迭代次数

    Returns:
        best_alpha: 找到的最佳 alpha
        selection_results: 专家选择结果
        avg_rate: 平均每层剪枝率
    """
    num_layers = df["Layer"].nunique()
    experts_per_layer = int(df.groupby("Layer").size().max())

    print(f"层数: {num_layers}")
    print(f"每层专家数: {experts_per_layer}")
    print(
        f"目标每层保留: {int(experts_per_layer * target_pruning_rate)} 个专家 ({target_pruning_rate:.1%})"
    )
    print(f"开始二分查找 alpha（基于激活频率）...")

    # 二分查找
    left, right = 0.0, 1.0
    best_alpha = 0.5
    best_selection = None
    best_diff = float("inf")

    for iteration in range(max_iterations):
        mid_alpha = (left + right) / 2
        selection, total_selected = select_by_activation_alpha(df, mid_alpha)

        # 计算平均每层保留的专家数
        avg_selected_per_layer = total_selected / num_layers
        target_per_layer = experts_per_layer * target_pruning_rate

        diff = avg_selected_per_layer - target_per_layer
        abs_diff = abs(diff)

        print(
            f"  迭代 {iteration+1}: alpha={mid_alpha:.4f}, 平均每层={avg_selected_per_layer:.1f}, 目标={target_per_layer:.1f}, 差={diff:.1f}"
        )

        # 更新最佳结果
        if abs_diff < best_diff:
            best_diff = abs_diff
            best_alpha = mid_alpha
            best_selection = selection

        # 检查是否达到容差
        actual_rate = avg_selected_per_layer / experts_per_layer
        if abs(actual_rate - target_pruning_rate) < tolerance:
            print(f"✓ 找到合适的 alpha={mid_alpha:.4f}")
            break

        # 调整搜索区间
        if avg_selected_per_layer < target_per_layer:
            left = mid_alpha
        else:
            right = mid_alpha

    avg_rate = (
        sum(len(experts) for experts in best_selection.values())
        / num_layers
        / experts_per_layer
    )

    print(f"\n✓ 最佳 alpha = {best_alpha:.4f}")
    print(
        f"  平均每层保留: {sum(len(experts) for experts in best_selection.values()) / num_layers:.1f}/{experts_per_layer} ({avg_rate:.1%})"
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
    """保存选择结果和统计信息"""

    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    rate_str = f"{int(pruning_rate * 100)}"
    output_json = os.path.join(
        output_dir, f"selected_experts_activation_{strategy}_rate{rate_str}.json"
    )
    output_csv = os.path.join(
        output_dir, f"selection_stats_activation_{strategy}_rate{rate_str}.csv"
    )

    # 保存选择结果（转换为字符串键以匹配现有格式）
    selection_results_str_keys = {str(k): v for k, v in selection_results.items()}
    with open(output_json, "w") as f:
        json.dump(selection_results_str_keys, f, indent=4)

    # 生成统计信息
    stats_data = []
    for layer_id, selected_experts in selection_results.items():
        layer_df = df[df["Layer"] == layer_id]
        total_experts = len(layer_df)
        total_activations = layer_df["Total_Activations"].sum()
        selected_activations = layer_df[layer_df["Expert_ID"].isin(selected_experts)][
            "Total_Activations"
        ].sum()
        activation_ratio = (
            selected_activations / total_activations if total_activations > 0 else 0
        )

        # Shapley 值（如果存在，这里通常为0）
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
                "Total_Activations": total_activations,
                "Selected_Activations": selected_activations,
                "Activation_Ratio": activation_ratio,
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

    # 保存统计
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_csv, index=False)

    # 计算总体统计
    total_experts = len(df)
    total_selected = sum(len(experts) for experts in selection_results.values())
    actual_global_rate = total_selected / total_experts

    print(f"\n{'='*70}")
    print(f"基于激活频率的选择完成！")
    print(f"{'='*70}")
    print(f"策略: {strategy} (激活频率)")
    print(f"最佳 alpha: {alpha:.4f}")
    print(f"目标剪枝率: {pruning_rate:.1%}")
    print(f"实际剪枝率: {actual_global_rate:.1%}")
    print(f"总专家数: {total_experts}")
    print(f"保留专家: {total_selected}")
    print(f"剪除专家: {total_experts - total_selected}")
    print(f"\n结果已保存:")
    print(f"  - {output_json}")
    print(f"  - {output_csv}")
    print(f"{'='*70}")

    # 打印每层摘要
    print("\n每层摘要:")
    print(
        stats_df[
            ["Layer", "Total_Experts", "Selected_Experts", "Pruning_Rate", "Activation_Ratio"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="基于激活频率的专家选择工具（对照实验）- 从 aggregated JSON 文件读取激活次数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
策略说明:

  global    - 全局剪枝率：在所有层中总共保留 X% 的专家（按激活频率排序）
  per_layer - 每层剪枝率：每层都保留 X% 的专家（按激活频率排序，推荐）

示例用法:

  # 每层按激活频率保留 50% 专家（推荐）
  python select_by_frequency.py \\
      --input ../../results/gpt-oss-20b/activations/gsm8k_25_shapley.json \\
      --output ../../results/gpt-oss-20b/selected_experts/frequency_gsm8k_25_rate0_5.json \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # 全局按激活频率保留 40% 专家
  python select_by_frequency.py \\
      --input ../../results/gpt-oss-20b/activations/gsm8k_25_shapley.json \\
      --output ../../results/gpt-oss-20b/selected_experts/frequency_gsm8k_25_rate0_4.json \\
      --pruning_rate 0.4 \\
      --strategy global
        """,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="aggregated JSON 文件路径（few-shot/results 目录中的文件）"
    )
    parser.add_argument(
        "--pruning_rate",
        type=float,
        required=True,
        help="保留率（保留的专家比例，0-1之间，例如 0.4 表示保留40%%，剪掉60%%）",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["global", "per_layer"],
        default="per_layer",
        help="剪枝策略（global: 全局剪枝率, per_layer: 每层剪枝率，默认: per_layer）",
    )
    parser.add_argument(
        "--output", type=str, default="results", help="输出目录（默认: results）"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.01, help="二分查找容差（默认: 0.01）"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="二分查找最大迭代次数（默认: 50）",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="（可选）每层专家总数。若提供或可推断，将对每层缺失专家补齐，避免 silent bug。",
    )
    parser.add_argument(
        "--strict_missing_experts",
        action="store_true",
        help="若发现某层缺失专家行，则直接报错退出（默认: 关闭，改为补齐 0）。",
    )

    args = parser.parse_args()

    # 验证剪枝率
    if not 0 < args.pruning_rate <= 1:
        parser.error("剪枝率必须在 (0, 1] 范围内")

    # 检查输入文件
    if not os.path.exists(args.input):
        parser.error(f"输入文件不存在: {args.input}")

    print("=" * 70)
    print("基于激活频率的专家选择（对照实验）")
    print("=" * 70)
    print(f"输入文件: {args.input}")
    print(f"目标剪枝率: {args.pruning_rate:.1%}")
    print(f"策略: {args.strategy} (激活频率)")
    print("=" * 70)

    # 解析 aggregated JSON 文件
    print(f"\n解析 aggregated JSON 文件...")
    df = parse_aggregated_json(args.input)
    print(f"✓ 解析完成: {len(df)} 条记录")
    print(f"  层数: {df['Layer'].nunique()}")
    print(f"  专家ID范围: {df['Expert_ID'].min()} - {df['Expert_ID'].max()}")
    print(f"  总激活次数: {df['Total_Activations'].sum():,}")

    # 补齐缺失专家（强烈建议）
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
            f"⚠️ 检测到缺失专家行：{affected}/{total_layers} 层存在缺失。最严重层 Layer {worst_layer[0]} 缺 {len(worst_layer[1])}/{inferred}。"
        )
    else:
        print("✓ 每层专家行完整，无缺失。")

    df = df_completed

    # 根据策略选择专家
    print(f"\n使用 {args.strategy} 策略进行二分查找（基于激活频率）...\n")

    if args.strategy == "global":
        # 全局剪枝率策略
        best_alpha, selection_results, actual_rate = find_alpha_for_global_pruning_rate(
            df, args.pruning_rate, args.tolerance, args.max_iterations
        )
    else:
        # 每层剪枝率策略
        best_alpha, selection_results, actual_rate = (
            find_alpha_for_per_layer_pruning_rate(
                df, args.pruning_rate, args.tolerance, args.max_iterations
            )
        )

    # 保存结果
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
