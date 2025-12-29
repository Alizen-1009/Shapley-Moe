#!/usr/bin/env python3
"""
随机剪枝专家选择工具（对照实验）

支持两种策略：
1. 全局剪枝率：在所有层中总共保留 X% 的专家（随机选择）
2. 每层剪枝率：每层都保留 X% 的专家（随机选择，推荐）

与基于 Shapley 值的选择不同，此工具完全随机选择专家，用于对照实验。
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
    从 shapley JSON 文件中解析专家激活次数
    
    Args:
        json_file: shapley JSON 文件路径
        
    Returns:
        DataFrame 包含 Layer, Expert_ID, Total_Activations 列
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 统计每层每个专家的激活次数
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
    - 缺失行补 Shapley_Value=0, Total_Activations=0（若列存在）
    - 返回补齐后的 df，以及每层缺失的 expert_id 列表
    """
    if "Layer" not in df.columns or "Expert_ID" not in df.columns:
        raise ValueError("输入 CSV 必须包含 Layer 与 Expert_ID 两列。")

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError("num_experts 必须为正整数。")

    missing_by_layer: Dict[int, List[int]] = {}
    layers = sorted(df["Layer"].unique().tolist())

    # 标准化列
    df = df.copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Expert_ID"] = df["Expert_ID"].astype(int)

    has_total_acts = "Total_Activations" in df.columns
    if "Shapley_Value" not in df.columns:
        # 如果没有 Shapley_Value 列，创建一个全0的列（随机剪枝不需要）
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
    全局随机剪枝：在所有层中总共保留 X% 的专家（随机选择）

    Args:
        df: 专家数据（需要包含 Layer 和 Expert_ID）
        target_pruning_rate: 目标剪枝率（保留的专家比例，例如 0.4 表示保留40%）
        seed: 随机种子（用于可复现性）

    Returns:
        selection_results: {layer_id: [expert_ids]}
        actual_rate: 实际剪枝率
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 计算总专家数
    total_experts = len(df)
    target_count = int(total_experts * target_pruning_rate)

    print(f"总专家数: {total_experts}")
    print(f"目标保留: {target_count} 个专家 ({target_pruning_rate:.1%})")
    print(f"使用随机选择...")

    # 获取所有专家（层ID, 专家ID）对
    all_experts = [(int(row["Layer"]), int(row["Expert_ID"])) for _, row in df.iterrows()]

    # 随机选择目标数量的专家
    selected_pairs = random.sample(all_experts, min(target_count, len(all_experts)))

    # 按层分组
    selection_results = {}
    for layer_id, expert_id in selected_pairs:
        if layer_id not in selection_results:
            selection_results[layer_id] = []
        selection_results[layer_id].append(expert_id)

    # 对每层的专家ID排序（便于查看）
    for layer_id in selection_results:
        selection_results[layer_id].sort()

    actual_count = len(selected_pairs)
    actual_rate = actual_count / total_experts

    print(f"✓ 实际保留: {actual_count}/{total_experts} ({actual_rate:.1%})")

    return selection_results, actual_rate


def select_random_by_per_layer_pruning_rate(
    df: pd.DataFrame,
    target_pruning_rate: float,
    seed: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], float]:
    """
    每层随机剪枝：每层都保留 X% 的专家（随机选择）

    Args:
        df: 专家数据（需要包含 Layer 和 Expert_ID）
        target_pruning_rate: 目标剪枝率（每层保留的专家比例）
        seed: 随机种子（用于可复现性）

    Returns:
        selection_results: {layer_id: [expert_ids]}
        avg_rate: 平均每层剪枝率
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_layers = df["Layer"].nunique()
    experts_per_layer = int(df.groupby("Layer").size().max())

    print(f"层数: {num_layers}")
    print(f"每层专家数: {experts_per_layer}")
    print(
        f"目标每层保留: {int(experts_per_layer * target_pruning_rate)} 个专家 ({target_pruning_rate:.1%})"
    )
    print(f"使用随机选择...")

    selection_results = {}
    total_selected = 0

    for layer_id, group in df.groupby("Layer"):
        layer_id = int(layer_id)
        all_expert_ids = group["Expert_ID"].astype(int).tolist()
        target_count = int(len(all_expert_ids) * target_pruning_rate)

        # 随机选择目标数量的专家
        selected_experts = random.sample(all_expert_ids, min(target_count, len(all_expert_ids)))
        selected_experts.sort()  # 排序便于查看

        selection_results[layer_id] = selected_experts
        total_selected += len(selected_experts)

    avg_rate = total_selected / num_layers / experts_per_layer

    print(f"✓ 平均每层保留: {total_selected / num_layers:.1f}/{experts_per_layer} ({avg_rate:.1%})")

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
    """保存选择结果和统计信息"""

    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    rate_str = f"{int(pruning_rate * 100)}"
    output_json = os.path.join(
        output_dir, f"selected_experts_random_{strategy}_rate{rate_str}.json"
    )
    output_csv = os.path.join(
        output_dir, f"selection_stats_random_{strategy}_rate{rate_str}.csv"
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

        # 计算选中专家的 Shapley 值（如果存在）
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

    # 保存统计
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_csv, index=False)

    # 计算总体统计
    total_experts = len(df)
    total_selected = sum(len(experts) for experts in selection_results.values())
    actual_global_rate = total_selected / total_experts

    print(f"\n{'='*70}")
    print(f"随机剪枝完成！")
    print(f"{'='*70}")
    print(f"策略: {strategy} (随机)")
    print(f"目标剪枝率: {pruning_rate:.1%}")
    print(f"实际剪枝率: {actual_global_rate:.1%}")
    if seed is not None:
        print(f"随机种子: {seed}")
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
            ["Layer", "Total_Experts", "Selected_Experts", "Pruning_Rate"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="随机剪枝专家选择工具（对照实验）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
策略说明:

  global    - 全局剪枝率：在所有层中总共保留 X% 的专家（随机选择）
  per_layer - 每层剪枝率：每层都保留 X% 的专家（随机选择，推荐）

示例用法:

  # 每层随机保留 50% 专家（推荐）
  python select_experts_random.py \\
      --input ../calc_shapley/results/expert_shapley_values_all_layers.csv \\
      --pruning_rate 0.5 \\
      --strategy per_layer

  # 全局随机保留 40% 专家
  python select_experts_random.py \\
      --input ../calc_shapley/results/expert_shapley_values_all_layers.csv \\
      --pruning_rate 0.4 \\
      --strategy global

  # 每层随机保留 30% 专家，设置随机种子
  python select_experts_random.py \\
      --input shapley_values.csv \\
      --pruning_rate 0.3 \\
      --strategy per_layer \\
      --seed 42 \\
      --output ./my_results
        """,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="输入文件路径（支持 CSV 或 JSON 格式）"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可复现性，默认: None，每次运行结果不同）",
    )

    args = parser.parse_args()

    # 验证剪枝率
    if not 0 < args.pruning_rate <= 1:
        parser.error("剪枝率必须在 (0, 1] 范围内")

    # 检查输入文件
    if not os.path.exists(args.input):
        parser.error(f"输入文件不存在: {args.input}")

    print("=" * 70)
    print("随机剪枝专家选择（对照实验）")
    print("=" * 70)
    print(f"输入文件: {args.input}")
    print(f"目标剪枝率: {args.pruning_rate:.1%}")
    print(f"策略: {args.strategy} (随机)")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print("=" * 70)

    # 读取数据
    print(f"\n读取数据...")
    if args.input.endswith('.json'):
        df = parse_shapley_json(args.input)
    else:
        df = pd.read_csv(args.input)
    print(f"✓ 读取完成: {len(df)} 条记录")

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

    # 根据策略随机选择专家
    print(f"\n使用 {args.strategy} 策略进行随机选择...\n")

    if args.strategy == "global":
        # 全局剪枝率策略
        selection_results, actual_rate = select_random_by_global_pruning_rate(
            df, args.pruning_rate, args.seed
        )
    else:
        # 每层剪枝率策略
        selection_results, actual_rate = select_random_by_per_layer_pruning_rate(
            df, args.pruning_rate, args.seed
        )

    # 保存结果
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

