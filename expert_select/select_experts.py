import pandas as pd
import argparse
import os
import json


def _infer_num_experts(df: pd.DataFrame) -> int:
    if "Expert_ID" not in df.columns or df["Expert_ID"].empty:
        raise ValueError("输入数据缺少 Expert_ID，无法推断 num_experts。")
    return int(df["Expert_ID"].max()) + 1


def _complete_layers_with_missing_experts(
    df: pd.DataFrame,
    num_experts: int,
    strict_missing: bool = False,
    missing_sample_limit: int = 20,
):
    if "Layer" not in df.columns or "Expert_ID" not in df.columns:
        raise ValueError("输入 CSV 必须包含 Layer 与 Expert_ID 两列。")
    if "Shapley_Value" not in df.columns:
        raise ValueError("输入 CSV 必须包含 Shapley_Value 列。")

    num_experts = int(num_experts)
    if num_experts <= 0:
        raise ValueError("num_experts 必须为正整数。")

    df = df.copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Expert_ID"] = df["Expert_ID"].astype(int)

    has_total_acts = "Total_Activations" in df.columns
    layers = sorted(df["Layer"].unique().tolist())
    full_index = list(range(num_experts))

    missing_by_layer = {}
    parts = []
    for layer_id in layers:
        layer_df = df[df["Layer"] == layer_id].copy()
        present = set(layer_df["Expert_ID"].tolist())
        missing = [i for i in full_index if i not in present]
        if missing:
            missing_by_layer[int(layer_id)] = missing
            if strict_missing:
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

        parts.append(layer_df)

    completed_df = pd.concat(parts, ignore_index=True)
    return completed_df, missing_by_layer


def select_experts(csv_path, alpha, output_dir):
    """
    根据 Shapley 值选择专家。
    策略：
    1. 在每一层中，按 Shapley 值降序排列专家。
    2. 计算该层专家 Shapley 值的总和。
    3. 设置阈值 target = total * alpha。
    4. 累加专家的 Shapley 值，直到累加值 >= target。
    5. 保留这些对总价值贡献最大的最少数量专家。
    """
    print(f"Reading input from {csv_path}")
    df = pd.read_csv(csv_path)
    inferred = _infer_num_experts(df)
    df, missing_by_layer = _complete_layers_with_missing_experts(
        df,
        num_experts=inferred,
        strict_missing=False,
    )
    if missing_by_layer:
        worst_layer = max(missing_by_layer.items(), key=lambda kv: len(kv[1]))
        print(
            f"⚠️ 检测到缺失专家行：{len(missing_by_layer)}/{df['Layer'].nunique()} 层存在缺失。最严重层 Layer {worst_layer[0]} 缺 {len(worst_layer[1])}/{inferred}。"
        )

    # Ensure results directory exists
    os.makedirs(output_dir, exist_ok=True)

    selection_results = {}
    stats_data = []

    # Process each layer
    for layer_id, group in df.groupby("Layer"):
        # Sort by Shapley Value descending
        group_sorted = group.sort_values("Shapley_Value", ascending=False)

        total_shapley = group_sorted["Shapley_Value"].sum()
        target_shapley = total_shapley * alpha

        cumulative_shapley = 0
        selected_experts = []

        for _, row in group_sorted.iterrows():
            cumulative_shapley += row["Shapley_Value"]
            # Using int() to convert numpy int to python int for JSON serialization
            selected_experts.append(int(row["Expert_ID"]))

            # Stop as soon as we meet the target
            if cumulative_shapley >= target_shapley:
                break

        # If alpha is large (e.g. 1.0) or numerical precision issues, we might miss the last one if strictly >,
        # but >= handles it. If alpha > 1, loop finishes and we take all.

        selection_results[int(layer_id)] = selected_experts

        stats_data.append(
            {
                "Layer": int(layer_id),
                "Total_Experts": len(group),
                "Selected_Experts": len(selected_experts),
                "Total_Shapley": total_shapley,
                "Selected_Shapley": cumulative_shapley,
                "Ratio": cumulative_shapley / total_shapley if total_shapley > 0 else 0,
                "Alpha": alpha,
                "Expected_Num_Experts": inferred,
                "Missing_Experts_Count": len(missing_by_layer.get(int(layer_id), [])),
                "Missing_Experts_Sample": ",".join(
                    map(str, missing_by_layer.get(int(layer_id), [])[:20])
                )
                if missing_by_layer.get(int(layer_id))
                else "",
            }
        )

    # Save selection map
    output_json = os.path.join(output_dir, f"selected_experts_alpha_{alpha}.json")
    with open(output_json, "w") as f:
        json.dump(selection_results, f, indent=4)

    # Save stats
    stats_df = pd.DataFrame(stats_data)
    output_csv = os.path.join(output_dir, f"selection_stats_alpha_{alpha}.csv")
    stats_df.to_csv(output_csv, index=False)

    print(f"Selection completed with alpha={alpha}")
    print(f"Results saved to:\n  - {output_json}\n  - {output_csv}")

    # Print a summary to console
    print("\nSummary per layer:")
    print(
        stats_df[["Layer", "Total_Experts", "Selected_Experts", "Ratio"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select experts based on Shapley values"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to expert shapley values CSV"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Cumulative Shapley value threshold factor (default: 0.8)",
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )

    args = parser.parse_args()

    select_experts(args.input, args.alpha, args.output)
