#!/bin/bash
# 基于剪枝率的专家选择脚本

# 默认配置
INPUT_FILE="../calc_shapley/results/qwen3-30b-a3b_gsm8k_25_shapley.csv"
KEEP_RATE=0.4           # 保留80%的专家（即剪掉20%）
STRATEGY="per_layer"    # per_layer 或 global
OUTPUT_DIR="results/qwen3-30b-a3b_gsm8k_25_shapley"
NUM_EXPERTS=""          # 可选：例如 128。留空则从 CSV 推断 max(Expert_ID)+1

# 帮助信息
show_help() {
    echo "================================"
    echo "基于保留率的专家选择"
    echo "================================"
    echo ""
    echo "用法:"
    echo "  $0 [保留率] [策略] [选项]"
    echo ""
    echo "参数:"
    echo "  保留率        保留的专家比例（0-1，默认: 0.4 即保留40%，剪掉60%）"
    echo "  策略          global 或 per_layer（默认: per_layer）"
    echo ""
    echo "选项:"
    echo "  --input FILE      输入CSV文件（默认: $INPUT_FILE）"
    echo "  --output DIR      输出目录（默认: $OUTPUT_DIR）"
    echo "  --num_experts N   （可选）每层专家总数（例如 qwen3 传 128）。缺失专家会自动补 Shapley=0。"
    echo "  --help            显示帮助"
    echo ""
    echo "策略说明:"
    echo "  per_layer  - 每层都保留相同比例的专家（推荐）"
    echo "  global     - 全局保留总专家数的指定比例"
    echo ""
    echo "示例:"
    echo "  $0                          # 每层保留40%（剪掉60%）"
    echo "  $0 0.5                      # 每层保留50%（剪掉50%）"
    echo "  $0 0.3 per_layer            # 每层保留30%（剪掉70%）"
    echo "  $0 0.4 global               # 全局保留40%（剪掉60%）"
    echo "  $0 0.6 per_layer --output ./my_results"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_experts)
            NUM_EXPERTS="$2"
            shift 2
            ;;
        global|per_layer)
            STRATEGY=$1
            shift
            ;;
        0.[0-9]*)
            KEEP_RATE=$1
            shift
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")"

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate lm-evaluation 2>/dev/null || conda activate lighteval 2>/dev/null || true
fi

echo "================================"
echo "专家选择配置"
echo "================================"
echo "输入文件: $INPUT_FILE"
echo "保留率: $KEEP_RATE (保留 $(echo "$KEEP_RATE * 100" | bc)%, 剪掉 $(echo "(1-$KEEP_RATE) * 100" | bc)%)"
echo "策略: $STRATEGY"
echo "输出目录: $OUTPUT_DIR"
if [ -n "$NUM_EXPERTS" ]; then
  echo "每层专家总数: $NUM_EXPERTS"
fi
echo "================================"
echo ""

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件不存在: $INPUT_FILE"
    echo ""
    echo "提示: 请先运行 Shapley 值计算"
    echo "  cd ../calc_shapley"
    echo "  bash calc_shapley.sh"
    exit 1
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 运行选择
python3 select_experts_by_pruning_rate.py \
    --input "$INPUT_FILE" \
    --pruning_rate "$KEEP_RATE" \
    --strategy "$STRATEGY" \
    --output "$OUTPUT_DIR" \
    ${NUM_EXPERTS:+--num_experts "$NUM_EXPERTS"}

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✓ 专家选择完成！"
    echo "================================"
    echo "查看结果:"
    echo "  ls -lh $OUTPUT_DIR"
    echo "  cat $OUTPUT_DIR/selected_experts_${STRATEGY}_rate*.json"
    echo ""
else
    echo ""
    echo "================================"
    echo "✗ 专家选择失败"
    echo "================================"
    exit 1
fi

