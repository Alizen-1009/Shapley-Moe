#!/bin/bash
# =============================================================================
# 剪枝模型脚本
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_CONFIG="${PROJECT_DIR}/configs/models.yaml"

# 从配置读取模型路径
get_model_path() {
    python3 -c "
import yaml
with open('$MODELS_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('models', {}).get('$1', {}).get('path', ''))
" 2>/dev/null
}

# 默认参数
MODEL=""
DATASET=""
METHOD="shapley"
STRATEGY="alpha_per_layer"
RATE="0.8"
DEVICE_MAP="auto"
PRUNE_STRATEGY="auto"  # 剪枝策略: auto(根据方法自动选择), zero_weights, gate_bias, both
OUTPUT_DIR=""  # 自定义输出目录（可选）

show_help() {
    echo "用法: $0 -m MODEL -d DATASET [-M METHOD] [-s STRATEGY] [-r RATE] [-p PRUNE_STRATEGY]"
    echo ""
    echo "选项:"
    echo "  -m MODEL           模型名称 (如 qwen3-30b-a3b)"
    echo "  -d DATASET         数据集名称 (如 gsm8k_25)"
    echo "  -M METHOD          剪枝方法 (默认: shapley)"
    echo "  -s STRATEGY        Shapley策略 (默认: alpha_per_layer)"
    echo "  -r RATE            保留率 (默认: 0.8)"
    echo "  -p PRUNE_STRATEGY  剪枝策略 (默认: auto)"
    echo "                     - auto: 根据剪枝方法自动选择 (默认)"
    echo "                     - zero_weights: 将专家权重置零"
    echo "                     - gate_bias: 修改gate使被剪掉的专家不被选中"
    echo "                     - both: 同时使用两种策略"
    echo "  -o OUTPUT_DIR      自定义输出目录 (可选，默认保存在原模型同级目录)"
    echo "  --device MAP       设备映射 (默认: auto)"
    echo ""
    echo "示例:"
    echo "  $0 -m qwen3-30b-a3b -d gsm8k_25 -r 0.8"
    echo "  $0 -m gpt-oss-20b -d arc_easy_25 -M easyep -r 0.6 -p both"
    echo ""
    echo "输出目录示例:"
    echo "  原模型: /root/hf_models/deepseekv2-lite-coder"
    echo "  剪枝后: /root/hf_models/deepseekv2-lite-coder_shapley_alpha_per_layer_gsm8k_25_rate0_6"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL="$2"; shift 2 ;;
        -d) DATASET="$2"; shift 2 ;;
        -M) METHOD="$2"; shift 2 ;;
        -s) STRATEGY="$2"; shift 2 ;;
        -r) RATE="$2"; shift 2 ;;
        -p) PRUNE_STRATEGY="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE_MAP="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "未知选项: $1"; show_help; exit 1 ;;
    esac
done

# 验证必需参数
if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "错误: 必须指定模型 (-m) 和数据集 (-d)"
    show_help
    exit 1
fi

# 获取模型路径
MODEL_PATH=$(get_model_path "$MODEL")
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 配置中未找到模型 '$MODEL'"
    exit 1
fi

# 构建选择文件路径
RATE_STR=$(echo "$RATE" | sed 's/\./_/g')
if [ "$METHOD" = "shapley" ]; then
    SELECTION_FILE="${PROJECT_DIR}/results/${MODEL}/selected_experts/${METHOD}_${STRATEGY}_${DATASET}_rate${RATE_STR}.json"
else
    SELECTION_FILE="${PROJECT_DIR}/results/${MODEL}/selected_experts/${METHOD}_${DATASET}_rate${RATE_STR}.json"
fi

# 检查选择文件
if [ ! -f "$SELECTION_FILE" ]; then
    echo "错误: 选择文件不存在: $SELECTION_FILE"
    exit 1
fi

# 构建输出目录名称
OUTPUT_SUFFIX=""
if [ "$METHOD" = "shapley" ]; then
    OUTPUT_SUFFIX="${METHOD}_${STRATEGY}_${DATASET}_rate${RATE_STR}"
else
    OUTPUT_SUFFIX="${METHOD}_${DATASET}_rate${RATE_STR}"
fi

# 如果没有指定输出目录，默认保存在原模型的同级目录
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_PARENT_DIR=$(dirname "$MODEL_PATH")
    MODEL_BASENAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="${MODEL_PARENT_DIR}/${MODEL_BASENAME}_${OUTPUT_SUFFIX}"
fi

echo "============================================"
echo "模型: $MODEL"
echo "模型路径: $MODEL_PATH"
echo "选择文件: $SELECTION_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "剪枝策略: $PRUNE_STRATEGY"
echo "============================================"

# 运行剪枝
python3 "${SCRIPT_DIR}/save_model.py" \
    --model_path "$MODEL_PATH" \
    --selection_file "$SELECTION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device_map "$DEVICE_MAP" \
    --strategy "$PRUNE_STRATEGY"

echo "✓ 完成: $OUTPUT_DIR"
