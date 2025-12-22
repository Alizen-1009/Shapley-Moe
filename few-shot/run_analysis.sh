#!/bin/bash
# 一键运行专家激活分析

# 配置参数
MODEL_PATH="/root/yuhao/hf_models/qwen3-30b-a3b"
DATA_FILE="../dateset/gsm8k_25.json"
OUTPUT_DIR="./results"
MAX_NEW_TOKENS=512
DEVICE="auto"

# 显示帮助
show_help() {
    echo "================================"
    echo "专家激活分析工具"
    echo "================================"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model PATH        模型路径（默认: $MODEL_PATH）"
    echo "  --data FILE         数据文件（默认: $DATA_FILE）"
    echo "  --output DIR        输出目录（默认: $OUTPUT_DIR）"
    echo "  --max-tokens NUM    最大生成token数（默认: $MAX_NEW_TOKENS）"
    echo "  --device DEVICE     设备（默认: $DEVICE）"
    echo "  --help              显示此帮助"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 --data ../dateset/gsm8k_50.json"
    echo "  $0 --model /path/to/model --data ../dateset/hellaswag_100.json"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")"

echo "================================"
echo "专家激活分析"
echo "================================"
echo "模型: $MODEL_PATH"
echo "数据: $DATA_FILE"
echo "输出: $OUTPUT_DIR"
echo "生成: $MAX_NEW_TOKENS tokens"
echo "设备: $DEVICE"
echo "================================"
echo ""

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 错误: 数据文件不存在: $DATA_FILE"
    echo ""
    echo "提示: 先下载数据集"
    echo "  cd ../dateset"
    echo "  bash download.sh gsm8k 25"
    exit 1
fi

# 运行分析
python3 analyze_and_aggregate.py \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✓ 分析完成！"
    echo "================================"
    echo "结果文件已保存到: $OUTPUT_DIR"
    echo ""
    echo "查看结果:"
    echo "  ls -lh $OUTPUT_DIR"
    echo ""
else
    echo ""
    echo "================================"
    echo "✗ 分析失败"
    echo "================================"
    exit 1
fi

