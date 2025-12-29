#!/bin/bash
# 通用数据集下载脚本

# 设置使用国内镜像（如果需要）
export HF_ENDPOINT=https://hf-mirror.com

cd "$(dirname "$0")"

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate lm-evaluation 2>/dev/null || conda activate lighteval 2>/dev/null || true
fi

# 显示帮助信息
show_help() {
    echo "================================"
    echo "通用数据集下载工具"
    echo "================================"
    echo ""
    echo "用法:"
    echo "  $0 [数据集名称] [样本数量] [选项]"
    echo ""
    echo "参数:"
    echo "  数据集名称    数据集名称（默认: gsm8k）"
    echo "  样本数量      要提取的样本数（默认: 25）"
    echo ""
    echo "选项:"
    echo "  --with-answers    包含答案（用于 few-shot）"
    echo "  --list           列出所有可用数据集"
    echo "  --help           显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                              # 下载 gsm8k 25 条"
    echo "  $0 gsm8k 50                     # 下载 gsm8k 50 条"
    echo "  $0 hellaswag 100                # 下载 hellaswag 100 条"
    echo "  $0 gsm8k 30 --with-answers      # 下载 gsm8k 30 条（含答案）"
    echo "  $0 --list                       # 列出可用数据集"
    echo ""
}

# 解析参数
DATASET="ontonotes5"
NUM_SAMPLES=25
WITH_ANSWERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --list)
            python3 download_dataset.py --list
            exit 0
            ;;
        --with-answers)
            WITH_ANSWERS="--with_answers"
            shift
            ;;
        [0-9]*)
            NUM_SAMPLES=$1
            shift
            ;;
        *)
            DATASET=$1
            shift
            ;;
    esac
done

echo "================================"
echo "下载数据集: $DATASET"
echo "样本数量: $NUM_SAMPLES"
if [ -n "$WITH_ANSWERS" ]; then
    echo "包含答案: 是"
fi
echo "================================"
echo ""

# 运行下载
python3 download_dataset.py \
    --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" \
    $WITH_ANSWERS

echo ""
echo "================================"
echo "✓ 完成！"
echo "================================"

