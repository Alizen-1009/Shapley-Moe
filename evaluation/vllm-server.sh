#!/bin/bash

# =============================================================================
# vLLM 服务器启动脚本
# 使用方式:
#   ./vllm-server.sh                              # 使用默认参数
#   ./vllm-server.sh --model qwen3-30b-a3b --rate 0.5
#   ./vllm-server.sh --method random --dataset arc_easy_25
# =============================================================================

# 默认参数
MODEL="deepseekv2-lite-coder"      # 模型名称
METHOD="shapley"                    # 剪枝方法: shapley, random, frequency, gating, easyep, reap
STRATEGY="alpha_per_layer"          # Shapley策略: alpha_per_layer, alpha_global, topk_per_layer, topk_global
DATASET="gsm8k_25"                  # 数据集名称
RATE="0.8"                          # 剪枝比例 (0.1-0.9)
PORT="8801"                         # 服务端口
TP_SIZE="8"                         # tensor-parallel-size
MODEL_BASE="/root/yuhao/Shapley-Moe/models"  # 模型基础目录

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)     MODEL="$2";     shift 2 ;;
        --method)       METHOD="$2";    shift 2 ;;
        --strategy|-s)  STRATEGY="$2";  shift 2 ;;
        --dataset|-d)   DATASET="$2";   shift 2 ;;
        --rate|-r)      RATE="$2";      shift 2 ;;
        --port|-p)      PORT="$2";      shift 2 ;;
        --tp)           TP_SIZE="$2";   shift 2 ;;
        --base)         MODEL_BASE="$2"; shift 2 ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model, -m     模型名称 (默认: $MODEL)"
            echo "  --method        剪枝方法: shapley|random|frequency|gating|easyep|reap (默认: $METHOD)"
            echo "  --strategy, -s  Shapley策略: alpha_per_layer|alpha_global|topk_per_layer|topk_global (默认: $STRATEGY)"
            echo "  --dataset, -d   数据集名称 (默认: $DATASET)"
            echo "  --rate, -r      剪枝比例 0.1-0.9 (默认: $RATE)"
            echo "  --port, -p      服务端口 (默认: $PORT)"
            echo "  --tp            tensor-parallel-size (默认: $TP_SIZE)"
            echo "  --base          模型基础目录 (默认: $MODEL_BASE)"
            echo ""
            echo "示例:"
            echo "  $0 --model qwen3-30b-a3b --rate 0.5"
            echo "  $0 --method random --dataset arc_easy_25"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 格式化剪枝比例 (0.8 -> 0_8)
RATE_FORMATTED=$(echo "$RATE" | tr '.' '_')

# 构建模型目录名称
if [[ "$METHOD" == "shapley" ]]; then
    # Shapley 方法包含策略
    MODEL_DIR="${MODEL}_${METHOD}_${STRATEGY}_${DATASET}_rate${RATE_FORMATTED}"
else
    # 其他方法不需要策略
    MODEL_DIR="${MODEL}_${METHOD}_${DATASET}_rate${RATE_FORMATTED}"
fi

MODEL_PATH="${MODEL_BASE}/${MODEL_DIR}"

# 显示配置
echo "=============================================="
echo "vLLM 服务器配置"
echo "=============================================="
echo "模型:       $MODEL"
echo "剪枝方法:   $METHOD"
[[ "$METHOD" == "shapley" ]] && echo "策略:       $STRATEGY"
echo "数据集:     $DATASET"
echo "剪枝比例:   $RATE"
echo "端口:       $PORT"
echo "TP Size:    $TP_SIZE"
echo "----------------------------------------------"
echo "模型路径:   $MODEL_PATH"
echo "=============================================="

# 检查模型是否存在
if [[ ! -d "$MODEL_PATH" ]]; then
    echo ""
    echo "⚠️  警告: 模型目录不存在!"
    echo "   请检查路径或先运行剪枝生成模型"
    echo ""
    # 列出可用模型
    echo "可用模型:"
    ls -1 "$MODEL_BASE" 2>/dev/null | head -10
    exit 1
fi

# 启动 vLLM 服务
echo ""
echo "🚀 启动 vLLM 服务..."

vllm serve \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --served-model-name "${MODEL}_pruned" \
    --trust-remote-code \
    --port "$PORT"
