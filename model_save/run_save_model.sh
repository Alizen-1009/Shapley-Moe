#!/bin/bash
# 批量保存剪枝后的MoE模型

# ============ 基础配置 ============
MODEL_NAME="gpt-oss-20b"           # 模型简称
DATASETS="truthful_qa"                # 数据集列表，逗号分隔（如: "humaneval,gsm8k,math_500"）
RATES="60,80"                          # 保留率列表，逗号分隔（如: "60,80"）
STRATEGY="per_layer"                # 策略

# ============ 设备配置 ============
# 设备映射策略:
#   auto          - 自动分配到可用GPU（推荐大模型）
#   balanced      - 均匀分配到所有GPU
#   cuda:0        - 仅使用单张指定GPU
#   cpu           - 仅使用CPU
DEVICE_MAP="auto"
FORCE=false                          # 是否强制重新保存（覆盖已有模型）

# 显示帮助
show_help() {
    echo "================================"
    echo "MoE模型专家剪枝 - 批量处理"
    echo "================================"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model NAME         模型名称（默认: $MODEL_NAME）"
    echo "  --datasets LIST      数据集列表，逗号分隔（默认: $DATASETS）"
    echo "  --rates LIST         保留率列表，逗号分隔（默认: $RATES）"
    echo "  --strategy STR       策略（默认: $STRATEGY）"
    echo "  --device-map MAP     设备映射（默认: $DEVICE_MAP）"
    echo "  -f, --force          强制重新保存（覆盖已有模型）"
    echo "  --help               显示此帮助"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 --datasets humaneval,gsm8k        # 处理多个数据集"
    echo "  $0 --rates 60,80                     # 处理多个保留率"
    echo "  $0 --datasets humaneval,gsm8k --rates 60,80  # 处理所有组合"
    echo "  $0 -f                                 # 强制重新保存"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --rates)
            RATES="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --device-map)
            DEVICE_MAP="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
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

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 自动生成路径 ============
MODEL_PATH="/root/yuhao/hf_models/${MODEL_NAME}"
SELECTION_BASE_DIR="$PROJECT_ROOT/expert_select/results"

# 将数据集和保留率字符串转换为数组
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra RATE_ARRAY <<< "$RATES"

echo "=========================================="
echo "MoE模型专家剪枝 - 批量处理"
echo "=========================================="
echo "原始模型: $MODEL_PATH"
echo "数据集: ${DATASET_ARRAY[*]}"
echo "保留率: ${RATE_ARRAY[*]}"
echo "策略: $STRATEGY"
echo "设备映射: $DEVICE_MAP"
if [ "$FORCE" = true ]; then
    echo "模式: 强制重新保存"
else
    echo "模式: 跳过已有模型"
fi
echo "=========================================="
echo ""

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 运行单个剪枝的函数
run_single_prune() {
    local dataset="$1"
    local rate="$2"
    local selection_file="${SELECTION_BASE_DIR}/${MODEL_NAME}_${dataset}_25_shapley/selected_experts_${STRATEGY}_rate${rate}.json"
    local output_dir="/root/yuhao/hf_models/${MODEL_NAME}-${dataset}-r${rate}"
    
    echo "--------------------------------"
    echo "数据集: $dataset, 保留率: $rate"
    echo "选择文件: $selection_file"
    echo "输出目录: $output_dir"
    echo "--------------------------------"
    
    # 检查选择文件是否存在
    if [ ! -f "$selection_file" ]; then
        echo "❌ 错误: 专家选择文件不存在: $selection_file"
        return 1
    fi
    
    # 检查输出目录是否已存在
    if [ -d "$output_dir" ] && [ "$FORCE" = false ]; then
        echo "✓ 模型已存在，跳过（使用 -f 强制重新保存）"
        return 0
    fi
    
    # 运行剪枝脚本
    python3 "$SCRIPT_DIR/save_pruned_model.py" \
        --model_path "$MODEL_PATH" \
        --selection_file "$selection_file" \
        --output_dir "$output_dir" \
        --device_map "$DEVICE_MAP"
    
    if [ $? -eq 0 ]; then
        echo "✓ 剪枝完成: $output_dir"
        return 0
    else
        echo "✗ 剪枝失败"
        return 1
    fi
}

# 统计信息
total=0
success=0
skipped=0
failed=0
failed_combos=()

# 遍历所有数据集和保留率的组合
for dataset in "${DATASET_ARRAY[@]}"; do
    for rate in "${RATE_ARRAY[@]}"; do
        total=$((total + 1))
        output_dir="/root/yuhao/hf_models/${MODEL_NAME}-${dataset}-r${rate}"
        
        # 检查是否已存在
        if [ -d "$output_dir" ] && [ "$FORCE" = false ]; then
            echo "--------------------------------"
            echo "数据集: $dataset, 保留率: $rate"
            echo "输出目录: $output_dir"
            echo "--------------------------------"
            echo "✓ 模型已存在，跳过（使用 -f 强制重新保存）"
            skipped=$((skipped + 1))
            success=$((success + 1))
            echo ""
            continue
        fi
        
        if run_single_prune "$dataset" "$rate"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
            failed_combos+=("${dataset}-r${rate}")
        fi
        
        echo ""
    done
done

# 输出总结
echo "=========================================="
echo "批量剪枝完成"
echo "=========================================="
echo "总计: $total 个组合"
echo "成功: $success 个"
echo "跳过: $skipped 个"
echo "失败: $failed 个"

if [ $failed -gt 0 ]; then
    echo ""
    echo "失败的组合:"
    for combo in "${failed_combos[@]}"; do
        echo "  - $combo"
    done
    echo ""
    exit 1
else
    echo ""
    echo "所有模型剪枝完成！"
    echo ""
    echo "剪枝后的模型保存在: /root/yuhao/hf_models/"
    echo ""
    echo "查看结果:"
    echo "  ls -lh /root/yuhao/hf_models/${MODEL_NAME}-*"
    echo ""
fi

