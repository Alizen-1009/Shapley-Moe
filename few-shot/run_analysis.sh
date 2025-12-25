#!/bin/bash
# 一键运行专家激活分析

# 配置参数
MODEL_PATH="/root/yuhao/hf_models/deepseekv2-lite-coder"
DATA_FILE="../dateset/results/ontonotes5_25.json"
DATA_DIR="../dateset/results"
OUTPUT_DIR="./results"
MAX_NEW_TOKENS=512
DEVICE="auto"
RUN_ALL=false
FORCE=false

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
    echo "  --data-dir DIR      数据文件目录（默认: $DATA_DIR）"
    echo "  --all               遍历所有数据文件（*.json）"
    echo "  --output DIR        输出目录（默认: $OUTPUT_DIR）"
    echo "  --max-tokens NUM    最大生成token数（默认: $MAX_NEW_TOKENS）"
    echo "  --device DEVICE     设备（默认: $DEVICE）"
    echo "  -f, --force         强制重新计算（覆盖已有结果）"
    echo "  --help              显示此帮助"
    echo ""
    echo "示例:"
    echo "  $0                                    # 运行单个默认数据集"
    echo "  $0 --data ../dateset/results/gsm8k_25.json   # 运行指定数据集"
    echo "  $0 --all                             # 遍历所有数据集（跳过已有）"
    echo "  $0 --all -f                          # 遍历所有数据集（强制重新计算）"
    echo "  $0 --all --data-dir ../dateset/results       # 指定目录遍历所有数据集"
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
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
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

cd "$(dirname "$0")"

# 检查 Python 依赖
check_dependencies() {
    local model_path="$1"
    local missing_deps=()
    
    # 检查模型是否需要 mamba-ssm
    # 通过检查模型配置或模型名称来判断
    if [[ "$model_path" == *"mamba"* ]] || [[ "$model_path" == *"Mamba"* ]] || \
       [[ "$model_path" == *"nemotron"* ]] || [[ "$model_path" == *"Nemotron"* ]]; then
        if ! python3 -c "import mamba_ssm" 2>/dev/null; then
            missing_deps+=("mamba-ssm")
        fi
    fi
    
    # 检查基础依赖
    if ! python3 -c "import torch" 2>/dev/null; then
        missing_deps+=("torch")
    fi
    if ! python3 -c "import transformers" 2>/dev/null; then
        missing_deps+=("transformers")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo "❌ 缺少必需的 Python 依赖: ${missing_deps[*]}"
        echo ""
        echo "请安装缺失的依赖："
        echo ""
        for dep in "${missing_deps[@]}"; do
            case "$dep" in
                mamba-ssm)
                    echo "  pip install mamba-ssm"
                    echo "  或者:"
                    echo "  pip install causal-conv1d>=1.2.0 mamba-ssm"
                    ;;
                torch)
                    echo "  pip install torch"
                    ;;
                transformers)
                    echo "  pip install transformers"
                    ;;
            esac
        done
        echo ""
        echo "如果仍然遇到问题，请参考模型文档安装所有依赖。"
        return 1
    fi
    
    return 0
}

# 运行单个数据文件的函数
run_single_analysis() {
    local data_file="$1"
    local dataset_name=$(basename "$data_file" .json)
    local model_name=$(basename "$MODEL_PATH")
    
    # 构建输出文件名（与 analyze_and_aggregate.py 保持一致）
    local output_file="${OUTPUT_DIR}/${model_name}_${dataset_name}_aggregated.json"
    
    echo ""
    echo "--------------------------------"
    echo "处理数据集: $dataset_name"
    echo "文件: $data_file"
    echo "输出: $output_file"
    echo "--------------------------------"
    
    # 检查数据文件是否存在
    if [ ! -f "$data_file" ]; then
        echo "❌ 错误: 数据文件不存在: $data_file"
        return 1
    fi
    
    # 检查输出文件是否已存在
    if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
        echo "✓ 已存在，跳过（使用 -f 强制重新计算）"
        return 0
    fi
    
    # 运行分析
    local exit_code=0
    python3 analyze_and_aggregate.py \
        --model "$MODEL_PATH" \
        --data "$data_file" \
        --output_dir "$OUTPUT_DIR" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --device "$DEVICE" 2>&1 | tee /tmp/analysis_output_$$.log
    exit_code=${PIPESTATUS[0]}
    
    # 检查是否是依赖缺失错误
    if [ $exit_code -ne 0 ]; then
        if grep -q "mamba-ssm.*cannot be imported\|mamba_ssm.*required" /tmp/analysis_output_$$.log 2>/dev/null; then
            echo ""
            echo "⚠️  检测到 mamba-ssm 依赖缺失"
            echo "请运行以下命令安装："
            echo "  pip install mamba-ssm"
            echo "  或者:"
            echo "  pip install causal-conv1d>=1.2.0 mamba-ssm"
            echo ""
        fi
    fi
    
    rm -f /tmp/analysis_output_$$.log
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $dataset_name 分析完成"
        return 0
    else
        echo "✗ $dataset_name 分析失败"
        return 1
    fi
}

# 主逻辑
# 检查依赖（只在第一次运行时检查）
if [ "$RUN_ALL" = true ]; then
    # 遍历所有数据文件
    echo "================================"
    echo "专家激活分析 - 批量处理"
    echo "================================"
    echo "模型: $MODEL_PATH"
    echo "数据目录: $DATA_DIR"
    echo "输出: $OUTPUT_DIR"
    echo "生成: $MAX_NEW_TOKENS tokens"
    echo "设备: $DEVICE"
    if [ "$FORCE" = true ]; then
        echo "模式: 强制重新计算"
    else
        echo "模式: 跳过已有结果"
    fi
    echo "================================"
    echo ""
    
    # 检查依赖（可选，如果检测失败会继续运行并在实际错误时提示）
    echo "检查依赖..."
    check_dependencies "$MODEL_PATH" || echo "⚠️  依赖检查失败，将继续尝试运行..."
    echo ""
    
    # 检查数据目录是否存在
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ 错误: 数据目录不存在: $DATA_DIR"
        exit 1
    fi
    
    # 查找所有JSON文件
    data_files=("$DATA_DIR"/*.json)
    
    if [ ! -e "${data_files[0]}" ]; then
        echo "❌ 错误: 在 $DATA_DIR 中未找到任何JSON文件"
        exit 1
    fi
    
    total=${#data_files[@]}
    success=0
    failed=0
    failed_files=()
    
    echo "找到 $total 个数据文件，开始处理..."
    echo ""
    
    # 遍历处理每个文件
    for data_file in "${data_files[@]}"; do
        if run_single_analysis "$data_file"; then
            ((success++))
        else
            ((failed++))
            failed_files+=("$data_file")
        fi
    done
    
    # 输出总结
    echo ""
    echo "================================"
    echo "批量处理完成"
    echo "================================"
    echo "总计: $total 个数据集"
    echo "成功: $success 个"
    echo "失败: $failed 个"
    
    if [ $failed -gt 0 ]; then
        echo ""
        echo "失败的文件:"
        for file in "${failed_files[@]}"; do
            echo "  - $file"
        done
        echo ""
        exit 1
    else
        echo ""
        echo "所有数据集处理完成！"
        echo "结果文件已保存到: $OUTPUT_DIR"
        echo ""
        echo "查看结果:"
        echo "  ls -lh $OUTPUT_DIR"
        echo ""
    fi
else
    # 单个文件处理（原有逻辑）
    echo "================================"
    echo "专家激活分析"
    echo "================================"
    echo "模型: $MODEL_PATH"
    echo "数据: $DATA_FILE"
    echo "输出: $OUTPUT_DIR"
    echo "生成: $MAX_NEW_TOKENS tokens"
    echo "设备: $DEVICE"
    if [ "$FORCE" = true ]; then
        echo "模式: 强制重新计算"
    fi
    echo "================================"
    echo ""
    
    # 检查依赖（可选，如果检测失败会继续运行并在实际错误时提示）
    echo "检查依赖..."
    check_dependencies "$MODEL_PATH" || echo "⚠️  依赖检查失败，将继续尝试运行..."
    echo ""
    
    if run_single_analysis "$DATA_FILE"; then
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
fi

