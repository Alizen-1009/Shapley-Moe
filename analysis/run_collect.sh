#!/bin/bash
# =============================================================================
# 一次 Few-Shot 收集所有剪枝信息 - 批处理脚本
# =============================================================================
#
# 功能：
#   在一次推理中同时收集 Shapley/Gating Score/EASYEP/REAP 四种剪枝方法需要的信息
#   结果按模型组织保存
#
# 输出目录结构:
#   results/{model_name}/activations/
#   ├── {dataset}_shapley.json   # Shapley 值计算
#   ├── {dataset}_gating.json    # Gating Score 剪枝
#   ├── {dataset}_easyep.json    # EASYEP 剪枝
#   └── {dataset}_reap.json      # REAP 剪枝
#
# 用法:
#   ./run_collect.sh [选项]
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 默认参数
MODEL_PATH="/root/yuhao/hf_models/deepseekv2-lite-coder"
DATA_FILE=""
DATA_DIR="${PROJECT_DIR}/data/calibration"
OUTPUT_DIR="${PROJECT_DIR}/results"
MAX_NEW_TOKENS=512
DEVICE="auto"
RUN_ALL=false
FORCE=false

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示帮助
show_help() {
    echo "============================================================================="
    echo "一次 Few-Shot 收集所有剪枝信息"
    echo "============================================================================="
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model PATH        模型路径（默认: $MODEL_PATH）"
    echo "  --data FILE         单个数据文件"
    echo "  --data-dir DIR      数据文件目录（默认: $DATA_DIR）"
    echo "  --all               遍历所有数据文件（*.json）"
    echo "  --output DIR        输出目录（默认: $OUTPUT_DIR）"
    echo "  --max-tokens NUM    最大生成token数（默认: $MAX_NEW_TOKENS）"
    echo "  --device DEVICE     设备（默认: $DEVICE）"
    echo "  -f, --force         强制重新计算（覆盖已有结果）"
    echo "  --help              显示此帮助"
    echo ""
    echo "输出目录结构:"
    echo "  results/{model_name}/activations/"
    echo "  ├── {dataset}_shapley.json   # Shapley 值计算"
    echo "  ├── {dataset}_gating.json    # Gating Score 剪枝"
    echo "  ├── {dataset}_easyep.json    # EASYEP 剪枝"
    echo "  └── {dataset}_reap.json      # REAP 剪枝"
    echo ""
    echo "示例:"
    echo "  # 处理单个数据集"
    echo "  $0 --model /path/to/model --data ${DATA_DIR}/gsm8k_25.json"
    echo ""
    echo "  # 遍历所有数据集"
    echo "  $0 --model /path/to/model --all"
    echo ""
    echo "  # 强制重新计算"
    echo "  $0 --model /path/to/model --all --force"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --data) DATA_FILE="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --max-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) log_error "未知参数: $1"; show_help; exit 1 ;;
    esac
done

cd "$SCRIPT_DIR"

# 运行单个数据文件的函数
run_single_analysis() {
    local data_file="$1"
    local dataset_name=$(basename "$data_file" .json)
    local model_name=$(basename "$MODEL_PATH")
    
    echo ""
    echo "--------------------------------"
    log_info "处理数据集: $dataset_name"
    log_info "模型: $model_name"
    echo "--------------------------------"
    
    # 检查数据文件是否存在
    if [ ! -f "$data_file" ]; then
        log_error "数据文件不存在: $data_file"
        return 1
    fi
    
    # 构建命令
    local cmd="python3 collect_activations.py"
    cmd="$cmd --model \"$MODEL_PATH\""
    cmd="$cmd --data \"$data_file\""
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    cmd="$cmd --max_new_tokens $MAX_NEW_TOKENS"
    cmd="$cmd --device $DEVICE"
    
    if [ "$FORCE" = true ]; then
        cmd="$cmd --force"
    fi
    
    # 运行分析
    if eval $cmd; then
        log_success "✓ $dataset_name 分析完成"
        return 0
    else
        log_error "✗ $dataset_name 分析失败"
        return 1
    fi
}

# =============================================================================
# 主逻辑
# =============================================================================

echo ""
echo "============================================================================="
log_info "一次 Few-Shot 收集所有剪枝信息"
echo "============================================================================="
log_info "模型: $MODEL_PATH"
log_info "输出目录: $OUTPUT_DIR"
log_info "生成: $MAX_NEW_TOKENS tokens"
log_info "设备: $DEVICE"
if [ "$FORCE" = true ]; then
    log_info "模式: 强制重新计算"
else
    log_info "模式: 跳过已有结果"
fi
echo "============================================================================="

if [ "$RUN_ALL" = true ]; then
    # 遍历所有数据文件
    log_info "数据目录: $DATA_DIR"
    
    # 检查数据目录是否存在
    if [ ! -d "$DATA_DIR" ]; then
        log_error "数据目录不存在: $DATA_DIR"
        exit 1
    fi
    
    # 查找所有JSON文件
    data_files=("$DATA_DIR"/*.json)
    
    if [ ! -e "${data_files[0]}" ]; then
        log_error "在 $DATA_DIR 中未找到任何JSON文件"
        exit 1
    fi
    
    total=${#data_files[@]}
    success=0
    failed=0
    failed_files=()
    
    log_info "找到 $total 个数据文件"
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
    echo "============================================================================="
    log_info "批量处理完成"
    echo "============================================================================="
    log_info "总计: $total 个数据集"
    log_success "成功: $success 个"
    
    if [ $failed -gt 0 ]; then
        log_error "失败: $failed 个"
        echo ""
        echo "失败的文件:"
        for file in "${failed_files[@]}"; do
            echo "  - $file"
        done
    fi
    
    model_name=$(basename "$MODEL_PATH")
    echo ""
    log_info "结果目录: ${OUTPUT_DIR}/${model_name}/activations/"
    echo ""
    
elif [ -n "$DATA_FILE" ]; then
    # 单个文件处理
    log_info "数据: $DATA_FILE"
    echo ""
    
    if run_single_analysis "$DATA_FILE"; then
        echo ""
        echo "============================================================================="
        log_success "分析完成！"
        echo "============================================================================="
        model_name=$(basename "$MODEL_PATH")
        log_info "结果目录: ${OUTPUT_DIR}/${model_name}/activations/"
        echo ""
    else
        exit 1
    fi
else
    log_error "请指定 --data 或 --all"
    show_help
    exit 1
fi
