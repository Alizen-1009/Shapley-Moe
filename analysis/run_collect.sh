#!/bin/bash
# =============================================================================
# 一次 Few-Shot 收集所有剪枝信息 - 批处理脚本
# =============================================================================
#
# 功能：
#   在一次推理中同时收集 Shapley/Gating Score/EASYEP/REAP 四种剪枝方法需要的信息
#   结果按模型组织保存
#   优先从 configs/ 读取配置
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
CONFIG_FILE="${PROJECT_DIR}/configs/experiments.yaml"
MODELS_CONFIG="${PROJECT_DIR}/configs/models.yaml"

# =============================================================================
# 从配置文件读取函数
# =============================================================================

# 通用配置读取函数
read_config() {
    local key="$1"
    local default="$2"
    local config_file="${3:-$CONFIG_FILE}"
    
    if [ ! -f "$config_file" ]; then
        echo "$default"
        return
    fi
    
    local value=$(python3 -c "
import yaml
try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    keys = '$key'.split('.')
    result = config
    for k in keys:
        result = result.get(k, None)
        if result is None:
            break
    if result is not None:
        if isinstance(result, list):
            print(' '.join(str(x) for x in result))
        else:
            print(result)
except:
    pass
" 2>/dev/null)
    
    if [ -n "$value" ]; then
        echo "$value"
    else
        echo "$default"
    fi
}

# 从 models.yaml 读取模型路径
get_model_path() {
    local model_name="$1"
    
    if [ ! -f "$MODELS_CONFIG" ]; then
        echo ""
        return
    fi
    
    python3 -c "
import yaml
try:
    with open('$MODELS_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    models = config.get('models', {})
    if '$model_name' in models:
        print(models['$model_name'].get('path', ''))
except:
    pass
" 2>/dev/null
}

# 列出所有可用模型
list_available_models() {
    if [ ! -f "$MODELS_CONFIG" ]; then
        echo ""
        return
    fi
    
    python3 -c "
import yaml
try:
    with open('$MODELS_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    models = config.get('models', {})
    for name in models.keys():
        print(name)
except:
    pass
" 2>/dev/null
}

# 从配置读取默认值
DEFAULT_MAX_TOKENS=$(read_config "defaults.max_new_tokens" "512")
DEFAULT_DEVICE=$(read_config "defaults.device" "auto")

# 默认参数
MODEL_PATH=""
MODEL_NAME=""
DATA_FILE=""
DATA_DIR="${PROJECT_DIR}/data/calibration"
OUTPUT_DIR="${PROJECT_DIR}/results"
MAX_NEW_TOKENS="$DEFAULT_MAX_TOKENS"
DEVICE="$DEFAULT_DEVICE"
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
    echo "  -m, --model NAME|PATH   模型名称（从配置读取路径）或完整路径"
    echo "  --data FILE             单个数据文件"
    echo "  --data-dir DIR          数据文件目录（默认: $DATA_DIR）"
    echo "  --all                   遍历所有数据文件（*.json）"
    echo "  --output DIR            输出目录（默认: $OUTPUT_DIR）"
    echo "  --max-tokens NUM        最大生成token数（默认: $DEFAULT_MAX_TOKENS）"
    echo "  --device DEVICE         设备（默认: $DEFAULT_DEVICE）"
    echo "  -f, --force             强制重新计算（覆盖已有结果）"
    echo "  --list-models           列出配置中的所有模型"
    echo "  --help                  显示此帮助"
    echo ""
    echo "配置文件:"
    echo "  模型配置: $MODELS_CONFIG"
    echo "  实验配置: $CONFIG_FILE"
    echo ""
    echo "示例:"
    echo "  # 使用模型名称（自动从配置读取路径）"
    echo "  $0 -m qwen3-30b-a3b --all"
    echo ""
    echo "  # 使用完整模型路径"
    echo "  $0 -m /path/to/model --data ${DATA_DIR}/gsm8k_25.json"
    echo ""
    echo "  # 列出可用模型"
    echo "  $0 --list-models"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) 
            MODEL_INPUT="$2"
            # 检查是路径还是模型名称
            if [[ "$MODEL_INPUT" == /* ]] || [[ "$MODEL_INPUT" == ./* ]]; then
                # 是路径
                MODEL_PATH="$MODEL_INPUT"
                MODEL_NAME=$(basename "$MODEL_PATH")
            else
                # 是模型名称，从配置读取路径
                MODEL_NAME="$MODEL_INPUT"
                MODEL_PATH=$(get_model_path "$MODEL_NAME")
                if [ -z "$MODEL_PATH" ]; then
                    log_warning "配置中未找到模型 '$MODEL_NAME' 的路径，将使用名称作为路径"
                    MODEL_PATH="$MODEL_NAME"
                fi
            fi
            shift 2 
            ;;
        --data) DATA_FILE="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --max-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        --list-models)
            echo "配置中的可用模型:"
            list_available_models | while read model; do
                path=$(get_model_path "$model")
                echo "  - $model: $path"
            done
            exit 0
            ;;
        --help|-h) show_help; exit 0 ;;
        *) log_error "未知参数: $1"; show_help; exit 1 ;;
    esac
done

# 验证模型参数
if [ -z "$MODEL_PATH" ]; then
    log_error "必须指定模型 (-m MODEL)"
    show_help
    exit 1
fi

cd "$SCRIPT_DIR"

# 运行单个数据文件的函数
run_single_analysis() {
    local data_file="$1"
    local dataset_name=$(basename "$data_file" .json)
    
    echo ""
    echo "--------------------------------"
    log_info "处理数据集: $dataset_name"
    log_info "模型: $MODEL_NAME"
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
log_info "模型名称: $MODEL_NAME"
log_info "模型路径: $MODEL_PATH"
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
    
    echo ""
    log_info "结果目录: ${OUTPUT_DIR}/${MODEL_NAME}/activations/"
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
        log_info "结果目录: ${OUTPUT_DIR}/${MODEL_NAME}/activations/"
        echo ""
    else
        exit 1
    fi
else
    log_error "请指定 --data 或 --all"
    show_help
    exit 1
fi
