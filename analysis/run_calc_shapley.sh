#!/bin/bash
# =============================================================================
# 批量计算 Shapley 值
# =============================================================================
#
# 功能：
#   从激活统计 JSON 文件计算每个专家的 Shapley 值
#
# 用法：
#   ./run_calc_shapley.sh [选项]
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 默认参数
MODEL=""
DATASET=""
NUM_EXPERTS=""
FORCE=false
RUN_ALL=false

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
    echo "批量计算 Shapley 值"
    echo "============================================================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model MODEL       模型名称 (必需)"
    echo "  -d, --dataset DATASET   数据集名称 (可选，默认处理所有)"
    echo "  -n, --num-experts NUM   每层专家数量 (可选，自动检测)"
    echo "  -f, --force             强制重新计算（覆盖已有文件）"
    echo "  --all                   处理所有数据集"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -m gpt-oss-20b                     # 计算所有数据集"
    echo "  $0 -m qwen3-30b-a3b -d gsm8k_25       # 计算单个数据集"
    echo "  $0 -m deepseekv2-lite-coder -f       # 强制重新计算"
    echo ""
}

# 从配置文件获取专家数量
get_num_experts_from_config() {
    local model_name="$1"
    
    local config_file="${PROJECT_DIR}/configs/models.yaml"
    if [ ! -f "$config_file" ]; then
        echo ""
        return
    fi
    
    # 使用 python 解析 YAML
    local num=$(python3 -c "
import yaml
try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    models = config.get('models', {})
    if '$model_name' in models:
        print(models['$model_name'].get('num_experts', ''))
except:
    pass
" 2>/dev/null)
    
    echo "$num"
}

# 自动检测专家数量（从 JSON 文件）
detect_num_experts_from_json() {
    local json_file="$1"
    
    # 从 JSON 文件中找到最大的专家 ID
    local max_expert=$(python3 -c "
import json
import ast
with open('$json_file', 'r') as f:
    data = json.load(f)
max_id = 0
for layer_data in data.get('layers', {}).values():
    for combo_str in layer_data.keys():
        try:
            combo = ast.literal_eval(combo_str)
            max_id = max(max_id, max(combo))
        except:
            pass
print(max_id + 1)
" 2>/dev/null)
    
    echo "$max_expert"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2 ;;
        -d|--dataset) DATASET="$2"; shift 2 ;;
        -n|--num-experts) NUM_EXPERTS="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        --all) RUN_ALL=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_error "未知选项: $1"; show_help; exit 1 ;;
    esac
done

# 验证参数
if [ -z "$MODEL" ]; then
    log_error "必须指定模型名称 (-m MODEL)"
    show_help
    exit 1
fi

# 设置目录
INPUT_DIR="${PROJECT_DIR}/results/${MODEL}/activations"
OUTPUT_DIR="${PROJECT_DIR}/results/${MODEL}/shapley_values"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    log_error "激活数据目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================================="
log_info "批量计算 Shapley 值"
echo "============================================================================="
log_info "模型: $MODEL"
log_info "输入目录: $INPUT_DIR"
log_info "输出目录: $OUTPUT_DIR"
if [ "$FORCE" = true ]; then
    log_warning "模式: 强制重新计算"
else
    log_info "模式: 跳过已有结果"
fi
echo "============================================================================="

# 获取数据集列表
if [ -n "$DATASET" ]; then
    # 指定了单个数据集
    DATASETS=("$DATASET")
else
    # 处理所有数据集
    DATASETS=($(ls ${INPUT_DIR}/*_shapley.json 2>/dev/null | xargs -n1 basename | sed 's/_shapley.json//' || echo ""))
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
    log_error "未找到任何激活数据文件"
    exit 1
fi

log_info "数据集: ${DATASETS[*]}"
echo ""

# 统计
total=0
success=0
skipped=0
failed=0

# 遍历处理
for ds in "${DATASETS[@]}"; do
    total=$((total + 1))
    
    input_file="${INPUT_DIR}/${ds}_shapley.json"
    output_file="${OUTPUT_DIR}/${ds}_shapley.csv"
    
    echo "----------------------------------------"
    log_info "处理: $ds"
    
    # 检查输入文件
    if [ ! -f "$input_file" ]; then
        log_error "输入文件不存在: $input_file"
        failed=$((failed + 1))
        continue
    fi
    
    # 检查输出文件是否已存在
    if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
        log_success "已存在，跳过"
        skipped=$((skipped + 1))
        continue
    fi
    
    # 获取专家数量
    if [ -z "$NUM_EXPERTS" ]; then
        # 1. 先尝试从配置文件获取
        num_exp=$(get_num_experts_from_config "$MODEL")
        if [ -n "$num_exp" ] && [ "$num_exp" -gt 0 ] 2>/dev/null; then
            log_info "从配置获取专家数量: $num_exp"
        else
            # 2. 从 JSON 文件检测
            detected=$(detect_num_experts_from_json "$input_file")
            if [ -n "$detected" ] && [ "$detected" -gt 0 ] && [ "$detected" -lt 1000 ]; then
                num_exp="$detected"
                log_info "检测到专家数量: $num_exp"
            else
                num_exp=64
                log_warning "无法获取专家数量，使用默认值: $num_exp"
            fi
        fi
    else
        num_exp="$NUM_EXPERTS"
        log_info "使用指定专家数量: $num_exp"
    fi
    
    # 运行计算
    log_info "开始计算..."
    if python3 "${SCRIPT_DIR}/calc_shapley.py" \
        --input_file "$input_file" \
        --output_csv "$output_file" \
        --num_experts "$num_exp" 2>&1 | grep -E "^(✓|计算|保存|Layer)" | tail -5; then
        
        if [ -f "$output_file" ]; then
            log_success "完成: $output_file"
            success=$((success + 1))
        else
            log_error "输出文件未生成"
            failed=$((failed + 1))
        fi
    else
        log_error "计算失败"
        failed=$((failed + 1))
    fi
done

echo ""
echo "============================================================================="
log_info "统计结果"
echo "============================================================================="
log_info "总计: $total"
log_success "成功: $success"
if [ $skipped -gt 0 ]; then
    log_info "跳过: $skipped"
fi
if [ $failed -gt 0 ]; then
    log_error "失败: $failed"
fi
echo ""

# 列出所有生成的文件
log_info "Shapley 值文件:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

echo ""
echo "============================================================================="
log_success "完成！"
echo "============================================================================="
