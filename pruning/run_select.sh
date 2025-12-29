#!/bin/bash
# =============================================================================
# 统一的专家选择脚本
# =============================================================================
#
# 功能：
#   根据指定的方法、模型、数据集和剪枝率进行专家选择
#
# 用法：
#   ./run_select.sh [选项]
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 默认参数
MODEL=""
DATASET=""
METHOD="shapley"
RATE="0.5"
STRATEGY="alpha_per_layer"
OUTPUT_DIR=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    echo "============================================================================="
    echo "统一的专家选择脚本"
    echo "============================================================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model MODEL       模型名称 (如 qwen3-30b-a3b, gpt-oss-20b)"
    echo "  -d, --dataset DATASET   数据集名称 (如 gsm8k_25, arc_easy_25)"
    echo "  -M, --method METHOD     剪枝方法 (shapley|easyep|reap|gating|frequency|random)"
    echo "  -r, --rate RATE         保留率 (0.0-1.0, 默认: 0.5)"
    echo "  -s, --strategy STRATEGY Shapley 策略:"
    echo "                          alpha_per_layer - 每层 Alpha 覆盖（默认，推荐）"
    echo "                          alpha_global    - 全局 Alpha 覆盖"
    echo "                          topk_per_layer  - 每层 Top-K"
    echo "                          topk_global     - 全局 Top-K"
    echo "  -o, --output DIR        输出目录 (默认: results/{model}/selected_experts/)"
    echo "  --all-datasets          处理所有数据集"
    echo "  --all-rates             使用所有默认剪枝率 (0.25,0.5,0.75)"
    echo "  --all-methods           使用所有剪枝方法"
    echo "  -h, --help              显示帮助"
    echo ""
    echo "剪枝方法:"
    echo "  shapley   - Shapley 值剪枝（基于边际贡献）"
    echo "  easyep    - EASYEP 剪枝（weight × (1-simibr) × norm）"
    echo "  reap      - REAP 剪枝（weight × norm）"
    echo "  gating    - Gating Score 剪枝（平均 softmax 分数）"
    echo "  frequency - 激活频率剪枝"
    echo "  random    - 随机剪枝（基线）"
    echo ""
    echo "示例:"
    echo "  $0 -m qwen3-30b-a3b -d gsm8k_25 -M shapley -r 0.5"
    echo "  $0 -m gpt-oss-20b --all-datasets -M easyep --all-rates"
    echo "  $0 -m deepseekv2-lite-coder -d arc_easy_25 --all-methods -r 0.5"
    echo ""
}

# 解析参数
ALL_DATASETS=false
ALL_RATES=false
ALL_METHODS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2 ;;
        -d|--dataset) DATASET="$2"; shift 2 ;;
        -M|--method) METHOD="$2"; shift 2 ;;
        -r|--rate) RATE="$2"; shift 2 ;;
        -s|--strategy) STRATEGY="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --all-datasets) ALL_DATASETS=true; shift ;;
        --all-rates) ALL_RATES=true; shift ;;
        --all-methods) ALL_METHODS=true; shift ;;
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

# 设置默认输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${PROJECT_DIR}/results/${MODEL}/selected_experts"
fi
mkdir -p "$OUTPUT_DIR"

# 获取数据集列表
if [ "$ALL_DATASETS" = true ]; then
    # 从 activations 目录获取数据集列表（新格式：{dataset}_shapley.json）
    DATASETS=($(ls ${PROJECT_DIR}/results/${MODEL}/activations/*_shapley.json 2>/dev/null | xargs -n1 basename | sed 's/_shapley.json//' || echo ""))
    if [ ${#DATASETS[@]} -eq 0 ]; then
        log_error "未找到模型 ${MODEL} 的激活数据"
        exit 1
    fi
else
    if [ -z "$DATASET" ]; then
        log_error "必须指定数据集 (-d DATASET) 或使用 --all-datasets"
        exit 1
    fi
    DATASETS=("$DATASET")
fi

# 获取剪枝率列表
if [ "$ALL_RATES" = true ]; then
    RATES=("0.25" "0.5" "0.75")
else
    RATES=("$RATE")
fi

# 获取方法列表
if [ "$ALL_METHODS" = true ]; then
    METHODS=("shapley" "easyep" "reap" "gating" "frequency" "random")
else
    METHODS=("$METHOD")
fi

# 执行专家选择
log_info "============================================================================="
log_info "专家选择"
log_info "============================================================================="
log_info "模型: $MODEL"
log_info "数据集: ${DATASETS[*]}"
log_info "方法: ${METHODS[*]}"
log_info "剪枝率: ${RATES[*]}"
log_info "输出目录: $OUTPUT_DIR"
log_info "============================================================================="

total=0
success=0
failed=0

for ds in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for rate in "${RATES[@]}"; do
            total=$((total + 1))
            
            # 确定输入文件（新的目录结构）
            case $method in
                shapley)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/shapley_values/${ds}_shapley.csv"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_shapley.py"
                    ;;
                easyep)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_easyep.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_easyep.py"
                    ;;
                reap)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_reap.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_reap.py"
                    ;;
                gating)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_gating.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_gating.py"
                    ;;
                frequency)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_shapley.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_frequency.py"
                    ;;
                random)
                    INPUT_FILE="${PROJECT_DIR}/results/${MODEL}/activations/${ds}_shapley.json"
                    SCRIPT="${SCRIPT_DIR}/methods/select_by_random.py"
                    ;;
            esac
            
            # 检查输入文件
            if [ ! -f "$INPUT_FILE" ]; then
                log_error "输入文件不存在: $INPUT_FILE"
                failed=$((failed + 1))
                continue
            fi
            
            # 生成输出文件名（包含策略信息）
            RATE_STR=$(echo "$rate" | sed 's/\./_/g')
            # Shapley 方法需要区分策略，其他方法使用 per_layer
            if [ "$method" = "shapley" ]; then
                OUTPUT_FILE="${OUTPUT_DIR}/${method}_${STRATEGY}_${ds}_rate${RATE_STR}.json"
            else
                OUTPUT_FILE="${OUTPUT_DIR}/${method}_${ds}_rate${RATE_STR}.json"
            fi
            
            log_info "处理: ${method} / ${ds} / rate=${rate}"
            
            # 运行脚本
            if python3 "$SCRIPT" \
                --input "$INPUT_FILE" \
                --output "$OUTPUT_FILE" \
                --pruning_rate "$rate" \
                --strategy "$STRATEGY" 2>&1 | grep -E "^(✓|选择|保留)" ; then
                log_success "  ✓ 完成: $OUTPUT_FILE"
                success=$((success + 1))
            else
                log_error "  ✗ 失败"
                failed=$((failed + 1))
            fi
        done
    done
done

log_info "============================================================================="
log_info "完成！成功: $success / $total, 失败: $failed"
log_info "============================================================================="

