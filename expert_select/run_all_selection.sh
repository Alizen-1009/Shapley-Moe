#!/bin/bash
# ================================================================
# 批量运行专家选择 - 处理所有 Shapley 值文件
# ================================================================

# 显示帮助
show_help() {
    echo "================================"
    echo "批量专家选择工具"
    echo "================================"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -r, --rates RATES   保留率列表，逗号分隔（默认: 0.8,0.6）"
    echo "  -s, --strategy STR  策略: per_layer 或 global（默认: per_layer）"
    echo "  -i, --input DIR     输入目录（默认: ../calc_shapley/results）"
    echo "  -o, --output DIR    输出目录（默认: ./results）"
    echo "  -n, --num_experts N 每层专家数量（默认: 128）"
    echo "  -f, --force         强制重新计算（覆盖已有结果）"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0                           # 使用默认配置（0.8 和 0.6）"
    echo "  $0 -r 0.9,0.8,0.7,0.6        # 自定义多个保留率"
    echo "  $0 -r 0.5 -s global          # 单个保留率，全局策略"
    echo "  $0 -f                        # 强制重新计算所有"
    echo ""
}

# 默认配置
INPUT_DIR="../calc_shapley/results"
OUTPUT_DIR="./results"
KEEP_RATES="0.8,0.6"
STRATEGY="per_layer"
NUM_EXPERTS=32
FORCE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rates)
            KEEP_RATES="$2"
            shift 2
            ;;
        -s|--strategy)
            STRATEGY="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--num_experts)
            NUM_EXPERTS="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

cd "$(dirname "$0")"

# 将保留率字符串转换为数组
IFS=',' read -ra RATES_ARRAY <<< "$KEEP_RATES"

echo "================================================================"
echo "🚀 批量专家选择"
echo "================================================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "保留率: ${RATES_ARRAY[*]}"
echo "策略: $STRATEGY"
echo "每层专家数: $NUM_EXPERTS"
if [ "$FORCE" = true ]; then
    echo -e "${YELLOW}模式: 强制重新计算${NC}"
else
    echo "模式: 跳过已有结果"
fi
echo "================================================================"
echo ""

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ 错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 统计
total=0
success=0
skipped=0
failed=0

# 遍历所有 shapley CSV 文件
for input_file in "$INPUT_DIR"/*_shapley.csv; do
    # 检查文件是否存在
    if [ ! -f "$input_file" ]; then
        echo -e "${YELLOW}⚠️  没有找到 shapley.csv 文件${NC}"
        exit 1
    fi
    
    # 提取基本名称 (例如: qwen3-30b-a3b_gsm8k_25_shapley)
    base_name=$(basename "$input_file" .csv)
    output_subdir="$OUTPUT_DIR/$base_name"
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}📊 处理: ${base_name}${NC}"
    echo "   输入: $input_file"
    echo ""
    
    # 遍历所有保留率
    for rate in "${RATES_ARRAY[@]}"; do
        total=$((total + 1))
        
        # 计算百分比（用于文件名）
        rate_percent=$(echo "$rate * 100" | bc | cut -d. -f1)
        output_file="$output_subdir/selected_experts_${STRATEGY}_rate${rate_percent}.json"
        
        echo -n "   保留率 $rate (${rate_percent}%): "
        
        # 检查输出文件是否已存在
        if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
            echo -e "${GREEN}✓ 已存在，跳过${NC}"
            skipped=$((skipped + 1))
            success=$((success + 1))
            continue
        fi
        
        # 确保子目录存在
        mkdir -p "$output_subdir"
        
        # 运行选择
        python3 select_experts_by_pruning_rate.py \
            --input "$input_file" \
            --pruning_rate "$rate" \
            --strategy "$STRATEGY" \
            --output "$output_subdir" \
            --num_experts "$NUM_EXPERTS" \
            > /dev/null 2>&1
        
        # 检查结果
        if [ $? -eq 0 ] && [ -f "$output_file" ]; then
            echo -e "${GREEN}✓ 完成${NC}"
            success=$((success + 1))
        else
            echo -e "${RED}✗ 失败${NC}"
            failed=$((failed + 1))
        fi
    done
    
    echo ""
done

echo "================================================================"
echo "📈 统计结果"
echo "================================================================"
echo "总计: $total 个任务"
echo -e "${GREEN}成功: $success${NC}"
if [ $skipped -gt 0 ]; then
    echo "  (其中跳过: $skipped)"
fi
if [ $failed -gt 0 ]; then
    echo -e "${RED}失败: $failed${NC}"
fi
echo ""

# 列出所有生成的目录
echo "📁 输出目录结构:"
for dir in "$OUTPUT_DIR"/*/; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        file_count=$(ls -1 "$dir"*.json 2>/dev/null | wc -l)
        echo "   📂 $dir_name/ ($file_count 个文件)"
    fi
done

echo ""
echo "================================================================"
echo "✅ 全部完成！"
echo "================================================================"
echo ""
echo "查看结果示例:"
echo "  ls -la $OUTPUT_DIR/*/"
echo "  cat $OUTPUT_DIR/*/selected_experts_*.json | head -50"
echo ""

