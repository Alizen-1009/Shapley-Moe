#!/bin/bash
# ================================================================
# 批量计算所有数据集的 Shapley 值
# ================================================================

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -f, --force      强制重新计算（覆盖已有文件）"
    echo "  -n, --num NUM    每层专家数量（默认: 128）"
    echo "  -h, --help       显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0               # 计算所有（跳过已有）"
    echo "  $0 -f            # 强制重新计算所有"
    echo "  $0 -n 64         # 指定每层64个专家"
}

# 默认配置
INPUT_DIR="few-shot/results"
OUTPUT_DIR="calc_shapley/results"
NUM_EXPERTS=32  # GPT-OSS-20B 每层专家数
FORCE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE=true
            shift
            ;;
        -n|--num)
            NUM_EXPERTS="$2"
            shift 2
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
NC='\033[0m' # No Color

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

echo "================================================================"
echo "🚀 批量计算 Shapley 值"
echo "================================================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "专家数量: $NUM_EXPERTS"
if [ "$FORCE" = true ]; then
    echo -e "${YELLOW}模式: 强制重新计算${NC}"
fi
echo "================================================================"
echo ""

# 统计
total=0
success=0
failed=0

# 遍历所有 aggregated.json 文件
for input_file in "$INPUT_DIR"/*_aggregated.json; do
    # 检查文件是否存在
    if [ ! -f "$input_file" ]; then
        echo -e "${YELLOW}⚠️  没有找到 aggregated.json 文件${NC}"
        exit 1
    fi
    
    # 提取基本名称 (例如: qwen3-30b-a3b_gsm8k_25)
    base_name=$(basename "$input_file" | sed 's/_aggregated\.json$//')
    output_file="$OUTPUT_DIR/${base_name}_shapley.csv"
    
    total=$((total + 1))
    
    echo -e "${YELLOW}📊 处理: ${base_name}${NC}"
    echo "   输入: $input_file"
    echo "   输出: $output_file"
    
    # 检查输出文件是否已存在
    if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
        echo -e "   ${GREEN}✓ 已存在，跳过（使用 -f 强制重新计算）${NC}"
        success=$((success + 1))
        echo ""
        continue
    fi
    
    # 运行计算
    python3 calc_shapley/calc_expert_shapley.py \
        --input_file "$input_file" \
        --output_csv "$output_file" \
        --num_experts "$NUM_EXPERTS" 2>&1 | tail -5
    
    # 检查结果
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        echo -e "   ${GREEN}✓ 完成${NC}"
        success=$((success + 1))
    else
        echo -e "   ${RED}✗ 失败${NC}"
        failed=$((failed + 1))
    fi
    
    echo ""
done

echo "================================================================"
echo "📈 统计结果"
echo "================================================================"
echo -e "总计: $total"
echo -e "${GREEN}成功: $success${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}失败: $failed${NC}"
fi
echo ""

# 列出所有生成的文件
echo "📁 生成的 Shapley 值文件:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null | awk '{print "   " $NF " (" $5 ")"}'

echo ""
echo "================================================================"
echo "✅ 全部完成！"
echo "================================================================"

