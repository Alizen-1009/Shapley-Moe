#!/bin/bash
# =============================================================================
# 通用数据集下载脚本
# =============================================================================
#
# 功能：
#   下载数据集，可从配置文件读取数据集列表
#
# 用法：
#   ./run_download.sh [数据集名称] [样本数量] [选项]
#
# =============================================================================

# 设置使用国内镜像（如果需要）
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/experiments.yaml"

cd "$SCRIPT_DIR"

# 激活 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate lm-evaluation 2>/dev/null || conda activate lighteval 2>/dev/null || true
fi

# =============================================================================
# 从配置文件读取函数
# =============================================================================

# 读取配置中的数据集列表
read_datasets_from_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo ""
        return
    fi
    
    python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    datasets = config.get('datasets', [])
    for ds in datasets:
        # 去掉 _25 后缀获取数据集名称
        name = ds.rsplit('_', 1)[0] if '_' in ds else ds
        print(name)
except:
    pass
" 2>/dev/null | sort -u
}

# =============================================================================
# 帮助信息
# =============================================================================

show_help() {
    echo "============================================================================="
    echo "通用数据集下载工具"
    echo "============================================================================="
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
    echo "  --all             下载配置文件中的所有数据集"
    echo "  --list            列出所有可用数据集"
    echo "  --list-config     列出配置文件中的数据集"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "配置文件: $CONFIG_FILE"
    echo ""
    echo "示例:"
    echo "  $0                              # 下载 gsm8k 25 条"
    echo "  $0 gsm8k 50                     # 下载 gsm8k 50 条"
    echo "  $0 hellaswag 100                # 下载 hellaswag 100 条"
    echo "  $0 gsm8k 30 --with-answers      # 下载 gsm8k 30 条（含答案）"
    echo "  $0 --all                        # 下载配置中的所有数据集"
    echo "  $0 --list                       # 列出可用数据集"
    echo ""
}

# =============================================================================
# 解析参数
# =============================================================================

DATASET=""
NUM_SAMPLES=25
WITH_ANSWERS=""
DOWNLOAD_ALL=false

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
        --list-config)
            echo "配置文件中的数据集:"
            read_datasets_from_config | while read ds; do
                echo "  - $ds"
            done
            exit 0
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --with-answers)
            WITH_ANSWERS="--with_answers"
            shift
            ;;
        [0-9]*)
            NUM_SAMPLES=$1
            shift
            ;;
        -*)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
        *)
            DATASET=$1
            shift
            ;;
    esac
done

# =============================================================================
# 下载逻辑
# =============================================================================

if [ "$DOWNLOAD_ALL" = true ]; then
    # 下载配置中的所有数据集
    echo "============================================================================="
    echo "批量下载配置中的所有数据集"
    echo "============================================================================="
    echo "样本数量: $NUM_SAMPLES"
    if [ -n "$WITH_ANSWERS" ]; then
        echo "包含答案: 是"
    fi
    echo "============================================================================="
    echo ""
    
    CONFIG_DATASETS=$(read_datasets_from_config)
    
    if [ -z "$CONFIG_DATASETS" ]; then
        echo "错误: 配置文件中未找到数据集列表"
        exit 1
    fi
    
    total=0
    success=0
    failed=0
    
    for ds in $CONFIG_DATASETS; do
        total=$((total + 1))
        echo ""
        echo "----------------------------------------"
        echo "下载: $ds ($NUM_SAMPLES 条)"
        echo "----------------------------------------"
        
        if python3 download_dataset.py \
            --dataset "$ds" \
            --num_samples "$NUM_SAMPLES" \
            $WITH_ANSWERS; then
            success=$((success + 1))
            echo "✓ $ds 下载完成"
        else
            failed=$((failed + 1))
            echo "✗ $ds 下载失败"
        fi
    done
    
    echo ""
    echo "============================================================================="
    echo "批量下载完成"
    echo "============================================================================="
    echo "总计: $total, 成功: $success, 失败: $failed"
    echo "============================================================================="
    
else
    # 下载单个数据集
    if [ -z "$DATASET" ]; then
        DATASET="gsm8k"
    fi
    
    echo "============================================================================="
echo "下载数据集: $DATASET"
echo "样本数量: $NUM_SAMPLES"
if [ -n "$WITH_ANSWERS" ]; then
    echo "包含答案: 是"
fi
    echo "============================================================================="
echo ""

# 运行下载
python3 download_dataset.py \
    --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" \
    $WITH_ANSWERS

echo ""
    echo "============================================================================="
echo "✓ 完成！"
    echo "============================================================================="
fi
