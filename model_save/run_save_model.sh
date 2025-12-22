#!/bin/bash
# 示例脚本：保存剪枝后的MoE模型

# 配置参数
MODEL_PATH="/root/yuhao/hf_models/qwen3-30b-a3b"
SELECTION_FILE="./expert_select/results/qwen3-30b-a3b_gsm8k_25_shapley/selected_experts_per_layer_rate60.json"
OUTPUT_DIR="/root/yuhao/hf_models/qwen3-30b-a3b-pruned-alpha0.6"
DEVICE="cuda:0"  # 使用GPU 1，如果GPU 0被占用

echo "=========================================="
echo "MoE模型专家剪枝"
echo "=========================================="
echo "原始模型: $MODEL_PATH"
echo "专家选择文件: $SELECTION_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "设备: $DEVICE"
echo "=========================================="

# 运行剪枝脚本
python ./model_save/save_pruned_model.py \
    --model_path "$MODEL_PATH" \
    --selection_file "$SELECTION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 剪枝完成!"
    echo "=========================================="
    echo "剪枝后的模型保存在: $OUTPUT_DIR"
    echo ""
    echo "测试模型:"
    echo "  python test_saved_model.py --model_path $OUTPUT_DIR"
    echo ""
    echo "使用lm-eval评测:"
    echo "  lm_eval --model vllm \\"
    echo "      --model_args pretrained=$OUTPUT_DIR,dtype=auto \\"
    echo "      --tasks gsm8k \\"
    echo "      --device cuda:0 \\"
    echo "      --batch_size 256"
else
    echo ""
    echo "=========================================="
    echo "✗ 剪枝失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

