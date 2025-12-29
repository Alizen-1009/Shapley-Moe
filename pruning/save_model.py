#!/usr/bin/env python3
"""
将MoE模型中未选中的专家权重置零并保存为safetensor格式

功能：
1. 加载原始模型
2. 根据expert selection JSON文件，将未选中的专家权重置零
3. 保存为safetensor格式的完整模型
4. 保存的模型可直接用于lm-evaluation-harness等评测工具
"""

import torch
import json
import os
import argparse
import logging
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelPruner:
    """MoE模型专家剪枝工具"""

    def __init__(
        self,
        model_path: str,
        selection_json_path: str,
        output_dir: str,
        device_map: str = "auto",
    ):
        """
        Args:
            model_path: 原始模型路径
            selection_json_path: 专家选择JSON文件路径
            output_dir: 输出模型保存目录
            device_map: 设备映射策略
                - "auto": 自动分配到可用GPU（支持多卡TP）
                - "balanced": 均匀分配到所有GPU
                - "cuda:0": 仅使用指定单卡
                - "cpu": 仅使用CPU
        """
        self.model_path = model_path
        self.selection_json_path = selection_json_path
        self.output_dir = output_dir
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self.selected_experts = None

    def load_selection_file(self) -> Dict[str, List[int]]:
        """加载专家选择文件"""
        logger.info(f"加载专家选择文件: {self.selection_json_path}")
        with open(self.selection_json_path, "r") as f:
            data = json.load(f)

        # 打印统计信息
        total_selected = sum(len(experts) for experts in data.values())
        total_layers = len(data)
        logger.info(f"共 {total_layers} 层，选中 {total_selected} 个专家")

        return data

    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"加载模型: {self.model_path}")
        logger.info(f"设备映射策略: {self.device_map}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型（支持多卡TP）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device_map,  # auto/balanced 会自动分配到多卡
            trust_remote_code=True,
        )
        
        # 显示设备分配情况
        if hasattr(self.model, 'hf_device_map'):
            devices_used = set(str(v) for v in self.model.hf_device_map.values())
            logger.info(f"模型分布在设备: {devices_used}")
        
        logger.info("模型加载成功")

    def zero_out_experts(self):
        """将未选中的专家权重置零"""
        logger.info("开始将未选中的专家权重置零...")

        total_zeroed_experts = 0
        total_zeroed_params = 0

        for name, module in self.model.named_modules():
            # 提取层索引
            layer_idx = self._extract_layer_index(name)

            if layer_idx is None or str(layer_idx) not in self.selected_experts:
                continue

            # 查找experts模块
            if hasattr(module, "experts"):
                experts = module.experts

                # 检查是否是GptOssExperts类型（权重打包在张量中）
                if hasattr(experts, "num_experts"):
                    num_experts = experts.num_experts
                    selected_indices = set(self.selected_experts[str(layer_idx)])
                    unselected_indices = [
                        i for i in range(num_experts) if i not in selected_indices
                    ]

                    if not unselected_indices:
                        logger.info(f"Layer {layer_idx}: 所有专家都被选中，跳过")
                        continue

                    # 置零专家权重参数
                    expert_params = [
                        "gate_up_proj",
                        "down_proj",
                        "gate_up_proj_bias",
                        "down_proj_bias",
                    ]

                    zeroed_params = 0
                    for param_name in expert_params:
                        if hasattr(experts, param_name):
                            param = getattr(experts, param_name)
                            if param is not None and isinstance(
                                param, torch.nn.Parameter
                            ):
                                with torch.no_grad():
                                    # 第一维是expert索引，置零未选中的专家
                                    param.data[unselected_indices] = 0.0
                                zeroed_params += 1

                    total_zeroed_experts += len(unselected_indices)
                    total_zeroed_params += zeroed_params

                    logger.info(
                        f"Layer {layer_idx}: 置零 {len(unselected_indices)} 个专家 "
                        f"(保留 {len(selected_indices)} 个), 处理 {zeroed_params} 个参数"
                    )

                # 检查是否是ModuleList类型（OLMOE等模型）
                elif isinstance(experts, torch.nn.ModuleList):
                    num_experts = len(experts)
                    selected_indices = set(self.selected_experts[str(layer_idx)])
                    unselected_indices = [
                        i for i in range(num_experts) if i not in selected_indices
                    ]

                    if not unselected_indices:
                        logger.info(f"Layer {layer_idx}: 所有专家都被选中，跳过")
                        continue

                    # 对于ModuleList，每个专家是独立的模块
                    zeroed_experts_in_layer = 0
                    zeroed_params_in_layer = 0

                    for expert_idx in unselected_indices:
                        if expert_idx < len(experts):
                            expert = experts[expert_idx]

                            # 置零专家的所有参数
                            for param_name, param in expert.named_parameters():
                                if param is not None and isinstance(
                                    param, torch.nn.Parameter
                                ):
                                    with torch.no_grad():
                                        param.data.zero_()
                                    zeroed_params_in_layer += 1

                            zeroed_experts_in_layer += 1

                    if zeroed_experts_in_layer > 0:
                        logger.info(
                            f"Layer {layer_idx}: 置零 {zeroed_experts_in_layer} 个专家 "
                            f"(保留 {len(selected_indices)} 个), 处理 {zeroed_params_in_layer} 个参数"
                        )
                        total_zeroed_experts += zeroed_experts_in_layer
                        total_zeroed_params += zeroed_params_in_layer
                    else:
                        logger.warning(f"Layer {layer_idx}: 未找到可置零的专家")

                else:
                    logger.warning(
                        f"Layer {layer_idx}: 不支持的专家模块类型: {type(experts)}"
                    )

        logger.info(
            f"完成! 共置零 {total_zeroed_experts} 个专家, "
            f"处理 {total_zeroed_params} 个参数张量"
        )

    def _extract_layer_index(self, module_name: str) -> int:
        """从模块名称中提取层索引"""
        parts = module_name.split(".")

        # 常见模式: model.layers.0.mlp 或 transformer.h.0.mlp
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts):
                try:
                    return int(parts[idx + 1])
                except ValueError:
                    pass

        if "h" in parts:
            idx = parts.index("h")
            if idx + 1 < len(parts):
                try:
                    return int(parts[idx + 1])
                except ValueError:
                    pass

        return None

    def save_model(self):
        """保存剪枝后的模型为safetensor格式"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")

        logger.info(f"保存剪枝后的模型到: {self.output_dir}")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存模型 (safetensor格式)
        self.model.save_pretrained(
            self.output_dir,
            safe_serialization=True,  # 使用safetensor格式
            max_shard_size="5GB",  # 大模型分片大小
        )
        logger.info("模型权重已保存 (safetensor格式)")

        # 保存tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Tokenizer已保存")

        # 保存剪枝信息
        pruning_info = {
            "original_model": self.model_path,
            "selection_file": self.selection_json_path,
            "selected_experts": self.selected_experts,
            "total_layers": len(self.selected_experts),
            "total_selected_experts": sum(
                len(experts) for experts in self.selected_experts.values()
            ),
        }

        info_path = os.path.join(self.output_dir, "pruning_info.json")
        with open(info_path, "w") as f:
            json.dump(pruning_info, f, indent=2)
        logger.info(f"剪枝信息已保存到: {info_path}")

    def run(self):
        """执行完整的剪枝流程"""
        logger.info("=" * 70)
        logger.info("开始MoE模型专家剪枝")
        logger.info("=" * 70)

        # 1. 加载专家选择
        self.selected_experts = self.load_selection_file()

        # 2. 加载模型
        self.load_model()

        # 3. 置零未选中的专家
        self.zero_out_experts()

        # 4. 保存模型
        self.save_model()

        logger.info("=" * 70)
        logger.info("剪枝完成!")
        logger.info("=" * 70)
        logger.info(f"剪枝后的模型已保存到: {self.output_dir}")
        logger.info("可以直接使用此模型进行评测或推理")


def main():
    parser = argparse.ArgumentParser(
        description="将MoE模型中未选中的专家权重置零并保存为safetensor格式"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="原始模型路径",
    )
    parser.add_argument(
        "--selection_file",
        type=str,
        required=True,
        help="专家选择JSON文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出模型保存目录",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="设备映射策略: auto(多卡自动分配), balanced(均匀分配), cuda:0(单卡), cpu",
    )
    # 兼容旧参数
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="(已废弃，请使用 --device_map)",
    )

    args = parser.parse_args()
    
    # 兼容旧的 --device 参数
    device_map = args.device_map
    if args.device is not None:
        logger.warning("--device 参数已废弃，请使用 --device_map")
        device_map = args.device

    # 检查输入文件是否存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return

    if not os.path.exists(args.selection_file):
        logger.error(f"专家选择文件不存在: {args.selection_file}")
        return

    # 执行剪枝
    pruner = ModelPruner(
        model_path=args.model_path,
        selection_json_path=args.selection_file,
        output_dir=args.output_dir,
        device_map=device_map,
    )

    pruner.run()


if __name__ == "__main__":
    main()
