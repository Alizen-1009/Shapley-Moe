#!/usr/bin/env python3
"""
MoE 模型专家剪枝工具

功能：
1. 加载原始模型
2. 根据 expert selection JSON 文件，修改 Gate/Router 使其不再选择被剪掉的专家
3. 可选：将未选中的专家权重置零（节省存储空间）
4. 保存为 safetensor 格式的完整模型

剪枝策略：
- gate_bias: 给被剪掉的专家在 gate 层添加大负偏置，使其不被选中（推荐）
- zero_weights: 将被剪掉的专家权重置零（旧方法，可能导致问题）
- both: 同时使用两种策略
"""

import torch
import json
import os
import argparse
import logging
from typing import Dict, List, Set, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Gate 偏置的大负数值，使被剪掉的专家选择概率接近 0
GATE_BIAS_VALUE = -1e9


class ModelPruner:
    """MoE 模型专家剪枝工具"""

    def __init__(
        self,
        model_path: str,
        selection_json_path: str,
        output_dir: str,
        device_map: str = "auto",
        pruning_strategy: str = "gate_bias",
    ):
        """
        Args:
            model_path: 原始模型路径
            selection_json_path: 专家选择 JSON 文件路径
            output_dir: 输出模型保存目录
            device_map: 设备映射策略
            pruning_strategy: 剪枝策略
                - "gate_bias": 修改 gate 偏置（推荐，确保被剪掉的专家不会被选中）
                - "zero_weights": 将专家权重置零（旧方法）
                - "both": 同时使用两种策略
        """
        self.model_path = model_path
        self.selection_json_path = selection_json_path
        self.output_dir = output_dir
        self.device_map = device_map
        self.pruning_strategy = pruning_strategy
        self.model = None
        self.tokenizer = None
        self.selected_experts: Dict[str, List[int]] = {}
        
        # 统计信息
        self.stats = {
            "gate_modified_layers": 0,
            "zeroed_experts": 0,
            "zeroed_params": 0,
        }

    def load_selection_file(self) -> Dict[str, List[int]]:
        """加载专家选择文件"""
        logger.info(f"加载专家选择文件: {self.selection_json_path}")
        with open(self.selection_json_path, "r") as f:
            data = json.load(f)

        # 打印统计信息
        total_selected = sum(len(experts) for experts in data.values())
        total_layers = len(data)
        avg_per_layer = total_selected / total_layers if total_layers > 0 else 0
        logger.info(f"共 {total_layers} 层，选中 {total_selected} 个专家 (平均每层 {avg_per_layer:.1f} 个)")

        return data

    def load_model(self):
        """加载模型和 tokenizer"""
        logger.info(f"加载模型: {self.model_path}")
        logger.info(f"设备映射策略: {self.device_map}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
        )
        
        # 显示设备分配情况
        if hasattr(self.model, 'hf_device_map'):
            devices_used = set(str(v) for v in self.model.hf_device_map.values())
            logger.info(f"模型分布在设备: {devices_used}")
        
        logger.info("模型加载成功")

    def _find_gate_module(self, moe_module) -> Optional[torch.nn.Module]:
        """查找 MoE 模块中的 Gate/Router 模块"""
        # 常见的 gate 属性名
        gate_names = ["gate", "router", "gate_proj", "wg"]
        
        for name in gate_names:
            if hasattr(moe_module, name):
                gate = getattr(moe_module, name)
                # 确保是一个有参数的模块
                if isinstance(gate, torch.nn.Module):
                    return gate
        
        return None

    def _get_num_experts(self, moe_module) -> Optional[int]:
        """获取 MoE 模块中的专家数量"""
        # 尝试从各种属性获取
        attr_names = ["num_experts", "n_routed_experts", "num_local_experts"]
        
        for attr in attr_names:
            if hasattr(moe_module, attr):
                return getattr(moe_module, attr)
        
        # 尝试从 experts 模块获取
        experts = None
        if hasattr(moe_module, "experts"):
            experts = moe_module.experts
        elif hasattr(moe_module, "routed_experts"):
            experts = moe_module.routed_experts
        
        if experts is not None:
            if hasattr(experts, "num_experts"):
                return experts.num_experts
            elif isinstance(experts, torch.nn.ModuleList):
                return len(experts)
        
        return None

    def _modify_gate_bias(self, gate_module: torch.nn.Module, 
                          num_experts: int,
                          selected_indices: Set[int],
                          layer_idx: int) -> bool:
        """
        修改 Gate 模块的偏置，使被剪掉的专家不被选中
        
        策略：给被剪掉专家对应的 gate 输出添加一个很大的负偏置
        这样 softmax/top-k 时这些专家的分数会非常低，不会被选中
        """
        unselected_indices = [i for i in range(num_experts) if i not in selected_indices]
        
        if not unselected_indices:
            logger.info(f"Layer {layer_idx}: 所有专家都被选中，无需修改 gate")
            return True
        
        modified = False
        
        # 方法 1: 修改 Linear 层的 bias
        if isinstance(gate_module, torch.nn.Linear):
            with torch.no_grad():
                # 如果没有 bias，创建一个
                if gate_module.bias is None:
                    gate_module.bias = torch.nn.Parameter(
                        torch.zeros(gate_module.out_features, 
                                   device=gate_module.weight.device,
                                   dtype=gate_module.weight.dtype)
                    )
                
                # 给被剪掉的专家添加大负偏置
                for idx in unselected_indices:
                    if idx < gate_module.bias.size(0):
                        gate_module.bias.data[idx] = GATE_BIAS_VALUE
                
                modified = True
                logger.info(f"Layer {layer_idx}: 修改 gate.bias，屏蔽 {len(unselected_indices)} 个专家")
        
        # 方法 2: 尝试查找 weight 参数并修改对应列（备用）
        elif hasattr(gate_module, "weight") and isinstance(gate_module.weight, torch.nn.Parameter):
            # 对于某些模型，可能需要修改 weight 而不是 bias
            # 但这种方法风险较大，仅作为备用
            with torch.no_grad():
                weight = gate_module.weight
                # 假设 weight 的形状是 [num_experts, hidden_size]
                if weight.dim() == 2 and weight.size(0) == num_experts:
                    # 将被剪掉专家的权重设为很小的值
                    for idx in unselected_indices:
                        weight.data[idx] = weight.data[idx] * 0.0 - 1e6
                    modified = True
                    logger.info(f"Layer {layer_idx}: 修改 gate.weight，屏蔽 {len(unselected_indices)} 个专家")
        
        if not modified:
            logger.warning(f"Layer {layer_idx}: 无法修改 gate 模块 (类型: {type(gate_module).__name__})")
        
        return modified

    def modify_gates(self):
        """修改所有 MoE 层的 gate，使被剪掉的专家不被选中"""
        logger.info("开始修改 Gate/Router 模块...")
        
        modified_layers = 0
        
        for name, module in self.model.named_modules():
            layer_idx = self._extract_layer_index(name)
            
            if layer_idx is None or str(layer_idx) not in self.selected_experts:
                continue
            
            # 检查是否是 MoE 模块
            is_moe = (hasattr(module, "experts") or 
                     hasattr(module, "routed_experts") or
                     hasattr(module, "gate"))
            
            if not is_moe:
                continue
            
            # 获取 gate 模块
            gate_module = self._find_gate_module(module)
            if gate_module is None:
                continue
            
            # 获取专家数量
            num_experts = self._get_num_experts(module)
            if num_experts is None:
                logger.warning(f"Layer {layer_idx}: 无法确定专家数量")
                continue
            
            # 获取选中的专家
            selected_indices = set(self.selected_experts[str(layer_idx)])
            
            # 修改 gate
            if self._modify_gate_bias(gate_module, num_experts, selected_indices, layer_idx):
                modified_layers += 1
        
        self.stats["gate_modified_layers"] = modified_layers
        logger.info(f"Gate 修改完成! 共修改 {modified_layers} 层")

    def zero_out_experts(self):
        """将未选中的专家权重置零"""
        logger.info("开始将未选中的专家权重置零...")

        total_zeroed_experts = 0
        total_zeroed_params = 0

        for name, module in self.model.named_modules():
            layer_idx = self._extract_layer_index(name)

            if layer_idx is None or str(layer_idx) not in self.selected_experts:
                continue

            # 查找 experts 模块（支持多种模型架构）
            experts = None
            if hasattr(module, "experts"):
                experts = module.experts
            elif hasattr(module, "routed_experts"):
                experts = module.routed_experts
            
            if experts is None:
                continue

            selected_indices = set(self.selected_experts[str(layer_idx)])

            # 检查是否是打包权重类型（如 GptOssExperts）
            if hasattr(experts, "num_experts"):
                num_experts = experts.num_experts
                unselected_indices = [
                    i for i in range(num_experts) if i not in selected_indices
                ]

                if not unselected_indices:
                    continue

                # 置零专家权重参数
                expert_params = [
                    "gate_up_proj", "down_proj",
                    "gate_up_proj_bias", "down_proj_bias",
                    "gate_proj", "up_proj",  # 某些模型使用这些名称
                    "w1", "w2", "w3",  # Llama 风格
                ]

                zeroed_params = 0
                for param_name in expert_params:
                    if hasattr(experts, param_name):
                        param = getattr(experts, param_name)
                        if param is not None and isinstance(param, torch.nn.Parameter):
                            with torch.no_grad():
                                param.data[unselected_indices] = 0.0
                            zeroed_params += 1

                if zeroed_params > 0:
                    total_zeroed_experts += len(unselected_indices)
                    total_zeroed_params += zeroed_params
                    logger.info(
                        f"Layer {layer_idx}: 置零 {len(unselected_indices)} 个专家, "
                        f"处理 {zeroed_params} 个参数"
                    )

            # 检查是否是 ModuleList 类型
            elif isinstance(experts, torch.nn.ModuleList):
                num_experts = len(experts)
                unselected_indices = [
                    i for i in range(num_experts) if i not in selected_indices
                ]

                if not unselected_indices:
                    continue

                zeroed_experts_in_layer = 0
                zeroed_params_in_layer = 0

                for expert_idx in unselected_indices:
                    if expert_idx < len(experts):
                        expert = experts[expert_idx]
                        for param_name, param in expert.named_parameters():
                            if isinstance(param, torch.nn.Parameter):
                                with torch.no_grad():
                                    param.data.zero_()
                                zeroed_params_in_layer += 1
                        zeroed_experts_in_layer += 1

                if zeroed_experts_in_layer > 0:
                    logger.info(
                        f"Layer {layer_idx}: 置零 {zeroed_experts_in_layer} 个专家, "
                        f"处理 {zeroed_params_in_layer} 个参数"
                    )
                    total_zeroed_experts += zeroed_experts_in_layer
                    total_zeroed_params += zeroed_params_in_layer

        self.stats["zeroed_experts"] = total_zeroed_experts
        self.stats["zeroed_params"] = total_zeroed_params
        logger.info(f"权重置零完成! 共置零 {total_zeroed_experts} 个专家")

    def _extract_layer_index(self, module_name: str) -> Optional[int]:
        """从模块名称中提取层索引"""
        parts = module_name.split(".")

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
        """保存剪枝后的模型为 safetensor 格式"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")

        logger.info(f"保存剪枝后的模型到: {self.output_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(
            self.output_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        logger.info("模型权重已保存 (safetensor 格式)")

        # 保存 tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Tokenizer 已保存")

        # 保存剪枝信息
        pruning_info = {
            "original_model": self.model_path,
            "selection_file": self.selection_json_path,
            "pruning_strategy": self.pruning_strategy,
            "selected_experts": self.selected_experts,
            "total_layers": len(self.selected_experts),
            "total_selected_experts": sum(
                len(experts) for experts in self.selected_experts.values()
            ),
            "stats": self.stats,
        }

        info_path = os.path.join(self.output_dir, "pruning_info.json")
        with open(info_path, "w") as f:
            json.dump(pruning_info, f, indent=2)
        logger.info(f"剪枝信息已保存到: {info_path}")

    def run(self):
        """执行完整的剪枝流程"""
        logger.info("=" * 70)
        logger.info("开始 MoE 模型专家剪枝")
        logger.info(f"剪枝策略: {self.pruning_strategy}")
        logger.info("=" * 70)

        # 1. 加载专家选择
        self.selected_experts = self.load_selection_file()

        # 2. 加载模型
        self.load_model()

        # 3. 根据策略执行剪枝
        if self.pruning_strategy in ["gate_bias", "both"]:
            self.modify_gates()
        
        if self.pruning_strategy in ["zero_weights", "both"]:
            self.zero_out_experts()

        # 4. 保存模型
        self.save_model()

        logger.info("=" * 70)
        logger.info("剪枝完成!")
        logger.info("=" * 70)
        logger.info(f"剪枝后的模型已保存到: {self.output_dir}")
        
        if self.pruning_strategy == "gate_bias":
            logger.info("✓ 使用 gate_bias 策略：被剪掉的专家将不会被 router 选中")
        elif self.pruning_strategy == "zero_weights":
            logger.info("⚠ 使用 zero_weights 策略：专家权重已置零，但 router 仍可能选中它们")
        else:
            logger.info("✓ 使用 both 策略：gate 已修改且权重已置零")


def main():
    parser = argparse.ArgumentParser(
        description="MoE 模型专家剪枝工具 - 支持多种剪枝策略"
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
        help="专家选择 JSON 文件路径",
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
        help="设备映射策略: auto, balanced, cuda:0, cpu",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="gate_bias",
        choices=["gate_bias", "zero_weights", "both"],
        help="剪枝策略: gate_bias(推荐), zero_weights, both",
    )
    # 兼容旧参数
    parser.add_argument("--device", type=str, default=None, help="(已废弃)")

    args = parser.parse_args()
    
    device_map = args.device_map
    if args.device is not None:
        logger.warning("--device 参数已废弃，请使用 --device_map")
        device_map = args.device

    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return

    if not os.path.exists(args.selection_file):
        logger.error(f"专家选择文件不存在: {args.selection_file}")
        return

    pruner = ModelPruner(
        model_path=args.model_path,
        selection_json_path=args.selection_file,
        output_dir=args.output_dir,
        device_map=device_map,
        pruning_strategy=args.strategy,
    )

    pruner.run()


if __name__ == "__main__":
    main()
