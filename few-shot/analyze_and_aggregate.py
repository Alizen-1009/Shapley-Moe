#!/usr/bin/env python3
"""
专家激活分析与聚合 - 一体化脚本

功能：
1. 加载模型和数据集
2. 分析每个样本的专家激活情况
3. 自动聚合统计结果
4. 输出聚合结果到 results 目录（仅保存 aggregated.json 文件）
"""

import torch
import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Dict
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExpertHook:
    """记录专家激活情况的Hook"""

    def __init__(self):
        self.expert_indices = []
        self.expert_weights = []

    def record(self, indices, weights):
        """记录专家索引和权重"""
        self.expert_indices.append(indices.detach().cpu())
        self.expert_weights.append(weights.detach().cpu())

    def clear(self):
        """清空记录"""
        self.expert_indices = []
        self.expert_weights = []


def register_expert_hooks(
    model, hooks: Dict[int, ExpertHook], exclude_shared_experts: bool = True
):
    """
    为模型的每个MoE层注册hook

    Args:
        model: 模型实例
        hooks: 字典，key为层索引，value为ExpertHook实例
        exclude_shared_experts: 是否排除共享专家（默认: True）

    Returns:
        handles: Hook句柄列表
        shared_expert_indices: 共享专家索引列表（如果存在）
    """
    handles = []
    layer_idx = 0
    shared_expert_indices = []

    # 尝试从配置中获取 experts_per_token (top-k) 和共享专家信息
    global_k = None
    if hasattr(model, "config") and hasattr(model.config, "experts_per_token"):
        global_k = model.config.experts_per_token
    elif hasattr(model, "config") and hasattr(model.config, "num_experts_per_tok"):
        global_k = model.config.num_experts_per_tok

    # 尝试获取共享专家索引（DeepSeekV2 等模型）
    if hasattr(model, "config"):
        if hasattr(model.config, "shared_expert_indices"):
            shared_expert_indices = model.config.shared_expert_indices
        elif (
            hasattr(model.config, "num_shared_experts")
            and model.config.num_shared_experts > 0
        ):
            # 通常共享专家是前几个专家
            shared_expert_indices = list(range(model.config.num_shared_experts))
        elif hasattr(model.config, "num_experts"):
            # DeepSeekV2: 共享专家通常是最后一个专家
            num_experts = model.config.num_experts
            if hasattr(model.config, "num_experts_to_route"):
                num_routed = model.config.num_experts_to_route
                if num_routed < num_experts:
                    # 共享专家是路由专家之外的那些
                    shared_expert_indices = list(range(num_routed, num_experts))

    if shared_expert_indices:
        logger.info(f"检测到共享专家索引: {shared_expert_indices}")

    for name, module in model.named_modules():
        # 跳过 expert 子模块（如 experts.0, experts.62, shared_experts 等）
        # 检查路径中是否包含 experts.数字 或 shared_experts
        if (
            ".experts." in name
            or ".shared_experts" in name
            or name.endswith(".experts")
        ):
            continue

        gate_module = None

        # 查找 gate 或 router 模块
        if hasattr(module, "gate") and isinstance(module.gate, torch.nn.Module):
            gate_module = module.gate
        elif hasattr(module, "router") and isinstance(module.router, torch.nn.Module):
            gate_module = module.router
        # DeepSeekV2 可能使用不同的命名
        elif hasattr(module, "gate_proj") and isinstance(
            module.gate_proj, torch.nn.Module
        ):
            # 检查是否是 MoE 层的一部分（但不是 expert 子模块）
            if (
                hasattr(module, "experts")
                or "mlp" in name.lower()
                or "moe" in name.lower()
            ) and ".experts." not in name:
                gate_module = module.gate_proj
        # 检查是否有 experts 属性（MoE 层的标志）
        elif (hasattr(module, "experts") or "moe" in name.lower()) and hasattr(
            module, "gate"
        ):
            if isinstance(module.gate, torch.nn.Module):
                gate_module = module.gate

        # DeepSeekV2 特定检测：查找包含 "mlp" 或 "moe" 的模块，检查其子模块
        if gate_module is None:
            if (
                "mlp" in name.lower() or "moe" in name.lower()
            ) and ".experts." not in name:
                # 检查子模块中是否有 gate 或 router
                for child_name, child_module in module.named_children():
                    if "gate" in child_name.lower() or "router" in child_name.lower():
                        if isinstance(child_module, torch.nn.Module):
                            gate_module = child_module
                            break
                    # 检查是否有 Linear 层作为 gate（DeepSeekV2 可能使用）
                    if isinstance(child_module, torch.nn.Linear) and (
                        "gate" in child_name.lower() or "router" in child_name.lower()
                    ):
                        gate_module = child_module
                        break

        # Fallback: 检查以 'mlp' 结尾的模块（但不是 expert 子模块）
        if gate_module is None and name.endswith("mlp") and ".experts." not in name:
            if hasattr(module, "router") and isinstance(module.router, torch.nn.Module):
                gate_module = module.router
            elif hasattr(module, "gate") and isinstance(module.gate, torch.nn.Module):
                gate_module = module.gate

        if gate_module is not None:
            # 确定 top-k 值
            k_val = getattr(module, "experts_per_token", None)
            if k_val is None:
                k_val = global_k

            if k_val is None:
                logger.warning(f"无法确定 experts_per_token，使用默认值 4")
                k_val = 4

            # 提取层索引
            try:
                parts = name.split(".")
                if "layers" in parts:
                    idx = parts.index("layers")
                    if idx + 1 < len(parts):
                        current_layer_idx = int(parts[idx + 1])
                    else:
                        current_layer_idx = layer_idx
                elif "h" in parts:
                    idx = parts.index("h")
                    if idx + 1 < len(parts):
                        current_layer_idx = int(parts[idx + 1])
                    else:
                        current_layer_idx = layer_idx
                else:
                    current_layer_idx = layer_idx
            except Exception:
                current_layer_idx = layer_idx

            logger.info(f"发现 MoE 层: {name} (Layer {current_layer_idx}, k={k_val})")

            hook_recorder = hooks[current_layer_idx]

            # 创建 hook 函数
            def create_hook(recorder, k, shared_experts, exclude_shared):
                def hook_fn(m, inp, out):
                    try:
                        with torch.no_grad():
                            # 处理不同的输出格式
                            if isinstance(out, tuple):
                                logits = out[0]
                            else:
                                logits = out

                            # 确保 logits 是浮点数类型
                            if not logits.dtype.is_floating_point:
                                logits = logits.float()

                            # 如果排除共享专家，先过滤 logits
                            if exclude_shared and shared_experts:
                                # 将共享专家的 logits 设为很小的值，这样它们不会被选中
                                filtered_logits = logits.clone()
                                for shared_idx in shared_experts:
                                    if shared_idx < filtered_logits.shape[-1]:
                                        filtered_logits[..., shared_idx] = float("-inf")
                                # 使用过滤后的 logits 获取 top-k
                                experts = torch.topk(
                                    filtered_logits, k=k, dim=-1, sorted=True
                                )
                            else:
                                # 获取 top-k 专家
                                experts = torch.topk(logits, k=k, dim=-1, sorted=True)

                            indices = experts.indices
                            # 确保 values 是浮点数类型，然后计算 softmax
                            values = (
                                experts.values.float()
                                if not experts.values.dtype.is_floating_point
                                else experts.values
                            )
                            weights = torch.softmax(values, dim=-1)

                            recorder.record(indices, weights)
                    except Exception as e:
                        logger.error(f"Hook 错误: {e}")

                return hook_fn

            handle = gate_module.register_forward_hook(
                create_hook(
                    hook_recorder, k_val, shared_expert_indices, exclude_shared_experts
                )
            )
            handles.append(handle)

            layer_idx += 1

    if layer_idx == 0:
        logger.warning("未找到 MoE 层！请检查模型结构")
        # 尝试打印模型结构以帮助调试
        logger.info("正在搜索可能的 MoE 相关模块...")
        moe_keywords = ["moe", "expert", "router", "gate", "mlp"]
        found_modules = []
        for name, module in model.named_modules():
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in moe_keywords):
                found_modules.append((name, type(module).__name__))
                if len(found_modules) >= 30:
                    break

        if found_modules:
            logger.info(f"找到 {len(found_modules)} 个可能相关的模块:")
            for name, module_type in found_modules[:20]:
                logger.info(f"  {name}: {module_type}")
                # 检查是否有 gate/router 属性
                try:
                    mod = dict(model.named_modules())[name]
                    attrs = [
                        attr
                        for attr in dir(mod)
                        if not attr.startswith("_")
                        and "gate" in attr.lower()
                        or "router" in attr.lower()
                    ]
                    if attrs:
                        logger.info(f"    可能的属性: {attrs[:5]}")
                except:
                    pass
        else:
            logger.info("未找到包含 'moe', 'expert', 'router', 'gate', 'mlp' 的模块")
            logger.info("模型结构（前30个模块）:")
            for i, (name, module) in enumerate(model.named_modules()):
                if i >= 30:
                    break
                logger.info(f"  {name}: {type(module).__name__}")
    else:
        logger.info(f"共注册 {layer_idx} 个 MoE 层的 hooks")
        if exclude_shared_experts and shared_expert_indices:
            logger.info(f"已配置排除共享专家: {shared_expert_indices}")

    return handles, shared_expert_indices


def remove_hooks(handles):
    """移除所有 hooks"""
    for handle in handles:
        handle.remove()


def analyze_and_aggregate(
    checkpoint: str,
    input_file: str,
    output_dir: str,
    max_new_tokens: int = 512,
    device: str = "auto",
):
    """
    分析专家激活并聚合结果

    Args:
        checkpoint: 模型路径
        input_file: 输入数据文件 (JSON格式)
        output_dir: 输出目录
        max_new_tokens: 最大生成token数
        device: 设备
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    model_name = os.path.basename(checkpoint)

    aggregated_file = os.path.join(
        output_dir, f"{model_name}_{dataset_name}_aggregated.json"
    )

    logger.info("=" * 70)
    logger.info("专家激活分析与聚合")
    logger.info("=" * 70)
    logger.info(f"模型: {checkpoint}")
    logger.info(f"数据: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    # 1. 加载模型
    logger.info("正在加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype="auto", device_map=device, trust_remote_code=True
        )
        logger.info("✓ 模型加载成功")
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {e}")
        return

    # 2. 注册 hooks
    hooks = defaultdict(ExpertHook)
    logger.info("正在注册 hooks...")
    handles, shared_expert_indices = register_expert_hooks(
        model, hooks, exclude_shared_experts=True
    )

    if not handles:
        logger.error("未找到任何 MoE 层，退出")
        return

    # 3. 加载数据
    logger.info(f"正在加载数据: {input_file}")
    prompts = []
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                prompts = json.loads(content)
            else:
                # JSONL 格式
                f.seek(0)
                for line in f:
                    if line.strip():
                        prompts.append(json.loads(line))
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return

    if not prompts:
        logger.error("未找到任何数据")
        return

    logger.info(f"✓ 加载 {len(prompts)} 条数据")

    # 4. 处理每个样本并收集统计
    logger.info("=" * 70)
    logger.info("开始分析专家激活...")
    logger.info("=" * 70)

    aggregated_layers = defaultdict(Counter)  # {layer_idx: Counter({combo: count})}

    for idx, item in enumerate(tqdm(prompts, desc="处理样本"), 1):
        text = item.get("text", "")
        if not text:
            continue

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 清空之前的记录
        for h in hooks.values():
            h.clear()

        # 生成（触发 hooks）
        try:
            with torch.no_grad():
                # 修复 DynamicCache 兼容性问题：不使用 past_key_values
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": False,  # 禁用缓存以避免 DynamicCache 兼容性问题
                }
                # 如果 pad_token_id 未设置，使用 eos_token_id
                if "pad_token_id" not in generate_kwargs:
                    if (
                        hasattr(tokenizer, "pad_token_id")
                        and tokenizer.pad_token_id is not None
                    ):
                        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
                    elif hasattr(tokenizer, "eos_token_id"):
                        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

                model.generate(**generate_kwargs)
        except Exception as e:
            logger.warning(f"样本 {idx} 生成失败: {e}")
            continue

        # 收集统计信息（不保存详细结果）
        for layer_idx, hook in hooks.items():
            if not hook.expert_indices:
                continue

            # 合并所有时间步的专家选择
            # 处理不同大小的张量：逐个处理每个时间步的记录
            for indices_tensor in hook.expert_indices:
                # 确保 indices_tensor 是 2D 的 (batch_size, k)
                if indices_tensor.dim() == 1:
                    indices_tensor = indices_tensor.unsqueeze(0)
                elif indices_tensor.dim() > 2:
                    # 如果是 3D 或更高维，reshape 为 2D
                    indices_tensor = indices_tensor.reshape(
                        -1, indices_tensor.shape[-1]
                    )

                # 统计每个专家组合的出现次数并聚合到全局统计
                for row in indices_tensor:
                    combo = tuple(sorted(row.tolist()))
                    combo_str = str(combo)
                    aggregated_layers[layer_idx][combo_str] += 1

    # 5. 保存聚合结果
    logger.info("正在保存聚合结果...")

    aggregated_data = {
        "model": checkpoint,
        "dataset": input_file,
        "total_samples": len(prompts),
        "total_layers": len(aggregated_layers),
        "layers": {},
    }

    # 转换为可序列化的格式
    for layer_idx, counter in aggregated_layers.items():
        aggregated_data["layers"][str(layer_idx)] = dict(counter)

    with open(aggregated_file, "w", encoding="utf-8") as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ 聚合结果已保存: {aggregated_file}")

    # 6. 清理
    remove_hooks(handles)

    # 7. 打印统计摘要
    logger.info("=" * 70)
    logger.info("分析完成！统计摘要：")
    logger.info("=" * 70)
    logger.info(f"总样本数: {len(prompts)}")
    logger.info(f"总层数: {len(aggregated_layers)}")

    # 打印每层最常见的专家组合
    logger.info("\n每层最常见的专家组合（Top 3）:")
    for layer_idx in sorted(aggregated_layers.keys()):
        top_combos = aggregated_layers[layer_idx].most_common(3)
        logger.info(f"\n  Layer {layer_idx}:")
        for combo_str, count in top_combos:
            logger.info(f"    {combo_str}: {count} 次")

    logger.info("=" * 70)
    logger.info(f"聚合结果: {aggregated_file}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="分析MoE模型的专家激活情况并聚合结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 基础用法
  python analyze_and_aggregate.py \\
      --model /root/yuhao/hf_models/gpt-oss-20b \\
      --data ../dateset/results/gsm8k_25.json

  # 指定输出目录
  python analyze_and_aggregate.py \\
      --model /root/yuhao/hf_models/gpt-oss-20b \\
      --data ../dateset/results/gsm8k_100.json \\
      --output_dir ./results

  # 自定义生成参数
  python analyze_and_aggregate.py \\
      --model /path/to/model \\
      --data ../dateset/results/hellaswag_50.json \\
      --max_new_tokens 256 \\
      --device cuda:0
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True, help="模型路径或 HuggingFace Hub ID"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="输入数据文件 (JSON 格式)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="输出目录（默认: ./results）"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="最大生成token数（默认: 512）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备（auto/cuda/cuda:0/cpu，默认: auto）",
    )

    args = parser.parse_args()

    # 设置默认输出目录
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, "results")

    # 检查输入文件
    if not os.path.exists(args.data):
        logger.error(f"输入文件不存在: {args.data}")
        return

    # 运行分析
    analyze_and_aggregate(
        checkpoint=args.model,
        input_file=args.data,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
