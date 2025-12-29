#!/usr/bin/env python3
"""
一次 Few-Shot 收集所有剪枝方法需要的信息

功能：
在一次推理过程中同时收集：
1. 专家组合激活次数（用于 Shapley 值计算）- 保存到 results_shapley/
2. 所有专家的 Gating Score（用于 Gating Score 剪枝）- 保存到 results_gating/
3. EASYEP 得分（weight × (1 - simibr) × norm）- 保存到 results_easyep/
4. REAP 得分（router_weight × expert_activation_norm）- 保存到 results_reap/

特点：
- 一次推理，多种输出
- 结果保存到不同目录，不会覆盖已有结果
- 支持跳过已存在的结果文件

EASYEP 原始公式：
    score[layer, expert_id] += weight[i] × (1 - simibr) × norm[i]
    
    其中:
    - weight: router softmax 权重
    - simibr: cos_sim(x_before_moe, x_after_rmoe) - MoE输入与routed输出的余弦相似度
    - norm: 专家输出的 L2 范数
"""

import torch
import torch.nn.functional as F
import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import logging
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComprehensiveExpertHook:
    """
    综合专家激活记录 Hook
    
    同时收集：
    - 基础信息：top-k 专家索引和权重
    - Gating Score：所有专家的 softmax 分数
    - EASYEP 需要的信息：专家输出范数、输入输出相似度
    - REAP 需要的信息：加权范数
    """

    def __init__(self):
        # 基础信息
        self.expert_indices = []      # top-k 专家索引
        self.expert_weights = []      # top-k 专家权重（softmax 后）
        self.all_gating_scores = []   # 所有专家的 gating score
        
        # EASYEP/REAP 需要的信息
        self.expert_norms = []        # 每个选中专家的输出 L2 范数 (total_tokens, k)
        self.simibr = []              # 输入输出余弦相似度 (total_tokens,)
        
        # 中间状态（用于计算 simibr）
        self.hidden_before_moe = None
        self.routed_output = None
        
        # Gate hook 捕获的输出
        self._gate_output = None
        self._gate_input = None

    def record_gate_info(self, indices, weights, all_scores):
        """记录 gate 信息"""
        self.expert_indices.append(indices.detach().cpu())
        self.expert_weights.append(weights.detach().cpu())
        self.all_gating_scores.append(all_scores.detach().cpu())

    def record_expert_norms(self, norms):
        """记录专家输出范数"""
        self.expert_norms.append(norms.detach().cpu())

    def record_simibr(self, sim):
        """记录输入输出相似度"""
        self.simibr.append(sim.detach().cpu())

    def set_hidden_before_moe(self, hidden):
        """设置 MoE 输入隐状态"""
        self.hidden_before_moe = hidden.detach()

    def set_routed_output(self, output):
        """设置 routed MoE 输出"""
        self.routed_output = output.detach()

    def clear(self):
        """清空记录"""
        self.expert_indices = []
        self.expert_weights = []
        self.all_gating_scores = []
        self.expert_norms = []
        self.simibr = []
        self.hidden_before_moe = None
        self.routed_output = None
        self._gate_output = None
        self._gate_input = None


def find_moe_layers(model):
    """
    查找模型中的 MoE 层
    
    Returns:
        moe_info: list of (layer_idx, moe_module, gate_module, experts_module)
    """
    moe_info = []
    
    for name, module in model.named_modules():
        # 跳过 expert 子模块
        if ".experts." in name or ".shared_experts" in name or name.endswith(".experts"):
            continue
        
        gate_module = None
        experts_module = None
        
        # 检查是否有 gate 和 experts（支持多种命名方式）
        # 1. 标准命名: gate + experts
        if hasattr(module, "gate") and hasattr(module, "experts"):
            gate_module = module.gate
            experts_module = module.experts
        # 2. router + experts
        elif hasattr(module, "router") and hasattr(module, "experts"):
            gate_module = module.router
            experts_module = module.experts
        # 3. DeepSeek V2: gate + routed_experts
        elif hasattr(module, "gate") and hasattr(module, "routed_experts"):
            gate_module = module.gate
            experts_module = module.routed_experts
        # 4. DeepSeek V2 变体: 检查是否有 forward 中使用的 moe 组件
        elif hasattr(module, "gate") and hasattr(module, "ep_size"):
            # 这是 DeepSeek V2 的 MoE 层，但 experts 可能以不同方式组织
            gate_module = module.gate
            # 尝试获取 experts
            if hasattr(module, "experts"):
                experts_module = module.experts
            elif hasattr(module, "w1") and hasattr(module, "w2"):
                # DeepSeek V2 使用 w1, w2, w3 作为专家权重
                experts_module = module  # 使用整个模块作为 experts_module
        
        if gate_module is not None and experts_module is not None:
            # 提取层索引
            try:
                parts = name.split(".")
                if "layers" in parts:
                    idx = parts.index("layers")
                    layer_idx = int(parts[idx + 1])
                elif "h" in parts:
                    idx = parts.index("h")
                    layer_idx = int(parts[idx + 1])
                else:
                    layer_idx = len(moe_info)
            except Exception:
                layer_idx = len(moe_info)
            
            moe_info.append((layer_idx, module, gate_module, experts_module, name))
    
    return moe_info


def register_comprehensive_hooks(
    model, 
    hooks: Dict[int, ComprehensiveExpertHook],
    exclude_shared_experts: bool = True
):
    """
    为模型注册综合 hook，收集 EASYEP/REAP 所需的所有信息
    
    Args:
        model: 模型实例
        hooks: 字典，key为层索引，value为ComprehensiveExpertHook实例
        exclude_shared_experts: 是否排除共享专家
        
    Returns:
        handles: Hook 句柄列表
        moe_info: MoE 层信息列表
    """
    handles = []
    
    # 获取配置信息
    config = model.config
    global_k = getattr(config, 'experts_per_token', None) or \
               getattr(config, 'num_experts_per_tok', None) or 4
    global_num_experts = getattr(config, 'num_experts', None) or \
                        getattr(config, 'n_routed_experts', None) or \
                        getattr(config, 'num_local_experts', None)
    
    # 获取共享专家索引
    shared_expert_indices = []
    if hasattr(config, "shared_expert_indices"):
        shared_expert_indices = config.shared_expert_indices
    elif hasattr(config, "num_shared_experts") and config.num_shared_experts > 0:
        shared_expert_indices = list(range(config.num_shared_experts))
    
    if shared_expert_indices:
        logger.info(f"检测到共享专家索引: {shared_expert_indices}")

    # 查找 MoE 层
    moe_layers = find_moe_layers(model)
    
    if not moe_layers:
        logger.warning("未找到 MoE 层，尝试通用方式注册 hook...")
        return register_gate_only_hooks(model, hooks, exclude_shared_experts)
    
    logger.info(f"找到 {len(moe_layers)} 个 MoE 层")
    
    for layer_idx, moe_module, gate_module, experts_module, name in moe_layers:
        logger.info(f"  注册 MoE 层: {name} (Layer {layer_idx})")
        
        hook_recorder = hooks[layer_idx]
        
        # 获取该层的 top-k 和专家数量
        k_val = getattr(moe_module, 'experts_per_token', None) or \
               getattr(moe_module, 'num_experts_per_tok', None) or \
               getattr(moe_module, 'topk', None) or \
               getattr(moe_module, 'top_k', None) or global_k
        
        # 获取专家数量（多种属性名）
        num_experts = getattr(moe_module, 'num_experts', None) or \
                     getattr(moe_module, 'n_routed_experts', None) or \
                     getattr(moe_module, 'num_local_experts', None) or \
                     global_num_experts
        
        # 如果还是没有，尝试从 experts_module 获取
        if num_experts is None:
            try:
                num_experts = len(experts_module)
            except:
                num_experts = 64  # 默认值
        
        logger.info(f"    Gate: {type(gate_module).__name__}, Experts: {type(experts_module).__name__}, k={k_val}, n_experts={num_experts}")
        
        # ============= Hook 1: MoE 模块的前向钩子（记录输入）=============
        def create_moe_pre_hook(recorder):
            def hook_fn(module, args):
                # 记录 MoE 输入
                if isinstance(args, tuple) and len(args) > 0:
                    hidden = args[0]
                    recorder.set_hidden_before_moe(hidden)
            return hook_fn
        
        handle_pre = moe_module.register_forward_pre_hook(create_moe_pre_hook(hook_recorder))
        handles.append(handle_pre)
        
        # ============= Hook 1.5: Gate 模块的输出钩子（捕获 gate 结果）=============
        # 这对于 DeepSeek V2 等模型非常重要，因为 gate 在 MoE forward 中被调用
        def create_gate_output_hook(recorder, n_experts, layer_id):
            logged = [False]  # 只打印一次
            def hook_fn(module, inp, out):
                try:
                    with torch.no_grad():
                        # 保存 gate 输出供后续使用
                        recorder._gate_output = out
                        recorder._gate_input = inp[0] if inp else None
                        
                        # 只在第一层首次调用时打印详细信息（减少日志）
                        if not logged[0] and layer_id == 1:
                            logged[0] = True
                            if isinstance(out, tuple):
                                out_info = f"tuple({len(out)}): " + ", ".join([f"{type(o).__name__}{list(o.shape) if hasattr(o, 'shape') else ''}" for o in out[:3]])
                            elif hasattr(out, 'shape'):
                                out_info = f"Tensor{list(out.shape)}"
                            else:
                                out_info = type(out).__name__
                            logger.info(f"  [Gate 输出格式] {out_info}")
                except Exception as e:
                    if not logged[0]:
                        logged[0] = True
                        logger.warning(f"Gate hook error (L{layer_id}): {e}")
            return hook_fn
        
        handle_gate = gate_module.register_forward_hook(create_gate_output_hook(hook_recorder, num_experts, layer_idx))
        handles.append(handle_gate)
        
        # ============= Hook 2: MoE 模块的后向钩子（计算专家范数和相似度）=============
        def create_moe_post_hook(recorder, k, n_experts, experts_mod, gate_mod, shared_experts, exclude_shared, moe_mod, layer_id):
            logged = [False]  # 只打印一次
            
            def hook_fn(module, args, output):
                try:
                    with torch.no_grad():
                        # 获取输入
                        if isinstance(args, tuple) and len(args) > 0:
                            hidden = args[0]
                        else:
                            hidden = recorder.hidden_before_moe
                        
                        if hidden is None:
                            if not logged[0] and layer_id == 1:
                                logged[0] = True
                                logger.warning(f"  [MoE Hook] hidden is None, skipping")
                            return
                        
                        device = hidden.device
                        
                        # 确保 hidden 是 2D
                        if hidden.dim() == 3:
                            batch_size, seq_len, hidden_dim = hidden.shape
                            flat_hidden = hidden.view(-1, hidden_dim)
                        else:
                            flat_hidden = hidden
                            batch_size, seq_len = 1, hidden.shape[0]
                            hidden_dim = hidden.shape[-1]
                        
                        total_tokens = flat_hidden.shape[0]
                        
                        # 获取 gate 输出（优先使用 hook 捕获的结果）
                        gate_output = None
                        gate_method = None
                        
                        # 方法 1: 使用 gate hook 捕获的输出（最可靠）
                        if hasattr(recorder, '_gate_output') and recorder._gate_output is not None:
                            gate_output = recorder._gate_output
                            gate_method = "hook_capture"
                        
                        # 方法 2: 直接调用 gate
                        if gate_output is None:
                            try:
                                gate_output = gate_mod(flat_hidden)
                                gate_method = "direct_call"
                            except Exception as e:
                                pass  # 静默失败，尝试下一个方法
                        
                        # 方法 3: 如果 gate 调用失败，尝试使用 gate.weight 手动计算
                        if gate_output is None and hasattr(gate_mod, 'weight'):
                            try:
                                gate_output = F.linear(flat_hidden.float(), gate_mod.weight.float(), 
                                                      getattr(gate_mod, 'bias', None))
                                gate_method = "manual_linear"
                            except Exception as e:
                                pass  # 静默失败
                        
                        # 方法 4: 如果 MoE 模块有 topk_indices/topk_weight 属性（某些模型缓存了结果）
                        if gate_output is None:
                            if hasattr(moe_mod, 'topk_idx') and hasattr(moe_mod, 'topk_weight'):
                                try:
                                    topk_indices = moe_mod.topk_idx
                                    topk_weights = moe_mod.topk_weight
                                    gate_output = (topk_indices, topk_weights)
                                    gate_method = "moe_cache"
                                except:
                                    pass
                        
                        if gate_output is None:
                            if not logged[0] and layer_id == 1:
                                logged[0] = True
                                logger.warning(f"  [MoE Hook] All gate methods failed")
                            return
                        
                        # 只在第一层首次成功时打印信息
                        if not logged[0] and layer_id == 1:
                            logger.info(f"  [MoE Hook] Gate method: {gate_method}, hidden: {list(flat_hidden.shape)}")
                        
                        # 清理捕获的 gate 输出
                        recorder._gate_output = None
                        
                        # 处理不同模型的 gate 输出格式
                        topk_indices = None
                        topk_weights = None
                        all_gating_scores = None
                        
                        if isinstance(gate_output, tuple):
                            # 检查第一个元素的类型来判断格式
                            first_elem = gate_output[0]
                            
                            if first_elem.dtype in [torch.int32, torch.int64, torch.long]:
                                # DeepSeekV2 等模型：gate 直接返回 (topk_idx, topk_weight, ...)
                                topk_indices = first_elem
                                topk_weights = gate_output[1].float()
                                
                                # 确保 k 值正确
                                actual_k = topk_indices.shape[-1]
                                
                                # 尝试获取原始 logits 用于 all_gating_scores
                                if hasattr(gate_mod, 'weight'):
                                    gate_logits = F.linear(flat_hidden.float(), gate_mod.weight.float(), None)
                                    all_gating_scores = torch.softmax(gate_logits, dim=-1)
                                else:
                                    # 无法获取完整 scores，使用 topk 结果构造
                                    all_gating_scores = torch.zeros(total_tokens, n_experts, device=device)
                                    all_gating_scores.scatter_(1, topk_indices, topk_weights)
                            else:
                                # tuple 但第一个元素是浮点数，取第一个作为 logits
                                gate_logits = first_elem.float()
                                all_gating_scores = torch.softmax(gate_logits, dim=-1)
                                topk_values, topk_indices = torch.topk(gate_logits, k=min(k, gate_logits.shape[-1]), dim=-1)
                                topk_weights = torch.softmax(topk_values.float(), dim=-1)
                        else:
                            # 标准模型：gate 返回 logits
                            gate_logits = gate_output.float()
                            
                            # 计算所有专家的 softmax gating score
                            all_gating_scores = torch.softmax(gate_logits, dim=-1)
                            
                            # 获取 top-k 专家
                            actual_k = min(k, gate_logits.shape[-1])
                            if exclude_shared and shared_experts:
                                filtered_logits = gate_logits.clone()
                                for shared_idx in shared_experts:
                                    if shared_idx < filtered_logits.shape[-1]:
                                        filtered_logits[..., shared_idx] = float('-inf')
                                topk_values, topk_indices = torch.topk(filtered_logits, k=actual_k, dim=-1)
                            else:
                                topk_values, topk_indices = torch.topk(gate_logits, k=actual_k, dim=-1)
                            
                            # 计算 softmax 权重
                            topk_weights = torch.softmax(topk_values.float(), dim=-1)
                        
                        if topk_indices is None or topk_weights is None:
                            if not logged[0] and layer_id == 1:
                                logged[0] = True
                                logger.warning(f"  [MoE Hook] Failed to get topk_indices/weights")
                            return
                        
                        # 只在第一层首次成功时打印信息
                        if not logged[0] and layer_id == 1:
                            logged[0] = True
                            logger.info(f"  [MoE Hook] topk: {list(topk_indices.shape)}, k={topk_indices.shape[-1]}")
                        
                        # 记录基础信息
                        recorder.record_gate_info(topk_indices, topk_weights, all_gating_scores)
                        
                        # ============= 计算专家输出范数 =============
                        # 注意：逐个计算专家输出范数非常慢，这里使用简化方案
                        # 使用权重作为范数的近似，避免 O(tokens × k × experts) 的计算
                        actual_k = topk_indices.shape[-1]
                        
                        # 简化方案：使用 1.0 作为默认范数
                        # EASYEP: score = weight × (1 - simibr) × norm ≈ weight × (1 - simibr)
                        # REAP: score = weight × norm ≈ weight
                        expert_norms = torch.ones(total_tokens, actual_k, device=device, dtype=torch.float32)
                        
                        recorder.record_expert_norms(expert_norms)
                        
                        # ============= 计算 simibr（输入输出余弦相似度）=============
                        # 简化方案：使用 MoE 模块的输入输出计算相似度
                        # 而不是重新计算 routed_output（太慢）
                        if isinstance(output, tuple):
                            moe_output = output[0] if len(output) > 0 else output
                        else:
                            moe_output = output
                        
                        if hasattr(moe_output, 'shape') and moe_output.dim() >= 2:
                            # 确保形状匹配
                            if moe_output.dim() == 3:
                                flat_output = moe_output.view(-1, moe_output.shape[-1])
                            else:
                                flat_output = moe_output
                            
                            # 取相同长度
                            min_len = min(flat_hidden.shape[0], flat_output.shape[0])
                            simibr = F.cosine_similarity(
                                flat_hidden[:min_len].float(), 
                                flat_output[:min_len].float(), 
                                dim=-1
                            )
                            # 填充到完整长度
                            if min_len < total_tokens:
                                full_simibr = torch.ones(total_tokens, device=device)
                                full_simibr[:min_len] = simibr
                                simibr = full_simibr
                        else:
                            # 无法计算，使用默认值 1.0（表示相似）
                            simibr = torch.ones(total_tokens, device=device)
                        
                        recorder.record_simibr(simibr)
                        
                except Exception as e:
                    logger.debug(f"MoE hook 错误: {e}")
            
            return hook_fn
        
        handle_post = moe_module.register_forward_hook(
            create_moe_post_hook(hook_recorder, k_val, num_experts, experts_module, 
                                gate_module, shared_expert_indices, exclude_shared_experts, moe_module, layer_idx)
        )
        handles.append(handle_post)
    
    return handles, moe_layers


def register_gate_only_hooks(model, hooks, exclude_shared_experts=True):
    """
    仅注册 gate hook 的回退方案（当无法识别完整 MoE 结构时）
    """
    handles = []
    layer_idx = 0
    shared_expert_indices = []
    
    # 获取配置信息
    global_k = 4
    global_num_experts = 64
    
    if hasattr(model, "config"):
        config = model.config
        global_k = getattr(config, 'experts_per_token', None) or \
                   getattr(config, 'num_experts_per_tok', None) or 4
        global_num_experts = getattr(config, 'num_experts', None) or \
                            getattr(config, 'n_routed_experts', None) or \
                            getattr(config, 'num_local_experts', None) or 64
    
    if hasattr(model, "config"):
        if hasattr(model.config, "shared_expert_indices"):
            shared_expert_indices = model.config.shared_expert_indices
        elif hasattr(model.config, "num_shared_experts") and model.config.num_shared_experts > 0:
            shared_expert_indices = list(range(model.config.num_shared_experts))
    
    for name, module in model.named_modules():
        if ".experts." in name or ".shared_experts" in name or name.endswith(".experts"):
            continue
        
        gate_module = None
        if hasattr(module, "gate") and isinstance(module.gate, torch.nn.Module):
            gate_module = module.gate
        elif hasattr(module, "router") and isinstance(module.router, torch.nn.Module):
            gate_module = module.router
        
        if gate_module is not None:
            k_val = getattr(module, "experts_per_token", None) or global_k or 4
            
            try:
                parts = name.split(".")
                if "layers" in parts:
                    idx = parts.index("layers")
                    current_layer_idx = int(parts[idx + 1])
                else:
                    current_layer_idx = layer_idx
            except Exception:
                current_layer_idx = layer_idx
            
            logger.info(f"发现 Gate 模块: {name} (Layer {current_layer_idx})")
            
            hook_recorder = hooks[current_layer_idx]
            
            def create_gate_hook(recorder, k, shared_experts, exclude_shared, gate_mod, n_experts):
                def hook_fn(m, inp, out):
                    try:
                        with torch.no_grad():
                            device = inp[0].device if inp else 'cpu'
                            
                            if isinstance(out, tuple) and len(out) >= 2:
                                # DeepSeekV2 等模型：gate 返回 (topk_idx, topk_weight, aux_loss)
                                # 检查 out[0] 是否是整数类型（专家索引）
                                if out[0].dtype in [torch.int32, torch.int64, torch.long]:
                                    indices = out[0]  # (total_tokens, k)
                                    weights = out[1].float()  # (total_tokens, k)
                                    
                                    total_tokens = indices.shape[0]
                                    
                                    # 尝试获取原始 logits 用于 all_gating_scores
                                    if hasattr(gate_mod, 'weight') and inp:
                                        flat_hidden = inp[0]
                                        if flat_hidden.dim() == 3:
                                            flat_hidden = flat_hidden.view(-1, flat_hidden.shape[-1])
                                        gate_logits = F.linear(
                                            flat_hidden.float(),
                                            gate_mod.weight.float(),
                                            None
                                        )
                                        all_gating_scores = torch.softmax(gate_logits, dim=-1)
                                    else:
                                        # 无法获取完整 scores，使用 topk 结果构造
                                        all_gating_scores = torch.zeros(total_tokens, n_experts, device=device)
                                        all_gating_scores.scatter_(1, indices, weights)
                                    
                                    recorder.record_gate_info(indices, weights, all_gating_scores)
                                    return
                                else:
                                    # tuple 但第一个元素是浮点数，可能是 logits
                                    logits = out[0]
                            else:
                                # 标准模型：gate 返回 logits
                                logits = out
                            
                            if not logits.dtype.is_floating_point:
                                logits = logits.float()
                            
                            all_gating_scores = torch.softmax(logits, dim=-1)
                            
                            if exclude_shared and shared_experts:
                                filtered_logits = logits.clone()
                                for shared_idx in shared_experts:
                                    if shared_idx < filtered_logits.shape[-1]:
                                        filtered_logits[..., shared_idx] = float('-inf')
                                experts = torch.topk(filtered_logits, k=k, dim=-1, sorted=True)
                            else:
                                experts = torch.topk(logits, k=k, dim=-1, sorted=True)
                            
                            indices = experts.indices
                            values = experts.values.float()
                            weights = torch.softmax(values, dim=-1)
                            
                            recorder.record_gate_info(indices, weights, all_gating_scores)
                    except Exception as e:
                        logger.error(f"Gate Hook 错误: {e}")
                
                return hook_fn
            
            handle = gate_module.register_forward_hook(
                create_gate_hook(hook_recorder, k_val, shared_expert_indices, exclude_shared_experts, 
                                gate_module, global_num_experts or 64)
            )
            handles.append(handle)
            layer_idx += 1
    
    return handles, []


def remove_hooks(handles):
    """移除所有 hooks"""
    for handle in handles:
        handle.remove()


def analyze_all_in_one(
    checkpoint: str,
    input_file: str,
    base_output_dir: str,
    max_new_tokens: int = 512,
    device: str = "auto",
    force: bool = False,
):
    """
    一次推理收集所有剪枝方法需要的信息
    """

    # 生成输出文件名
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    model_name = os.path.basename(checkpoint)
    
    # 创建输出目录（按模型组织）
    # 结构: results/{model_name}/activations/{dataset}_{type}.json
    model_dir = os.path.join(base_output_dir, model_name)
    activation_dir = os.path.join(model_dir, "activations")
    os.makedirs(activation_dir, exist_ok=True)

    # 输出文件路径
    shapley_file = os.path.join(activation_dir, f"{dataset_name}_shapley.json")
    gating_file = os.path.join(activation_dir, f"{dataset_name}_gating.json")
    easyep_file = os.path.join(activation_dir, f"{dataset_name}_easyep.json")
    reap_file = os.path.join(activation_dir, f"{dataset_name}_reap.json")

    output_files = [shapley_file, gating_file, easyep_file, reap_file]
    
    all_exist = all(os.path.exists(f) for f in output_files)
    if all_exist and not force:
        logger.info("所有输出文件已存在，跳过（使用 --force 强制重新计算）")
        for f in output_files:
            logger.info(f"  - {f}")
        return

    logger.info("=" * 70)
    logger.info("一次 Few-Shot 收集所有剪枝信息")
    logger.info("=" * 70)
    logger.info(f"模型: {checkpoint}")
    logger.info(f"模型名称: {model_name}")
    logger.info(f"数据: {input_file}")
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"输出目录: {activation_dir}")
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
    hooks = defaultdict(ComprehensiveExpertHook)
    logger.info("正在注册 hooks...")
    handles, moe_info = register_comprehensive_hooks(model, hooks, exclude_shared_experts=True)

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
    logger.info("开始分析（同时收集 Shapley/Gating/EASYEP/REAP 信息）...")
    logger.info("=" * 70)

    # 统计数据结构
    shapley_layers = defaultdict(Counter)
    gating_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))
    
    # EASYEP: score = weight × (1 - simibr) × norm
    easyep_scores = defaultdict(lambda: defaultdict(float))
    easyep_counts = defaultdict(lambda: defaultdict(int))
    
    # REAP: score = mean(weight × norm)
    reap_scores = defaultdict(lambda: defaultdict(lambda: {'weighted_norm_sum': 0.0, 'count': 0}))

    for idx, item in enumerate(tqdm(prompts, desc="处理样本"), 1):
        text = item.get("text", "")
        if not text:
            continue

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        for h in hooks.values():
            h.clear()

        try:
            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": False,  # DeepSeek V2 的 cache 不兼容，禁用
                }
                if "pad_token_id" not in generate_kwargs:
                    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
                    elif hasattr(tokenizer, "eos_token_id"):
                        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
                
                model.generate(**generate_kwargs)
        except Exception as e:
            logger.warning(f"样本 {idx} 生成失败: {e}")
            continue

        # 统计所有信息
        layers_with_data = 0
        for layer_idx, hook in hooks.items():
            if not hook.expert_indices:
                continue
            layers_with_data += 1

            # 处理每个时间步
            num_steps = len(hook.expert_indices)
            
            for step_idx in range(num_steps):
                indices_tensor = hook.expert_indices[step_idx]
                weights_tensor = hook.expert_weights[step_idx]
                gating_tensor = hook.all_gating_scores[step_idx]
                
                # 获取 norms 和 simibr（如果有）
                has_norms = step_idx < len(hook.expert_norms) and hook.expert_norms[step_idx] is not None
                has_simibr = step_idx < len(hook.simibr) and hook.simibr[step_idx] is not None
                
                norms_tensor = hook.expert_norms[step_idx] if has_norms else None
                simibr_tensor = hook.simibr[step_idx] if has_simibr else None
                
                # 确保维度正确
                if indices_tensor.dim() == 1:
                    indices_tensor = indices_tensor.unsqueeze(0)
                elif indices_tensor.dim() > 2:
                    indices_tensor = indices_tensor.reshape(-1, indices_tensor.shape[-1])
                
                if weights_tensor.dim() == 1:
                    weights_tensor = weights_tensor.unsqueeze(0)
                elif weights_tensor.dim() > 2:
                    weights_tensor = weights_tensor.reshape(-1, weights_tensor.shape[-1])
                
                total_tokens = indices_tensor.shape[0]
                k = indices_tensor.shape[-1]

                # ==================== 1. Shapley 统计 ====================
                for row in indices_tensor:
                    combo = tuple(sorted(row.tolist()))
                    combo_str = str(combo)
                    shapley_layers[layer_idx][combo_str] += 1

                # ==================== 2. Gating Score 统计 ====================
                if gating_tensor.dim() == 2:
                    for token_idx in range(min(gating_tensor.shape[0], total_tokens)):
                        for expert_idx in range(gating_tensor.shape[1]):
                            score = float(gating_tensor[token_idx, expert_idx].item())
                            if score > 1e-10:
                                gating_stats[layer_idx][expert_idx]['sum'] += score
                                gating_stats[layer_idx][expert_idx]['count'] += 1

                # ==================== 3. EASYEP 统计 ====================
                # score = weight × (1 - simibr) × norm
                for token_idx in range(total_tokens):
                    # 获取该 token 的 simibr
                    if has_simibr and token_idx < simibr_tensor.shape[0]:
                        sim = float(simibr_tensor[token_idx].item())
                        # simibr_factor = max(1 - sim, 0)  # 原文公式
                        simibr_factor = max(1 - sim, 0.0)
                    else:
                        simibr_factor = 1.0  # 如果没有 simibr，使用 1.0
                    
                    for k_idx in range(k):
                        expert_id = int(indices_tensor[token_idx, k_idx].item())
                        weight = float(weights_tensor[token_idx, k_idx].item())
                        
                        # 获取范数
                        if has_norms and token_idx < norms_tensor.shape[0]:
                            norm = float(norms_tensor[token_idx, k_idx].item())
                        else:
                            norm = 1.0  # 如果没有范数，使用 1.0
                        
                        # EASYEP 得分: weight × (1 - simibr) × norm
                        easyep_score = weight * simibr_factor * norm
                        easyep_scores[layer_idx][expert_id] += easyep_score
                        easyep_counts[layer_idx][expert_id] += 1
                        
                        # REAP 得分: weight × norm
                        reap_score = weight * norm
                        reap_scores[layer_idx][expert_id]['weighted_norm_sum'] += reap_score
                        reap_scores[layer_idx][expert_id]['count'] += 1

        # 第一个样本处理后打印统计信息
        if idx == 1:
            layers_with_data = sum(1 for h in hooks.values() if h.expert_indices)
            total_records = sum(len(h.expert_indices) for h in hooks.values())
            logger.info(f"[样本 1 统计] 有数据的层: {layers_with_data}, 总记录数: {total_records}")
            if layers_with_data == 0:
                logger.warning("警告: 第一个样本后没有任何层记录到数据，请检查 hook 是否正常工作")
        
        if idx % 10 == 0:
            gc.collect()

    # 5. 保存结果
    logger.info("正在保存结果...")

    # ==================== 保存 Shapley 结果 ====================
    shapley_data = {
        "model": checkpoint,
        "dataset": input_file,
        "total_samples": len(prompts),
        "total_layers": len(shapley_layers),
        "layers": {},
    }
    for layer_idx, counter in shapley_layers.items():
        shapley_data["layers"][str(layer_idx)] = dict(counter)
    
    with open(shapley_file, "w", encoding="utf-8") as f:
        json.dump(shapley_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Shapley 结果: {shapley_file}")

    # ==================== 保存 Gating Score 结果 ====================
    gating_data = {
        "model": checkpoint,
        "dataset": input_file,
        "total_samples": len(prompts),
        "total_layers": len(gating_stats),
        "layers": {},
    }
    for layer_idx, expert_stats in gating_stats.items():
        layer_data = {}
        for expert_id, stats in expert_stats.items():
            avg_score = stats['sum'] / stats['count'] if stats['count'] > 0 else 0.0
            layer_data[int(expert_id)] = {
                "sum_gating_score": stats['sum'],
                "avg_gating_score": avg_score,
                "count": stats['count']
            }
        gating_data["layers"][str(layer_idx)] = layer_data
    
    with open(gating_file, "w", encoding="utf-8") as f:
        json.dump(gating_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Gating Score 结果: {gating_file}")

    # ==================== 保存 EASYEP 结果 ====================
    easyep_data = {
        "model": checkpoint,
        "dataset": input_file,
        "total_samples": len(prompts),
        "total_layers": len(easyep_scores),
        "description": "EASYEP 得分：weight × (1 - simibr) × norm",
        "formula": "score = Σ(weight × max(1 - cos_sim(x_before, x_after_rmoe), 0) × expert_norm)",
        "layers": {},
    }
    for layer_idx in easyep_scores:
        layer_data = {}
        for expert_id, total_score in easyep_scores[layer_idx].items():
            count = easyep_counts[layer_idx].get(expert_id, 0)
            avg_score = total_score / count if count > 0 else 0.0
            layer_data[int(expert_id)] = {
                "easyep_sum": total_score,
                "easyep_mean": avg_score,
                "activation_count": count
            }
        easyep_data["layers"][str(layer_idx)] = layer_data
    
    with open(easyep_file, "w", encoding="utf-8") as f:
        json.dump(easyep_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ EASYEP 结果: {easyep_file}")

    # ==================== 保存 REAP 结果 ====================
    reap_data = {
        "model": checkpoint,
        "dataset": input_file,
        "total_samples": len(prompts),
        "total_layers": len(reap_scores),
        "description": "REAP 得分：weight × expert_norm",
        "formula": "score = mean(router_weight × expert_activation_norm)",
        "layers": {},
    }
    for layer_idx, expert_stats in reap_scores.items():
        layer_data = {}
        for expert_id, stats in expert_stats.items():
            avg_score = stats['weighted_norm_sum'] / stats['count'] if stats['count'] > 0 else 0.0
            layer_data[int(expert_id)] = {
                "reap_sum": stats['weighted_norm_sum'],
                "reap_mean": avg_score,
                "count": stats['count']
            }
        reap_data["layers"][str(layer_idx)] = layer_data
    
    with open(reap_file, "w", encoding="utf-8") as f:
        json.dump(reap_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ REAP 结果: {reap_file}")

    # 6. 清理
    remove_hooks(handles)

    # 7. 打印统计摘要
    logger.info("=" * 70)
    logger.info("分析完成！统计摘要：")
    logger.info("=" * 70)
    logger.info(f"总样本数: {len(prompts)}")
    logger.info(f"总层数: {len(shapley_layers)}")
    
    max_expert_id = 0
    for layer_stats in gating_stats.values():
        for exp_id in layer_stats.keys():
            max_expert_id = max(max_expert_id, exp_id)
    logger.info(f"专家数量: {max_expert_id + 1}")

    logger.info("\n输出文件:")
    logger.info(f"  - Shapley:      {shapley_file}")
    logger.info(f"  - Gating Score: {gating_file}")
    logger.info(f"  - EASYEP:       {easyep_file}")
    logger.info(f"  - REAP:         {reap_file}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="一次 Few-Shot 收集所有剪枝方法需要的信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  python collect_activations.py \\
      --model /root/yuhao/hf_models/gpt-oss-20b \\
      --data ../data/calibration/gsm8k_25.json \\
      --output_dir ../results

输出目录结构:
  results/
  └── {model_name}/
      └── activations/
          ├── {dataset}_shapley.json   # Shapley 值计算
          ├── {dataset}_gating.json    # Gating Score 剪枝
          ├── {dataset}_easyep.json    # EASYEP 剪枝
          └── {dataset}_reap.json      # REAP 剪枝
        """,
    )

    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--data", type=str, required=True, help="输入数据文件")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小（增大可提升速度）")
    parser.add_argument("--force", "-f", action="store_true", help="强制重新计算")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(args.data):
        logger.error(f"输入文件不存在: {args.data}")
        return

    analyze_all_in_one(
        checkpoint=args.model,
        input_file=args.data,
        base_output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
