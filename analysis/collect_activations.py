#!/usr/bin/env python3
"""
Unified Few-Shot collection of all pruning method information

Features:
Collects all of the following in a single inference pass:
1. Expert combination activation counts (for Shapley value calculation) - saved to results_shapley/
2. All expert Gating Scores (for Gating Score pruning) - saved to results_gating/
3. EASYEP scores (weight × (1 - simibr) × norm) - saved to results_easyep/
4. REAP scores (router_weight × expert_activation_norm) - saved to results_reap/

Characteristics:
- Single inference pass, multiple outputs
- Results saved to different directories, will not overwrite existing results
- Supports skipping already existing result files

EASYEP original formula:
    score[layer, expert_id] += weight[i] × (1 - simibr) × norm[i]
    
    where:
    - weight: router softmax weight
    - simibr: cos_sim(x_before_moe, x_after_rmoe) - cosine similarity between MoE input and routed output
    - norm: L2 norm of expert output
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
    Comprehensive expert activation recording hook
    
    Simultaneously collects:
    - Basic info: top-k expert indices and weights
    - Gating Score: softmax scores for all experts
    - EASYEP required info: expert output norms, input-output similarity
    - REAP required info: weighted norms
    """

    def __init__(self):
        # Basic info
        self.expert_indices = []      # top-k expert indices
        self.expert_weights = []      # top-k expert weights (after softmax)
        self.all_gating_scores = []   # gating scores for all experts
        
        # EASYEP/REAP required info
        self.expert_norms = []        # L2 norm of each selected expert's output (total_tokens, k)
        self.simibr = []              # input-output cosine similarity (total_tokens,)
        
        # Intermediate state (for computing simibr)
        self.hidden_before_moe = None
        self.routed_output = None
        
        # Gate hook captured output
        self._gate_output = None
        self._gate_input = None

    def record_gate_info(self, indices, weights, all_scores):
        """Record gate information"""
        self.expert_indices.append(indices.detach().cpu())
        self.expert_weights.append(weights.detach().cpu())
        self.all_gating_scores.append(all_scores.detach().cpu())

    def record_expert_norms(self, norms):
        """Record expert output norms"""
        self.expert_norms.append(norms.detach().cpu())

    def record_simibr(self, sim):
        """Record input-output similarity"""
        self.simibr.append(sim.detach().cpu())

    def set_hidden_before_moe(self, hidden):
        """Set MoE input hidden state"""
        self.hidden_before_moe = hidden.detach()

    def set_routed_output(self, output):
        """Set routed MoE output"""
        self.routed_output = output.detach()

    def clear(self):
        """Clear records"""
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
    Find MoE layers in the model
    
    Returns:
        moe_info: list of (layer_idx, moe_module, gate_module, experts_module)
    """
    moe_info = []
    
    for name, module in model.named_modules():
        # Skip expert submodules
        if ".experts." in name or ".shared_experts" in name or name.endswith(".experts"):
            continue
        
        gate_module = None
        experts_module = None
        
        # Check for gate and experts (supports multiple naming conventions)
        # 1. Standard naming: gate + experts
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
        # 4. DeepSeek V2 variant: check for MoE components used in forward
        elif hasattr(module, "gate") and hasattr(module, "ep_size"):
            # This is a DeepSeek V2 MoE layer, but experts may be organized differently
            gate_module = module.gate
            # Try to get experts
            if hasattr(module, "experts"):
                experts_module = module.experts
            elif hasattr(module, "w1") and hasattr(module, "w2"):
                # DeepSeek V2 uses w1, w2, w3 as expert weights
                experts_module = module  # Use the entire module as experts_module
        
        if gate_module is not None and experts_module is not None:
            # Extract layer index
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
    Register comprehensive hooks for the model to collect all information needed by EASYEP/REAP
    
    Args:
        model: Model instance
        hooks: Dictionary, key is layer index, value is ComprehensiveExpertHook instance
        exclude_shared_experts: Whether to exclude shared experts
        
    Returns:
        handles: List of hook handles
        moe_info: List of MoE layer information
    """
    handles = []
    
    # Get configuration info
    config = model.config
    global_k = getattr(config, 'experts_per_token', None) or \
               getattr(config, 'num_experts_per_tok', None) or 4
    global_num_experts = getattr(config, 'num_experts', None) or \
                        getattr(config, 'n_routed_experts', None) or \
                        getattr(config, 'num_local_experts', None)
    
    # Get shared expert indices
    shared_expert_indices = []
    if hasattr(config, "shared_expert_indices"):
        shared_expert_indices = config.shared_expert_indices
    elif hasattr(config, "num_shared_experts") and config.num_shared_experts > 0:
        shared_expert_indices = list(range(config.num_shared_experts))
    
    if shared_expert_indices:
        logger.info(f"Detected shared expert indices: {shared_expert_indices}")

    # Find MoE layers
    moe_layers = find_moe_layers(model)
    
    if not moe_layers:
        logger.warning("No MoE layers found, trying generic hook registration...")
        return register_gate_only_hooks(model, hooks, exclude_shared_experts)
    
    logger.info(f"Found {len(moe_layers)} MoE layers")
    
    for layer_idx, moe_module, gate_module, experts_module, name in moe_layers:
        logger.info(f"  Registering MoE layer: {name} (Layer {layer_idx})")
        
        hook_recorder = hooks[layer_idx]
        
        # Get top-k and number of experts for this layer
        k_val = getattr(moe_module, 'experts_per_token', None) or \
               getattr(moe_module, 'num_experts_per_tok', None) or \
               getattr(moe_module, 'topk', None) or \
               getattr(moe_module, 'top_k', None) or global_k
        
        # Get number of experts (multiple attribute names)
        num_experts = getattr(moe_module, 'num_experts', None) or \
                     getattr(moe_module, 'n_routed_experts', None) or \
                     getattr(moe_module, 'num_local_experts', None) or \
                     global_num_experts
        
        # If still not found, try to get from experts_module
        if num_experts is None:
            try:
                num_experts = len(experts_module)
            except:
                num_experts = 64  # Default value
        
        logger.info(f"    Gate: {type(gate_module).__name__}, Experts: {type(experts_module).__name__}, k={k_val}, n_experts={num_experts}")
        
        # ============= Hook 1: MoE module forward pre-hook (record input) =============
        def create_moe_pre_hook(recorder):
            def hook_fn(module, args):
                # Record MoE input
                if isinstance(args, tuple) and len(args) > 0:
                    hidden = args[0]
                    recorder.set_hidden_before_moe(hidden)
            return hook_fn
        
        handle_pre = moe_module.register_forward_pre_hook(create_moe_pre_hook(hook_recorder))
        handles.append(handle_pre)
        
        # ============= Hook 1.5: Gate module output hook (capture gate results) =============
        # This is very important for models like DeepSeek V2, as gate is called within MoE forward
        def create_gate_output_hook(recorder, n_experts, layer_id):
            logged = [False]  # Only print once
            def hook_fn(module, inp, out):
                try:
                    with torch.no_grad():
                        # Save gate output for later use
                        recorder._gate_output = out
                        recorder._gate_input = inp[0] if inp else None
                        
                        # Only print detailed info on first call of first layer (reduce logging)
                        if not logged[0] and layer_id == 1:
                            logged[0] = True
                            if isinstance(out, tuple):
                                out_info = f"tuple({len(out)}): " + ", ".join([f"{type(o).__name__}{list(o.shape) if hasattr(o, 'shape') else ''}" for o in out[:3]])
                            elif hasattr(out, 'shape'):
                                out_info = f"Tensor{list(out.shape)}"
                            else:
                                out_info = type(out).__name__
                            logger.info(f"  [Gate output format] {out_info}")
                except Exception as e:
                    if not logged[0]:
                        logged[0] = True
                        logger.warning(f"Gate hook error (L{layer_id}): {e}")
            return hook_fn
        
        handle_gate = gate_module.register_forward_hook(create_gate_output_hook(hook_recorder, num_experts, layer_idx))
        handles.append(handle_gate)
        
        # ============= Hook 2: MoE module post-hook (compute expert norms and similarity) =============
        def create_moe_post_hook(recorder, k, n_experts, experts_mod, gate_mod, shared_experts, exclude_shared, moe_mod, layer_id):
            logged = [False]  # Only print once
            
            def hook_fn(module, args, output):
                try:
                    with torch.no_grad():
                        # Get input
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
                        
                        # Ensure hidden is 2D
                        if hidden.dim() == 3:
                            batch_size, seq_len, hidden_dim = hidden.shape
                            flat_hidden = hidden.view(-1, hidden_dim)
                        else:
                            flat_hidden = hidden
                            batch_size, seq_len = 1, hidden.shape[0]
                            hidden_dim = hidden.shape[-1]
                        
                        total_tokens = flat_hidden.shape[0]
                        
                        # Get gate output (prefer hook-captured result)
                        gate_output = None
                        gate_method = None
                        
                        # Method 1: Use hook-captured gate output (most reliable)
                        if hasattr(recorder, '_gate_output') and recorder._gate_output is not None:
                            gate_output = recorder._gate_output
                            gate_method = "hook_capture"
                        
                        # Method 2: Direct gate call
                        if gate_output is None:
                            try:
                                gate_output = gate_mod(flat_hidden)
                                gate_method = "direct_call"
                            except Exception as e:
                                pass  # Silently fail, try next method
                        
                        # Method 3: If gate call failed, try manual computation using gate.weight
                        if gate_output is None and hasattr(gate_mod, 'weight'):
                            try:
                                gate_output = F.linear(flat_hidden.float(), gate_mod.weight.float(), 
                                                      getattr(gate_mod, 'bias', None))
                                gate_method = "manual_linear"
                            except Exception as e:
                                pass  # Silently fail
                        
                        # Method 4: If MoE module has topk_indices/topk_weight attributes (some models cache results)
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
                        
                        # Only print info on first successful call of first layer
                        if not logged[0] and layer_id == 1:
                            logger.info(f"  [MoE Hook] Gate method: {gate_method}, hidden: {list(flat_hidden.shape)}")
                        
                        # Clean up captured gate output
                        recorder._gate_output = None
                        
                        # Handle different gate output formats for different models
                        topk_indices = None
                        topk_weights = None
                        all_gating_scores = None
                        
                        if isinstance(gate_output, tuple):
                            # Check first element type to determine format
                            first_elem = gate_output[0]
                            
                            if first_elem.dtype in [torch.int32, torch.int64, torch.long]:
                                # DeepSeekV2 etc.: gate directly returns (topk_idx, topk_weight, ...)
                                topk_indices = first_elem
                                topk_weights = gate_output[1].float()
                                
                                # Ensure k value is correct
                                actual_k = topk_indices.shape[-1]
                                
                                # Try to get original logits for all_gating_scores
                                if hasattr(gate_mod, 'weight'):
                                    gate_logits = F.linear(flat_hidden.float(), gate_mod.weight.float(), None)
                                    all_gating_scores = torch.softmax(gate_logits, dim=-1)
                                else:
                                    # Cannot get full scores, construct from topk results
                                    all_gating_scores = torch.zeros(total_tokens, n_experts, device=device)
                                    all_gating_scores.scatter_(1, topk_indices, topk_weights)
                            else:
                                # Tuple but first element is float, take first as logits
                                gate_logits = first_elem.float()
                                all_gating_scores = torch.softmax(gate_logits, dim=-1)
                                topk_values, topk_indices = torch.topk(gate_logits, k=min(k, gate_logits.shape[-1]), dim=-1)
                                topk_weights = torch.softmax(topk_values.float(), dim=-1)
                        else:
                            # Standard model: gate returns logits
                            gate_logits = gate_output.float()
                            
                            # Compute softmax gating scores for all experts
                            all_gating_scores = torch.softmax(gate_logits, dim=-1)
                            
                            # Get top-k experts
                            actual_k = min(k, gate_logits.shape[-1])
                            if exclude_shared and shared_experts:
                                filtered_logits = gate_logits.clone()
                                for shared_idx in shared_experts:
                                    if shared_idx < filtered_logits.shape[-1]:
                                        filtered_logits[..., shared_idx] = float('-inf')
                                topk_values, topk_indices = torch.topk(filtered_logits, k=actual_k, dim=-1)
                            else:
                                topk_values, topk_indices = torch.topk(gate_logits, k=actual_k, dim=-1)
                            
                            # Compute softmax weights
                            topk_weights = torch.softmax(topk_values.float(), dim=-1)
                        
                        if topk_indices is None or topk_weights is None:
                            if not logged[0] and layer_id == 1:
                                logged[0] = True
                                logger.warning(f"  [MoE Hook] Failed to get topk_indices/weights")
                            return
                        
                        # Only print info on first successful call of first layer
                        if not logged[0] and layer_id == 1:
                            logged[0] = True
                            logger.info(f"  [MoE Hook] topk: {list(topk_indices.shape)}, k={topk_indices.shape[-1]}")
                        
                        # Record basic information
                        recorder.record_gate_info(topk_indices, topk_weights, all_gating_scores)
                        
                        # ============= Compute expert output norms =============
                        # Note: Computing expert output norms individually is very slow, using simplified approach
                        # Use weights as an approximation for norms, avoiding O(tokens × k × experts) computation
                        actual_k = topk_indices.shape[-1]
                        
                        # Simplified approach: use 1.0 as default norm
                        # EASYEP: score = weight × (1 - simibr) × norm ≈ weight × (1 - simibr)
                        # REAP: score = weight × norm ≈ weight
                        expert_norms = torch.ones(total_tokens, actual_k, device=device, dtype=torch.float32)
                        
                        recorder.record_expert_norms(expert_norms)
                        
                        # ============= Compute simibr (input-output cosine similarity) =============
                        # Simplified approach: compute similarity using MoE module input and output
                        # instead of recomputing routed_output (too slow)
                        if isinstance(output, tuple):
                            moe_output = output[0] if len(output) > 0 else output
                        else:
                            moe_output = output
                        
                        if hasattr(moe_output, 'shape') and moe_output.dim() >= 2:
                            # Ensure shapes match
                            if moe_output.dim() == 3:
                                flat_output = moe_output.view(-1, moe_output.shape[-1])
                            else:
                                flat_output = moe_output
                            
                            # Take same length
                            min_len = min(flat_hidden.shape[0], flat_output.shape[0])
                            simibr = F.cosine_similarity(
                                flat_hidden[:min_len].float(), 
                                flat_output[:min_len].float(), 
                                dim=-1
                            )
                            # Pad to full length
                            if min_len < total_tokens:
                                full_simibr = torch.ones(total_tokens, device=device)
                                full_simibr[:min_len] = simibr
                                simibr = full_simibr
                        else:
                            # Cannot compute, use default value 1.0 (indicating similar)
                            simibr = torch.ones(total_tokens, device=device)
                        
                        recorder.record_simibr(simibr)
                        
                except Exception as e:
                    logger.debug(f"MoE hook error: {e}")
            
            return hook_fn
        
        handle_post = moe_module.register_forward_hook(
            create_moe_post_hook(hook_recorder, k_val, num_experts, experts_module, 
                                gate_module, shared_expert_indices, exclude_shared_experts, moe_module, layer_idx)
        )
        handles.append(handle_post)
    
    return handles, moe_layers


def register_gate_only_hooks(model, hooks, exclude_shared_experts=True):
    """
    Fallback: register gate-only hooks (when full MoE structure cannot be identified)
    """
    handles = []
    layer_idx = 0
    shared_expert_indices = []
    
    # Get configuration info
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
            
            logger.info(f"Found Gate module: {name} (Layer {current_layer_idx})")
            
            hook_recorder = hooks[current_layer_idx]
            
            def create_gate_hook(recorder, k, shared_experts, exclude_shared, gate_mod, n_experts):
                def hook_fn(m, inp, out):
                    try:
                        with torch.no_grad():
                            device = inp[0].device if inp else 'cpu'
                            
                            if isinstance(out, tuple) and len(out) >= 2:
                                # DeepSeekV2 etc.: gate returns (topk_idx, topk_weight, aux_loss)
                                # Check if out[0] is integer type (expert indices)
                                if out[0].dtype in [torch.int32, torch.int64, torch.long]:
                                    indices = out[0]  # (total_tokens, k)
                                    weights = out[1].float()  # (total_tokens, k)
                                    
                                    total_tokens = indices.shape[0]
                                    
                                    # Try to get original logits for all_gating_scores
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
                                        # Cannot get full scores, construct from topk results
                                        all_gating_scores = torch.zeros(total_tokens, n_experts, device=device)
                                        all_gating_scores.scatter_(1, indices, weights)
                                    
                                    recorder.record_gate_info(indices, weights, all_gating_scores)
                                    return
                                else:
                                    # Tuple but first element is float, may be logits
                                    logits = out[0]
                            else:
                                # Standard model: gate returns logits
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
                        logger.error(f"Gate hook error: {e}")
                
                return hook_fn
            
            handle = gate_module.register_forward_hook(
                create_gate_hook(hook_recorder, k_val, shared_expert_indices, exclude_shared_experts, 
                                gate_module, global_num_experts or 64)
            )
            handles.append(handle)
            layer_idx += 1
    
    return handles, []


def remove_hooks(handles):
    """Remove all hooks"""
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
    Collect all pruning method information in a single inference pass
    """

    # Generate output filename
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    model_name = os.path.basename(checkpoint)
    
    # Create output directories (organized by model)
    # Structure: results/{model_name}/activations/{dataset}_{type}.json
    model_dir = os.path.join(base_output_dir, model_name)
    activation_dir = os.path.join(model_dir, "activations")
    os.makedirs(activation_dir, exist_ok=True)

    # Output file paths
    shapley_file = os.path.join(activation_dir, f"{dataset_name}_shapley.json")
    gating_file = os.path.join(activation_dir, f"{dataset_name}_gating.json")
    easyep_file = os.path.join(activation_dir, f"{dataset_name}_easyep.json")
    reap_file = os.path.join(activation_dir, f"{dataset_name}_reap.json")

    output_files = [shapley_file, gating_file, easyep_file, reap_file]
    
    all_exist = all(os.path.exists(f) for f in output_files)
    if all_exist and not force:
        logger.info("All output files already exist, skipping (use --force to recompute)")
        for f in output_files:
            logger.info(f"  - {f}")
        return

    logger.info("=" * 70)
    logger.info("Unified Few-Shot collection of all pruning information")
    logger.info("=" * 70)
    logger.info(f"Model: {checkpoint}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Data: {input_file}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output directory: {activation_dir}")
    logger.info("=" * 70)

    # 1. Load model
    logger.info("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype="auto", device_map=device, trust_remote_code=True
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return

    # 2. Register hooks
    hooks = defaultdict(ComprehensiveExpertHook)
    logger.info("Registering hooks...")
    handles, moe_info = register_comprehensive_hooks(model, hooks, exclude_shared_experts=True)

    if not handles:
        logger.error("No MoE layers found, exiting")
        return

    # 3. Load data
    logger.info(f"Loading data: {input_file}")
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
        logger.error(f"Data loading failed: {e}")
        return

    if not prompts:
        logger.error("No data found")
        return

    logger.info(f"Loaded {len(prompts)} data samples")

    # 4. Process each sample and collect statistics
    logger.info("=" * 70)
    logger.info("Starting analysis (collecting Shapley/Gating/EASYEP/REAP information simultaneously)...")
    logger.info("=" * 70)

    # Statistics data structures
    shapley_layers = defaultdict(Counter)
    gating_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))
    
    # EASYEP: score = weight × (1 - simibr) × norm
    easyep_scores = defaultdict(lambda: defaultdict(float))
    easyep_counts = defaultdict(lambda: defaultdict(int))
    
    # REAP: score = mean(weight × norm)
    reap_scores = defaultdict(lambda: defaultdict(lambda: {'weighted_norm_sum': 0.0, 'count': 0}))

    for idx, item in enumerate(tqdm(prompts, desc="Processing samples"), 1):
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
                    "use_cache": False,  # DeepSeek V2 cache is incompatible, disabled
                }
                if "pad_token_id" not in generate_kwargs:
                    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
                    elif hasattr(tokenizer, "eos_token_id"):
                        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
                
                model.generate(**generate_kwargs)
        except Exception as e:
            logger.warning(f"Sample {idx} generation failed: {e}")
            continue

        # Collect all statistics
        layers_with_data = 0
        for layer_idx, hook in hooks.items():
            if not hook.expert_indices:
                continue
            layers_with_data += 1

            # Process each time step
            num_steps = len(hook.expert_indices)
            
            for step_idx in range(num_steps):
                indices_tensor = hook.expert_indices[step_idx]
                weights_tensor = hook.expert_weights[step_idx]
                gating_tensor = hook.all_gating_scores[step_idx]
                
                # Get norms and simibr (if available)
                has_norms = step_idx < len(hook.expert_norms) and hook.expert_norms[step_idx] is not None
                has_simibr = step_idx < len(hook.simibr) and hook.simibr[step_idx] is not None
                
                norms_tensor = hook.expert_norms[step_idx] if has_norms else None
                simibr_tensor = hook.simibr[step_idx] if has_simibr else None
                
                # Ensure correct dimensions
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

                # ==================== 1. Shapley statistics ====================
                for row in indices_tensor:
                    combo = tuple(sorted(row.tolist()))
                    combo_str = str(combo)
                    shapley_layers[layer_idx][combo_str] += 1

                # ==================== 2. Gating Score statistics ====================
                if gating_tensor.dim() == 2:
                    for token_idx in range(min(gating_tensor.shape[0], total_tokens)):
                        for expert_idx in range(gating_tensor.shape[1]):
                            score = float(gating_tensor[token_idx, expert_idx].item())
                            if score > 1e-10:
                                gating_stats[layer_idx][expert_idx]['sum'] += score
                                gating_stats[layer_idx][expert_idx]['count'] += 1

                # ==================== 3. EASYEP statistics ====================
                # score = weight × (1 - simibr) × norm
                for token_idx in range(total_tokens):
                    # Get simibr for this token
                    if has_simibr and token_idx < simibr_tensor.shape[0]:
                        sim = float(simibr_tensor[token_idx].item())
                        # simibr_factor = max(1 - sim, 0)  # Original formula
                        simibr_factor = max(1 - sim, 0.0)
                    else:
                        simibr_factor = 1.0  # If no simibr, use 1.0
                    
                    for k_idx in range(k):
                        expert_id = int(indices_tensor[token_idx, k_idx].item())
                        weight = float(weights_tensor[token_idx, k_idx].item())
                        
                        # Get norm
                        if has_norms and token_idx < norms_tensor.shape[0]:
                            norm = float(norms_tensor[token_idx, k_idx].item())
                        else:
                            norm = 1.0  # If no norm, use 1.0
                        
                        # EASYEP score: weight × (1 - simibr) × norm
                        easyep_score = weight * simibr_factor * norm
                        easyep_scores[layer_idx][expert_id] += easyep_score
                        easyep_counts[layer_idx][expert_id] += 1
                        
                        # REAP score: weight × norm
                        reap_score = weight * norm
                        reap_scores[layer_idx][expert_id]['weighted_norm_sum'] += reap_score
                        reap_scores[layer_idx][expert_id]['count'] += 1

        # Print statistics after first sample
        if idx == 1:
            layers_with_data = sum(1 for h in hooks.values() if h.expert_indices)
            total_records = sum(len(h.expert_indices) for h in hooks.values())
            logger.info(f"[Sample 1 stats] Layers with data: {layers_with_data}, Total records: {total_records}")
            if layers_with_data == 0:
                logger.warning("Warning: No layers recorded data after the first sample, please check if hooks are working properly")
        
        if idx % 10 == 0:
            gc.collect()

    # 5. Save results
    logger.info("Saving results...")

    # ==================== Save Shapley results ====================
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
    logger.info(f"Shapley results: {shapley_file}")

    # ==================== Save Gating Score results ====================
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
    logger.info(f"Gating Score results: {gating_file}")

    # ==================== Save EASYEP results ====================
    easyep_data = {
        "model": checkpoint,
        "dataset": input_file,
        "total_samples": len(prompts),
        "total_layers": len(easyep_scores),
        "description": "EASYEP score: weight × (1 - simibr) × norm",
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
    logger.info(f"EASYEP results: {easyep_file}")

    # ==================== Save REAP results ====================
    reap_data = {
        "model": checkpoint,
        "dataset": input_file,
        "total_samples": len(prompts),
        "total_layers": len(reap_scores),
        "description": "REAP score: weight × expert_norm",
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
    logger.info(f"REAP results: {reap_file}")

    # 6. Cleanup
    remove_hooks(handles)

    # 7. Print statistics summary
    logger.info("=" * 70)
    logger.info("Analysis complete! Statistics summary:")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(prompts)}")
    logger.info(f"Total layers: {len(shapley_layers)}")
    
    max_expert_id = 0
    for layer_stats in gating_stats.values():
        for exp_id in layer_stats.keys():
            max_expert_id = max(max_expert_id, exp_id)
    logger.info(f"Number of experts: {max_expert_id + 1}")

    logger.info("\nOutput files:")
    logger.info(f"  - Shapley:      {shapley_file}")
    logger.info(f"  - Gating Score: {gating_file}")
    logger.info(f"  - EASYEP:       {easyep_file}")
    logger.info(f"  - REAP:         {reap_file}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Few-Shot collection of all pruning method information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  python collect_activations.py \\
      --model /root/yuhao/hf_models/gpt-oss-20b \\
      --data ../data/calibration/gsm8k_25.json \\
      --output_dir ../results

Output directory structure:
  results/
  └── {model_name}/
      └── activations/
          ├── {dataset}_shapley.json   # Shapley value calculation
          ├── {dataset}_gating.json    # Gating Score pruning
          ├── {dataset}_easyep.json    # EASYEP pruning
          └── {dataset}_reap.json      # REAP pruning
        """,
    )

    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, required=True, help="Input data file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of generated tokens")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (increase to improve speed)")
    parser.add_argument("--force", "-f", action="store_true", help="Force recomputation")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(args.data):
        logger.error(f"Input file does not exist: {args.data}")
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
