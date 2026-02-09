#!/usr/bin/env python3
"""
MoE Model Expert Pruning Tool

Features:
1. Load original model
2. Process pruned experts based on expert selection JSON file
3. Save as a complete model in safetensor format

Pruning strategies:
- zero_weights: Zero out the weights of pruned experts (default)
- gate_bias: Add a large negative bias to pruned experts in the gate layer so they won't be selected
- both: Use both strategies simultaneously
- auto: Automatically select strategy based on pruning method

Default strategies for different pruning methods:
- shapley: zero_weights
- easyep: zero_weights
- reap: zero_weights
- gating: zero_weights
- frequency: zero_weights
- random: zero_weights
"""

import torch
import json
import os
import argparse
import logging
import re
from typing import Dict, List, Set, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Large negative value for gate bias, making pruned experts' selection probability close to 0
GATE_BIAS_VALUE = -1e9

# Default strategies for different pruning methods
# Modify the default strategy for each method as needed
METHOD_DEFAULT_STRATEGIES = {
    "shapley": "zero_weights",
    "easyep": "zero_weights",
    "reap": "zero_weights",
    "gating": "zero_weights",
    "frequency": "zero_weights",
    "random": "zero_weights",
}

def detect_method_from_filename(filename: str) -> Optional[str]:
    """Detect pruning method from selection filename"""
    basename = os.path.basename(filename).lower()
    
    # Detect various methods
    if basename.startswith("shapley"):
        return "shapley"
    elif basename.startswith("easyep") or "easyep" in basename:
        return "easyep"
    elif basename.startswith("reap") or "reap" in basename:
        return "reap"
    elif basename.startswith("gating") or "gating" in basename:
        return "gating"
    elif basename.startswith("frequency") or "frequency" in basename:
        return "frequency"
    elif basename.startswith("random") or "random" in basename:
        return "random"
    
    return None

def get_strategy_for_method(method: Optional[str], user_strategy: str) -> str:
    """Determine the final strategy based on method and user-specified strategy"""
    # If user explicitly specified a strategy (not auto), use it directly
    if user_strategy != "auto":
        return user_strategy
    
    # Auto mode: automatically select based on method
    if method and method in METHOD_DEFAULT_STRATEGIES:
        return METHOD_DEFAULT_STRATEGIES[method]
    
    # Default to zero_weights
    return "zero_weights"


class ModelPruner:
    """MoE Model Expert Pruning Tool"""

    def __init__(
        self,
        model_path: str,
        selection_json_path: str,
        output_dir: str,
        device_map: str = "auto",
        pruning_strategy: str = "auto",
    ):
        """
        Args:
            model_path: Original model path
            selection_json_path: Expert selection JSON file path
            output_dir: Output model save directory
            device_map: Device mapping strategy
            pruning_strategy: Pruning strategy
                - "auto": Automatically select based on pruning method (default)
                - "zero_weights": Zero out expert weights
                - "gate_bias": Modify gate bias so pruned experts won't be selected
                - "both": Use both strategies simultaneously
        """
        self.model_path = model_path
        self.selection_json_path = selection_json_path
        self.output_dir = output_dir
        self.device_map = device_map
        self.user_pruning_strategy = pruning_strategy  # User-specified strategy
        self.pruning_strategy = pruning_strategy  # Final strategy (may be auto-adjusted)
        self.detected_method = None  # Detected pruning method
        self.model = None
        self.tokenizer = None
        self.selected_experts: Dict[str, List[int]] = {}
        
        # Statistics
        self.stats = {
            "gate_modified_layers": 0,
            "zeroed_experts": 0,
            "zeroed_params": 0,
        }

    def load_selection_file(self) -> Dict[str, List[int]]:
        """Load expert selection file"""
        logger.info(f"Loading expert selection file: {self.selection_json_path}")
        with open(self.selection_json_path, "r") as f:
            data = json.load(f)

        # Print statistics
        total_selected = sum(len(experts) for experts in data.values())
        total_layers = len(data)
        avg_per_layer = total_selected / total_layers if total_layers > 0 else 0
        logger.info(f"Total {total_layers} layers, selected {total_selected} experts (average {avg_per_layer:.1f} per layer)")

        return data

    def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_path}")
        logger.info(f"Device mapping strategy: {self.device_map}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
        )
        
        # Display device allocation
        if hasattr(self.model, 'hf_device_map'):
            devices_used = set(str(v) for v in self.model.hf_device_map.values())
            logger.info(f"Model distributed across devices: {devices_used}")
        
        logger.info("Model loaded successfully")

    def _find_gate_module(self, moe_module) -> Optional[torch.nn.Module]:
        """Find the Gate/Router module in a MoE module"""
        # Common gate attribute names
        gate_names = ["gate", "router", "gate_proj", "wg"]
        
        for name in gate_names:
            if hasattr(moe_module, name):
                gate = getattr(moe_module, name)
                # Ensure it is a module with parameters
                if isinstance(gate, torch.nn.Module):
                    return gate
        
        return None

    def _get_num_experts(self, moe_module) -> Optional[int]:
        """Get the number of experts in a MoE module"""
        # Try to get from various attributes
        attr_names = ["num_experts", "n_routed_experts", "num_local_experts"]
        
        for attr in attr_names:
            if hasattr(moe_module, attr):
                return getattr(moe_module, attr)
        
        # Try to get from experts module
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
        Modify Gate module so pruned experts won't be selected
        
        Strategy priority:
        1. If Gate already has bias, modify bias
        2. If Gate is Linear without bias, try to add bias
        3. Zero out pruned experts' weight rows (as fallback)
        """
        unselected_indices = [i for i in range(num_experts) if i not in selected_indices]
        
        if not unselected_indices:
            logger.info(f"Layer {layer_idx}: All experts selected, no gate modification needed")
            return True
        
        modified = False
        method_used = ""
        
        # Method 1: Modify Linear layer
        if isinstance(gate_module, torch.nn.Linear):
            with torch.no_grad():
                # 1a. If bias already exists, modify it directly
                if gate_module.bias is not None:
                    for idx in unselected_indices:
                        if idx < gate_module.bias.size(0):
                            gate_module.bias.data[idx] = GATE_BIAS_VALUE
                    modified = True
                    method_used = "modified existing bias"
                else:
                    # 1b. No bias, try to add one
                    # Note: This requires the model's forward function to support bias
                    # Most nn.Linear forward functions will automatically use bias (if it exists)
                    try:
                        new_bias = torch.zeros(
                            gate_module.out_features, 
                            device=gate_module.weight.device,
                            dtype=gate_module.weight.dtype
                        )
                        for idx in unselected_indices:
                            new_bias[idx] = GATE_BIAS_VALUE
                        
                        # Register as new bias parameter
                        gate_module.register_parameter('bias', torch.nn.Parameter(new_bias))
                        modified = True
                        method_used = "added new bias"
                        logger.info(f"Layer {layer_idx}: Gate originally had no bias, added one")
                    except Exception as e:
                        logger.warning(f"Layer {layer_idx}: Failed to add bias: {e}")
                        
                        # 1c. If adding bias failed, zero out pruned experts' weight rows
                        # This makes these experts' gate scores close to 0 (depends on input)
                        weight = gate_module.weight
                        if weight.dim() == 2 and weight.size(0) == num_experts:
                            for idx in unselected_indices:
                                weight.data[idx] = 0.0
                            modified = True
                            method_used = "weight zeroed out"
        
        # Method 2: For non-Linear gate modules
        elif hasattr(gate_module, "weight") and isinstance(gate_module.weight, torch.nn.Parameter):
            with torch.no_grad():
                weight = gate_module.weight
                # Assume weight shape is [num_experts, hidden_size]
                if weight.dim() == 2 and weight.size(0) == num_experts:
                    for idx in unselected_indices:
                        weight.data[idx] = 0.0
                    modified = True
                    method_used = "weight zeroed out (non-Linear)"
        
        if modified:
            logger.info(f"Layer {layer_idx}: Masked {len(unselected_indices)} experts ({method_used})")
        else:
            logger.warning(f"Layer {layer_idx}: Unable to modify gate module (type: {type(gate_module).__name__})")
        
        return modified

    def modify_gates(self):
        """Modify gates in all MoE layers so pruned experts won't be selected"""
        logger.info("Starting to modify Gate/Router modules...")
        
        modified_layers = 0
        
        for name, module in self.model.named_modules():
            layer_idx = self._extract_layer_index(name)
            
            if layer_idx is None or str(layer_idx) not in self.selected_experts:
                continue
            
            # Check if it is a MoE module
            is_moe = (hasattr(module, "experts") or 
                     hasattr(module, "routed_experts") or
                     hasattr(module, "gate"))
            
            if not is_moe:
                continue
            
            # Get gate module
            gate_module = self._find_gate_module(module)
            if gate_module is None:
                continue
            
            # Get number of experts
            num_experts = self._get_num_experts(module)
            if num_experts is None:
                logger.warning(f"Layer {layer_idx}: Unable to determine number of experts")
                continue
            
            # Get selected experts
            selected_indices = set(self.selected_experts[str(layer_idx)])
            
            # Modify gate
            if self._modify_gate_bias(gate_module, num_experts, selected_indices, layer_idx):
                modified_layers += 1
        
        self.stats["gate_modified_layers"] = modified_layers
        logger.info(f"Gate modification done! Modified {modified_layers} layers")

    def zero_out_experts(self):
        """Zero out weights of unselected experts"""
        logger.info("Starting to zero out unselected expert weights...")

        total_zeroed_experts = 0
        total_zeroed_params = 0

        for name, module in self.model.named_modules():
            layer_idx = self._extract_layer_index(name)

            if layer_idx is None or str(layer_idx) not in self.selected_experts:
                continue

            # Find experts module (supports multiple model architectures)
            experts = None
            if hasattr(module, "experts"):
                experts = module.experts
            elif hasattr(module, "routed_experts"):
                experts = module.routed_experts
            
            if experts is None:
                continue

            selected_indices = set(self.selected_experts[str(layer_idx)])

            # Check if it is a packed weight type (e.g., GptOssExperts)
            if hasattr(experts, "num_experts"):
                num_experts = experts.num_experts
                unselected_indices = [
                    i for i in range(num_experts) if i not in selected_indices
                ]

                if not unselected_indices:
                    continue

                # Zero out expert weight parameters
                expert_params = [
                    "gate_up_proj", "down_proj",
                    "gate_up_proj_bias", "down_proj_bias",
                    "gate_proj", "up_proj",  # Some models use these names
                    "w1", "w2", "w3",  # Llama style
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
                        f"Layer {layer_idx}: Zeroed out {len(unselected_indices)} experts, "
                        f"processed {zeroed_params} parameters"
                    )

            # Check if it is a ModuleList type
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
                        f"Layer {layer_idx}: Zeroed out {zeroed_experts_in_layer} experts, "
                        f"processed {zeroed_params_in_layer} parameters"
                    )
                    total_zeroed_experts += zeroed_experts_in_layer
                    total_zeroed_params += zeroed_params_in_layer

        self.stats["zeroed_experts"] = total_zeroed_experts
        self.stats["zeroed_params"] = total_zeroed_params
        logger.info(f"Weight zeroing done! Zeroed out {total_zeroed_experts} experts")

    def _extract_layer_index(self, module_name: str) -> Optional[int]:
        """Extract layer index from module name"""
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
        """Save the pruned model in safetensor format"""
        if self.model is None:
            raise ValueError("Model not loaded, please call load_model() first")

        logger.info(f"Saving pruned model to: {self.output_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(
            self.output_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        logger.info("Model weights saved (safetensor format)")

        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Tokenizer saved")

        # Save pruning info
        pruning_info = {
            "original_model": self.model_path,
            "selection_file": self.selection_json_path,
            "detected_method": self.detected_method,
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
        logger.info(f"Pruning info saved to: {info_path}")

    def run(self):
        """Execute the complete pruning workflow"""
        logger.info("=" * 70)
        logger.info("Starting MoE Model Expert Pruning")
        logger.info("=" * 70)
        
        # 0. Detect pruning method and determine strategy
        self.detected_method = detect_method_from_filename(self.selection_json_path)
        self.pruning_strategy = get_strategy_for_method(
            self.detected_method, 
            self.user_pruning_strategy
        )
        
        if self.detected_method:
            logger.info(f"Detected pruning method: {self.detected_method}")
        if self.user_pruning_strategy == "auto":
            logger.info(f"Auto-selected strategy: {self.pruning_strategy}")
        else:
            logger.info(f"Using specified strategy: {self.pruning_strategy}")

        # 1. Load expert selection
        self.selected_experts = self.load_selection_file()

        # 2. Load model
        self.load_model()

        # 3. Execute pruning based on strategy
        if self.pruning_strategy in ["gate_bias", "both"]:
            self.modify_gates()
        
        if self.pruning_strategy in ["zero_weights", "both"]:
            self.zero_out_experts()

        # 4. Save model
        self.save_model()

        logger.info("=" * 70)
        logger.info("Pruning completed!")
        logger.info("=" * 70)
        logger.info(f"Pruned model saved to: {self.output_dir}")
        
        if self.pruning_strategy == "zero_weights":
            logger.info("✓ Used zero_weights strategy: expert weights have been zeroed out")
        elif self.pruning_strategy == "gate_bias":
            logger.info("✓ Used gate_bias strategy: pruned experts will not be selected by router")
        else:
            logger.info("✓ Used both strategy: gate modified and weights zeroed out")


def main():
    parser = argparse.ArgumentParser(
        description="MoE Model Expert Pruning Tool - Supports multiple pruning strategies"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Original model path",
    )
    parser.add_argument(
        "--selection_file",
        type=str,
        required=True,
        help="Expert selection JSON file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output model save directory",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy: auto, balanced, cuda:0, cpu",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "zero_weights", "gate_bias", "both"],
        help="Pruning strategy: auto (auto-select based on method), zero_weights, gate_bias, both",
    )
    # Backward compatible parameter
    parser.add_argument("--device", type=str, default=None, help="(deprecated)")

    args = parser.parse_args()
    
    device_map = args.device_map
    if args.device is not None:
        logger.warning("--device parameter is deprecated, please use --device_map")
        device_map = args.device

    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return

    if not os.path.exists(args.selection_file):
        logger.error(f"Expert selection file does not exist: {args.selection_file}")
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
