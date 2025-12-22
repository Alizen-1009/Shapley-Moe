import torch
import json
import os
import logging
from typing import Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExpertInference:
    def __init__(self, model_path: str, selection_json_path: str, device: str = "auto"):
        self.model_path = model_path
        self.selection_json_path = selection_json_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.selected_experts = self._load_selection_file()
        self.hooks = []

    def _load_selection_file(self) -> Dict[str, List[int]]:
        logger.info(f"Loading selected experts from {self.selection_json_path}")
        with open(self.selection_json_path, "r") as f:
            data = json.load(f)
        return data

    def load_model(self):
        logger.info(f"Loading model from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=self.device,
                trust_remote_code=True,
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        logger.info("All hooks removed.")

    def prune_experts(self):

        for name, module in self.model.named_modules():

            gate_module = None
            if hasattr(module, "gate") and isinstance(module.gate, torch.nn.Module):
                gate_module = module.gate
            elif hasattr(module, "router") and isinstance(
                module.router, torch.nn.Module
            ):
                gate_module = module.router
            if gate_module is None and name.endswith("mlp"):
                if hasattr(module, "router") and isinstance(
                    module.router, torch.nn.Module
                ):
                    gate_module = module.router
                elif hasattr(module, "gate") and isinstance(
                    module.gate, torch.nn.Module
                ):
                    gate_module = module.gate

            if gate_module is not None:
                layer_idx = self._extract_layer_index(name)

                if layer_idx is not None and str(layer_idx) in self.selected_experts:
                    selected = set(self.selected_experts[str(layer_idx)])
                    self._register_router_weight_zero_hook(
                        gate_module, selected, layer_idx
                    )

    def _register_router_weight_zero_hook(
        self, gate_module: torch.nn.Module, selected_indices: set, layer_idx: int
    ):
        """
        Register a hook on the router/gate module that zeros out the routing weights
        for unselected experts. This allows the original top-k selection to proceed,
        but ensures unselected experts have zero weight in the final output.
        """
        num_experts = gate_module.weight.shape[0]
        unselected_indices = [
            i for i in range(num_experts) if i not in selected_indices
        ]

        def hook_fn(module, args, output):
            # The router typically outputs (routing_weights, selected_experts) or just routing_weights
            # We need to zero out the weights for unselected experts

            if isinstance(output, tuple):
                # Format: (routing_weights, selected_experts, ...)
                routing_weights = output[0]

                # Zero out unselected experts' weights
                # routing_weights shape is typically [batch_size, seq_len, num_experts] or [batch_size * seq_len, num_experts]
                if unselected_indices:
                    routing_weights = routing_weights.clone()
                    routing_weights[..., unselected_indices] = 0.0

                return (routing_weights,) + output[1:]
            else:
                # Just routing weights
                routing_weights = output.clone()
                if unselected_indices:
                    routing_weights[..., unselected_indices] = 0.0
                return routing_weights

        handle = gate_module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        logger.info(
            f"Layer {layer_idx}: Registered hook to zero routing weights for {len(unselected_indices)} unselected experts."
        )

    def _extract_layer_index(self, module_name: str) -> Optional[int]:
        parts = module_name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts):
                return int(parts[idx + 1])
        if "h" in parts:
            idx = parts.index("h")
            if idx + 1 < len(parts):
                return int(parts[idx + 1])

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs):
        """
        Generate text using the model with proper chat template formatting.

        Args:
            prompt: Input text or question
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional arguments to pass to model.generate()
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Format as chat message if tokenizer has chat template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_input = prompt

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(
            self.model.device
        )

        # Default generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # Greedy decoding by default
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part (exclude input)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return generated_text


if __name__ == "__main__":
    # Example usage
    MODEL_ID = "/root/yuhao/hf_models/OLMOE-7B-Instruct"
    SELECTION_FILE = "expert_select/results/OLMOE-7B-Instruct_gsm8k_25_shapley/selected_experts_per_layer_rate60.json"

    if os.path.exists(SELECTION_FILE):
        inference = ExpertInference(MODEL_ID, SELECTION_FILE)
        inference.load_model()

        # Prune experts by zeroing routing weights
        inference.prune_experts()

        # Test generation
        print("\n" + "=" * 50)
        print("Test Generation with Expert Pruning")
        print("=" * 50)

        test_prompts = [
            "What is the capital of France?",
            # "What is 1 + 1 = ?",
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            print(f"Response: ", end="")
            res = inference.generate(prompt, max_new_tokens=128)
            print(res)
            print("-" * 50)
    else:
        print(f"Selection file not found: {SELECTION_FILE}")
