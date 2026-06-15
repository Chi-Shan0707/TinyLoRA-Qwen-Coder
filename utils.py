"""
tiny_lora_utils.py - Shared utilities for TinyLoRA training, validation, and testing.

This module contains:
- TinyLoRAGlobalParams and TinyLoRALinear classes
- apply_tiny_lora function for injecting TinyLoRA into a model
- compile_and_run function for C++ code evaluation
- Model and tokenizer loading utilities
"""

import os
import re
import sys
import subprocess
import tempfile
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


# ========== TinyLoRA Classes ==========

class TinyLoRAGlobalParams(nn.Module):
    """Container for the global shared vector."""
    def __init__(self, u_dim=16, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        # CRITICAL: Must init to zeros so initial DeltaW=0 and model starts unperturbed
        self.global_v = nn.Parameter(torch.zeros(u_dim, device=device, dtype=dtype))
    
    def forward(self):
        """Return the registered global vector."""
        return self.global_v

    def get_projection_bank(self, rank, device, dtype):
        """
        Return the shared fixed TinyLoRA projection bank for a rank.

        The bank is sampled once and then reused by every TinyLoRA-wrapped layer,
        so the only trainable degrees of freedom remain the shared global_v vector.
        """
        buffer_name = f"projection_bank_rank_{rank}"
        if not hasattr(self, buffer_name):
            u_dim = self.global_v.shape[0]
            projection_bank = torch.randn(u_dim, rank, rank, dtype=torch.float32) / (rank ** 0.5)
            self.register_buffer(buffer_name, projection_bank, persistent=False)
        return getattr(self, buffer_name).to(device=device, dtype=dtype)


class TinyLoRALinear(nn.Module):
    def __init__(self, original_layer, rank=2, u=None, global_params_ref=None):
        """
        TinyLoRA Linear Layer with global parameter sharing
        
        Args:
            original_layer: Original Linear layer to be replaced
            rank: Rank for TinyLoRA (default=2)
            u: Dimension of global shared vector
            global_params_ref: Reference to TinyLoRAGlobalParams container
        """
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        
        if global_params_ref is None:
            raise ValueError("global_params_ref cannot be None")
        self.global_params_ref = global_params_ref
        
        # Get weight from original layer
        W = original_layer.weight.data
        device = W.device
        buffer_dtype = torch.bfloat16
        
        # Detect if this is a quantized layer
        self._is_quantized = hasattr(original_layer, 'quant_state')
        
        # Handle 4-bit quantized weights
        if self._is_quantized:
            # [KEY FIX - Multi-GPU] Keep original bnb Linear4bit layer and delegate
            # base forward pass to bitsandbytes (internally handles distributed memory layout).
            # No more manual dequantize + F.linear, avoiding CUBLAS_STATUS_NOT_SUPPORTED on multi-GPU.
            self.base_layer = original_layer
            
            # Dequantize only for SVD computation (one-time, on CPU)
            from bitsandbytes.functional import dequantize_4bit
            qs = original_layer.quant_state
            W_dequant = dequantize_4bit(
                W, quant_state=qs, quant_type="nf4"
            ).to(torch.float32).cpu()
        else:
            W_dequant = W.to(torch.float32).cpu()
            buffer_dtype = W.dtype
            self.register_buffer('W_base', W.detach().clone().to(buffer_dtype))
            
            if original_layer.bias is not None:
                self.register_buffer('bias', original_layer.bias.data.detach().clone().to(buffer_dtype))
            else:
                self.bias = None
        
        # Perform SVD (deterministic operation)
        try:
            U, S, Vh = torch.linalg.svd(W_dequant, full_matrices=False)
        except Exception as e:
            print(f"[WARN] SVD failed, using zeros: {e}")
            U = torch.zeros(self.out_features, min(self.out_features, self.in_features))
            S = torch.zeros(min(self.out_features, self.in_features))
            Vh = torch.zeros(min(self.out_features, self.in_features), self.in_features)
        
        # Keep only top 'rank' components
        U_r = U[:, :self.rank]
        S_r = S[:self.rank]
        Vh_r = Vh[:self.rank, :]
        
        # Register as buffers (frozen, not trainable)
        self.register_buffer('U', U_r.to(buffer_dtype).to(device))
        self.register_buffer('S', S_r.to(buffer_dtype).to(device))
        self.register_buffer('Vh', Vh_r.to(buffer_dtype).to(device))
        
        # Reuse the shared fixed random projection banks P_i.
        # NOTE: This uses the global random seed set before apply_tiny_lora
        # Each P_i is an r x r matrix, matching the TinyLoRA paper:
        #   R = sum_i v_i P_i, DeltaW = U @ diag(S) @ R @ Vh
        # Scale by 1/sqrt(rank) to control variance.
        P = global_params_ref.get_projection_bank(self.rank, device=device, dtype=buffer_dtype)
        self.register_buffer('P', P)

    def _compute_delta_output(self, x, compute_dtype):
        """
        Compute TinyLoRA delta output without materializing the full delta weight.

        Paper form:
            R = sum_i v_i P_i
            DeltaW = U @ diag(S) @ R @ Vh
            Deltay = x @ DeltaW.T

        Association used here:
            Deltay = (((x @ Vh.T) @ R.T) * S) @ U.T
        
        All intermediate tensors are made contiguous for cuBLAS compatibility
        in distributed multi-GPU or cluster environments.
        """
        v = self.global_params_ref.global_v.to(compute_dtype)
        R = torch.einsum("u,urs->rs", v, self.P.to(compute_dtype)).contiguous()
        h = torch.nn.functional.linear(x, self.Vh.to(compute_dtype), None).contiguous()
        h = torch.nn.functional.linear(h, R, None).contiguous()
        h = h * self.S.to(compute_dtype)
        return torch.nn.functional.linear(h, self.U.to(compute_dtype), None)

    def forward(self, x):
        """Forward pass with TinyLoRA delta."""
        orig_dtype = x.dtype

        if self._is_quantized:
            # === Quantized path (multi-GPU safe) ===
            # [KEY] Let bitsandbytes Linear4bit handle base forward internally.
            # It handles quantization/dequantization and distributed memory compatibility,
            # avoiding CUBLAS_STATUS_NOT_SUPPORTED errors on multi-GPU.
            out = self.base_layer(x)
            compute_dtype = out.dtype

            # Compute TinyLoRA delta with contiguous tensors for distributed safety
            x_cast = x.to(compute_dtype).contiguous()
            out = out + self._compute_delta_output(x_cast, compute_dtype)

            return out.to(orig_dtype)
        else:
            # === Non-quantized path ===
            # Use W_base dtype (bfloat16) as compute dtype
            compute_dtype = self.W_base.dtype
            x_cast = x.to(compute_dtype).contiguous()
            out = torch.nn.functional.linear(x_cast, self.W_base.contiguous(), None)

            # Compute TinyLoRA delta with contiguous tensors
            out = out + self._compute_delta_output(x_cast, compute_dtype)

            if self.bias is not None:
                out = out + self.bias.to(compute_dtype)

            return out.to(orig_dtype)


def apply_tiny_lora(model, global_params_ref, rank=2):
    """
    Traverse the model and replace all target Linear layers with TinyLoRALinear,
    passing reference to global_params container to achieve Tiling (full parameter sharing) from the paper.
    
    Args:
        rank: Rank for TinyLoRA SVD decomposition (default=2)
    """
    # Qwen/Llama target module names
    target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    replaced_count = 0
    
    # Recursive function to traverse submodules
    for name, child in model.named_children():
        # Check if this is a target layer
        if isinstance(child, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes):
            # Replace with TinyLoRALinear
            tiny_lora_layer = TinyLoRALinear(child, rank=rank, global_params_ref=global_params_ref)
            setattr(model, name, tiny_lora_layer)
            replaced_count += 1
            print(f"  - Replaced: {name}")
        else:
            # Recursively process child modules
            replaced_count += apply_tiny_lora(child, global_params_ref, rank=rank)
            
    return replaced_count


# ========== Code Compilation and Execution ==========

def compile_and_run(code, test_cases, timeout=2):
    """
    Compile and run C++ code against multiple test cases, then return reward.
    
    Args:
        code: C++ source code
        test_cases: list of dicts, each containing 'input' and 'output'
        timeout: execution timeout in seconds
    
    Returns:
        float: 0.0 (compile fail) / 0.5 (partial pass) / 1.0 (all pass)
    """
    # Remove freopen statements
    code = re.sub(r'freopen\s*\(.*?\);', '', code, flags=re.IGNORECASE)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        src_path = os.path.join(temp_dir, "solution.cpp")
        exe_path = os.path.join(temp_dir, "solution.out")
        
        # Write source file
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        # Compile
        compile_cmd = ["g++", "-O2", "-std=c++17", src_path, "-o", exe_path]
        try:
            result = subprocess.run(
                compile_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )
            if result.returncode != 0:
                return 0.0  # Compilation failed
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0
        
        # Run test cases
        passed = 0
        total = len(test_cases)
        
        if total == 0:
            return 0.5  # No test cases, give partial credit
        
        for tc in test_cases:
            input_data = tc.get('input', '')
            expected_output = tc.get('output', '').strip()
            
            try:
                result = subprocess.run(
                    [exe_path],
                    input=input_data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    text=True
                )
                
                actual_output = result.stdout.strip()
                
                if actual_output == expected_output:
                    passed += 1
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue
        
        # Calculate reward
        # Base: 0.5 for compile success, then 0.5 * (passed / total) for test cases
        if passed == 0:
            return 0.5  # Compiled but no tests passed
        elif passed == total:
            return 1.0  # All tests passed
        else:
            # Partial pass: 0.5 + 0.5 * (passed / total)
            return 0.5 + 0.5 * (passed / total)


def convert_hf_tests_to_list(hf_tests):
    """
    Convert HuggingFace test format to a list of test cases.
    
    Args:
        hf_tests: dict with 'input' and 'output' as lists
    
    Returns:
        list of dicts, each with 'input' and 'output'
    """
    if not isinstance(hf_tests, dict):
        return []
    
    inputs = hf_tests.get('input', [])
    outputs = hf_tests.get('output', [])
    
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []
    
    return [
        {'input': inp, 'output': out}
        for inp, out in zip(inputs, outputs)
    ]


# ========== Model Loading Utilities ==========

def get_model_and_tokenizer(model_path, use_4bit=True, for_inference=False):
    """
    Load model and tokenizer, optionally with 4-bit quantization.
    
    Args:
        model_path: Path to the model
        use_4bit: Whether to use 4-bit quantization
        for_inference: If True, enable KV cache and skip gradient checkpointing
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"\n{'='*60}")
    print(f"[LOAD] Loading model from: {model_path}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure device placement. Multi-GPU DDP uses LOCAL_RANK so each rank
    # holds the full model on its own GPU. CPU fallback keeps lightweight smoke
    # tests usable on machines without an NVIDIA driver.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    has_cuda = torch.cuda.is_available()
    device_map = {"": local_rank} if has_cuda else {"": "cpu"}
    model_dtype = torch.bfloat16 if has_cuda else torch.float32

    if use_4bit and not has_cuda:
        print("[WARN] CUDA is unavailable; disabling 4-bit quantization for CPU loading.")
        use_4bit = False
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            dtype=model_dtype,
        )
        
        if for_inference:
            # Inference mode: enable KV cache, skip gradient checkpointing
            model.config.use_cache = True
        else:
            # Training mode: disable KV cache, prepare for kbit training
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True,
            dtype=model_dtype,
        )
    
    print(f"[OK] Model and tokenizer loaded successfully")
    
    return model, tokenizer


def extract_code_from_response(response):
    """
    Extract C++ code from a model response.
    
    Args:
        response: Model generated text
    
    Returns:
        str: Extracted C++ code or empty string
    """
    # Try to extract code from markdown code block
    patterns = [
        r'```cpp\s*(.*?)\s*```',
        r'```c\+\+\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no code block found, try to find code with #include
    if '#include' in response:
        # Extract from first #include to the end
        start = response.find('#include')
        return response[start:].strip()
    
    return ""


def apply_chat_template(example, tokenizer):
    """
    Build prompt from problem description and public test cases.
    Supports both:
    - deepmind/code_contests dataset structure (public_tests, private_tests, generated_tests)
    - DeepCoder dataset structure (input_output)

    Args:
        example: Dataset example
        tokenizer: Tokenizer for applying chat template

    Returns:
        dict: Example with 'prompt' field added
    """
    # Extract problem description
    # Truncate long descriptions to save memory
    description = example.get('description', '')
    if len(description) > 8000:  # Limit description length
        description = description[:8000] + "...(truncated)"

    # Combine into final prompt
    final_prompt = f"""You will be given a programming contest problem. Please reason step by step and provide a complete C++ implementation.
Output the solution in a code block. Do not include debugging info or extra output. Limit reasoning to 128 tokens.


[Problem Description]
{description}

Please provide your C++ solution:"""
    
    # Build Qwen chat template format
    messages = [
        {"role": "system", "content": "You are an expert competitive programmer. Output valid C++ code that compiles and solves the problem correctly."},
        {"role": "user", "content": final_prompt}
    ]
    
    # Apply chat template using tokenizer
    example['prompt'] = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return example
