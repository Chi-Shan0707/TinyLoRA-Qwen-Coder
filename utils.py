"""
tiny_lora_utils.py - Shared Utilities for TinyLoRA Training, Validation and Testing
TinyLoRA 共享工具模块 - 用于训练、验证和测试

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
    """专门用于注册全局共享向量的容器 / Container for global shared vector"""
    def __init__(self, u_dim=16, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        # 【关键】必须初始化为零，确保初始 ΔW=0，模型从未扰动状态开始
        # CRITICAL: Must init to zeros so initial ΔW=0 and model starts unperturbed
        self.global_v = nn.Parameter(torch.zeros(u_dim, device=device, dtype=dtype))
    
    def forward(self):
        """不实际调用，仅用于注册参数 / Not actually called, only for parameter registration"""
        return self.global_v


class TinyLoRALinear(nn.Module):
    def __init__(self, original_layer, rank=2, u=None, global_params_ref=None):
        """
        TinyLoRA Linear Layer with global parameter sharing / 支持全局参数共享的 TinyLoRA 线性层
        
        Args:
            original_layer: Original Linear layer to be replaced / 被替换的原始线性层
            rank: Rank for TinyLoRA (default=2) / TinyLoRA 的秩（默认为 2）
            u: Dimension of global shared vector / 全局共享向量的维度
            global_params_ref: Reference to TinyLoRAGlobalParams container / TinyLoRAGlobalParams 容器的引用
        """
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        
        if global_params_ref is None:
            raise ValueError("global_params_ref cannot be None / global_params_ref 不能为 None")
        self.global_params_ref = global_params_ref
        u_dim = global_params_ref.global_v.shape[0]
        
        # Get weight from original layer / 从原始层获取权重
        W = original_layer.weight.data
        device = W.device
        
        # Detect if this is a quantized layer / 检测是否为量化层
        self._is_quantized = hasattr(original_layer, 'quant_state')
        
        # Handle 4-bit quantized weights / 处理 4-bit 量化权重
        if self._is_quantized:
            # 【关键修改 - 多卡兼容】保留原始 bnb Linear4bit 层，
            # 让 bitsandbytes 自己处理前向传播（内部已兼容分布式内存布局）。
            # 不再手动 dequantize + F.linear，避免多卡下 CUBLAS_STATUS_NOT_SUPPORTED。
            # [KEY FIX - Multi-GPU] Keep original bnb Linear4bit layer and delegate
            # base forward pass to bitsandbytes (internally handles distributed memory layout).
            # No more manual dequantize + F.linear, avoiding CUBLAS_STATUS_NOT_SUPPORTED on multi-GPU.
            self.base_layer = original_layer
            
            # Dequantize only for SVD computation (one-time, on CPU)
            # 仅为 SVD 计算进行反量化（一次性，在 CPU 上）
            from bitsandbytes.functional import dequantize_4bit
            qs = original_layer.quant_state
            W_dequant = dequantize_4bit(
                W, quant_state=qs, quant_type="nf4"
            ).to(torch.float32).cpu()
        else:
            W_dequant = W.to(torch.float32).cpu()
            self.register_buffer('W_base', W.to(torch.bfloat16))
            
            if original_layer.bias is not None:
                self.register_buffer('bias', original_layer.bias.data.to(torch.bfloat16))
            else:
                self.bias = None
        
        # Perform SVD (deterministic operation) / 执行 SVD（确定性运算）
        try:
            U, S, Vh = torch.linalg.svd(W_dequant, full_matrices=False)
        except Exception as e:
            print(f"⚠️ SVD failed, using zeros / SVD 失败，使用零矩阵: {e}")
            U = torch.zeros(self.out_features, min(self.out_features, self.in_features))
            S = torch.zeros(min(self.out_features, self.in_features))
            Vh = torch.zeros(min(self.out_features, self.in_features), self.in_features)
        
        # Keep only top 'rank' components / 仅保留前 'rank' 个成分
        U_r = U[:, :self.rank]
        S_r = S[:self.rank]
        Vh_r = Vh[:self.rank, :]
        
        # Register as buffers (frozen, not trainable) / 注册为 buffer（冻结，不可训练）
        self.register_buffer('U', U_r.to(torch.bfloat16).to(device))
        self.register_buffer('S', S_r.to(torch.bfloat16).to(device))
        self.register_buffer('Vh', Vh_r.to(torch.bfloat16).to(device))
        
        # Generate fixed random projection matrix P / 生成固定随机投影矩阵 P
        # NOTE: This uses the global random seed set before apply_tiny_lora
        # 注意：这使用在 apply_tiny_lora 之前设置的全局随机种子
        # Scale by 1/sqrt(rank) per paper recommendation to control variance
        # 按论文建议除以 sqrt(rank) 以控制方差
        P = torch.randn(u_dim, self.rank, device=device, dtype=torch.bfloat16) / (self.rank ** 0.5)
        self.register_buffer('P', P)

    def _compute_delta(self, compute_dtype):
        """
        Compute TinyLoRA delta weight matrix: ΔW = U @ diag(S * (v @ P)) @ Vh
        计算 TinyLoRA 增量权重矩阵
        
        All intermediate tensors are made .contiguous() to ensure cuBLAS compatibility
        in distributed (multi-GPU / cluster) environments.
        所有中间张量都调用 .contiguous() 以确保在分布式（多卡/集群）环境下与 cuBLAS 兼容。
        """
        v = self.global_params_ref.global_v.to(compute_dtype)
        vP = v @ self.P.to(compute_dtype)
        S_vP = self.S.to(compute_dtype) * vP
        delta = (self.U.to(compute_dtype) @ torch.diag(S_vP) @ self.Vh.to(compute_dtype)).contiguous()
        return delta

    def forward(self, x):
        """Forward pass with TinyLoRA delta / TinyLoRA 增量的前向传播"""
        orig_dtype = x.dtype

        if self._is_quantized:
            # === Quantized path (multi-GPU safe) ===
            # 【关键】让 bitsandbytes 的 Linear4bit 自己处理 base forward，
            # 它内部已处理量化/反量化以及分布式内存兼容性，不会触发 cuBLAS 错误。
            # [KEY] Let bitsandbytes Linear4bit handle base forward internally.
            # It handles quantization/dequantization and distributed memory compatibility,
            # avoiding CUBLAS_STATUS_NOT_SUPPORTED errors on multi-GPU.
            out = self.base_layer(x)
            compute_dtype = out.dtype

            # Compute TinyLoRA delta with contiguous tensors for distributed safety
            # 用 contiguous 张量计算 TinyLoRA delta 以确保分布式安全
            x_cast = x.to(compute_dtype).contiguous()
            delta = self._compute_delta(compute_dtype)

            out = out + torch.nn.functional.linear(x_cast, delta, None)

            return out.to(orig_dtype)
        else:
            # === Non-quantized path ===
            # Use W_base dtype (bfloat16) as compute dtype
            compute_dtype = self.W_base.dtype
            x_cast = x.to(compute_dtype).contiguous()
            out = torch.nn.functional.linear(x_cast, self.W_base.contiguous(), None)

            # Compute TinyLoRA delta with contiguous tensors
            delta = self._compute_delta(compute_dtype)

            out = out + torch.nn.functional.linear(x_cast, delta, None)

            if self.bias is not None:
                out = out + self.bias.to(compute_dtype)

            return out.to(orig_dtype)


def apply_tiny_lora(model, global_params_ref, rank=2):
    """
    遍历模型，将所有目标 Linear 层替换为 TinyLoRALinear，
    并传入对 global_params 容器的引用，实现论文中的 Tiling (全参数共享)。
    
    Traverse the model and replace all target Linear layers with TinyLoRALinear,
    passing reference to global_params container to achieve Tiling (full parameter sharing) from the paper.
    
    Args:
        rank: Rank for TinyLoRA SVD decomposition (default=2) / TinyLoRA SVD 分解的秩（默认为 2）
    """
    # Qwen/Llama target module names / Qwen/Llama 目标模块名称
    target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    replaced_count = 0
    
    # Recursive function to traverse submodules / 递归函数遍历子模块
    for name, child in model.named_children():
        # Check if this is a target layer / 检查是否为目标层
        if isinstance(child, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes):
            # Replace with TinyLoRALinear / 替换为 TinyLoRALinear
            tiny_lora_layer = TinyLoRALinear(child, rank=rank, global_params_ref=global_params_ref)
            setattr(model, name, tiny_lora_layer)
            replaced_count += 1
            print(f"  ✓ Replaced / 已替换: {name}")
        else:
            # Recursively process child modules / 递归处理子模块
            replaced_count += apply_tiny_lora(child, global_params_ref, rank=rank)
            
    return replaced_count


# ========== Code Compilation and Execution ==========

def compile_and_run(code, test_cases, timeout=2):
    """
    Compile and run C++ code against multiple test cases, return reward
    编译并运行 C++ 代码，对多个测试用例进行评测，返回奖励
    
    Args:
        code: C++ source code / C++ 源代码
        test_cases: list of dicts, each containing 'input' and 'output' / 测试用例列表
        timeout: execution timeout in seconds / 执行超时时间（秒）
    
    Returns:
        float: 0.0 (compile fail) / 0.5 (partial pass) / 1.0 (all pass)
               0.0（编译失败） / 0.5（部分通过） / 1.0（全部通过）
    """
    # Remove freopen statements / 移除 freopen 语句
    code = re.sub(r'freopen\s*\(.*?\);', '', code, flags=re.IGNORECASE)
    
    # Create temp directory / 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        src_path = os.path.join(temp_dir, "solution.cpp")
        exe_path = os.path.join(temp_dir, "solution.out")
        
        # Write source file / 写入源文件
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        # Compile / 编译
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
                return 0.0  # Compilation failed / 编译失败
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0
        
        # Run test cases / 运行测试用例
        passed = 0
        total = len(test_cases)
        
        if total == 0:
            return 0.5  # No test cases, give partial credit / 无测试用例，给部分分数
        
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
        
        # Calculate reward / 计算奖励
        # Base: 0.5 for compile success, then 0.5 * (passed / total) for test cases
        # 基础：编译成功后 0.5 分，然后每个测试用例通过加 0.5 * (k/N)
        if passed == 0:
            return 0.5  # Compiled but no tests passed / 编译成功但未通过测试
        elif passed == total:
            return 1.0  # All tests passed / 全部通过
        else:
            # Partial pass: 0.5 + 0.5 * (passed / total)
            return 0.5 + 0.5 * (passed / total)


def convert_hf_tests_to_list(hf_tests):
    """
    Convert HuggingFace test format to list of test cases
    将 HuggingFace 测试格式转换为测试用例列表
    
    Args:
        hf_tests: dict with 'input' and 'output' as lists / 包含 'input' 和 'output' 列表的字典
    
    Returns:
        list of dicts, each with 'input' and 'output' / 测试用例列表
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
    Load model and tokenizer with 4-bit quantization
    加载 4-bit 量化的模型和分词器
    
    Args:
        model_path: Path to the model / 模型路径
        use_4bit: Whether to use 4-bit quantization / 是否使用 4-bit 量化
        for_inference: If True, enable KV cache and skip gradient checkpointing
                       如果为 True，启用 KV 缓存并跳过梯度检查点
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"\n{'='*60}")
    print(f"📦 Loading model from / 从以下路径加载模型: {model_path}")
    print(f"{'='*60}\n")
    
    # Load tokenizer / 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure quantization / 配置量化
    # Multi-GPU DDP: use LOCAL_RANK to place full model on each rank's GPU
    # 多卡 DDP：使用 LOCAL_RANK 将完整模型放置在每个 rank 的 GPU 上
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}
    
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
            dtype=torch.bfloat16,
        )
        
        if for_inference:
            # Inference mode: enable KV cache, skip gradient checkpointing
            # 推理模式：启用 KV 缓存，跳过梯度检查点
            model.config.use_cache = True
        else:
            # Training mode: disable KV cache, prepare for kbit training
            # 训练模式：禁用 KV 缓存，准备 k-bit 训练
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    
    print(f"✅ Model and tokenizer loaded successfully / 模型和分词器加载成功\n")
    
    return model, tokenizer


def extract_code_from_response(response):
    """
    Extract C++ code from model response
    从模型响应中提取 C++ 代码
    
    Args:
        response: Model generated text / 模型生成的文本
    
    Returns:
        str: Extracted C++ code or empty string / 提取的 C++ 代码或空字符串
    """
    # Try to extract code from markdown code block / 尝试从 markdown 代码块提取
    patterns = [
        r'```cpp\s*(.*?)\s*```',
        r'```c\+\+\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no code block found, try to find code with #include / 如果没有找到代码块，尝试查找包含 #include 的代码
    if '#include' in response:
        # Extract from first #include to the end / 从第一个 #include 提取到末尾
        start = response.find('#include')
        return response[start:].strip()
    
    return ""


def apply_chat_template(example, tokenizer):
    """
    Build prompt from problem description and public test cases.
    Supports both:
    - deepmind/code_contests dataset structure (public_tests, private_tests, generated_tests)
    - DeepCoder dataset structure (input_output)

    从问题描述和公开测试用例构建提示。
    支持两种数据集格式。

    Args:
        example: Dataset example / 数据集样本
        tokenizer: Tokenizer for applying chat template / 用于应用聊天模板的分词器

    Returns:
        dict: Example with 'prompt' field added / 添加了 'prompt' 字段的样本
    """
    # Extract problem description / 提取问题描述
    # Truncate long descriptions to save memory
    description = example.get('description', '')
    if len(description) > 8000:  # Limit description length
        description = description[:8000] + "...(truncated)"

    # Combine into final prompt / 组合成最终提示
    # 注意：不拼接输入输出样例，避免某些样本的测试用例巨大导致 prompt 过长
    final_prompt = f"""You will be given a programming contest problem. Please reason step by step and provide a complete C++ implementation.
Output the solution in a code block. Do not include debugging info or extra output. Limit reasoning to 128 tokens.


【Problem Description】
{description}

Please provide your C++ solution:"""
    
    # Build Qwen chat template format / 构建 Qwen 聊天模板格式
    messages = [
        {"role": "system", "content": "You are an expert competitive programmer. Output valid C++ code that compiles and solves the problem correctly."},
        {"role": "user", "content": final_prompt}
    ]
    
    # Apply chat template using tokenizer / 使用分词器应用聊天模板
    example['prompt'] = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return example
