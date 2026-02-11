import torch
import torch.nn as nn
import os
import sys
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from modelscope.hub.snapshot_download import snapshot_download
import bitsandbytes as bnb

print("✅ All libraries imported successfully! / 所有库导入成功！\n📝 Usage example: python train_rl.py 16 1000\n(First arg: TinyLoRA u value, Second arg: max training samples)")

# ========== argument parsing ==========
# ========== 命令行参数：u 值 和 最大样本数 ==========
U_VALUE = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAX_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

print(f"TinyLoRA u value / u值: {U_VALUE}")
if MAX_SAMPLES is not None:
    print(f"Max training samples / 最大训练样本数: {MAX_SAMPLES}")
else:
    print(f"Max training samples / 最大训练样本数: unlimited")

# ========== Dataset Configuration ==========
# ========== 数据集配置 ==========
# Filter configuration for deepmind/code_contests dataset
# 用于 deepmind/code_contests 数据集的过滤配置
# source: The original source of the problem, with possible values including UNKNOWN_SOURCE (0),CODECHEF (1), CODEFORCES (2), HACKEREARTH (3), CODEJAM (4), ATCODER (5) and AIZU (6).
# difficulty: A representation of the difficulty of the problem with possible values including UNKNOWN_DIFFICULTY (0), EASY (1), MEDIUM (2), HARD (3), HARDER (4), HARDEST (5), EXTERNAL (6), A (7), B (8), C (9), D (10), E (11), F (12), G (13), H (14), I (15), J (16), K (17), L (18), M (19), N (20), O (21), P (22), Q (23), R (24), S (25), T (26), U (27) and V (28). Note that different sources use different, non-comparable gradings. For Codeforces problems, cf_rating is a more reliable measure of difficulty when available.
# Structure / 结构:
#   Key: source ID (integer) / 键：数据源 ID（整数）
#   Value: list of allowed difficulty IDs / 值：允许的难度 ID 列表
#

# Difficulty mapping reference (ignored cf_rating for now to keep it simple):
# 难度映射参考（为简化起见暂时忽略 cf_rating）：
#   Source 2 (Codeforces) & Source 5 (AtCoder):
#     7=A, 8=B, 9=C, 10=D, 11=E, 12=F, 13=G, 14=H...
#   Source 1 & Source 3 (Other platforms):
#     1=EASY, 2=MEDIUM, 3=HARD, 4=VERY_HARD...
#
DATASET_CONFIG = {
    2: [7, 8],      # Codeforces: A-B level (Introductory) / Codeforces：A-B 级别（入门）
    5: [7, 8],      # AtCoder: A-B level (Introductory) / AtCoder：A-B 级别（入门）
    1: [1],         # General platforms: EASY only / 通用平台：仅简单难度
    3: [1],         # General platforms: EASY only / 通用平台：仅简单难度
}

# ========== Reward Scaling Configuration ==========
# ========== 奖励缩放配置 ==========
# Hierarchical scaling: Source (1st) -> Difficulty (2nd)
# 层级缩放：数据源（第一关键字） -> 难度（第二关键字）
# Note: These multipliers are applied to the base reward (0.5 for compile, up to 1.0 for pass)
# 注意：这些倍数应用于基础奖励（编译成功 0.5，通过所有测试最高 1.0）
REWARD_SCALING_CONFIG = {
    2: {          # Codeforces
        7: 1.0,   # A level: baseline / A级：基准
        8: 1.1,   # B level: slightly higher / B级：略高
    },
    5: {          # AtCoder
        7: 1.0,   # A level: baseline / A级：基准
        8: 1.1,   # B level: slightly higher / B级：略高
    },
    1: { 1: 1.0 }, # General platforms: EASY / 通用平台：简单
    3: { 1: 1.0 }, # General platforms: EASY / 通用平台：简单
}

# ========== Model Configuration ==========
# ========== 模型配置 ==========
MS_MODEL_ID = "qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-3B-Instruct"
OUTPUT_DIR = "./output/luoguqwencoder-lora"



#  Qwen2.5-Coder-3B-Instruct
# ========== 下载模型 ==========
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"Download from ModelScope/从ModelScope下载模型 {MS_MODEL_ID} 到 {LOCAL_MODEL_DIR}...")
    snapshot_download(
        repo_id=MS_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print("模型下载完成！")
else:
    print(f"Load from local/本地已存在模型，直接加载：{LOCAL_MODEL_DIR}")

# ========== Load tokenizer =========
# ========== 加载 tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ========== Load model =====

# ========== 加载模型（4bit 量化）==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16,
    dtype=torch.bfloat16,
)
model.config.use_cache = False

# 准备模型进行 k-bit 训练
model = prepare_model_for_kbit_training(model)


# ========== Define TinyLoRA Layers ==========
# ========== 定义 TinyLoRA 层 ==========

# 获取模型第一层的设备 (通常是 cuda:0)
device = model.model.layers[0].self_attn.q_proj.weight.device
print(f"Model device/模型主设备: {device}")

# 【修复错误1】创建一个 wrapper module 来正确注册 global_v
class TinyLoRAGlobalParams(nn.Module):
    """专门用于注册全局共享向量的容器"""
    def __init__(self, u_dim=16, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        # 这样注册才会被 model.named_parameters() 识别
        self.global_v = nn.Parameter(torch.zeros(u_dim, device=device, dtype=dtype))
    
    def forward(self):
        # 容器模块不需要实际的前向逻辑
        return self.global_v

# 创建全局参数容器
global_params = TinyLoRAGlobalParams(u_dim=U_VALUE, device=device, dtype=torch.bfloat16)


class TinyLoRALinear(nn.Module):
    def __init__(self, original_layer, rank = 2, u = None, global_params_ref=None):
        if u is None:
            u = U_VALUE

    # R= v_1 P_1 + v_2 P_2 + ... + v_u P_u
    # v都是scalar
    # P都是rank x rank的矩阵
    # global_params_ref: 指向包含 global_v 的容器模块

        super().__init__()
        # 必先继承父类的初始化函数，才能使用 nn.Module 的功能（例如注册参数和缓冲区）。
        
        #  super().__init__() 是什么？
        # 这是 Python 面向对象编程（OOP）的标准写法。
        # 含义：调用父类（Parent Class）的初始化函数。
        # 在这里的作用：你的类 TinyLoRALinear 继承自 nn.Module（PyTorch 的神经网络基类）。执行 super().__init__() 是为了让 PyTorch 的机制生效，比如：
        # 注册你定义的 self.v 为可训练参数。
        # 注册 self.U, self.S 等为 Buffer（不训练的参数）。


        print(f"original_layer.device: {original_layer.weight.device}, dtype: {original_layer.weight.dtype}")

        original_device = original_layer.weight.device # 记录原device


        self.base_layer = original_layer
        
      
        if global_params_ref is None:
            raise RuntimeError("必须传入 global_params_ref！")
        self.global_params_ref = global_params_ref

        W = original_layer.weight.data.float()
        if hasattr(original_layer.weight, "quant_state"):
         
            W_real = bnb.functional.dequantize_4bit(
                original_layer.weight.data, 
                original_layer.weight.quant_state,
                quant_type="nf4"  # 与 BitsAndBytesConfig 中的配置一致
            )
        else:
            # 非量化情况
            W_real = original_layer.weight.data


        W_real_on_cpu = W_real.float().cpu()

        U, S ,Vh = torch.linalg.svd( W_real_on_cpu ,full_matrices=False)

        # SVD 分解 W 矩阵
        # W = U S Vh 
        # Vh是 V的Hermitian transposed，共轭转置
        # 冻结 U, S, V (LoRA-XS 的骨架)

        

        # 将结果转回 BFloat16 并移回 GPU
        # 截断并注册(即固定住)
        # 建议转回 bf16 省显存
        # 
        # 这一步也是为了让 TinyLoRA 的参数和主模型精度保持一致
        
        target_dtype = torch.bfloat16

        self.register_buffer('U', U[:, :rank].to(original_device).to(target_dtype)) 
        self.register_buffer('S', torch.diag(S[:rank]).to(original_device).to(target_dtype))
        self.register_buffer('Vh', Vh[:rank, :].to(original_device).to(target_dtype))
        
        # 固定随机矩阵 P  (For TinyLoRA)
        self.register_buffer('P', torch.randn(u, rank, rank, device=original_device, dtype=target_dtype))

    def forward(self, x):
        # 动态从容器中获取 global_v，而不是作为自己的属性
        # 这样确保 v 只被 model.tiny_lora_params 注册一次
        v = self.global_params_ref.global_v
        
        # 计算 TinyLoRA 的增量矩阵 R = sum_i(v_i * P_i)
        # 注意：不能用 'u, urr -> rr'，因为 einsum 输出中同一下标不能重复
        # 必须用不同字母区分两个 rank 维度
        R = torch.einsum('u, uij -> ij', v, self.P)
        # 重组增量权重
        delta_W = self.U @ self.S @ R @ self.Vh
        # 前向传播：x * (W + delta_W)^T
        return self.base_layer(x) + x @ delta_W.t()


def apply_tiny_lora(model, global_params_ref):
    """
    遍历模型，将所有目标 Linear 层替换为 TinyLoRALinear，
    并传入对 global_params 容器的引用，实现论文中的 Tiling (全参数共享)。
    """
    # Qwen/Llama 的目标模块名称通常包含这些
    target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 计数器
    replaced_count = 0
    
    # 递归函数：遍历子模块
    for name, child in model.named_children():
        # 如果是目标 Linear 层
        if isinstance(child, (nn.Linear, bnb.nn.Linear4bit)) and any(name.endswith(s) for s in target_suffixes):
            # 1. 创建 TinyLoRA 层，传入 global_params 容器的引用
            new_layer = TinyLoRALinear(child, rank=2, u=U_VALUE, global_params_ref=global_params_ref)
            
            # 2. 替换掉原模块 (Monkey Patch)
            setattr(model, name, new_layer)
            replaced_count += 1
            print(f"✅ Replace successfully/已替换: {name} -> TinyLoRA (Shared)")
            
        else:
            # 继续递归遍历子模块 (例如 model.layers.0.self_attn...)
            replaced_count += apply_tiny_lora(child, global_params_ref)
            
    return replaced_count

# ========== 执行替换 ==========
print("Start replacing/正在应用 TinyLoRA Tiling (参数共享)...")

print("It's normal to see many lines of 'replace'./看到很多替换日志是正常的。")
# 【关键】固定随机种子，确保 P 矩阵可复现
# 保存模型时只存 v 向量，加载时需要用相同种子重建 P 矩阵
TINYLORA_SEED = 42
torch.manual_seed(TINYLORA_SEED)
torch.cuda.manual_seed(TINYLORA_SEED)
print(f"✅ Fix TinyLoRA seed/已固定 TinyLoRA 随机种子: {TINYLORA_SEED}")

# 【关键修复】先将 global_params 注册为模型的子模块
# 这样在层替换时，TinyLoRALinear 就能通过引用访问到已注册的 global_v
model.tiny_lora_params = global_params
print(f"✅ Register global_params to model/已将 global_params 注册到模型")

# 然后再进行层替换，传入 global_params 容器本身
total_replaced = apply_tiny_lora(model, global_params)
print(f"✅ Replace completed/替换完成！共替换了 {total_replaced} 个模块。")

# ========== 关键步骤：冻结除 v 以外的所有参数 ==========
print("Freezing parameters/正在冻结模型参数...")

# 【更优雅的方案】直接通过对象引用操作，不依赖字符串匹配
# 1. 第一步：全局冻结所有参数
model.requires_grad_(False)

# 2. 第二步：精准解冻 global_v
# 直接通过对象引用操作，绝对稳健
global_params.global_v.requires_grad = True
print(f"✅ Trainable parameter/可训练参数: global_v, shape={global_params.global_v.shape}")

# 验证可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters/总参数量: {all_params:,}")
print(f"Trainable parameters/可训练参数量: {trainable_params}")
if trainable_params != U_VALUE:
    raise RuntimeError(f"警告：可训练参数数量为 {trainable_params}，预期为 {U_VALUE}！")

import re
import subprocess
import tempfile
import os

def compile_and_run(code, test_cases):
    """
    Compile and run code against multiple test cases, return reward / 编译并运行代码
    test_cases: list of dicts, each containing 'input' and 'output' / 测试用例列表
    Return: 0.0 (compile fail) / 0.5 (partial pass) / 1.0 (all pass)
    返回：0.0（编译失败） / 0.5（部分通过） / 1.0（全部通过）
    """
    code = re.sub(r'freopen\s*\(.*?\);', '', code, flags=re.IGNORECASE)
    
    # Create temp directory / 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "solution.cpp")
        exe_file = os.path.join(temp_dir, "solution")
        
        # Write C++ code / 写入 C++ 代码
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write(code)
            
        # Compile with -O2 optimization / 编译
        try:
            compile_result = subprocess.run(
                ['g++', src_file, '-o', exe_file, '-O2'],
                capture_output=True, text=True, timeout=5
            )
            if compile_result.returncode != 0:
                return 0.0  # Compile failed / 编译失败
        except subprocess.TimeoutExpired:
            return 0.0  # Compile timeout / 编译超时

        # Run all test cases / 运行所有测试用例
        passed = 0
        for test_case in test_cases:
            input_data = test_case['input']
            expected_output = test_case['output'].strip()
            
            try:
                run_result = subprocess.run(
                    [exe_file],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=2 
                )
                
                actual_output = run_result.stdout.strip()
                
                if actual_output == expected_output:
                    passed += 1
                    
            except (subprocess.TimeoutExpired, Exception):
                pass  # Test case failed / 测试用例失败
        
        # Return score: 1.0 if all passed, 0.5 if partial, 0.0 if none
        if passed == len(test_cases):
            return 1.0
        elif passed > 0:
            return 0.6 + float(passed/len(test_cases))*0.4  # Partial pass / 部分通过
        else:
            return 0.5  # At least compiled / 至少编译成功

def code_reward_func(prompts, completions, public_tests=None, private_tests=None, generated_tests=None, source=None, difficulty=None, **kwargs):
    """
    GRPO reward function for code evaluation / GRPO 的奖励函数
    
    For deepmind/code_contests dataset:
    - public_tests, private_tests, generated_tests: dicts with 'input' and 'output' as lists
    - We evaluate against public_tests, private_tests, and generated_tests
    
    对于 deepmind/code_contests 数据集：
    - public_tests, private_tests, generated_tests：包含 'input' 和 'output' 列表的字典
    - 对 public_tests, private_tests 和 generated_tests 进行评估
    
    Reward rules / 奖励规则：
    - Compile fail or invalid format: 0.0 / 编译失败或无效格式：0.0
    - Compile success, partial tests pass: 0.5 / 编译成功，部分测试通过：0.5
    - All tests pass: 1.0 / 所有测试通过：1.0
    """
    rewards = []
    
    # Convert None to empty lists / 将 None 转换为空列表
    if public_tests is None:
        public_tests = [None] * len(completions)
    if private_tests is None:
        private_tests = [None] * len(completions)
    if generated_tests is None:
        generated_tests = [None] * len(completions)
    if source is None:
        source = [0] * len(completions)
    if difficulty is None:
        difficulty = [0] * len(completions)
    
    # Iterate through each generated completion / 遍历每一条生成的回复
    for completion, pub_test, priv_test, gen_test, src, diff in zip(
        completions, public_tests, private_tests, generated_tests, source, difficulty
    ):
        # 1. Extract code block / 提取代码块
        match = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", completion, re.DOTALL)
        
        if not match:
            # Fallback: look for raw code with #include / 备选：寻找包含 #include 的裸代码
            if "#include" in completion:
                code = completion
            else:
                rewards.append(0.0)  # Invalid format / 格式完全不对
                continue
        else:
            code = match.group(1)

        # 2. Parse test cases from dict format / 从字典格式解析测试用例
        test_cases_list = []
        
        # Public tests: dict with 'input' and 'output' as lists / 公开测试
        if pub_test and isinstance(pub_test, dict) and 'input' in pub_test and 'output' in pub_test:
            inputs = pub_test['input'] if isinstance(pub_test['input'], list) else [pub_test['input']]
            outputs = pub_test['output'] if isinstance(pub_test['output'], list) else [pub_test['output']]
            for inp, out in zip(inputs, outputs):
                test_cases_list.append({'input': inp, 'output': out})
        
        # Private tests: dict with 'input' and 'output' as lists / 私有测试
        if priv_test and isinstance(priv_test, dict) and 'input' in priv_test and 'output' in priv_test:
            inputs = priv_test['input'] if isinstance(priv_test['input'], list) else [priv_test['input']]
            outputs = priv_test['output'] if isinstance(priv_test['output'], list) else [priv_test['output']]
            for inp, out in zip(inputs, outputs):
                test_cases_list.append({'input': inp, 'output': out})
        
        # Generated tests: dict with 'input' and 'output' as lists / 生成的测试
        if gen_test and isinstance(gen_test, dict) and 'input' in gen_test and 'output' in gen_test:
            inputs = gen_test['input'] if isinstance(gen_test['input'], list) else [gen_test['input']]
            outputs = gen_test['output'] if isinstance(gen_test['output'], list) else [gen_test['output']]
            for inp, out in zip(inputs, outputs):
                test_cases_list.append({'input': inp, 'output': out})
        
        # If no test cases extracted, give penalty / 如果没有提取到测试用例，给予惩罚
        if not test_cases_list:
            rewards.append(0.0)
            continue
        
        # 3. Run code against all test cases / 对所有测试用例运行代码
        base_reward = compile_and_run(code, test_cases_list)
        
        # 4. REWARD SCALING - Adjust based on source and difficulty
        # 奖励缩放 - 根据源和难度从 REWARD_SCALING_CONFIG 查找
        # ============================================================================
        # Hierarchical lookup: Source -> Difficulty
        reward_multiplier = 1.0
        
        if src in REWARD_SCALING_CONFIG:
            source_scaling = REWARD_SCALING_CONFIG[src]
            if diff in source_scaling:
                reward_multiplier = source_scaling[diff]
            elif -1 in source_scaling: # Fallback for unknown difficulty in known source
                reward_multiplier = source_scaling[-1]
        
        # ============================================================================
        # Apply multiplier to base reward / 对基础奖励应用倍数
        final_reward = base_reward * reward_multiplier
        rewards.append(final_reward)
        
    return rewards
    

# 【最终方案】绕过 Trainer 对纯量化模型的检查
# Trainer 的检查逻辑 (transformers/trainer.py):
#   _is_quantized_and_base_model = model.is_quantized AND NOT model._hf_peft_config_loaded
#   if _is_quantized_and_base_model and not isinstance(model, PeftModel): raise ValueError
#
# 我们的 TinyLoRA 是合法的 adapter（只训练 16 个参数），但不是标准 PeftModel。
# 设置 _hf_peft_config_loaded = True 让第一道检查直接为 False，不会走到 isinstance 判断。
# 这不影响实际计算——权重已经在内存中量化，TinyLoRA 层正确处理了反量化。
model._hf_peft_config_loaded = True

print("✅ Set _hf_peft_config_loaded=True / 已设置 _hf_peft_config_loaded=True：bypass Trainer quantization check")

def filter_dataset(dataset, config, max_samples, seed=42):
    """
    Filter dataset based on source and difficulty configuration.
    根据数据源和难度配置过滤数据集。
    
    Args:
        dataset: HuggingFace Dataset object / HuggingFace 数据集对象
        config: Dict mapping source IDs to allowed difficulty lists / 将数据源 ID 映射到允许的难度列表的字典
        max_samples: Maximum number of samples after filtering / 过滤后的最大样本数
        seed: Random seed for shuffling / 用于打乱的随机种子
    
    Returns:
        Filtered and sampled dataset / 过滤并采样后的数据集
    """
    print("\n" + "="*60)
    print("🔍 Filtering dataset based on configuration...")
    print("🔍 根据配置过滤数据集...")
    print("="*60)
    
    # Log configuration / 记录配置
    source_names = {
        1: "General Platform 1 / 通用平台 1",
        2: "Codeforces / Codeforces",
        3: "General Platform 3 / 通用平台 3",
        5: "AtCoder / AtCoder",
    }
    
    for source_id, difficulties in config.items():
        source_name = source_names.get(source_id, f"Source {source_id} / 数据源 {source_id}")
        print(f"📌 {source_name}: Keeping difficulties {difficulties} / 保留难度 {difficulties}")
    
    # Filter function / 过滤函数
    def should_keep(example):
        source = example.get('source', -1)
        difficulty = example.get('difficulty', -1)
        
        # Check if source is in config / 检查数据源是否在配置中
        if source not in config:
            return False
        
        # Check if difficulty is allowed for this source / 检查该数据源是否允许此难度
        if difficulty not in config[source]:
            return False
        
        return True
    
    # Apply filter / 应用过滤
    print("\n⏳ Filtering in progress... / 正在过滤...")
    original_size = len(dataset)
    filtered_dataset = dataset.filter(should_keep)
    filtered_size = len(filtered_dataset)
    
    print(f"✅ Original dataset size / 原始数据集大小: {original_size:,}")
    print(f"✅ After filtering / 过滤后: {filtered_size:,} samples / 样本")
    print(f"📊 Retention rate / 保留率: {filtered_size/original_size*100:.2f}%")
    
    # Apply max_samples limit with shuffling / 应用最大样本数限制并打乱
    if filtered_size > max_samples:
        print(f"\n🎲 Shuffling and sampling {max_samples:,} from {filtered_size:,}...")
        print(f"🎲 打乱并从 {filtered_size:,} 中采样 {max_samples:,} 个...")
        filtered_dataset = filtered_dataset.shuffle(seed=seed).select(range(max_samples))
        final_size = len(filtered_dataset)
        print(f"✅ Final training set size / 最终训练集大小: {final_size:,}")
    else:
        print(f"\n✅ All {filtered_size:,} filtered samples will be used (below max_samples limit).")
        print(f"✅ 将使用全部 {filtered_size:,} 个过滤后的样本（低于最大样本数限制）。")
        final_size = filtered_size
    
    print("="*60 + "\n")
    return filtered_dataset


def apply_chat_template(example):
    """
    Build prompt from problem description and public test cases.
    For deepmind/code_contests dataset structure.
    
    从问题描述和公开测试用例构建提示。
    适用于 deepmind/code_contests 数据集结构。
    """
    # Extract problem description / 提取问题描述
    description = example.get('description', '')
    
    # Build public test cases section / 构建公开测试用例部分
    public_tests_section = ""
    public_tests = example.get('public_tests', {})
    
    if isinstance(public_tests, dict) and 'input' in public_tests and 'output' in public_tests:
        inputs = public_tests['input'] if isinstance(public_tests['input'], list) else [public_tests['input']]
        outputs = public_tests['output'] if isinstance(public_tests['output'], list) else [public_tests['output']]
        
        if inputs and outputs:
            public_tests_section = "\n【Cases】\n"
            for i, (inp, out) in enumerate(zip(inputs, outputs), 1):
                public_tests_section += f"Test {i}:\n"
                public_tests_section += f"Input :\n{inp}\n"
                public_tests_section += f"Output:\n{out}\n"
    
    # Combine into final prompt / 组合成最终提示
    final_prompt = f"""You will be given a programming contest problem. Please reason step by step and provide a complete C++ implementation.
Output the solution in a code block. Do not include debugging info or extra output. Limit reasoning to 128 tokens.


【Problem Description 】
{description}

{public_tests_section}

Please provide your C++ solution :"""
    
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



# ========== Load dataset ==========
# When using load_dataset("json", data_files="....jsonl"),
# HuggingFace defaults to classifying the file as 'train' split.
# Note: data_files points to processed file path from download_dataset.py
# split="train" is important! Trainer needs Dataset object, not DatasetDict
# 
# ========== 加载数据集 ==========

rl_dataset = load_dataset(
    "json", 
    data_files="./local_code_contests/code_contests_train.jsonl",
    split="train"
)

# Apply filtering based on source and difficulty configuration
# 根据数据源和难度配置应用过滤
rl_dataset = filter_dataset(
    dataset=rl_dataset,
    config=DATASET_CONFIG,
    max_samples=MAX_SAMPLES,
    seed=TINYLORA_SEED
)



# Apply template / 应用模版
rl_dataset = rl_dataset.map(apply_chat_template)

# Print sample to verify / 打印一条数据验证
print(f"✅ Dataset loaded successfully! / 数据加载成功！Total samples / 样本数量: {len(rl_dataset)}")
print(f"📝 Sample data / 样例数据: {rl_dataset[0]}")



# ========== Configure and start GRPO training ==========
# Configure GRPO / 配置 GRPO
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Set to 1 if GPU memory insufficient / 显存不足设为 1
    gradient_accumulation_steps=8,  # Accumulate to simulate large batch / 累积梯度模拟大 Batch
    learning_rate=1e-5,             # RL learning rate usually small / RL 学习率通常要小
    num_generations=4,              # Group Size (G): sample 4 answers per iteration / 每次采样 4 个答案
    max_completion_length=1024,     # Max generation length / 生成的最大长度
    logging_steps=1,
    bf16=True,                      # Enable BF16 acceleration / 开启 BF16 加速
    save_strategy="no",             # Disable auto checkpoint (TinyLoRA is non-standard PEFT)
)

# Initialize trainer / 初始化训练器
trainer = GRPOTrainer(
    model=model,
    reward_funcs=code_reward_func,  # Your judge function / 你的判题函数
    args=training_args,
    train_dataset=rl_dataset,       # Processed data / 处理好的数据
    processing_class=tokenizer,     # Tokenizer
)

# Start training! / 开始训练！
print("🚀 Starting TinyLoRA-RL training... / 开始 TinyLoRA-RL 训练...")
trainer.train()

# Save training results / 保存训练结果
# Note: peft's save_pretrained may not recognize custom layers
# Manually save global_v and metadata needed to rebuild model
# 注意：peft 的 save_pretrained 可能不认你的自定义层
# 手动保存 global_v 以及重建模型所需的元信息
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_dict = {
    "global_v": global_params.global_v.data,  # Trained v vector / 训练好的 v 向量
    "u_value": U_VALUE,                        # Dimension of v / v 的维度
    "rank": 2,                                 # TinyLoRA rank
    "seed": TINYLORA_SEED,                     # P matrix random seed (for reproducibility)
    "model_id": MS_MODEL_ID,                   # Base model ID / 基座模型 ID
    "total_replaced_layers": total_replaced,   # Number of replaced layers / 替换的层数
}
torch.save(save_dict, f"{OUTPUT_DIR}/tiny_lora_v.pt")
print(f"✅ Training complete! / 训练完成！Parameters saved to / 参数已保存至 {OUTPUT_DIR}/tiny_lora_v.pt")
print(f"📊 Save contents / 保存内容: global_v (shape={global_params.global_v.shape}), u={U_VALUE}, rank=2, seed={TINYLORA_SEED}")