import torch
import torch.nn as nn
import os
import sys
import re
import subprocess
import tempfile
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from modelscope.hub.snapshot_download import snapshot_download
import bitsandbytes as bnb

# Import shared utilities / 导入共享工具
from utils import (
    TinyLoRAGlobalParams,
    TinyLoRALinear,
    apply_tiny_lora,
    compile_and_run,
    convert_hf_tests_to_list,
    extract_code_from_response,
    apply_chat_template,
)

print("✅ All libraries imported successfully! / 所有库导入成功！")

# ========== argument parsing with argparse ==========
import argparse

parser = argparse.ArgumentParser(
    description="TinyLoRA-RL Training Script / TinyLoRA-RL 训练脚本",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples / 示例:
  # Basic training
  python train_rl.py 32 2000

  # Training with validation
  python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10

  # Training without quantization
  python train_rl.py 32 2000 --no_quant

  # Training with custom rank
  python train_rl.py 32 2000 --rank 4

  # Training with DeepCoder dataset
  python train_rl.py 32 2000 --dataset deepcoder
    """
)

parser.add_argument(
    "u_value",
    nargs="?",
    type=int,
    default=16,
    help="TinyLoRA parameter count / TinyLoRA 参数数量 (default: 16)"
)
parser.add_argument(
    "max_samples",
    nargs="?",
    type=int,
    default=2000,
    help="Maximum training samples / 最大训练样本数 (default: 2000)"
)
parser.add_argument(
    "--do_validate",
    action="store_true",
    help="Enable validation during training / 开启训练中验证"
)
parser.add_argument(
    "--val_steps",
    type=int,
    default=100,
    help="Run validation every N steps / 每N步运行验证 (default: 100)"
)
parser.add_argument(
    "--val_samples",
    type=int,
    default=10,
    help="Number of validation samples / 验证样本数 (default: 10)"
)
parser.add_argument(
    "--rank",
    type=int,
    default=2,
    help="TinyLoRA SVD rank / TinyLoRA SVD 秩 (default: 2)"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="code_contests",
    choices=["code_contests", "deepcoder"],
    help="Dataset to use / 要使用的数据集: 'code_contests' or 'deepcoder' (default: code_contests)"
)
parser.add_argument(
    "--no_quant",
    action="store_true",
    help="Disable 4-bit quantization, load model in BF16 / 禁用4-bit量化，以BF16加载模型"
)

args = parser.parse_args()

# Assign parsed arguments / 分配解析后的参数
U_VALUE = args.u_value
MAX_SAMPLES = args.max_samples
DO_VALIDATE = args.do_validate
VAL_STEPS = args.val_steps
VAL_SAMPLES = args.val_samples
TINYLORA_RANK = args.rank
DATASET_NAME = args.dataset
USE_QUANT = not args.no_quant

print(f"\n{'='*60}")
print(f"📋 Training Configuration / 训练配置")
print(f"{'='*60}")
print(f"TinyLoRA u value / u值: {U_VALUE}")
print(f"TinyLoRA rank / TinyLoRA 秩: {TINYLORA_RANK}")
if MAX_SAMPLES is not None:
    print(f"Max training samples / 最大训练样本数: {MAX_SAMPLES}")
else:
    print(f"Max training samples / 最大训练样本数: unlimited")
print(f"Dataset / 数据集: {DATASET_NAME}")
print(f"Quantization / 量化加载: {'4-bit (NF4)' if USE_QUANT else 'BF16 (no quant)'}")
print(f"Validation enabled / 启用验证: {DO_VALIDATE}")
if DO_VALIDATE:
    print(f"Validation frequency / 验证频率: every {VAL_STEPS} steps / 每 {VAL_STEPS} 步")
    print(f"Validation samples / 验证样本数: {VAL_SAMPLES}")
print(f"{'='*60}\n")

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

MS_MODEL_ID = "qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-3B-Instruct"
OUTPUT_DIR = "./output/luoguqwencoder-lora"



# ============ Load Model =============
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
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ========== Multi-GPU / DDP Support ==========
# When using torchrun (DDP), each process must load the FULL model on its own GPU.
# device_map="auto" would split the model across GPUs, conflicting with DDP.
# 使用 torchrun (DDP) 时，每个进程必须在自己的 GPU 上加载完整模型。
# device_map="auto" 会将模型分片到多张 GPU 上，与 DDP 冲突。
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
print(f"🖥️ LOCAL_RANK: {LOCAL_RANK}")

# ========== Load model and whether USE_QUANT =====


if USE_QUANT:
    print("📦 Loading model with 4-bit quantization / 以 4-bit 量化加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        quantization_config=bnb_config,
        device_map={"":  LOCAL_RANK},  
        # 多卡DDP: 每个rank加载完整模型到自己的GPU / Multi-GPU DDP: each rank loads full model on its own GPU
        trust_remote_code=True,
        dtype=torch.bfloat16,
        # dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
else:
    print("📦 Loading model without quantization (BF16) / 以 BF16 无量化加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map={"":  LOCAL_RANK},
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False


# ========== Define TinyLoRA Layers ==========
# ========== 定义 TinyLoRA 层 ==========
# Note: TinyLoRA classes are now imported from utils.py
# 注意：TinyLoRA 类现在从 utils.py 导入

# 获取当前 rank 对应的 GPU 设备
# Get the GPU device for current rank
device = torch.device(f"cuda:{LOCAL_RANK}")
print(f"Model device/模型主设备: {device}")

# 创建全局参数容器
global_params = TinyLoRAGlobalParams(u_dim=U_VALUE, device=device, dtype=torch.bfloat16)

# ========== 执行替换 ==========
print("Start replacing/正在应用 TinyLoRA Tiling (参数共享)...")

print("It's normal to see many lines of 'replace'./看到很多替换日志是正常的。")
# 【关键】固定随机种子，确保 P 矩阵可复现
# 保存模型时只存 v 向量，加载时需要用相同种子重建 P 矩阵

# =======  SEED FIX FOR REPRODUCIBILITY / 可复现性种子固定 =======
TINYLORA_SEED = 20260306
torch.manual_seed(TINYLORA_SEED)
torch.cuda.manual_seed(TINYLORA_SEED)
print(f"✅ Fix TinyLoRA seed/已固定 TinyLoRA 随机种子: {TINYLORA_SEED}")

# 【关键修复】先将 global_params 注册为模型的子模块
# 这样在层替换时，TinyLoRALinear 就能通过引用访问到已注册的 global_v
model.tiny_lora_params = global_params
print(f"✅ Register global_params to model/已将 global_params 注册到模型")

# 然后再进行层替换，传入 global_params 容器本身
total_replaced = apply_tiny_lora(model, global_params, rank=TINYLORA_RANK)
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


# ========== Code Reward Function ==========
# Note: compile_and_run is now imported from utils.py / 注意：compile_and_run 现在从 utils.py 导入


def convert_io_to_test_cases(io_data):
    """
    Convert DeepCoder's tests format to test cases list.
    DeepCoder format: {"inputs": [...], "outputs": [...]}
    Returns: [{"input": ..., "output": ...}, ...]
    """
    if not io_data:
        return []

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])

    # Ensure lists
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(outputs, list):
        outputs = [outputs]

    test_cases = []
    for inp, out in zip(inputs, outputs):
        test_cases.append({"input": inp, "output": out})

    return test_cases


def code_reward_func(prompts, completions, public_tests=None, private_tests=None, generated_tests=None, source=None, difficulty=None, input_output=None, **kwargs):
    """
    GRPO reward function for code evaluation / GRPO 的奖励函数

    Supports both:
    1. deepmind/code_contests dataset:
       - public_tests, private_tests, generated_tests: dicts with 'input' and 'output' as lists

    2. DeepCoder dataset (agentica-org/DeepCoder-Preview-Dataset):
       - input_output: dict with 'inputs' and 'outputs' as lists

    Reward rules / 奖励规则：
    - Compile fail or invalid format: 0.0 / 编译失败或无效格式：0.0
    - Compile success, partial tests pass: 0.5-0.99 / 编译成功，部分测试通过：0.5-0.99
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
    if input_output is None:
        input_output = [None] * len(completions)
    if source is None:
        source = [0] * len(completions)
    if difficulty is None:
        difficulty = [0] * len(completions)

    # Iterate through each generated completion / 遍历每一条生成的回复
    for i, (completion, pub_test, priv_test, gen_test, io_test, src, diff) in enumerate(zip(
        completions, public_tests, private_tests, generated_tests, input_output, source, difficulty
    )):
        # 1. Extract code using utility function / 使用工具函数提取代码
        code = extract_code_from_response(completion)

        if not code:
            rewards.append(0.0)  # No valid code found / 未找到有效代码
            continue

        # 2. Parse test cases - support both dataset formats
        test_cases_list = []

        # Try DeepCoder format first (input_output)
        if io_test is not None:
            test_cases_list.extend(convert_io_to_test_cases(io_test))

        # Fallback to CodeContests format (public/private/generated tests)
        if not test_cases_list:
            test_cases_list.extend(convert_hf_tests_to_list(pub_test))
            test_cases_list.extend(convert_hf_tests_to_list(priv_test))
            test_cases_list.extend(convert_hf_tests_to_list(gen_test))

        # If no test cases extracted, give penalty / 如果没有提取到测试用例，给予惩罚
        if not test_cases_list:
            rewards.append(0.0)
            continue

        # Limit test cases during training to avoid excessive subprocess fork pressure
        # (each subprocess.run forks the large model process; too many forks → OOM killer)
        # 限制测试用例数量避免大量fork导致OOM：训练中5个测试用例足以提供奖励信号
        test_cases_list = test_cases_list[:5]

        # 3. Run code against all test cases / 对所有测试用例运行代码
        base_reward = compile_and_run(code, test_cases_list)

        # 4. REWARD SCALING - Adjust based on source and difficulty
        # 奖励缩放 - 根据源和难度从 REWARD_SCALING_CONFIG 查找
        reward_multiplier = 1.0

        # DeepCoder dataset doesn't have source/difficulty, skip scaling
        if src != 0 or diff != 0:  # Only apply scaling for CodeContests
            if src in REWARD_SCALING_CONFIG:
                source_scaling = REWARD_SCALING_CONFIG[src]
                if diff in source_scaling:
                    reward_multiplier = source_scaling[diff]
                elif -1 in source_scaling:  # Fallback for unknown difficulty
                    reward_multiplier = source_scaling[-1]

        final_reward = base_reward * reward_multiplier
        rewards.append(final_reward)

    return rewards
        
       

# 绕过 Trainer 对纯量化模型的检查（仅在量化模式下需要）
# Trainer 的检查逻辑 (transformers/trainer.py):
#   _is_quantized_and_base_model = model.is_quantized AND NOT model._hf_peft_config_loaded
#   if _is_quantized_and_base_model and not isinstance(model, PeftModel): raise ValueError
#
# 我们的 TinyLoRA 是合法的 adapter（只训练 16 个参数），但不是标准 PeftModel。
# 设置 _hf_peft_config_loaded = True 让第一道检查直接为 False，不会走到 isinstance 判断。
# 这不影响实际计算——权重已经在内存中量化，TinyLoRA 层正确处理了反量化。
if USE_QUANT:
    model._hf_peft_config_loaded = True
    print("✅ Set _hf_peft_config_loaded=True / 已设置 _hf_peft_config_loaded=True：bypass Trainer quantization check")
else:
    print("ℹ️ Non-quantized mode, no need to bypass Trainer check / 非量化模式，无需绕过 Trainer 检查")

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



# ========== Load dataset ==========
# When using load_dataset("json", data_files="....jsonl"),
# HuggingFace defaults to classifying the file as 'train' split.
# Note: data_files points to processed file path from download_dataset.py
# split="train" is important! Trainer needs Dataset object, not DatasetDict
#


# Select dataset based on DATASET_NAME

if DATASET_NAME == 'deepcoder':
    # DeepCoder has multiple configs: codeforces, lcbv5, primeintellect, taco
    print(f"📂 Loading DeepCoder dataset from ./local_Deep-Coder-Preview-Dataset/")

    # Load all config files
    deepcoder_files = [
        "./local_Deep-Coder-Preview-Dataset/deepcoder_codeforces_train.jsonl",
        "./local_Deep-Coder-Preview-Dataset/deepcoder_lcbv5_train.jsonl",
        "./local_Deep-Coder-Preview-Dataset/deepcoder_primeintellect_train.jsonl",
        "./local_Deep-Coder-Preview-Dataset/deepcoder_taco_train.jsonl",
    ]

    # Check which files exist
    existing_files = [f for f in deepcoder_files if os.path.exists(f)]
    if not existing_files:
        raise FileNotFoundError(
            f"DeepCoder dataset files not found! Please run download_DeepCoder-Preview-Dataset.py first.\n"
            f"Expected files: {deepcoder_files}"
        )

    print(f"   Loading from: {[os.path.basename(f) for f in existing_files]}")

    # Use streaming to avoid OOM - load samples one by one
    # 使用streaming避免OOM - 逐个加载样本
    from datasets import Dataset

    # First, count total samples to estimate memory needs
    print("   Counting total samples (streaming)...")
    total_samples = 0
    for f in existing_files:
        with open(f, 'r', encoding='utf-8') as fp:
            for _ in fp:
                total_samples += 1
    print(f"   Total samples available: {total_samples}")

    # Sample MAX_SAMPLES directly using streaming
    # 直接使用streaming采样MAX_SAMPLES个样本
    print(f"   Streaming and sampling {MAX_SAMPLES} samples...")

    import random
    import json as _json
    random.seed(TINYLORA_SEED)

    # Reservoir sampling via direct file I/O (avoids HuggingFace Arrow overhead)
    # 通过直接文件读取做reservoir采样（避免HuggingFace Arrow格式转换开销）
    reservoir = []
    sample_count = 0
    for f in existing_files:
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                sample = _json.loads(line)

                # Trim input_output to first 5 test cases at load time
                # Some samples carry huge test cases (200K+ chars each), causing
                # multi-GB memory bloat → OOM kill.  5 cases suffice for reward signal.
                io = sample.get("input_output")
                if isinstance(io, dict):
                    inputs  = io.get("inputs",  [])[:5]
                    outputs = io.get("outputs", [])[:5]
                    sample["input_output"] = {"inputs": inputs, "outputs": outputs}

                sample_count += 1
                if len(reservoir) < MAX_SAMPLES:
                    reservoir.append(sample)
                else:
                    idx = random.randint(0, sample_count - 1)
                    if idx < MAX_SAMPLES:
                        reservoir[idx] = sample
    rl_dataset = Dataset.from_list(reservoir)

    print(f"✅ Loaded {len(rl_dataset)} DeepCoder samples (streaming)")

elif DATASET_NAME == 'code_contests':
    print(f"📂 Loading CodeContests dataset from ./local_code_contests/code_contests_train.jsonl")
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
    print(f"✅ Loaded {len(rl_dataset)} CodeContests samples")

else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}. Choose 'deepcoder' or 'code_contests'")

# Apply template / 应用模版
# Use batched=False and num_proc=1 to reduce memory usage
print("⏳ Applying chat template...")
# Columns needed by code_reward_func must be preserved after map
# reward함수에 필요한 컬럼은 map 이후에도 보존해야 함
_reward_columns = {'prompt', 'input_output', 'public_tests', 'private_tests', 'generated_tests', 'source', 'difficulty'}
rl_dataset = rl_dataset.map(
    lambda x: apply_chat_template(x, tokenizer),
    batched=False,
    num_proc=1,
    remove_columns=[c for c in rl_dataset.column_names if c not in _reward_columns]
)

# Print sample to verify / 打印一条数据验证
print(f"✅ Dataset loaded successfully! / 数据加载成功！Total samples / 样本数量: {len(rl_dataset)}")
print(f"📝 Sample data / 样例数据: {rl_dataset[0]}")



# ========== Load Validation Dataset (if enabled) ==========

val_dataset = None
if DO_VALIDATE:
    print(f"\n{'='*60}")
    print(f"📊 Loading validation dataset / 正在加载验证数据集...")
    print(f"{'='*60}\n")

    # DeepCoder dataset does not have validation split
    if DATASET_NAME == 'deepcoder':
        print("⚠️  Warning / 警告:")
        print("   DeepCoder dataset does not have validation data.")
        print("   DeepCoder 数据集没有验证集，已自动跳过验证。")
        print("   Validation is only available for code_contests dataset.")
        print("   验证功能仅支持 code_contests 数据集。")
        print(f"{'='*60}\n")
        DO_VALIDATE = False  # Disable validation for deepcoder
    else:
        # CodeContests dataset has validation data
        val_dataset = load_dataset(
            "json",
            data_files="./local_code_contests/code_contests_valid.jsonl",
            split="train"
        )

        # Apply chat template to validation dataset / 对验证数据集应用模板
        val_dataset = val_dataset.map(lambda x: apply_chat_template(x, tokenizer))

        print(f"✅ Validation dataset loaded / 验证数据集加载成功: {len(val_dataset)} samples / 样本\n")


# ========== Define Validation Callback ==========

class ValidationCallback(TrainerCallback):
    """
    Custom callback for validation during training / 训练期间验证的自定义回调
    Tracks best model and saves checkpoint / 跟踪最佳模型并保存检查点
    """
    def __init__(self, val_dataset, val_steps, val_samples, output_dir, global_params, u_value, rank, seed, model_id, total_replaced):
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.val_samples = val_samples
        self.output_dir = output_dir
        self.global_params = global_params
        self.u_value = u_value
        self.rank = rank
        self.seed = seed
        self.model_id = model_id
        self.total_replaced = total_replaced
        self.best_score = 0.0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Run validation at specified intervals / 在指定间隔运行验证"""
        if state.global_step % self.val_steps == 0 and state.global_step > 0:
            print(f"\n{'='*80}")
            print(f"🔍 Running validation at step {state.global_step} / 在第 {state.global_step} 步运行验证")
            print(f"{'='*80}\n")
            
            # Import validation function / 导入验证函数
            from validate import run_validation
            
            # Get model and tokenizer from kwargs / 从 kwargs 获取模型和分词器
            model = kwargs.get('model')
            tokenizer = kwargs.get('processing_class') or kwargs.get('tokenizer')
            
            # Run validation / 运行验证
            results = run_validation(
                model=model,
                tokenizer=tokenizer,
                dataset=self.val_dataset,
                num_samples=self.val_samples,
            )
            
            # Check if this is the best model / 检查是否是最佳模型
            current_score = results['pass_at_1']
            
            if current_score > self.best_score:
                self.best_score = current_score
                print(f"\n{'='*80}")
                print(f"🎉 New best model! / 新的最佳模型！")
                print(f"   Previous best Pass@1 / 之前最佳通过率: {self.best_score:.2%}")
                print(f"   Current Pass@1 / 当前通过率: {current_score:.2%}")
                print(f"{'='*80}\n")
                
                # Save best model / 保存最佳模型
                best_save_dict = {
                    "global_v": self.global_params.global_v.data.clone(),
                    "u_value": self.u_value,
                    "rank": self.rank,
                    "seed": self.seed,
                    "model_id": self.model_id,
                    "total_replaced_layers": self.total_replaced,
                    "is_quantized": USE_QUANT,
                    "validation_score": current_score,
                    "step": state.global_step,
                }
                
                best_path = f"{self.output_dir}/best_tiny_lora_v.pt"
                torch.save(best_save_dict, best_path)
                print(f"💾 Best model saved to / 最佳模型已保存至: {best_path}")
                print(f"📊 Validation Pass@1 / 验证通过率: {current_score:.2%}\n")
            else:
                print(f"📊 Current Pass@1: {current_score:.2%} (Best: {self.best_score:.2%})\n")
        
        return control

# Prepare callbacks / 准备回调
callbacks = []
if DO_VALIDATE and val_dataset is not None:
    validation_callback = ValidationCallback(
        val_dataset=val_dataset,
        val_steps=VAL_STEPS,
        val_samples=VAL_SAMPLES,
        output_dir=OUTPUT_DIR,
        global_params=global_params,
        u_value=U_VALUE,
        rank=TINYLORA_RANK,
        seed=TINYLORA_SEED,
        model_id=MS_MODEL_ID,
        total_replaced=total_replaced,
    )
    callbacks.append(validation_callback)
    print(f"✅ Validation callback registered / 验证回调已注册")
    print(f"   Validation frequency / 验证频率: every {VAL_STEPS} steps / 每 {VAL_STEPS} 步")
    print(f"   Validation samples / 验证样本数: {VAL_SAMPLES}\n")



# ========== Configure and start GRPO training ==========
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



    # 👇======================================================👇
    # updated in v3.2 - Clip High (from DeepCoder/DAPO paper)
    # Clip High: asymmetric clipping - increase upper bound to encourage more exploration
    # Standard GRPO: clip range [1-ε, 1+ε], typically ε=0.2 or 0.3
    # Clip High: clip range [1-ε, 1+ε_high] where ε_high > ε, enabling larger policy updates
    # when correct solutions are found, preventing premature convergence
    beta=0.0,                       # deprecate KL penalty: empower the model to explore more without constraints
    epsilon=0.05,                    # lower bound clipping (1 - epsilon)
    epsilon_high=0.95,               # upper bound clipping (1 + epsilon_high) - Clip High!

    # deprecated in v3.1+, kept for backward compatibility
    # clip_range=0.3,               # expand Clip range / 扩大 Clip 范围
    

    # 👇=======================================================👇
    # updated in v4.0 - num_iterations should be larger than 1
    num_iterations=4,              # Total training iterations / 总训练迭代次数
)

# Initialize trainer / 初始化训练器
trainer = GRPOTrainer(
    model=model,
    reward_funcs=code_reward_func,  # Your judge function / 你的判题函数
    args=training_args,
    train_dataset=rl_dataset,       # Processed data / 处理好的数据
    processing_class=tokenizer,     # Tokenizer
    callbacks=callbacks,            # Add validation callback / 添加验证回调
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
    "rank": TINYLORA_RANK,                     # TinyLoRA rank / TinyLoRA 秩
    "seed": TINYLORA_SEED,                     # P matrix random seed (for reproducibility)
    "model_id": MS_MODEL_ID,                   # Base model ID / 基座模型 ID
    "total_replaced_layers": total_replaced,   # Number of replaced layers / 替换的层数
    "is_quantized": USE_QUANT,                 # Whether model was loaded with 4-bit quantization / 模型是否以 4-bit 量化加载
}
torch.save(save_dict, f"{OUTPUT_DIR}/tiny_lora_v.pt")
print(f"✅ Training complete! / 训练完成！Parameters saved to / 参数已保存至 {OUTPUT_DIR}/tiny_lora_v.pt")
print(f"📊 Save contents / 保存内容: global_v (shape={global_params.global_v.shape}), u={U_VALUE}, rank={TINYLORA_RANK}, seed={TINYLORA_SEED}")

# Print all GRPO-related parameters / 输出所有GRPO相关参数
print("\n" + "="*50)
print("📋 GRPO Parameters / GRPO 参数:")
print("="*50)
print(f"  global_v shape: {global_params.global_v.shape}")
print(f"  global_v values: {global_params.global_v.data}")
print(f"  u_dim (U_VALUE): {U_VALUE}")
print(f"  rank (TINYLORA_RANK): {TINYLORA_RANK}")
print(f"  seed (TINYLORA_SEED): {TINYLORA_SEED}")
print(f"  total_replaced_layers: {total_replaced}")
print(f"  is_quantized: {USE_QUANT}")
print(f"  model_id: {MS_MODEL_ID}")
print("="*50)