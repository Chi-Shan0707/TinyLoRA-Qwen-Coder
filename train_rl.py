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

# Import shared utilities
from utils import (
    TinyLoRAGlobalParams,
    TinyLoRALinear,
    apply_tiny_lora,
    compile_and_run,
    convert_hf_tests_to_list,
    extract_code_from_response,
    apply_chat_template,
)

print("[OK] All libraries imported successfully!")

# ========== argument parsing with argparse ==========
import argparse

parser = argparse.ArgumentParser(
    description="TinyLoRA-RL training script",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Basic training
  python train_rl.py 32 2000

  # Training with validation
  python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10

  # Training without quantization
  python train_rl.py 32 2000 --no_quant

  # Small-model smoke run
  python train_rl.py 8 2 --model_id Qwen/Qwen3-0.6B --local_model_dir ./models/Qwen3-0.6B --no_quant

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
    help="TinyLoRA parameter count (default: 16)"
)
parser.add_argument(
    "max_samples",
    nargs="?",
    type=int,
    default=2000,
    help="Maximum training samples (default: 2000)"
)
parser.add_argument(
    "--do_validate",
    action="store_true",
    help="Enable validation during training"
)
parser.add_argument(
    "--val_steps",
    type=int,
    default=100,
    help="Run validation every N steps (default: 100)"
)
parser.add_argument(
    "--val_samples",
    type=int,
    default=10,
    help="Number of validation samples (default: 10)"
)
parser.add_argument(
    "--rank",
    type=int,
    default=2,
    help="TinyLoRA SVD rank (default: 2)"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="code_contests",
    choices=["code_contests", "deepcoder"],
    help="Dataset to use: 'code_contests' or 'deepcoder' (default: code_contests)"
)
parser.add_argument(
    "--no_quant",
    action="store_true",
    help="Disable 4-bit quantization, load model in BF16"
)
parser.add_argument(
    "--model_id",
    type=str,
    default="qwen/Qwen2.5-Coder-3B-Instruct",
    help="ModelScope model ID to download when local_model_dir is missing"
)
parser.add_argument(
    "--local_model_dir",
    type=str,
    default="./models/Qwen2.5-Coder-3B-Instruct",
    help="Local model directory"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./output/luoguqwencoder-lora",
    help="Directory for TinyLoRA checkpoints and trainer outputs"
)

args = parser.parse_args()

# Assign parsed arguments
U_VALUE = args.u_value
MAX_SAMPLES = args.max_samples
DO_VALIDATE = args.do_validate
VAL_STEPS = args.val_steps
VAL_SAMPLES = args.val_samples
TINYLORA_RANK = args.rank
DATASET_NAME = args.dataset
USE_QUANT = not args.no_quant
MS_MODEL_ID = args.model_id
LOCAL_MODEL_DIR = args.local_model_dir
OUTPUT_DIR = args.output_dir

if U_VALUE <= 0:
    raise ValueError("u_value must be positive")
if MAX_SAMPLES <= 0:
    raise ValueError("max_samples must be positive")
if VAL_STEPS <= 0:
    raise ValueError("val_steps must be positive")
if VAL_SAMPLES <= 0:
    raise ValueError("val_samples must be positive")
if TINYLORA_RANK <= 0:
    raise ValueError("rank must be positive")

print(f"\n{'='*60}")
print(f"[CONFIG] Training Configuration")
print(f"{'='*60}")
print(f"TinyLoRA u value: {U_VALUE}")
print(f"TinyLoRA rank: {TINYLORA_RANK}")
if MAX_SAMPLES is not None:
    print(f"Max training samples: {MAX_SAMPLES}")
else:
    print(f"Max training samples: unlimited")
print(f"Dataset: {DATASET_NAME}")
print(f"Model ID: {MS_MODEL_ID}")
print(f"Local model directory: {LOCAL_MODEL_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Quantization: {'4-bit (NF4)' if USE_QUANT else 'BF16 (no quant)'}")
print(f"Validation enabled: {DO_VALIDATE}")
if DO_VALIDATE:
    print(f"Validation frequency: every {VAL_STEPS} steps")
    print(f"Validation samples: {VAL_SAMPLES}")
print(f"{'='*60}\n")

# ========== Dataset Configuration ==========
# Filter configuration for deepmind/code_contests dataset
# Filter examples
# source: The original source of the problem, with possible values including UNKNOWN_SOURCE (0),CODECHEF (1), CODEFORCES (2), HACKEREARTH (3), CODEJAM (4), ATCODER (5) and AIZU (6).
# difficulty: A representation of the difficulty of the problem with possible values including UNKNOWN_DIFFICULTY (0), EASY (1), MEDIUM (2), HARD (3), HARDER (4), HARDEST (5), EXTERNAL (6), A (7), B (8), C (9), D (10), E (11), F (12), G (13), H (14), I (15), J (16), K (17), L (18), M (19), N (20), O (21), P (22), Q (23), R (24), S (25), T (26), U (27) and V (28). Note that different sources use different, non-comparable gradings. For Codeforces problems, cf_rating is a more reliable measure of difficulty when available.
# Structure
#   Key: source ID (integer)
#   Value: list of allowed difficulty IDs
#

# Difficulty mapping reference (ignored cf_rating for now to keep it simple):
#   Source 2 (Codeforces) & Source 5 (AtCoder):
#     7=A, 8=B, 9=C, 10=D, 11=E, 12=F, 13=G, 14=H...
#   Source 1 & Source 3 (Other platforms):
#     1=EASY, 2=MEDIUM, 3=HARD, 4=VERY_HARD...
#
DATASET_CONFIG = {
    2: [7, 8],      # Codeforces: A-B level (introductory)
    5: [7, 8],      # AtCoder: A-B level (introductory)
    1: [1],         # General platforms: EASY only
    3: [1],         # General platforms: EASY only
}

# ========== Reward Scaling Configuration ==========
# Hierarchical scaling: Source (1st) -> Difficulty (2nd)
# Note: These multipliers are applied to the base reward (0.5 for compile, up to 1.0 for pass)
REWARD_SCALING_CONFIG = {
    2: {          # Codeforces
        7: 1.0,   # A level: baseline
        8: 1.1,   # B level: slightly higher
    },
    5: {          # AtCoder
        7: 1.0,   # A level: baseline
        8: 1.1,   # B level: slightly higher
    },
    1: { 1: 1.0 }, # General platforms: EASY
    3: { 1: 1.0 }, # General platforms: EASY
}

# ============ Load Model =============
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"Downloading model from ModelScope: {MS_MODEL_ID} -> {LOCAL_MODEL_DIR}...")
    snapshot_download(
        repo_id=MS_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print("Model download complete.")
else:
    print(f"Loading existing local model from: {LOCAL_MODEL_DIR}")

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
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
print(f"[DEVICE] LOCAL_RANK: {LOCAL_RANK}")

# ========== Load model and whether USE_QUANT =====


if USE_QUANT:
    print("[LOAD] Loading model with 4-bit quantization")
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
        # Multi-GPU DDP: each rank loads the full model on its own GPU
        trust_remote_code=True,
        dtype=torch.bfloat16,
        # dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
else:
    print("[LOAD] Loading model without quantization (BF16)")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map={"":  LOCAL_RANK},
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False


# ========== Define TinyLoRA Layers ==========
# Note: TinyLoRA classes are now imported from utils.py

# Get the GPU device for current rank
device = torch.device(f"cuda:{LOCAL_RANK}")
print(f"Model device: {device}")

global_params = TinyLoRAGlobalParams(u_dim=U_VALUE, device=device, dtype=torch.bfloat16)

print("Start replacing")

print("It's normal to see many lines of 'replace'.")

# =======  SEED FIX FOR REPRODUCIBILITY
TINYLORA_SEED = 20260306
torch.manual_seed(TINYLORA_SEED)
torch.cuda.manual_seed(TINYLORA_SEED)
print(f"[OK] Fix TinyLoRA seed: {TINYLORA_SEED}")

model.tiny_lora_params = global_params
print(f"[OK] Register global_params to model")

total_replaced = apply_tiny_lora(model, global_params, rank=TINYLORA_RANK)
print(f"[OK] Replace completed")

print("Freezing parameters")

model.requires_grad_(False)

global_params.global_v.requires_grad = True
print(f"[OK] Trainable parameter: global_v, shape={global_params.global_v.shape}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {all_params:,}")
print(f"Trainable parameters: {trainable_params}")
if trainable_params != U_VALUE:
    raise RuntimeError(f"Trainable parameter count is {trainable_params}, expected {U_VALUE}.")


# ========== Code Reward Function ==========
# Note: compile_and_run is now imported from utils.py


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
    GRPO reward function for code evaluation.

    Supports both:
    1. deepmind/code_contests dataset:
       - public_tests, private_tests, generated_tests: dicts with 'input' and 'output' as lists

    2. DeepCoder dataset (agentica-org/DeepCoder-Preview-Dataset):
       - input_output: dict with 'inputs' and 'outputs' as lists

    Reward rules
    - Compile fail or invalid format: 0.0
    - Compile success, partial tests pass: 0.5-0.99
    - All tests pass: 1.0
    """
    rewards = []

    # Convert None to empty lists
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

    # Iterate through each generated completion
    for i, (completion, pub_test, priv_test, gen_test, io_test, src, diff) in enumerate(zip(
        completions, public_tests, private_tests, generated_tests, input_output, source, difficulty
    )):
        # 1. Extract code using utility function
        code = extract_code_from_response(completion)

        if not code:
            rewards.append(0.0)  # No valid code found
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

        # If no test cases extracted, give penalty
        if not test_cases_list:
            rewards.append(0.0)
            continue

        # Limit test cases during training to avoid excessive subprocess fork pressure
        # (each subprocess.run forks the large model process; too many forks -> OOM killer)
        test_cases_list = test_cases_list[:5]

        # 3. Run code against all test cases
        base_reward = compile_and_run(code, test_cases_list)

        # 4. REWARD SCALING - Adjust based on source and difficulty
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
        
       

#   _is_quantized_and_base_model = model.is_quantized AND NOT model._hf_peft_config_loaded
#   if _is_quantized_and_base_model and not isinstance(model, PeftModel): raise ValueError
#
if USE_QUANT:
    model._hf_peft_config_loaded = True
    print("[OK] Set _hf_peft_config_loaded=True")
else:
    print("[INFO] Non-quantized mode, no need to bypass Trainer check")

def filter_dataset(dataset, config, max_samples, seed=42):
    """
    Filter dataset based on source and difficulty configuration.
    Filter the dataset by source and difficulty configuration.
    
    Args:
        dataset: HuggingFace Dataset object
        config: Dict mapping source IDs to allowed difficulty lists
        max_samples: Maximum number of samples after filtering
        seed: Random seed for shuffling
    
    Returns:
        Filtered and sampled dataset
    """
    print("\n" + "="*60)
    print("[CHECK] Filtering dataset based on configuration...")
    print("Filtering dataset according to configuration...")
    print("="*60)
    
    # Log configuration
    source_names = {
        1: "General Platform 1",
        2: "Codeforces / Codeforces",
        3: "General Platform 3",
        5: "AtCoder / AtCoder",
    }
    
    for source_id, difficulties in config.items():
        source_name = source_names.get(source_id, f"Source {source_id}")
        print(f"[NOTE] {source_name}: Keeping difficulties {difficulties}")
    
    # Filter examples
    def should_keep(example):
        source = example.get('source', -1)
        difficulty = example.get('difficulty', -1)
        
        # Check if source is in config
        if source not in config:
            return False
        
        # Check if difficulty is allowed for this source
        if difficulty not in config[source]:
            return False
        
        return True
    
    # Filter examples
    print("\n[WAIT] Filtering in progress...")
    original_size = len(dataset)
    filtered_dataset = dataset.filter(should_keep)
    filtered_size = len(filtered_dataset)
    
    print(f"[OK] Original dataset size: {original_size:,}")
    print(f"[OK] After filtering: {filtered_size:,} samples")
    retention_rate = (filtered_size / original_size * 100) if original_size else 0.0
    print(f"[STATS] Retention rate: {retention_rate:.2f}%")
    
    # Apply max_samples limit with shuffling
    if filtered_size > max_samples:
        print(f"\n[SEED] Shuffling and sampling {max_samples:,} from {filtered_size:,}...")
        filtered_dataset = filtered_dataset.shuffle(seed=seed).select(range(max_samples))
        final_size = len(filtered_dataset)
        print(f"[OK] Final training set size: {final_size:,}")
    else:
        print(f"\n[OK] All {filtered_size:,} filtered samples will be used (below max_samples limit).")
        print(f"Using all {filtered_size:,} filtered samples because this is below the max_samples limit.")
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
    print(f"[DIR] Loading DeepCoder dataset from ./local_Deep-Coder-Preview-Dataset/")

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
    print(f"   Streaming and sampling {MAX_SAMPLES} samples...")

    import random
    import json as _json
    random.seed(TINYLORA_SEED)

    # Reservoir sampling via direct file I/O (avoids HuggingFace Arrow overhead)
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
                # multi-GB memory bloat -> OOM kill.  5 cases suffice for reward signal.
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

    print(f"[OK] Loaded {len(rl_dataset)} DeepCoder samples (streaming)")

elif DATASET_NAME == 'code_contests':
    print(f"[DIR] Loading CodeContests dataset from ./local_code_contests/code_contests_train.jsonl")
    rl_dataset = load_dataset(
        "json",
        data_files="./local_code_contests/code_contests_train.jsonl",
        split="train"
    )
    # Apply filtering based on source and difficulty configuration
    # Filter examples
    rl_dataset = filter_dataset(
        dataset=rl_dataset,
        config=DATASET_CONFIG,
        max_samples=MAX_SAMPLES,
        seed=TINYLORA_SEED
    )
    print(f"[OK] Loaded {len(rl_dataset)} CodeContests samples")

else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}. Choose 'deepcoder' or 'code_contests'")

if len(rl_dataset) == 0:
    raise ValueError(
        "No training samples available after dataset loading/filtering. "
        "Please check the selected dataset and filters."
    )

# Apply template
# Use batched=False and num_proc=1 to reduce memory usage
print("[WAIT] Applying chat template...")
# Columns needed by code_reward_func must be preserved after map
# Keep reward columns after map.
_reward_columns = {'prompt', 'input_output', 'public_tests', 'private_tests', 'generated_tests', 'source', 'difficulty'}
rl_dataset = rl_dataset.map(
    lambda x: apply_chat_template(x, tokenizer),
    batched=False,
    num_proc=1,
    remove_columns=[c for c in rl_dataset.column_names if c not in _reward_columns]
)

# Print sample to verify
print(f"[OK] Dataset loaded successfully!: {len(rl_dataset)}")
print(f"[INFO] Sample data: {rl_dataset[0]}")



# ========== Load Validation Dataset (if enabled) ==========

val_dataset = None
if DO_VALIDATE:
    print(f"\n{'='*60}")
    print(f"[STATS] Loading validation dataset")
    print(f"{'='*60}\n")

    # DeepCoder dataset does not have validation split
    if DATASET_NAME == 'deepcoder':
        print("[WARN]  Warning:")
        print("   DeepCoder dataset does not have validation data.")
        print("   DeepCoder dataset has no validation split, so validation is skipped automatically.")
        print("   Validation is only available for code_contests dataset.")
        print("   Validation is only supported for the code_contests dataset.")
        print(f"{'='*60}\n")
        DO_VALIDATE = False  # Disable validation for deepcoder
    else:
        # CodeContests dataset has validation data
        val_dataset = load_dataset(
            "json",
            data_files="./local_code_contests/code_contests_valid.jsonl",
            split="train"
        )

        # Apply chat template to validation dataset
        val_dataset = val_dataset.map(lambda x: apply_chat_template(x, tokenizer))

        print(f"[OK] Validation dataset loaded: {len(val_dataset)} samples")


# ========== Define Validation Callback ==========

class ValidationCallback(TrainerCallback):
    """
    Custom callback for validation during training
    Tracks best model and saves checkpoint
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
        """Run validation at specified intervals"""
        if state.global_step % self.val_steps == 0 and state.global_step > 0:
            print(f"\n{'='*80}")
            print(f"[CHECK] Running validation at step {state.global_step}")
            print(f"{'='*80}\n")
            
            # Import validation function
            from validate import run_validation
            
            # Get model and tokenizer from kwargs
            model = kwargs.get('model')
            tokenizer = kwargs.get('processing_class') or kwargs.get('tokenizer')
            
            # Run validation
            results = run_validation(
                model=model,
                tokenizer=tokenizer,
                dataset=self.val_dataset,
                num_samples=self.val_samples,
            )
            
            # Check if this is the best model
            current_score = results['pass_at_1']
            
            if current_score > self.best_score:
                previous_best = self.best_score
                self.best_score = current_score
                print(f"\n{'='*80}")
                print(f"[BEST] New best model!")
                print(f"   Previous best Pass@1: {previous_best:.2%}")
                print(f"   Current Pass@1: {current_score:.2%}")
                print(f"{'='*80}\n")
                
                # Save best model
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
                print(f"[SAVE] Best model saved to: {best_path}")
                print(f"[STATS] Validation Pass@1: {current_score:.2%}\n")
            else:
                print(f"[STATS] Current Pass@1: {current_score:.2%} (Best: {self.best_score:.2%})\n")
        
        return control

# Prepare callbacks
callbacks = []
if DO_VALIDATE and val_dataset is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    print(f"[OK] Validation callback registered")
    print(f"   Validation frequency: every {VAL_STEPS} steps")
    print(f"   Validation samples: {VAL_SAMPLES}\n")



# ========== Configure and start GRPO training ==========
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Set to 1 if GPU memory insufficient
    gradient_accumulation_steps=8,  # Accumulate to simulate large batch
    learning_rate=1e-5,             # RL learning rate is usually small
    num_generations=4,              # Group Size (G): sample 4 answers per iteration
    max_completion_length=1024,     # Max generation length
    logging_steps=1,
    bf16=True,                      # Enable BF16 acceleration
    save_strategy="no",             # Disable auto checkpoint (TinyLoRA is non-standard PEFT)



    # ======================================================
    # updated in v3.2 - Clip High (from DeepCoder/DAPO paper)
    # Clip High: asymmetric clipping - increase upper bound to encourage more exploration
    # Standard GRPO: clip range [1-epsilon, 1+epsilon], typically epsilon=0.2 or 0.3
    # Clip High: clip range [1-epsilon, 1+epsilon_high] where epsilon_high > epsilon, enabling larger policy updates
    # when correct solutions are found, preventing premature convergence
    beta=0.0,                       # deprecate KL penalty: empower the model to explore more without constraints
    epsilon=0.05,                    # lower bound clipping (1 - epsilon)
    epsilon_high=0.95,               # upper bound clipping (1 + epsilon_high) - Clip High!

    # deprecated in v3.1+, kept for backward compatibility
    # clip_range=0.3,               # expand Clip range
    

    # =======================================================
    # updated in v4.0 - num_iterations should be larger than 1
    num_iterations=4,              # Total training iterations
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=code_reward_func,  # Your judge function
    args=training_args,
    train_dataset=rl_dataset,       # Processed data
    processing_class=tokenizer,     # Tokenizer
    callbacks=callbacks,            # Add validation callback
)

# Start training!
print("[START] Starting TinyLoRA-RL training...")
trainer.train()

# Save training results
# Note: peft's save_pretrained may not recognize custom layers
# Manually save global_v and metadata needed to rebuild model
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_dict = {
    "global_v": global_params.global_v.data,  # Trained v vector
    "u_value": U_VALUE,                        # Dimension of v
    "rank": TINYLORA_RANK,                     # TinyLoRA rank
    "seed": TINYLORA_SEED,                     # P matrix random seed (for reproducibility)
    "model_id": MS_MODEL_ID,                   # Base model ID
    "total_replaced_layers": total_replaced,   # Number of replaced layers
    "is_quantized": USE_QUANT,                 # Whether model was loaded with 4-bit quantization
}
torch.save(save_dict, f"{OUTPUT_DIR}/tiny_lora_v.pt")
print(f"[OK] Training complete!")
print(f"[STATS] Save contents: global_v (shape={global_params.global_v.shape}), u={U_VALUE}, rank={TINYLORA_RANK}, seed={TINYLORA_SEED}")

# Print all GRPO-related parameters
print("\n" + "="*50)
print("[CONFIG] GRPO Parameters:")
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
