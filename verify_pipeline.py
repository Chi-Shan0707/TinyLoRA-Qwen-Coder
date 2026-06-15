import torch
import os
import subprocess

# Import all shared utilities from utils.py - single source of truth for all pipeline stages
# Import shared utilities from utils.py
from utils import (
    compile_and_run,
    extract_code_from_response,
    convert_hf_tests_to_list,
    apply_chat_template,
    get_model_and_tokenizer,
)

# Configuration section
MS_MODEL_ID = "qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-3B-Instruct"

# Multi-GPU / DDP: LOCAL_RANK is read from environment by get_model_and_tokenizer internally.
# It uses device_map={"":LOCAL_RANK} so each rank holds the full model on its own GPU.
# See utils.py get_model_and_tokenizer for the full DDP-safe implementation.
# Used here for logging only
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

# Use CodeContests-style data for testing
# Use a simple sample problem for testing
TEST_DATA_JSON = {
    "description": """Problem description.
You are smart. You like maths. You are given a problem.
Input

Two integers(1<=x,y<=200)
Output:

The first line: their sum
The second line: their product




Explanation
Example is self-explanatory.""",
    "public_tests": {
        "input": ["1 2"],
        "output": ["3\n2"]
    },
    "private_tests": {
        "input": ["2 4"],
        "output": ["6\n8"]
    },
    "generated_tests": {
        "input": ["2  100"],
        "output": ["102\n200"]
    },
    "source": 0,
    "difficulty": 1
}
# =================================================

def print_step(title):
    print(f"\n{'='*10} {title} {'='*10}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify pipeline script - test the full training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_pipeline.py
  python verify_pipeline.py --no_quant
  python verify_pipeline.py --max_tokens 512
  python verify_pipeline.py --model ./models/Qwen3-0.6B --no_quant --max_tokens 128
  python verify_pipeline.py --model ./models/Qwen3-0.6B --no_quant --skip_generation
        """
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Load model in BF16 without quantization"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Local model directory or model ID. Defaults to the configured local Qwen2.5-Coder path if present."
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Stop after model/tokenizer loading and chat-template validation."
    )

    args = parser.parse_args()

    print_step("STEP 1: Load model and tokenizer")
    
    # Check g++
    try:
        subprocess.run(['g++', '--version'], capture_output=True)
        print("g++ compiler detected.")
    except:
        print("g++ was not detected. Please install it first, for example: sudo apt install g++")
        return

    model_path = args.model or (LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else MS_MODEL_ID)
    print(f"[DEVICE] LOCAL_RANK: {LOCAL_RANK}")

    # Use get_model_and_tokenizer from utils.py:
    #   - device_map={{"":LOCAL_RANK}}: DDP-safe, each process holds full model on its own GPU
    #   - torch_dtype=bfloat16: non-quantized layers (Embedding/LayerNorm) use BF16, not FP32
    #   - for_inference=True: enables KV cache, skips gradient checkpointing
    model, tokenizer = get_model_and_tokenizer(model_path, use_4bit=not args.no_quant, for_inference=True)

    # ------------------------------------------------------------------
    print_step("STEP 2: Verify chat template (JSON -> Qwen prompt)")
    
    # Use apply_chat_template from utils.py - identical logic to training, guarantees distribution consistency
    final_prompt = apply_chat_template(TEST_DATA_JSON, tokenizer)['prompt']
    
    print("--- Prompt prefix sent to the model ---")
    print(final_prompt[:300] + "...\n")
    print("--- Prompt suffix sent to the model ---")
    print("..." + final_prompt[-100:])
    
    if "<|im_start|>system" in final_prompt and "<|im_start|>assistant" in final_prompt:
        print("\nChat template check passed: Qwen ChatML tags detected.")
    else:
        print("\nWarning: ChatML tags were not detected. Check tokenizer_config.json.")

    if args.skip_generation:
        print("\nSkipping generation as requested. Load and prompt checks completed.")
        return

    # ------------------------------------------------------------------
    print_step("STEP 3: Run model generation")
    
    inputs = tokenizer([final_prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"Prompt token length: {inputs['input_ids'].shape[1]}")
    print(f"Generating (max {args.max_tokens} tokens)...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,     
            temperature=args.temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Keep only the generated part
    if "<|im_start|>assistant" in full_response:
        response_only = full_response.split("<|im_start|>assistant")[-1]
    else:
        response_only = full_response
    
    print("\n--- Generated code preview (first 1000 characters) ---")
    print(response_only[:1000] + "..." if len(response_only)>500 else response_only)

    # ------------------------------------------------------------------
    print_step("STEP 4: Verify code extraction and evaluation (CodeContests format)")
    
    # Use extract_code_from_response from utils.py (returns str, empty string if not found)
    extracted_code = extract_code_from_response(response_only)

    # Use convert_hf_tests_to_list from utils.py - same parsing as training reward function
    test_cases = []
    for test_type in ['public_tests', 'private_tests', 'generated_tests']:
        test_cases.extend(convert_hf_tests_to_list(TEST_DATA_JSON.get(test_type, {})))

    if extracted_code:
        print("Code extracted successfully.")
        print(f"Evaluating with {len(test_cases)} test cases...")

        # Use compile_and_run from utils.py (returns float: 0.0 / 0.5 / 1.0)
        score = compile_and_run(extracted_code, test_cases)

        print(f"\nFinal reward: {score}")

        if score == 1.0:
            print("Pipeline passed perfectly: the model solved the problem.")
        elif score > 0.0:
            print("Pipeline is functional: generated code runs, but some test cases failed. RL training is expected to improve this.")
        else:
            print("Generated code failed to compile or failed all test cases.")
            print("Note: an unfine-tuned 3B model may not solve a contest problem on the first try. If compilation and judging run, the pipeline is working.")
    else:
        print("Code extraction failed. The model may not have generated a code block.")

if __name__ == "__main__":
    main()
