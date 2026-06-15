"""
test.py - Standalone Testing Script for TinyLoRA Model

Usage:
    python test.py [checkpoint_path] [num_samples]
    python test.py --baseline [num_samples]
    
Example:
    python test.py ./output/luoguqwencoder-lora/tiny_lora_v.pt 50
    python test.py ./output/luoguqwencoder-lora/best_tiny_lora_v.pt 100
    python test.py --baseline 50          # Test base model for comparison
"""

import os
import sys
import torch
from datasets import load_dataset
from utils import (
    get_model_and_tokenizer,
    TinyLoRAGlobalParams,
    apply_tiny_lora,
    compile_and_run,
    convert_hf_tests_to_list,
    extract_code_from_response,
    apply_chat_template,
)
from tqdm import tqdm


def test_model(checkpoint_path, num_samples=50, test_data_path="./local_code_contests/code_contests_test.jsonl", baseline=False, test_seed=42, use_quant=True):
    """
    Test a trained TinyLoRA model or baseline model on the test dataset
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        num_samples: Number of samples to test
        test_data_path: Path to test dataset
        baseline: If True, test base model without TinyLoRA
        test_seed: Random seed for evaluation (sample selection & generation)
        use_quant: Whether to load model with 4-bit quantization
    
    Returns:
        dict: Test metrics
    """
    print(f"\n{'='*80}")
    mode_str = "Baseline Model" if baseline else "TinyLoRA Model"
    print(f"[TEST] {mode_str} Testing")
    print(f"{'='*80}\n")
    
    # ========== Step 1: Load checkpoint or set seed
    if not baseline:
        print(f"[LOAD] Loading checkpoint")
        print(f"   Path: {checkpoint_path}\n")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract metadata
        u_value = checkpoint['u_value']
        rank = checkpoint['rank']
        seed = checkpoint['seed']
        model_id = checkpoint.get('model_id', 'qwen/Qwen2.5-Coder-3B-Instruct')
        global_v = checkpoint['global_v']
        
        print(f"[OK] Checkpoint loaded")
        print(f"   - u_value: {u_value}")
        print(f"   - rank: {rank}")
        print(f"   - seed: {seed}")
        print(f"   - model_id: {model_id}")
        print(f"   - global_v shape: {global_v.shape}\n")
        
        # Check quantization state from checkpoint
        trained_with_quant = checkpoint.get("is_quantized", False)
        print(f"[NOTE] Checkpoint quantization: {'4-bit' if trained_with_quant else 'BF16/FP16'}")
        if use_quant != trained_with_quant:
            print("[WARN] Warning: evaluation quantization differs from training; small numeric differences are possible.")
        
        # ========== Step 2: Note training seed (used only for P matrix reproduction)
        print(f"[SEED] Checkpoint training seed: {seed} (used for TinyLoRA P matrix reproduction)")
    
    # ========== Step 3: Load base model
    # Check if model exists locally
    local_model_path = "./models/Qwen2.5-Coder-3B-Instruct"
    if os.path.exists(local_model_path):
        model_path = local_model_path
    else:
        model_id = 'qwen/Qwen2.5-Coder-3B-Instruct' if baseline else model_id
        model_path = model_id
    
    model, tokenizer = get_model_and_tokenizer(model_path, use_4bit=use_quant, for_inference=True)
    
    # ========== Step 4: Conditionally inject TinyLoRA
    if not baseline:
        print(f"[SETUP] Injecting TinyLoRA layers")
        
        # Get device
        device = model.model.layers[0].self_attn.q_proj.weight.device
        
        # Create global params container
        global_params = TinyLoRAGlobalParams(u_dim=u_value, device=device, dtype=torch.bfloat16)
        
        # Register to model
        model.tiny_lora_params = global_params
        
        # CRITICAL: Re-seed right before apply_tiny_lora to match training P matrices
        # In train_rl.py, seed is set RIGHT BEFORE apply_tiny_lora, not before model loading
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Apply TinyLoRA (this will use the fixed random seed)
        total_replaced = apply_tiny_lora(model, global_params, rank=rank)
        print(f"   [OK] TinyLoRA injected: {total_replaced} layers replaced")
        
        # ========== Step 5: Load trained weights
        print(f"[SAVE] Loading trained weights")
        with torch.no_grad():
            global_params.global_v.copy_(global_v.to(global_params.global_v.dtype).to(device))
        print(f"   [OK] Trained weights loaded")
    else:
        print(f"[SKIP]  Skipping TinyLoRA injection (baseline mode)")
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"[STATS] Model Statistics:")
    print(f"   - Total parameters: {all_params:,}")
    if not baseline:
        print(f"   - Trainable parameters: {trainable_params}")
        print(f"   - Compression ratio: {all_params / trainable_params:.1f}x\n")
    
    # ========== Step 5b: Set test seed for evaluation
    print(f"[SEED] Setting test seed for evaluation: {test_seed}")
    torch.manual_seed(test_seed)
    torch.cuda.manual_seed(test_seed)
    import random, numpy as np
    random.seed(test_seed)
    np.random.seed(test_seed)
    print(f"   [OK] Test seed set")

    # ========== Step 6: Load test dataset
    print(f"[FILE] Loading test dataset")
    print(f"   Path: {test_data_path}\n")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test dataset not found: {test_data_path}")
    
    test_dataset = load_dataset("json", data_files=test_data_path, split="train")
    
    # Limit samples
    num_samples = min(num_samples, len(test_dataset))
    print(f"   [OK] Dataset loaded: {len(test_dataset)} total samples")
    print(f"   [STATS] Testing on: {num_samples} samples")
    
    # Apply chat template if needed
    if 'prompt' not in test_dataset.column_names:
        print(f"[RUN] Applying chat template...")
        test_dataset = test_dataset.map(lambda x: apply_chat_template(x, tokenizer))
        print(f"   [OK] Chat template applied")
    
    # ========== Step 7: Run evaluation
    print(f"{'='*80}")
    print(f"[START] Starting evaluation")
    print(f"{'='*80}\n")
    
    model.eval()
    
    total_score = 0.0
    compile_success = 0
    partial_pass = 0
    full_pass = 0
    no_code_extracted = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Testing"):
            sample = test_dataset[i]
            
            # Get prompt
            prompt = sample.get('prompt', '')
            if not prompt:
                continue
            
            # Prepare test cases
            public_tests = sample.get('public_tests', {})
            private_tests = sample.get('private_tests', {})
            generated_tests = sample.get('generated_tests', {})
            
            all_test_cases = []
            all_test_cases.extend(convert_hf_tests_to_list(public_tests))
            all_test_cases.extend(convert_hf_tests_to_list(private_tests))
            all_test_cases.extend(convert_hf_tests_to_list(generated_tests))
            
            if not all_test_cases:
                continue
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                # Decode generated tokens only
                input_len = inputs["input_ids"].shape[-1]
                response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                
                # Extract code
                code = extract_code_from_response(response)
                
                if not code:
                    no_code_extracted += 1
                    continue
                
                # Compile and run
                reward = compile_and_run(code, all_test_cases)
                
                total_score += reward
                
                # Track statistics
                if reward >= 0.5:
                    compile_success += 1
                if reward == 0.5:
                    partial_pass += 1
                elif reward == 1.0:
                    full_pass += 1
                    
            except Exception as e:
                print(f"\n[WARN] Error during generation: {e}")
                continue
    
    # ========== Step 8: Calculate and report metrics
    print(f"\n{'='*80}")
    print(f"[OK] Evaluation complete")
    print(f"{'='*80}\n")
    
    avg_score = total_score / num_samples if num_samples > 0 else 0.0
    compile_rate = compile_success / num_samples if num_samples > 0 else 0.0
    pass_at_1 = full_pass / num_samples if num_samples > 0 else 0.0
    
    print(f"[STATS] Test Results:")
    print(f"   - Total Samples: {num_samples}")
    print(f"   - Average Score: {avg_score:.4f}")
    print(f"   - Compile Rate: {compile_rate:.2%} ({compile_success}/{num_samples})")
    print(f"   - Pass@1: {pass_at_1:.2%} ({full_pass}/{num_samples})")
    print(f"   - Partial Pass: {partial_pass}/{num_samples}")
    print(f"   - No Code Extracted: {no_code_extracted}/{num_samples}")
    print(f"{'='*80}\n")
    
    results = {
        'total_samples': num_samples,
        'avg_score': avg_score,
        'compile_rate': compile_rate,
        'pass_at_1': pass_at_1,
        'compile_success': compile_success,
        'partial_pass': partial_pass,
        'full_pass': full_pass,
        'no_code_extracted': no_code_extracted,
    }
    
    return results


if __name__ == "__main__":
    """
    Main entry point for standalone testing
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test a trained TinyLoRA model"
    )
    parser.add_argument(
        "--checkpoint_path",
        nargs='?',
        default="./output/luoguqwencoder-lora/tiny_lora_v.pt",
        help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--num_samples",
        nargs='?',
        type=int,
        default=50,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./local_code_contests/code_contests_test.jsonl",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Test base model without TinyLoRA (for comparison)"
    )
    parser.add_argument(
        "--test_seed",
        type=int,
        default=42,
        help="Random seed for evaluation (sample selection & generation)"
    )
    parser.add_argument(
        "--use_quant",
        action="store_true",
        default=True,
        help="Load model with 4-bit quantization (default: True)"
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Disable 4-bit quantization, load in BF16"
    )
    
    args = parser.parse_args()
    
    # Resolve use_quant: --no_quant overrides --use_quant
    use_quant = not args.no_quant
    
    mode_str = "Baseline Model" if args.baseline else "TinyLoRA Model"
    print(f"\n[RESULT] {mode_str} Testing")
    if not args.baseline:
        print(f"   Checkpoint: {args.checkpoint_path}")
    print(f"   Samples: {args.num_samples}")
    print(f"   Test Data: {args.test_data}")
    print(f"   Test Seed: {args.test_seed}")
    print(f"   Quantization: {'4-bit' if use_quant else 'BF16 (no quant)'}\n")
    
    # Run testing
    results = test_model(
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        test_data_path=args.test_data,
        baseline=args.baseline,
        test_seed=args.test_seed,
        use_quant=use_quant,
    )
    
    print(f"[OK] Testing complete!")
    print(f"[RESULT] Final Pass@1: {results['pass_at_1']:.2%}\n")
