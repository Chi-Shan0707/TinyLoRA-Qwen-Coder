"""
test.py - Standalone Testing Script for TinyLoRA Model
测试脚本 - 用于 TinyLoRA 模型的独立测试

Usage / 用法:
    python test.py [checkpoint_path] [num_samples]
    python test.py --baseline [num_samples]
    
Example / 示例:
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
    在测试数据集上测试训练好的 TinyLoRA 模型或基座模型
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file / .pt 检查点文件的路径
        num_samples: Number of samples to test / 要测试的样本数
        test_data_path: Path to test dataset / 测试数据集路径
        baseline: If True, test base model without TinyLoRA / 如果为 True，测试基座模型（不含 TinyLoRA）
        test_seed: Random seed for evaluation (sample selection & generation) / 评估用随机种子（样本选择和生成）
        use_quant: Whether to load model with 4-bit quantization / 是否以 4-bit 量化加载模型
    
    Returns:
        dict: Test metrics / 测试指标
    """
    print(f"\n{'='*80}")
    mode_str = "Baseline Model" if baseline else "TinyLoRA Model"
    print(f"🧪 {mode_str} Testing / {mode_str} 测试")
    print(f"{'='*80}\n")
    
    # ========== Step 1: Load checkpoint or set seed / 加载检查点或设置种子 ==========
    if not baseline:
        print(f"📦 Loading checkpoint / 正在加载检查点...")
        print(f"   Path / 路径: {checkpoint_path}\n")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found / 检查点未找到: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract metadata / 提取元数据
        u_value = checkpoint['u_value']
        rank = checkpoint['rank']
        seed = checkpoint['seed']
        model_id = checkpoint.get('model_id', 'qwen/Qwen2.5-Coder-3B-Instruct')
        global_v = checkpoint['global_v']
        
        print(f"✅ Checkpoint loaded / 检查点加载成功")
        print(f"   • u_value / u 值: {u_value}")
        print(f"   • rank / 秩: {rank}")
        print(f"   • seed / 随机种子: {seed}")
        print(f"   • model_id / 模型 ID: {model_id}")
        print(f"   • global_v shape / global_v 形状: {global_v.shape}\n")
        
        # Check quantization state from checkpoint / 从检查点读取量化状态
        trained_with_quant = checkpoint.get("is_quantized", False)
        print(f"📌 模型微调时的量化状态: {'4-bit' if trained_with_quant else 'BF16/FP16'}")
        if use_quant != trained_with_quant:
            print("⚠️ 警告: 当前测试时的量化状态(--use_quant)与训练时不同，可能会有微小精度差异。")
        
        # ========== Step 2: Note training seed (used only for P matrix reproduction) / 记录训练种子（仅用于 P 矩阵复现） ==========
        print(f"🎲 Checkpoint training seed / 检查点训练随机种子: {seed} (used for TinyLoRA P matrix reproduction / 用于 TinyLoRA P 矩阵复现)")
    
    # ========== Step 3: Load base model / 加载基座模型 ==========
    # Check if model exists locally / 检查模型是否存在于本地
    local_model_path = "./models/Qwen2.5-Coder-3B-Instruct"
    if os.path.exists(local_model_path):
        model_path = local_model_path
    else:
        model_id = 'qwen/Qwen2.5-Coder-3B-Instruct' if baseline else model_id
        model_path = model_id
    
    model, tokenizer = get_model_and_tokenizer(model_path, use_4bit=use_quant, for_inference=True)
    
    # ========== Step 4: Conditionally inject TinyLoRA / 条件性注入 TinyLoRA ==========
    if not baseline:
        print(f"🔧 Injecting TinyLoRA layers / 正在注入 TinyLoRA 层...")
        
        # Get device / 获取设备
        device = model.model.layers[0].self_attn.q_proj.weight.device
        
        # Create global params container / 创建全局参数容器
        global_params = TinyLoRAGlobalParams(u_dim=u_value, device=device, dtype=torch.bfloat16)
        
        # Register to model / 注册到模型
        model.tiny_lora_params = global_params
        
        # 【关键】重新设置随机种子，确保 P 矩阵与训练时一致
        # CRITICAL: Re-seed right before apply_tiny_lora to match training P matrices
        # In train_rl.py, seed is set RIGHT BEFORE apply_tiny_lora, not before model loading
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Apply TinyLoRA (this will use the fixed random seed) / 应用 TinyLoRA（会使用固定的随机种子）
        total_replaced = apply_tiny_lora(model, global_params)
        print(f"   ✅ TinyLoRA injected / TinyLoRA 已注入: {total_replaced} layers replaced / 层已替换\n")
        
        # ========== Step 5: Load trained weights / 加载训练权重 ==========
        print(f"💾 Loading trained weights / 正在加载训练权重...")
        with torch.no_grad():
            global_params.global_v.copy_(global_v.to(global_params.global_v.dtype).to(device))
        print(f"   ✅ Trained weights loaded / 训练权重已加载\n")
    else:
        print(f"⏭️  Skipping TinyLoRA injection (baseline mode) / 跳过 TinyLoRA 注入（基座模型模式）\n")
    
    # Verify trainable parameters / 验证可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model Statistics / 模型统计:")
    print(f"   • Total parameters / 总参数: {all_params:,}")
    if not baseline:
        print(f"   • Trainable parameters / 可训练参数: {trainable_params}")
        print(f"   • Compression ratio / 压缩比: {all_params / trainable_params:.1f}x\n")
    
    # ========== Step 5b: Set test seed for evaluation / 设置评估用随机种子 ==========
    print(f"🎲 Setting test seed for evaluation / 正在设置评估随机种子: {test_seed}")
    torch.manual_seed(test_seed)
    torch.cuda.manual_seed(test_seed)
    import random, numpy as np
    random.seed(test_seed)
    np.random.seed(test_seed)
    print(f"   ✅ Test seed set / 评估随机种子已设置\n")

    # ========== Step 6: Load test dataset / 加载测试数据集 ===========
    print(f"📁 Loading test dataset / 正在加载测试数据集...")
    print(f"   Path / 路径: {test_data_path}\n")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test dataset not found / 测试数据集未找到: {test_data_path}")
    
    test_dataset = load_dataset("json", data_files=test_data_path, split="train")
    
    # Limit samples / 限制样本数
    num_samples = min(num_samples, len(test_dataset))
    print(f"   ✅ Dataset loaded / 数据集已加载: {len(test_dataset)} total samples / 总样本")
    print(f"   📊 Testing on / 测试样本数: {num_samples} samples / 样本\n")
    
    # Apply chat template if needed / 如果需要，应用聊天模板
    if 'prompt' not in test_dataset.column_names:
        print(f"🔄 Applying chat template... / 正在应用聊天模板...")
        test_dataset = test_dataset.map(lambda x: apply_chat_template(x, tokenizer))
        print(f"   ✅ Chat template applied / 聊天模板已应用\n")
    
    # ========== Step 7: Run evaluation / 运行评估 ==========
    print(f"{'='*80}")
    print(f"🚀 Starting evaluation / 开始评估...")
    print(f"{'='*80}\n")
    
    model.eval()
    
    total_score = 0.0
    compile_success = 0
    partial_pass = 0
    full_pass = 0
    no_code_extracted = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Testing / 测试中"):
            sample = test_dataset[i]
            
            # Get prompt / 获取提示
            prompt = sample.get('prompt', '')
            if not prompt:
                continue
            
            # Prepare test cases / 准备测试用例
            public_tests = sample.get('public_tests', {})
            private_tests = sample.get('private_tests', {})
            generated_tests = sample.get('generated_tests', {})
            
            all_test_cases = []
            all_test_cases.extend(convert_hf_tests_to_list(public_tests))
            all_test_cases.extend(convert_hf_tests_to_list(private_tests))
            all_test_cases.extend(convert_hf_tests_to_list(generated_tests))
            
            if not all_test_cases:
                continue
            
            # Tokenize and generate / 分词并生成
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
                
                # Decode response / 解码响应
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove prompt from response / 从响应中移除提示
                if prompt in response:
                    response = response[len(prompt):].strip()
                
                # Extract code / 提取代码
                code = extract_code_from_response(response)
                
                if not code:
                    no_code_extracted += 1
                    continue
                
                # Compile and run / 编译并运行
                reward = compile_and_run(code, all_test_cases)
                
                total_score += reward
                
                # Track statistics / 跟踪统计信息
                if reward >= 0.5:
                    compile_success += 1
                if reward == 0.5:
                    partial_pass += 1
                elif reward == 1.0:
                    full_pass += 1
                    
            except Exception as e:
                print(f"\n⚠️ Error during generation / 生成时出错: {e}")
                continue
    
    # ========== Step 8: Calculate and report metrics / 计算并报告指标 ==========
    print(f"\n{'='*80}")
    print(f"✅ Evaluation complete / 评估完成")
    print(f"{'='*80}\n")
    
    avg_score = total_score / num_samples if num_samples > 0 else 0.0
    compile_rate = compile_success / num_samples if num_samples > 0 else 0.0
    pass_at_1 = full_pass / num_samples if num_samples > 0 else 0.0
    
    print(f"📊 Test Results / 测试结果:")
    print(f"   • Total Samples / 总样本数: {num_samples}")
    print(f"   • Average Score / 平均分数: {avg_score:.4f}")
    print(f"   • Compile Rate / 编译成功率: {compile_rate:.2%} ({compile_success}/{num_samples})")
    print(f"   • Pass@1 / 完全通过率: {pass_at_1:.2%} ({full_pass}/{num_samples})")
    print(f"   • Partial Pass / 部分通过: {partial_pass}/{num_samples}")
    print(f"   • No Code Extracted / 未提取到代码: {no_code_extracted}/{num_samples}")
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
    独立测试的主入口
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test a trained TinyLoRA model / 测试训练好的 TinyLoRA 模型"
    )
    parser.add_argument(
        "--checkpoint_path",
        nargs='?',
        default="./output/luoguqwencoder-lora/tiny_lora_v.pt",
        help="Path to checkpoint .pt file / 检查点 .pt 文件的路径"
    )
    parser.add_argument(
        "--num_samples",
        nargs='?',
        type=int,
        default=50,
        help="Number of samples to test / 要测试的样本数"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./local_code_contests/code_contests_test.jsonl",
        help="Path to test dataset / 测试数据集路径"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Test base model without TinyLoRA (for comparison) / 测试基座模型（不含 TinyLoRA，用于对比）"
    )
    parser.add_argument(
        "--test_seed",
        type=int,
        default=42,
        help="Random seed for evaluation (sample selection & generation) / 评估用随机种子（样本选择和生成），默认 42"
    )
    parser.add_argument(
        "--use_quant",
        action="store_true",
        default=True,
        help="Load model with 4-bit quantization (default: True) / 以 4-bit 量化加载模型（默认：是）"
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Disable 4-bit quantization, load in BF16 / 禁用 4-bit 量化，以 BF16 加载"
    )
    
    args = parser.parse_args()
    
    # Resolve use_quant: --no_quant overrides --use_quant
    use_quant = not args.no_quant
    
    mode_str = "Baseline Model" if args.baseline else "TinyLoRA Model"
    print(f"\n🎯 {mode_str} Testing / {mode_str} 测试")
    if not args.baseline:
        print(f"   Checkpoint / 检查点: {args.checkpoint_path}")
    print(f"   Samples / 样本数: {args.num_samples}")
    print(f"   Test Data / 测试数据: {args.test_data}")
    print(f"   Test Seed / 评估随机种子: {args.test_seed}")
    print(f"   Quantization / 量化加载: {'4-bit' if use_quant else 'BF16 (no quant)'}\n")
    
    # Run testing / 运行测试
    results = test_model(
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        test_data_path=args.test_data,
        baseline=args.baseline,
        test_seed=args.test_seed,
        use_quant=use_quant,
    )
    
    print(f"✅ Testing complete! / 测试完成！")
    print(f"🎯 Final Pass@1 / 最终通过率: {results['pass_at_1']:.2%}\n")
