"""
validate.py - Validation Script for TinyLoRA Model
验证脚本 - 用于 TinyLoRA 模型验证

This script provides validation functionality during training.
本脚本在训练期间提供验证功能。
"""

import torch
import re
from tqdm import tqdm
from utils import compile_and_run, convert_hf_tests_to_list, extract_code_from_response, apply_chat_template


def run_validation(model, tokenizer, dataset, num_samples=10, max_length=1024, temperature=0.7):
    """
    Run validation on a subset of the dataset
    在数据集的子集上运行验证
    
    Args:
        model: The model to validate / 要验证的模型
        tokenizer: Tokenizer for the model / 模型的分词器
        dataset: Validation dataset / 验证数据集
        num_samples: Number of samples to validate / 要验证的样本数
        max_length: Maximum generation length / 最大生成长度
        temperature: Sampling temperature / 采样温度
    
    Returns:
        dict: Validation metrics including Pass@1 / 验证指标，包括 Pass@1
    """
    print(f"\n{'='*60}")
    print(f"🔍 Starting validation... / 开始验证...")
    print(f"📊 Samples to validate / 验证样本数: {num_samples}")
    print(f"{'='*60}\n")
    
    # Ensure we don't exceed dataset size / 确保不超过数据集大小
    num_samples = min(num_samples, len(dataset))
    
    # Set model to eval mode / 将模型设置为评估模式
    model.eval()
    
    total_score = 0.0
    compile_success = 0
    partial_pass = 0
    full_pass = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Validating / 验证中"):
            sample = dataset[i]
            
            # Get prompt / 获取提示
            prompt = sample.get('prompt', '')
            if not prompt:
                # If prompt is not pre-generated, skip / 如果提示未预生成，跳过
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
            # Ensure inputs are float32 for 4-bit quantized model / 确保输入为 float32 以适配 4-bit 量化模型
            inputs = {k: v.to(torch.float32) if v.dtype != torch.int64 else v for k, v in inputs.items()}
            
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
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
                    # No valid code found / 未找到有效代码
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
                print(f"⚠️ Error during generation / 生成时出错: {e}")
                continue
    
    # Calculate metrics / 计算指标
    avg_score = total_score / num_samples if num_samples > 0 else 0.0
    compile_rate = compile_success / num_samples if num_samples > 0 else 0.0
    pass_at_1 = full_pass / num_samples if num_samples > 0 else 0.0
    
    results = {
        'avg_score': avg_score,
        'compile_rate': compile_rate,
        'pass_at_1': pass_at_1,
        'compile_success': compile_success,
        'partial_pass': partial_pass,
        'full_pass': full_pass,
        'total_samples': num_samples,
    }
    
    print(f"\n{'='*60}")
    print(f"✅ Validation complete / 验证完成")
    print(f"{'='*60}")
    print(f"📊 Validation Results / 验证结果:")
    print(f"  • Average Score / 平均分数: {avg_score:.4f}")
    print(f"  • Compile Rate / 编译成功率: {compile_rate:.2%}")
    print(f"  • Pass@1 / 通过率: {pass_at_1:.2%}")
    print(f"  • Compile Success / 编译成功: {compile_success}/{num_samples}")
    print(f"  • Partial Pass / 部分通过: {partial_pass}/{num_samples}")
    print(f"  • Full Pass / 完全通过: {full_pass}/{num_samples}")
    print(f"{'='*60}\n")
    
    # Set model back to train mode / 将模型设回训练模式
    model.train()
    
    return results


if __name__ == "__main__":
    """
    Standalone validation script / 独立验证脚本
    Usage / 用法: python validate.py
    """
    import sys
    from datasets import load_dataset
    from utils import get_model_and_tokenizer, TinyLoRAGlobalParams, apply_tiny_lora
    
    # Configuration / 配置
    MODEL_PATH = "./models/Qwen2.5-Coder-3B-Instruct"
    CHECKPOINT_PATH = "./output/luoguqwencoder-lora/tiny_lora_v.pt"
    VAL_DATA_PATH = "./local_code_contests/code_contests_valid.jsonl"
    NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    print(f"\n🚀 Standalone Validation Mode / 独立验证模式")
    print(f"📦 Model Path / 模型路径: {MODEL_PATH}")
    print(f"💾 Checkpoint Path / 检查点路径: {CHECKPOINT_PATH}")
    print(f"📁 Validation Data / 验证数据: {VAL_DATA_PATH}")
    print(f"🔢 Validation Samples / 验证样本数: {NUM_SAMPLES}\n")
    
    # Load checkpoint / 加载检查点
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    u_value = checkpoint['u_value']
    seed = checkpoint['seed']
    
    # Set random seed / 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"✅ Random seed set / 已设置随机种子: {seed}")
    
    # Load model and tokenizer / 加载模型和分词器
    model, tokenizer = get_model_and_tokenizer(MODEL_PATH)
    
    # Create global params / 创建全局参数
    device = model.model.layers[0].self_attn.q_proj.weight.device
    global_params = TinyLoRAGlobalParams(u_dim=u_value, device=device, dtype=torch.bfloat16)
    
    # Inject TinyLoRA / 注入 TinyLoRA
    model.tiny_lora_params = global_params
    print(f"🔧 Injecting TinyLoRA... / 正在注入 TinyLoRA...")
    total_replaced = apply_tiny_lora(model, global_params)
    print(f"✅ TinyLoRA injected / TinyLoRA 注入完成: {total_replaced} layers / 层")
    
    # Load trained weights / 加载训练好的权重
    with torch.no_grad():
        global_params.global_v.copy_(checkpoint['global_v'].to(global_params.global_v.dtype).to(device))
    print(f"✅ Loaded trained weights / 已加载训练权重: global_v shape={global_params.global_v.shape}")
    
    # Load validation dataset / 加载验证数据集
    val_dataset = load_dataset("json", data_files=VAL_DATA_PATH, split="train")
    
    # Apply chat template if needed / 如果需要，应用聊天模板
    if 'prompt' not in val_dataset.column_names:
        print(f"🔄 Applying chat template... / 正在应用聊天模板...")
        val_dataset = val_dataset.map(lambda x: apply_chat_template(x, tokenizer))
    
    # Run validation / 运行验证
    results = run_validation(model, tokenizer, val_dataset, num_samples=NUM_SAMPLES)
    
    print(f"\n✅ Validation finished! / 验证完成！")
    print(f"📊 Final Pass@1 / 最终通过率: {results['pass_at_1']:.2%}")
