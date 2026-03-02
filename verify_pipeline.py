import torch
import os
import subprocess

# Import all shared utilities from utils.py — single source of truth for all pipeline stages
# 从 utils.py 导入所有共享工具 — 训练/验证/测试/流水线验证的单一真相来源
from utils import (
    compile_and_run,
    extract_code_from_response,
    convert_hf_tests_to_list,
    apply_chat_template,
    get_model_and_tokenizer,
)

# ==================== 配置区域 ====================
MS_MODEL_ID = "qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-3B-Instruct"

# Multi-GPU / DDP: LOCAL_RANK is read from environment by get_model_and_tokenizer internally.
# It uses device_map={"":LOCAL_RANK} so each rank holds the full model on its own GPU.
# See utils.py get_model_and_tokenizer for the full DDP-safe implementation.
# 多卡 / DDP：LOCAL_RANK 由 get_model_and_tokenizer 内部读取并处理，这里仅用于日志显示。
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

# 【关键】使用 CodeContests 数据结构进行测试
# 这里使用一个简单的括号匹配问题作为测试题（来自实际数据集）
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
    print_step("STEP 1: 加载模型与Tokenizer")
    
    # 检查 g++
    try:
        subprocess.run(['g++', '--version'], capture_output=True)
        print("✅ 检测到 g++ 编译器")
    except:
        print("❌ 未检测到 g++，请先安装 (sudo apt install g++)")
        return

    model_path = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else MS_MODEL_ID
    print(f"🖥️ LOCAL_RANK: {LOCAL_RANK}")

    # Use get_model_and_tokenizer from utils.py:
    #   - device_map={{"":LOCAL_RANK}}: DDP-safe, each process holds full model on its own GPU
    #   - torch_dtype=bfloat16: non-quantized layers (Embedding/LayerNorm) use BF16, not FP32
    #   - for_inference=True: enables KV cache, skips gradient checkpointing
    # 使用 utils.py 的 get_model_and_tokenizer：
    #   - device_map={{"":LOCAL_RANK}}：DDP 安全，每个进程在自己 GPU 持有完整模型
    #   - torch_dtype=bfloat16：非量化层使用 BF16，不退化为 FP32
    #   - for_inference=True：启用 KV cache，跳过梯度检查点
    model, tokenizer = get_model_and_tokenizer(model_path, use_4bit=True, for_inference=True)

    # ------------------------------------------------------------------
    print_step("STEP 2: 验证 Chat Template (JSON -> Qwen Prompt)")
    
    # Use apply_chat_template from utils.py — identical logic to training, guarantees distribution consistency
    # 使用 utils.py 的 apply_chat_template — 与训练时完全相同的逻辑，保证 prompt 分布一致
    final_prompt = apply_chat_template(TEST_DATA_JSON, tokenizer)['prompt']
    
    print("--- 最终输入给模型的 Prompt 开头部分 ---")
    print(final_prompt[:300] + "...\n")
    print("--- 最终输入给模型的 Prompt 结尾部分 ---")
    print("..." + final_prompt[-100:])
    
    # 检查关键标签
    if "<|im_start|>system" in final_prompt and "<|im_start|>assistant" in final_prompt:
        print("\n✅ 模版格式检查通过 (检测到 Qwen ChatML 标签)")
    else:
        print("\n❌ 警告：未检测到 ChatML 标签，请检查 tokenizer_config.json")

    # ------------------------------------------------------------------
    print_step("STEP 3: 执行模型生成")
    
    inputs = tokenizer([final_prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"Prompt token 长度: {inputs.input_ids.shape[1]}")
    print("正在生成 (Max 1024 tokens)...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,     
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码
    full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # 只要生成部分
    if "<|im_start|>assistant" in full_response:
        response_only = full_response.split("<|im_start|>assistant")[-1]
    else:
        response_only = full_response
    
    print("\n--- 模型生成的代码部分 (前1000字符) ---")
    print(response_only[:1000] + "..." if len(response_only)>500 else response_only)

    # ------------------------------------------------------------------
    print_step("STEP 4: 验证代码提取与评测 (基于 CodeContests 格式)")
    
    # Use extract_code_from_response from utils.py (returns str, empty string if not found)
    # 使用 utils.py 的 extract_code_from_response（返回 str，未找到时返回空字符串）
    extracted_code = extract_code_from_response(response_only)

    # Use convert_hf_tests_to_list from utils.py — same parsing as training reward function
    # 使用 utils.py 的 convert_hf_tests_to_list — 与训练奖励函数解析逻辑完全一致
    test_cases = []
    for test_type in ['public_tests', 'private_tests', 'generated_tests']:
        test_cases.extend(convert_hf_tests_to_list(TEST_DATA_JSON.get(test_type, {})))

    if extracted_code:
        print(f"✅ 成功提取代码")
        print(f"正在使用 {len(test_cases)} 个测试用例进行评测...")

        # Use compile_and_run from utils.py (returns float: 0.0 / 0.5 / 1.0)
        # 使用 utils.py 的 compile_and_run（返回 float：0.0 / 0.5 / 1.0），与训练奖励函数完全相同
        score = compile_and_run(extracted_code, test_cases)

        print(f"\n📊 最终得分 (Reward): {score}")

        if score == 1.0:
            print("🎉 结论：Pipeline 完美通过！模型成功解出了题目。")
        elif score > 0.0:
            print("⚠️ 结论：Pipeline 通畅，代码可运行，但部分用例未通过 (这是 RL 训练需要解决的问题)。")
        else:
            print("⚠️ 结论：代码编译失败或运行全错。")
            print("注意：对于未微调的 3B 模型，第一次做对竞赛题目可能有挑战。只要编译过程没报错，Pipeline 就是好的。")
    else:
        print("❌ 代码提取失败！模型可能没生成代码块。")

if __name__ == "__main__":
    main()