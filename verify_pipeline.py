import torch
import os
import re
import json
import subprocess
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==================== 配置区域 ====================
MS_MODEL_ID = "qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-3B-Instruct"

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

def extract_code(completion):
    """从回复中提取代码，逻辑同 train_rl.py"""
    # 优先匹配代码块
    match = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1), "Code Block"
    # 兜底匹配 #include
    elif "#include" in completion:
        return completion, "Raw Text"
    else:
        return None, "Failed"

def compile_and_run(code, test_cases):
    """编译并运行，逻辑同 train_rl.py"""
    # 移除 freopen，防止卡死
    code = re.sub(r'freopen\s*\(.*?\);', '', code, flags=re.IGNORECASE)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "solution.cpp")
        exe_file = os.path.join(temp_dir, "solution")
        
        # 写入
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write(code)
            
        print(f"   -> 正在编译临时文件...")
        # 编译
        try:
            res = subprocess.run(
                ['g++', src_file, '-o', exe_file, '-O2'],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode != 0:
                return 0.0, f"编译失败:\n{res.stderr}"
        except Exception as e:
            return 0.0, f"编译异常: {e}"

        # 运行测试用例
        passed = 0
        total = len(test_cases)
        for i, case in enumerate(test_cases):
            input_data = case['input']
            expected_output = case['output'].strip()
            
            try:
                res = subprocess.run(
                    [exe_file],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=2 # 2秒超时
                )
                actual_output = res.stdout.strip()
                
                if actual_output == expected_output:
                    print(f"   -> Case {i+1}: ✅ 通过 (输入: '{input_data.strip()}' | 预期: '{expected_output}' | 实际: '{actual_output}')")
                    passed += 1
                else:
                    print(f"   -> Case {i+1}: ❌ 失败 (输入: '{input_data.strip()}' | 预期: '{expected_output}' | 实际: '{actual_output}')")
            except subprocess.TimeoutExpired:
                print(f"   -> Case {i+1}: ⚠️ 运行超时 (Timeout)")
            except Exception as e:
                print(f"   -> Case {i+1}: ⚠️ 运行错误 {e}")
        
        return passed / total, "Success"

def main():
    print_step("STEP 1: 加载模型与Tokenizer")
    
    # 检查 g++
    try:
        subprocess.run(['g++', '--version'], capture_output=True)
        print("✅ 检测到 g++ 编译器")
    except:
        print("❌ 未检测到 g++，请先安装 (sudo apt install g++)")
        return

    # 加载 Tokenizer
    model_path = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else MS_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 (4-bit)
    print(f"正在加载模型: {model_path} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ 模型加载完成")

    # ------------------------------------------------------------------
    print_step("STEP 2: 验证 Chat Template (JSON -> Qwen Prompt)")
    
    # 模拟 train_rl.py 中的 apply_chat_template 逻辑
    description = TEST_DATA_JSON.get('description', '')
    public_tests = TEST_DATA_JSON.get('public_tests', {})
    
    # Build public test cases section / 构建公开测试用例部分
    public_tests_section = ""
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
    raw_prompt = f"""You will be given a programming contest problem. Please reason step by step and provide a complete C++ implementation.
Output the solution in a code block. Do not include debugging info or extra output. Limit reasoning to 128 tokens.


【Problem Description 】
{description}

{public_tests_section}

Please provide your C++ solution :"""
    
    messages = [
        {"role": "system", "content": "You are an expert competitive programmer. Output valid C++ code that compiles and solves the problem correctly."},
        {"role": "user", "content": raw_prompt}
    ]
    
    # 应用模版
    final_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
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
    
    extracted_code, method = extract_code(response_only)
    
    # Parse test cases from CodeContests format / 从 CodeContests 格式解析测试用例
    test_cases = []
    for test_type in ['public_tests', 'private_tests', 'generated_tests']:
        test_data = TEST_DATA_JSON.get(test_type, {})
        if isinstance(test_data, dict) and 'input' in test_data and 'output' in test_data:
            inputs = test_data['input'] if isinstance(test_data['input'], list) else [test_data['input']]
            outputs = test_data['output'] if isinstance(test_data['output'], list) else [test_data['output']]
            for inp, out in zip(inputs, outputs):
                test_cases.append({'input': inp, 'output': out})
    
    if extracted_code:
        print(f"✅ 成功提取代码 (方式: {method})")
        print(f"正在使用 {len(test_cases)} 个测试用例进行评测...")
        
        # 实际运行评测
        score, msg = compile_and_run(extracted_code, test_cases)
        
        print(f"\n📊 最终得分 (Reward): {score}")
        
        if score == 1.0:
            print("🎉 结论：Pipeline 完美通过！模型成功解出了题目。")
        elif score > 0.0:
            print("⚠️ 结论：Pipeline 通畅，代码可运行，但部分用例未通过 (这是 RL 训练需要解决的问题)。")
        else:
            print(f"⚠️ 结论：代码编译失败或运行全错。详细信息: {msg}")
            print("注意：对于未微调的 3B 模型，第一次做对竞赛题目可能有挑战。只要编译过程没报错，Pipeline 就是好的。")
    else:
        print("❌ 代码提取失败！模型可能没生成代码块。")

if __name__ == "__main__":
    main()