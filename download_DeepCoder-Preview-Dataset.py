"""
Download and preprocess DeepCoder-Preview-Dataset from HuggingFace cache.

Dataset: agentica-org/DeepCoder-Preview-Dataset
- Format: parquet
- Configs: codeforces, lcbv5, primeintellect, taco
- Each has 'train' split
- Fields: problem, tests, metadata

This script reads from local cache if available, or downloads from HuggingFace.

Usage:
    python download_DeepCoder-Preview-Dataset.py
"""

import json
import os
from tqdm import tqdm

# 定义输出目录
OUTPUT_DIR = "./local_Deep-Coder-Preview-Dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DeepCoder数据集配置 - 根据需要选择
DEEPCODER_CONFIGS = ["lcbv5"]  # 可选: ["codeforces", "lcbv5", "primeintellect", "taco"]

# 是否下载solutions（默认不下载）
DOWNLOAD_SOLUTIONS = False


def get_cache_path(config_name):
    """Get the local cache path for a config."""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--agentica-org--DeepCoder-Preview-Dataset/snapshots")

    if not os.path.exists(cache_base):
        return None

    # 查找最新的snapshot目录
    snapshots = [d for d in os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, d))]
    if not snapshots:
        return None

    snapshot_path = os.path.join(cache_base, snapshots[0], config_name)
    return snapshot_path if os.path.exists(snapshot_path) else None


def convert_tests_format(tests_data):
    """
    Convert DeepCoder's tests format to standard format.

    Input formats:
    1. [{"input": ..., "output": ...}, ...] - list of test cases
    2. {"inputs": [...], "outputs": [...]} - dict format

    Output: {"inputs": [...], "outputs": [...]}
    """
    if not tests_data:
        return None

    # tests字段是JSON字符串，需要先解析
    if isinstance(tests_data, str):
        try:
            tests_data = json.loads(tests_data)
        except (json.JSONDecodeError, ValueError):
            return None

    # 格式1: 列表格式 [{"input": ..., "output": ...}, ...]
    if isinstance(tests_data, list) and len(tests_data) > 0:
        # 检查第一个元素是否是字典
        if isinstance(tests_data[0], dict):
            inputs = []
            outputs = []
            for tc in tests_data:
                if isinstance(tc, dict):
                    inp = tc.get("input", "")
                    out = tc.get("output", "")
                    if inp is not None:
                        inputs.append(inp)
                    if out is not None:
                        outputs.append(out)

            if len(inputs) >= 5 and len(outputs) >= 5:
                return {"inputs": inputs, "outputs": outputs}

    # 格式2: 字典格式 {"inputs": [...], "outputs": [...]}
    if isinstance(tests_data, dict):
        inputs = tests_data.get("inputs", [])
        outputs = tests_data.get("outputs", [])

        # 确保是列表
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        # 检查是否有足够的测试用例
        if len(inputs) >= 5 and len(outputs) >= 5:
            return {"inputs": inputs, "outputs": outputs}

    return None


def process_config(config_name):
    """Process a single config from DeepCoder dataset."""
    print(f"\n{'=' * 60}")
    print(f"📂 Processing config: {config_name}")
    print(f"{'=' * 60}")

    # 尝试从本地缓存读取
    cache_path = get_cache_path(config_name)

    if cache_path is None:
        print(f"   ❌ No local cache found for config: {config_name}")
        print(f"   Please ensure the dataset is downloaded first.")
        return 0, 0, 0

    print(f"   Using local cache: {cache_path}")

    # 获取所有parquet文件
    parquet_files = sorted([f for f in os.listdir(cache_path) if f.endswith('.parquet')])

    if not parquet_files:
        print(f"   ❌ No parquet files found in cache")
        return 0, 0, 0

    print(f"   Found {len(parquet_files)} parquet files")

    output_file = os.path.join(OUTPUT_DIR, f"deepcoder_{config_name}_train.jsonl")

    count_total = 0
    count_filtered = 0
    count_saved = 0

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import pandas as pd

    with open(output_file, "w", encoding="utf-8") as f:
        for parquet_file in tqdm(parquet_files, desc=f"Processing {config_name}"):
            parquet_path = os.path.join(cache_path, parquet_file)

            # 读取parquet文件
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as e:
                print(f"   ⚠️ Error reading {parquet_file}: {e}")
                continue

            # 处理每一行
            for idx, row in df.iterrows():
                count_total += 1

                # 获取problem字段
                problem = row.get("problem", "")
                if not problem or not isinstance(problem, str):
                    count_filtered += 1
                    continue

                # 获取tests字段
                tests_data = row.get("tests", None)

                # 转换测试用例格式
                converted_tests = convert_tests_format(tests_data)

                # 过滤：DeepCoder要求至少5个测试用例
                if converted_tests is None:
                    count_filtered += 1
                    continue

                # 构建保留字段 - 与CodeContests格式兼容
                filtered_sample = {
                    "description": problem,
                    "input_output": converted_tests,
                    "config": config_name,
                }

                # 可选：添加solutions
                if DOWNLOAD_SOLUTIONS:
                    metadata = row.get("metadata", {})
                    if isinstance(metadata, dict):
                        solutions = metadata.get("solutions", [])
                        if solutions:
                            filtered_sample["solutions"] = solutions

                # 写入JSONL
                f.write(json.dumps(filtered_sample, ensure_ascii=False) + "\n")
                count_saved += 1

    print(f"   ✅ Complete: {count_saved} samples saved (filtered: {count_filtered}, total: {count_total})")
    print(f"   📁 Output: {output_file}")

    return count_total, count_filtered, count_saved


def process_deepcoder_dataset():
    """Process all configs from DeepCoder dataset."""
    print("=" * 60)
    print("🚀 DeepCoder-Preview-Dataset Downloader (from local cache)")
    print("=" * 60)
    print(f"   Dataset: agentica-org/DeepCoder-Preview-Dataset")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Configs: {DEEPCODER_CONFIGS}")
    print(f"   Download solutions: {DOWNLOAD_SOLUTIONS}")
    print()

    total_total = 0
    total_filtered = 0
    total_saved = 0

    for config in DEEPCODER_CONFIGS:
        t, f, s = process_config(config)
        total_total += t
        total_filtered += f
        total_saved += s

    print()
    print("=" * 60)
    print("✅ All configs processed!")
    print(f"   Total samples processed: {total_total}")
    print(f"   Filtered (less than 5 tests): {total_filtered}")
    print(f"   Saved: {total_saved}")
    print("=" * 60)
    print()
    print("📝 Usage in training:")
    print("   python train_rl.py 32 2000 --dataset deepcoder")


if __name__ == "__main__":
    process_deepcoder_dataset()
