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

# Define output directory
OUTPUT_DIR = "./local_Deep-Coder-Preview-Dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DeepCoder dataset configs; adjust as needed
DEEPCODER_CONFIGS = ["lcbv5"]  # Options: ["codeforces", "lcbv5", "primeintellect", "taco"]

# Whether to include solutions; disabled by default
DOWNLOAD_SOLUTIONS = False


def get_cache_path(config_name):
    """Get the local cache path for a config."""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--agentica-org--DeepCoder-Preview-Dataset/snapshots")

    if not os.path.exists(cache_base):
        return None

    # Find the latest snapshot directory
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

    # The tests field is a JSON string and must be parsed first
    if isinstance(tests_data, str):
        try:
            tests_data = json.loads(tests_data)
        except (json.JSONDecodeError, ValueError):
            return None

    # Format 1: list format
    if isinstance(tests_data, list) and len(tests_data) > 0:
        # Check whether the first item is a dict
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

    # Format 2: dict format
    if isinstance(tests_data, dict):
        inputs = tests_data.get("inputs", [])
        outputs = tests_data.get("outputs", [])

        # Ensure values are lists
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        # Check whether enough test cases are available
        if len(inputs) >= 5 and len(outputs) >= 5:
            return {"inputs": inputs, "outputs": outputs}

    return None


def process_config(config_name):
    """Process a single config from DeepCoder dataset."""
    print(f"\n{'=' * 60}")
    print(f"[DIR] Processing config: {config_name}")
    print(f"{'=' * 60}")

    # Try reading from the local cache
    cache_path = get_cache_path(config_name)

    if cache_path is None:
        print(f"   [ERROR] No local cache found for config: {config_name}")
        print(f"   Please ensure the dataset is downloaded first.")
        return 0, 0, 0

    print(f"   Using local cache: {cache_path}")

    # Collect all parquet files
    parquet_files = sorted([f for f in os.listdir(cache_path) if f.endswith('.parquet')])

    if not parquet_files:
        print(f"   [ERROR] No parquet files found in cache")
        return 0, 0, 0

    print(f"   Found {len(parquet_files)} parquet files")

    output_file = os.path.join(OUTPUT_DIR, f"deepcoder_{config_name}_train.jsonl")

    count_total = 0
    count_filtered = 0
    count_saved = 0

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import pandas as pd

    with open(output_file, "w", encoding="utf-8") as f:
        for parquet_file in tqdm(parquet_files, desc=f"Processing {config_name}"):
            parquet_path = os.path.join(cache_path, parquet_file)

            # Read parquet file
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as e:
                print(f"   [WARN] Error reading {parquet_file}: {e}")
                continue

            # Process each row
            for idx, row in df.iterrows():
                count_total += 1

                # Read the problem field
                problem = row.get("problem", "")
                if not problem or not isinstance(problem, str):
                    count_filtered += 1
                    continue

                # Read the tests field
                tests_data = row.get("tests", None)

                # Convert test case format
                converted_tests = convert_tests_format(tests_data)

                # Filter examples
                if converted_tests is None:
                    count_filtered += 1
                    continue

                # Build retained fields compatible with CodeContests format
                filtered_sample = {
                    "description": problem,
                    "input_output": converted_tests,
                    "config": config_name,
                }

                # Optionally add solutions
                if DOWNLOAD_SOLUTIONS:
                    metadata = row.get("metadata", {})
                    if isinstance(metadata, dict):
                        solutions = metadata.get("solutions", [])
                        if solutions:
                            filtered_sample["solutions"] = solutions

                # Write JSONL
                f.write(json.dumps(filtered_sample, ensure_ascii=False) + "\n")
                count_saved += 1

    print(f"   [OK] Complete: {count_saved} samples saved (filtered: {count_filtered}, total: {count_total})")
    print(f"   [FILE] Output: {output_file}")

    return count_total, count_filtered, count_saved


def process_deepcoder_dataset():
    """Process all configs from DeepCoder dataset."""
    print("=" * 60)
    print("[START] DeepCoder-Preview-Dataset Downloader (from local cache)")
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
    print("[OK] All configs processed!")
    print(f"   Total samples processed: {total_total}")
    print(f"   Filtered (less than 5 tests): {total_filtered}")
    print(f"   Saved: {total_saved}")
    print("=" * 60)
    print()
    print("[INFO] Usage in training:")
    print("   python train_rl.py 32 2000 --dataset deepcoder")


if __name__ == "__main__":
    process_deepcoder_dataset()
