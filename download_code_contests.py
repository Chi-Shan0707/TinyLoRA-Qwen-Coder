"""
Download and preprocess CodeContests dataset from HuggingFace.

Dataset: deepmind/code_contests
- Source: Codeforces, AtCoder, and other platforms
- Fields: description, public_tests, private_tests, generated_tests, source, difficulty

Usage:
    python download_code_contests.py
    python download_code_contests.py --splits train valid test
    python download_code_contests.py --output_dir ./my_data

Example / 示例:
    # Download all splits
    python download_code_contests.py --splits train valid test

    # Download only train split
    python download_code_contests.py --splits train
"""

import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 默认保留的字段
DEFAULT_KEEP_COLUMNS = [
    "description",
    "public_tests",
    "private_tests",
    "generated_tests",
    "source",
    "difficulty"
]

# 默认输出目录
DEFAULT_OUTPUT_DIR = "./local_code_contests"

# 默认下载的分区
DEFAULT_SPLITS = ["train"]


def process_and_save_split(split_name, keep_columns, output_dir):
    """Download and save a specific split of the dataset."""
    print(f"\n🚀 正在处理分区: {split_name} ...")

    # streaming=True: 边下边处理，不占用大量磁盘空间
    dataset = load_dataset("deepmind/code_contests", split=split_name, streaming=True)

    output_file = os.path.join(output_dir, f"code_contests_{split_name}.jsonl")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(dataset, desc=f"Saving {split_name}"):
            # 过滤：只保留需要的字段
            filtered_sample = {key: sample[key] for key in keep_columns if key in sample}

            # 写入 JSONL
            f.write(json.dumps(filtered_sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ 分区 {split_name} 完成！共保存 {count} 条数据到 {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess CodeContests dataset / 下载并预处理 CodeContests 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  python download_code_contests.py                        # Download train split only
  python download_code_contests.py --splits train valid test  # Download all splits
  python download_code_contests.py --output_dir ./my_data    # Custom output directory
        """
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=["train", "valid", "test"],
        help="Splits to download / 要下载的分区 (default: train)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory / 输出目录 (default: ./local_code_contests)"
    )
    parser.add_argument(
        "--keep_columns",
        nargs="+",
        default=DEFAULT_KEEP_COLUMNS,
        help="Columns to keep / 要保留的字段 (default: description public_tests private_tests generated_tests source difficulty)"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CodeContests Dataset Downloader")
    print("=" * 60)
    print(f"   Dataset: deepmind/code_contests")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Splits: {args.splits}")
    print(f"   Keep columns: {args.keep_columns}")
    print()

    total_count = 0
    for split in args.splits:
        count = process_and_save_split(split, args.keep_columns, args.output_dir)
        total_count += count

    print()
    print("=" * 60)
    print(f"✅ All splits processed! Total samples: {total_count}")
    print(f"   Data saved to: {args.output_dir}")
    print("=" * 60)
    print()
    print("📝 Usage in training:")
    print("   python train_rl.py 32 2000")


if __name__ == "__main__":
    main()
