import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 1. 定义你需要保留的字段
KEEP_COLUMNS = [
    "description", 
    "public_tests", 
    "private_tests", 
    "generated_tests", 
    "source", 
    "difficulty"
]

# 2. 定义要下载的分区
# SPLITS = ["train", "valid", "test"]
# SPLITS = ["valid","test"]
SPLITS = ["train"]  # 先测试一个分区，确认无误后再全部下载

# 定义输出目录
OUTPUT_DIR = "./local_code_contests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_save_split(split_name):
    print(f"🚀 正在处理分区: {split_name} ...")
    
    # 关键：streaming=True
    # 这意味着我们不会下载几百 GB 的原始文件，而是像看视频一样边下边处理
    dataset = load_dataset("deepmind/code_contests", split=split_name, streaming=True)
    
    output_file = os.path.join(OUTPUT_DIR, f"code_contests_{split_name}.jsonl")
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        # 使用 tqdm 显示进度（由于流式加载不知道总长度，只显示处理条数）
        for sample in tqdm(dataset, desc=f"Saving {split_name}"):
            # 过滤：只构建包含目标字段的字典
            filtered_sample = {key: sample[key] for key in KEEP_COLUMNS}
            
            # 写入 JSONL
            # ensure_ascii=False 保证中文描述（如果有）能正常显示
            f.write(json.dumps(filtered_sample, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"✅ 分区 {split_name} 完成！共保存 {count} 条数据到 {output_file}\n")

if __name__ == "__main__":
    print("开始轻量化下载 deepmind/code_contests 数据集...")
    print(f"保留字段: {KEEP_COLUMNS}")
    
    for split in SPLITS:
        process_and_save_split(split)
        
    print("🎉 所有数据下载并清洗完成！")
    print(f"数据保存在: {OUTPUT_DIR}")