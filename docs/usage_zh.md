# 使用指南（中文）

## 1）端到端流程

1. 安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 准备数据：

```bash
python download_dataset.py
```

3. 可选流水线自检：

```bash
python verify_pipeline.py
```

4. 开始训练：

```bash
python train_rl.py 32 2000
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
python train_rl.py 32 2000 --no_quant
```

**训练命令行参数：**
- `u_value`（第一个位置参数）：TinyLoRA参数数量（默认：16）
- `max_samples`（第二个位置参数）：最大训练样本数（默认：2000）
- `--do_validate`：开启训练中验证
- `--val_steps N`：每N步运行验证（默认：100）
- `--val_samples N`：验证样本数（默认：10）
- `--no_quant`：禁用4-bit量化，以BF16加载模型

5. 评估：

```bash
python validate.py 50
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
python test.py --baseline --num_samples 50
```

**验证命令行参数：**
- `num_samples`（位置参数）：验证样本数（默认：10）
- `--no_quant`：禁用4-bit量化进行验证

**测试命令行参数：**
- `--checkpoint_path`：检查点.pt文件路径（默认：./output/luoguqwencoder-lora/tiny_lora_v.pt）
- `--num_samples`：测试样本数（默认：50）
- `--test_data`：测试数据集路径（默认：./local_code_contests/code_contests_test.jsonl）
- `--baseline`：测试基座模型（不含TinyLoRA）
- `--test_seed`：评估随机种子（默认：42）
- `--use_quant`：以4-bit量化加载模型（默认：是）
- `--no_quant`：禁用4-bit量化，以BF16加载

---

## 2）数据与流水线说明

### 数据文件

- `local_code_contests/code_contests_train.jsonl`
- `local_code_contests/code_contests_valid.jsonl`
- `local_code_contests/code_contests_test.jsonl`

### 预处理入口

- 脚本：`download_dataset.py`
- 结果：在本地生成 train/valid/test 三份 jsonl，供 `train_rl.py`、`validate.py`、`test.py` 使用。

### 训练数据控制项

`train_rl.py` 中的关键项：

- `DATASET_CONFIG`：数据源与筛选策略
- `MAX_SAMPLES`：训练样本上限
- `TINYLORA_SEED`：随机投影/采样可复现
- `filter_dataset(...)`：难度/来源过滤入口。

---

## 3）验证与测试说明

### 训练中验证

```bash
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
```

- `--do_validate`：开启周期性验证
- `--val_steps`：验证间隔步数
- `--val_samples`：每次验证样本数。

### 训练后验证

```bash
python validate.py 50
```

- 对指定样本规模进行验证评估。

### 基座与微调对比测试

```bash
python test.py --baseline --num_samples 50
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
```

- `--baseline`：直接评估基座模型
- `--checkpoint_path`：评估 TinyLoRA 微调 checkpoint。

---

## 4）量化与非量化路径

### 默认量化路径

- 默认采用 4-bit 量化加载，显存更友好。

### BF16 路径

```bash
python train_rl.py 32 2000 --no_quant
```

- `--no_quant` 切换到 BF16 流程。

### Checkpoint 元信息

- checkpoint 写入 `is_quantized`。
- 加载时依据该字段避免量化状态不匹配。

---

## 5）常见问题排查

- CUDA OOM：
  - 减少验证频率/样本规模，降低显存压力
  - 默认优先使用量化路径。
- 奖励阶段提取不到代码：
  - 检查 `apply_chat_template` 的提示词结构
  - 检查奖励解析逻辑的格式假设。
- 编译运行超时：
  - 调整 `compile_and_run(..., timeout=2)` 的超时设置。
