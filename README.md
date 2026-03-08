<div align="center">

# 🧬 TinyLoRA-Qwen-Coder

**Fine-tune a code model with configurable tiny parameters (default: 32) — and it actually works.**

[![Paper](https://img.shields.io/badge/Paper-Learning_to_Reason_in_13_Parameters-blue)](./paper-Learning%20to%20Reason%20in%2013%20Parameters/README.md)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-green)](./LICENSE)
[![Model](https://img.shields.io/badge/Base-Qwen2.5--Coder--3B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
[![Dataset](https://img.shields.io/badge/Data-CodeContests-orange)](https://huggingface.co/datasets/deepmind/code_contests)
[![Dataset](https://img.shields.io/badge/Data-DeepCoder-blue)](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)
![Version](https://img.shields.io/badge/Version-v4.0-red)

<br>

| 🔢 Parameters | 🧠 Base Model | 🎯 Task | ⚡ Method | 💾 VRAM |
| :---: | :---: | :---: | :---: | :---: |
| **u=32 (adjustable)** | Qwen2.5-Coder-3B(adjustable) | C++ Code Gen | GRPO (RL) | 16GB+ |

</div>

> **v4.0** — Increased `num_iterations=4` (making clip_high more effective) · DeepCoder-Preview-Dataset (lcbv5, 28 samples) · Pass@1 +100% · Training time -73%

> We adapt TinyLoRA from math reasoning to competitive programming: inject tiny shared parameters into Qwen2.5-Coder-3B, train with GRPO, and reward real `g++` compile-and-run correctness.
>
> If this project is useful to you, please give it a ⭐ Star.<br>
> 如果你觉得我的项目有意思的话，可否留下一个star呢(✿◠‿◠)?

**Language / 语言**: [English](#english) | [中文](#中文)

---

## English

### 🚀 Project Intro

- **Task**: competitive C++ code generation with verifiable compile-and-run rewards.
- **Core method**: TinyLoRA + GRPO on Qwen2.5-Coder-3B-Instruct(configurable).
- **Default tiny setup**: `u=32` shared trainable scalars (configurable).
- **Runtime modes**: 4-bit quantized (default) and BF16 (`--no_quant`).
- **CLI Help**: All scripts support `--help` for detailed usage information.

### 📖 Training Method Overview (Based on Paper)

This section describes how TinyLoRA works, as introduced in the paper ["Learning to Reason in 13 Parameters"](https://arxiv.org/abs/2602.04118).<br>
Technical Guide (EN default): [TECHNICAL_GUIDE.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/TECHNICAL_GUIDE.md)

#### 1. Shared Weight Layer with Random Projection

TinyLoRA freezes the pretrained model's weights and injects a **tiny trainable parameter layer** using **Low-Rank Adaptation (LoRA)** with a key twist — **parameter sharing**:

- **Core Equation**: $W' = W + U\Sigma\left(\sum_{i=1}^{u} v_i P_i\right)V^\top$
  - $W$: Frozen pretrained weight matrix
  - $U, \Sigma, V$: Frozen SVD skeleton (obtained via SVD decomposition of $W$)
  - $P_i$: Fixed random projection matrices (generated once and frozen)
  - $v_i$: **Trainable tiny scalar vector** (the only parameters updated during training)

- **Parameter Sharing**: Instead of training separate low-rank matrices for each layer, all layers share the **same random projection bases ($P_i$)** and only differ in their **trainable scalar vector ($v$)**. This dramatically reduces the number of trainable parameters from $O(d_{model} \times rank \times num\_layers)$ to just $O(u)$.

- **How it works**: For each layer with weight $W$, we:
  1. Compute SVD: $W = U\Sigma V^\top$
  2. Generate random projection matrix $P$ (fixed throughout training)
  3. Compute delta: $\Delta W = U\Sigma (v \cdot P) V^\top$
  4. Final weight: $W' = W + \Delta W$

#### 2. Fine-tuning the Vector v with GRPO

Only the vector $v$ (dimension $u$, typically 16-32) is trainable
- All other parameters (base model weights $W$, SVD components $U,\Sigma,V$, projection matrices $P$) remain frozen
- This is extremely parameter-efficient: training just 16-32 scalars can influence the entire model behavior

#### 3. Reward Calculation

The reward function evaluates code quality through actual compilation and execution:

| Condition | Score |
| :--- | :---: |
| Compile failed | `0.0` |
| Compile success (0 tests passed) | `0.5` |
| Partial pass (k/N tests passed) | `0.5 + 0.5 × (k/N)` |
| All tests passed | `1.0` |

- **Compilation**: Uses `g++ -O2 -std=c++17`
- **Execution**: Runs against test cases with 2-second timeout
- **Output comparison**: Exact match after stripping whitespace
- **Difficulty scaling**: Different sources/difficulties may have reward multipliers (e.g., Codeforces B-level × 1.1)

### ⚡ Quick Start (Install + Configure + Run)

1) Environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Download and preprocess dataset:

```bash
# Option A: CodeContests dataset (default)
python download_code_contests.py

# Option B: DeepCoder dataset (from agentica-org/DeepCoder-Preview-Dataset, parquet format)
python download_DeepCoder-Preview-Dataset.py
```

3) Optional end-to-end sanity check:

```bash
python verify_pipeline.py
```

4) Start RL training:

> args:
  u_value: the first argument value (TinyLoRA parameter count, default: 16)<br>
  max_samples: the second argument value (max training samples, default: 2000)<br>
  --do_validate: enable validation during training<br>
  --val_steps N: run validation every N steps (default: 100)<br>
  --val_samples N: number of validation samples (default: 10)<br>
  --no_quant: disable 4-bit quantization, load model in BF16<br>
  --rank N: TinyLoRA SVD rank (default: 2)<br>
  --dataset NAME: choose dataset - 'code_contests' (default) or 'deepcoder'<br>

```bash
# Using CodeContests dataset (default)
python train_rl.py 32 2000
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
python train_rl.py 32 2000 --no_quant
python train_rl.py 32 2000 --rank 4

# Using DeepCoder dataset
python train_rl.py 32 2000 --dataset deepcoder
```

5) Evaluate:

```bash
python validate.py 50
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
python test.py --baseline --num_samples 50
```

### 🧭 Configurable Controls (System Blocks)

This section is organized by code-level **control blocks** (not flat knobs), matching your Python implementation.

#### 🎯 Block 1: Reward System (`train_rl.py` + `utils.py`)

- **Entry points**: `code_reward_func` in `train_rl.py`, `compile_and_run` in `utils.py`.
- **Current behavior**:
  - compile fail / invalid code → `0.0`
  - compile success (partial) → `0.5`
  - full pass → `1.0`
- **Controllable scope**:
  - reward shape (discrete vs continuous)
  - test source mix (`public + private + generated`)
  - runtime timeout (`compile_and_run(..., timeout=2)`)
  - difficulty/source reward scaling (`REWARD_SCALING_CONFIG`)
  - no-code penalty policy.

#### 📚 Block 2: Training Data Selection (`train_rl.py`)

- **Entry points**: `DATASET_CONFIG`, `filter_dataset`, `MAX_SAMPLES`, `TINYLORA_SEED`.
- **Controllable scope**:
  - platform coverage (e.g., add CodeJam/AIZU)
  - difficulty window expansion (e.g., include C)
  - sample cap and shuffle seed strategy
  - train/valid/test file replacement.

#### 🔧 Block 3: TinyLoRA Architecture (`utils.py` + `train_rl.py`)

- **Entry points**: `TinyLoRAGlobalParams`, `TinyLoRALinear`, `apply_tiny_lora`.
- **Controllable scope**:
  - parameter count via CLI `u`
  - `rank`: SVD rank via CLI `--rank N` (default: 2), controls capacity/stability tradeoff
  - replacement scope (all proj vs attention-only)
  - projection seed via `TINYLORA_SEED`.

#### ⚙️ Block 4: GRPO Optimization (`train_rl.py`)

- **Entry points**: `GRPOConfig` in `train_rl.py`.
- **Current defaults**:
  - `num_generations=4`
  - `learning_rate=1e-5`
  - `gradient_accumulation_steps=8`
  - `max_completion_length=1024`
  - `num_train_epochs=1`.

#### 💬 Block 5: Prompt Construction (`utils.py`)

- **Entry point**: `apply_chat_template`.
- **Controllable scope**:
  - system style and reasoning constraints
  - whether to expose public tests
  - language/task template variants.

### 🧩 Docs Index

- Detailed Usage Guide (includes data pipeline + validation/testing): [docs/usage_en.md](./docs/usage_en.md)
- Changelog (detailed): [docs/changelog_en.md](./docs/changelog_en.md)
- Known Pitfalls & Notes: [docs/warning_en.md](./docs/warning_en.md)
- FAQ: [docs/faq_en.md](./docs/faq_en.md)
- Paper Hub: [paper-Learning to Reason in 13 Parameters/README.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/README.md)
- Technical Guide (EN default): [TECHNICAL_GUIDE.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/TECHNICAL_GUIDE.md)
- Technical Guide (CN): [TECHNICAL_GUIDE_CN.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/TECHNICAL_GUIDE_CN.md)

### 📈 Evidence of Change

**v4.0: Increased `num_iterations=4` + DeepCoder Dataset (lcbv5)**

Strict A/B comparison with identical test conditions:

- test seed: `42`
- same test dataset: `code_contests_test.jsonl`
- same sample count: 165
- Key change: Increased `num_iterations` from 1 to 4 (making clip_high / DeepCoder method more effective)

Training comparison:

| Config | Old (v3.x) | New (v4.0) |
| :--- | :--- | :--- |
| Training Dataset | code_contests | lcbv5 (DeepCoder-Preview-Dataset) |
| `num_iterations` | 1 | 4 |
| Training Samples | 13,328 | 28 |
| Training Time | ~4h 24m | ~1h 12m |

Test results:

| Metric | Old Training | New Training (v4.0) | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Samples** | 165 | 165 | — |
| **Pass@1** | 1.82% (3/165) | 3.64% (6/165) | **+100%** |
| **Compile Rate** | 73.33% (121/165) | 76.36% (126/165) | **+4.13%** |
| **Average Score** | 0.4274 | 0.4489 | **+5.03%** |

v4.0 demonstrates that:
- Using higher quality training data (lcbv5) significantly improves model performance
- Increasing `num_iterations` makes clip_high (DeepCoder method) more effective
- Training data reduced by **99.8%** (13328 → 28)
- Training time reduced by **73%** (4h24m → 1h12m)

[Detailed comparison](./docs/comparison_en.md)

---

**Earlier Results (v3.x baseline):**

Strict A/B comparison with identical settings:

- test seed: `42`
- same sample order
- same 10 test samples from `code_contests_test.jsonl`
- training command:

```bash
python train_rl.py 32 20 --do_validate --val_steps 10 --val_samples 10
```

Training snapshot:

| Config | Value |
| :--- | :--- |
| Trainable vector dim `u` | 32 |
| TinyLoRA rank | 2 |
| Training samples | 20 |
| Checkpoint seed | 212 |
| `global_v` shape | `torch.Size([32])` |

Test comparison:

| Metric | Baseline (Base Model) | TinyLoRA Fine-tuned (`u=32`) | Delta |
| :--- | :---: | :---: | :---: |
| **Total Samples** | 10 | 10 | — |
| **Average Score** | 0.4500 | 0.4000 | -0.05 |
| **Compile Rate** | 80.00% (8/10) | 80.00% (8/10) | Same |
| **Pass@1** | 10.00% (1/10) | 0.00% (0/10) | -10% |
| **Partial Pass** | **7/10** | **8/10** | **+1** |
| **No Code Extracted** | 0/10 | 0/10 | Same |

Interpretation:

- tiny-parameter RL already changes model behavior under strict controls;
- early-stage gains can first appear as partial-pass improvements before full-pass convergence.

### 📜 License

- Repository scripts: CC BY 4.0.
- Please also follow upstream model/dataset licenses.

### 📚 Citation / BibTeX

```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```

```bibtex@misc{deepcoder2025,
  title={DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level},
  author={Michael Luo and Sijun Tan and Roy Huang and Ameen Patel and Alpay Ariyak and Qingyang Wu and Xiaoxiang Shi and Rachel Xin and Colin Cai and Maurice Weber and Ce Zhang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51}},
  note={Notion Blog},
  year={2025}
}
```

```bibtex@article{li2022competition,
  title={Competition-Level Code Generation with AlphaCode},
  author={Li, Yujia and Choi, David and Chung, Junyoung and Kushman, Nate and
    Schrittwieser, Julian and Leblond, R{\'e}mi and Eccles, Tom and
    Keeling, James and Gimeno, Felix and Dal Lago, Agustin and
    Hubert, Thomas and Choy, Peter and de Masson d'Autume, Cyprien and
    Babuschkin, Igor and Chen, Xinyun and Huang, Po-Sen and Welbl, Johannes and
    Gowal, Sven and Cherepanov, Alexey and Molloy, James and
    Mankowitz, Daniel and Sutherland Robson, Esme and Kohli, Pushmeet and
    de Freitas, Nando and Kavukcuoglu, Koray and Vinyals, Oriol},
  journal={arXiv preprint arXiv:2203.07814},
  year={2022}
}
```

---

## 中文

### 🚀 项目介绍

- **任务**：面向竞赛题的 C++ 代码生成（可编译、可运行、可验证）。
- **核心方法**：在 Qwen2.5-Coder-3B-Instruct （可替换）上做 TinyLoRA + GRPO。
- **默认微调规模**：`u=32`（可调），支持 4-bit 与 BF16 两种流程。


### 📖 训练方法概述（基于论文）

本节介绍 TinyLoRA 的工作原理，源自论文 ["Learning to Reason in 13 Parameters"](https://arxiv.org/abs/2602.04118)。<br>
技术文档（中文）： [TECHNICAL_GUIDE_CN.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/TECHNICAL_GUIDE_CN.md)

#### 1. 共享权重层与随机投影

TinyLoRA 冻结预训练模型的权重，并注入一个**极小的可训练参数层**，使用**低秩适配（LoRA）**的关键技巧 —— **参数共享**：

- **核心公式**: $W' = W + U\Sigma\left(\sum_{i=1}^{u} v_i P_i\right)V^\top$
  - $W$: 冻结的预训练权重矩阵
  - $U, \Sigma, V$: 冻结的 SVD 骨架（通过 $W$ 的 SVD 分解获得）
  - $P_i$: 固定随机投影矩阵（生成一次后冻结）
  - $v_i$: **可训练极小标量向量**（训练期间唯一更新的参数）

- **参数共享**: 不为每个层训练独立的低秩矩阵，所有层共享**相同的随机投影基 ($P_i$)**，仅通过**可训练标量向量 ($v$)** 来区分。这将可训练参数数量从 $O(d_{model} \times rank \times num\_layers)$ 大幅减少到仅 $O(u)$。

- **工作原理**: 对于每个有权重 $W$ 的层：
  1. 计算 SVD: $W = U\Sigma V^\top$
  2. 生成随机投影矩阵 $P$（训练期间固定）
  3. 计算增量: $\Delta W = U\Sigma (v \cdot P) V^\top$
  4. 最终权重: $W' = W + \Delta W$

#### 2. 使用 GRPO 微调向量 v

- GRPO 训练期间，仅向量 $v$（维度 $u$，通常 16-32）可训练
- 其他所有参数（基础模型权重 $W$、SVD 分量 $U,\Sigma,V$、投影矩阵 $P$）保持冻结
- 这极其参数高效：仅训练 16-32 个标量就能影响整个模型行为

#### 3. 奖励计算

奖励函数通过实际编译和执行来评估代码质量：

| 条件 | 分数 |
| :--- | :---: |
| 编译失败 | `0.0` |
| 编译成功（0 个测试通过） | `0.5` |
| 部分通过（k/N 个测试通过） | `0.5 + 0.5 × (k/N)` |
| 全部测试通过 | `1.0` |

- **编译**: 使用 `g++ -O2 -std=c++17`
- **执行**: 对测试用例运行，超时 2 秒
- **输出比较**: 去除空白后精确匹配
- **难度缩放**: 不同来源/难度可能有奖励倍数（例如 Codeforces B 级 × 1.1）

### ⚡ 快速开始（安装 + 配置 + 启动）

1）环境准备：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2）下载并预处理数据：

```bash
python download_dataset.py
```

3）可选流水线自检：

```bash
python verify_pipeline.py
```

4）开始训练：

> args: 
  u_value: 第一个参数（TinyLoRA参数数量，默认：16）<br>
  max_samples: 第二个参数（最大训练样本数，默认：2000）<br>
  --do_validate: 开启训练中验证<br>
  --val_steps N: 每N步运行验证（默认：100）<br>
  --val_samples N: 验证样本数（默认：10）<br>
  --no_quant: 禁用4-bit量化，以BF16加载模型<br>
  --rank N: TinyLoRA SVD 秩（默认：2）<br>

```bash
python train_rl.py 32 2000
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
python train_rl.py 32 2000 --no_quant
python train_rl.py 32 2000 --rank 4
```

5）评估：

```bash
python validate.py 50
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
python test.py --baseline --num_samples 50
```

### 🧭 可调控参数（按“块”组织）

下面按代码结构给出 5 个可调控块，避免“散点式参数罗列”。

#### 🎯 块 1：奖励系统（`train_rl.py` + `utils.py`）

- **入口**：`code_reward_func` + `compile_and_run`
- **当前机制**：编译失败 `0.0`，部分通过 `0.5`，全部通过 `1.0`
- **可调范围**：
  - 离散/连续奖励形状
  - `public/private/generated` 测试源组合
  - 运行超时（`timeout=2`）
  - `REWARD_SCALING_CONFIG` 难度缩放
  - 无代码提取惩罚策略。

#### 📚 块 2：训练数据选择（`train_rl.py`）

- **入口**：`DATASET_CONFIG`、`filter_dataset`、`MAX_SAMPLES`、`TINYLORA_SEED`
- **可调范围**：
  - 平台范围（可加 CodeJam/AIZU）
  - 难度窗口（可扩到 C 级）
  - 采样上限与随机种子
  - 数据文件替换策略。

#### 🔧 块 3：TinyLoRA 架构（`utils.py` + `train_rl.py`）

- **入口**：`TinyLoRAGlobalParams`、`TinyLoRALinear`、`apply_tiny_lora`
- **可调范围**：
  - `u`（可训练参数总量）
  - `rank`：SVD 秩，通过 `--rank N` 配置（默认 2），控制容量/稳定性权衡
  - 注入层范围（全量/attention-only）
  - `TINYLORA_SEED`（随机投影基）。

#### ⚙️ 块 4：GRPO 优化器配置（`train_rl.py`）

- **入口**：`GRPOConfig`
- **当前默认**：`num_generations=4`、`learning_rate=1e-5`、`gradient_accumulation_steps=8`、`max_completion_length=1024`、`num_train_epochs=1`

#### 💬 块 5：Prompt 模板系统（`utils.py`）

- **入口**：`apply_chat_template`
- **可调范围**：
  - system prompt 与推理提示
  - 是否展示 public tests
  - 语言模板（C++ / Python 等）。

### 🧩 文档索引

- 详细使用指南（含数据流水线 + 验证测试细节）： [docs/usage_zh.md](./docs/usage_zh.md)
- 更新日志（详细版）： [docs/changelog_zn.md](./docs/changelog_zn.md)
- 已知坑点与注意事项： [docs/warning_zn.md](./docs/warning_zn.md)
- 常见问题： [docs/faq_zh.md](./docs/faq_zh.md)
- 论文入口： [paper-Learning to Reason in 13 Parameters/README.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/README.md)
- 技术文档（英文默认）： [TECHNICAL_GUIDE.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/TECHNICAL_GUIDE.md)
- 技术文档（中文）： [TECHNICAL_GUIDE_CN.md](./paper-Learning%20to%20Reason%20in%2013%20Parameters/TECHNICAL_GUIDE_CN.md)

### 💭 心路历程（展开看作者的踩坑之旅 PwP）

<details>
<summary>点击展开</summary>

*SFT 时代的故事*：

> 什么，你问我为什么要挑选 Qwen2.5-1.5B-Instruct 进行微调？—— 那当然是因为它参数量小啦。<br>
> 什么，你继续问我为什么不挑选 Qwen2.5-Coder-1.5B-Instruct？<br>
> ~~其实是我问千问推荐了这个，然后忘记继续搜集信息直接开搞，训练到一半才刷到 Coder 版本 PwP~~<br>
> ~~第一遍实在太差了，换 Coder 吧~~ → 这个也太差劲了，上 7B 吧 PwP<br>
> *不对，为什么疯狂报 mismatch 啊？从 1.5B→7B 我啥都没改啊？疯狂 debug……*<br>
> 7B 根本跑不动，只能 3B → ~~训练完了参数上传不动 PwP~~

然后，6号晚上，~~天助我也~~，我看到了 TinyLoRA 的论文：

- 基座：Qwen2.5-Coder-3B-Instruct，4bit 量化
- 训练：不用 SFT，用 RL（GRPO）
- 参数：全模型只保留极少可训练标量参数
- 任务：编译+运行 C++ 代码的强化学习

在 [Qwen4Luogu-RL](https://github.com/Chi-Shan0707/Qwen4Luogu-RL) 中能成功通过样例测试的十不存一（并没有夸张），于是换到了 [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests) 数据集 —— 题量大、英语环境、难度可控、测试用例超丰富。

</details>

### 📈 实验结果

**v4.0: `num_iterations=4` + DeepCoder 数据集 (lcbv5)**

严格控制变量（相同测试条件）下：

- 测试种子：`42`
- 测试数据集：`code_contests_test.jsonl`
- 样本数量：165
- 关键改动：`num_iterations` 从 1 提升到 4（使 clip_high / DeepCoder 方法更有效）

训练对比：

| 配置项 | 旧版 (v3.x) | 新版 (v4.0) |
| :--- | :--- | :--- |
| 训练数据集 | code_contests | lcbv5 (DeepCoder-Preview-Dataset) |
| `num_iterations` | 1 | 4 |
| 训练样本数 | 13,328 | 28 |
| 训练时间 | ~4小时24分 | ~1小时12分 |

测试结果：

| 指标 | 旧训练 | 新训练 (v4.0) | 提升 |
| :--- | :---: | :---: | :---: |
| **总样本数** | 165 | 165 | — |
| **Pass@1** | 1.82% (3/165) | 3.64% (6/165) | **+100%** |
| **编译成功率** | 73.33% (121/165) | 76.36% (126/165) | **+4.13%** |
| **平均分数** | 0.4274 | 0.4489 | **+5.03%** |

v4.0 表明：
- 使用更高质量的训练数据 (lcbv5) 显著提升模型性能
- 增加 `num_iterations` 使 clip_high (DeepCoder 方法) 更有效
- 训练数据减少 **99.8%** (13328 → 28)
- 训练时间减少 **73%** (4小时24分 → 1小时12分)

[详细对比](./docs/comparison_zh.md)

---

**早期结果 (v3.x 基线)：**

严格控制变量（相同测试种子与样本顺序）下：

- 测试种子：`42`
- 测试样本：`code_contests_test.jsonl` 同一批 10 条
- 训练命令：

```bash
python train_rl.py 32 20 --do_validate --val_steps 10 --val_samples 10
```

训练配置快照：

| 配置项 | 值 |
| :--- | :--- |
| 可训练参数维度 `u` | 32 |
| TinyLoRA rank | 2 |
| 训练样本数 | 20 |
| Checkpoint seed | 212 |
| `global_v` shape | `torch.Size([32])` |

测试对比：

| 指标 | Baseline（基座模型） | TinyLoRA 微调后（`u=32`） | 变化 |
| :--- | :---: | :---: | :---: |
| **总样本数** | 10 | 10 | — |
| **平均分数** | 0.4500 | 0.4000 | -0.05 |
| **编译成功率** | 80.00% (8/10) | 80.00% (8/10) | 持平 |
| **Pass@1** | 10.00% (1/10) | 0.00% (0/10) | -10% |
| **部分通过** | **7/10** | **8/10** | **+1** |
| **未提取到代码** | 0/10 | 0/10 | 持平 |

解读：

- 极小参数 RL 已在严格对照下改变模型行为；
- 小样本阶段通常先出现”部分通过增加”，再向 Pass@1 收敛。

### 📜 许可证

- 仓库脚本：CC BY 4.0。
- 同时请遵守上游模型与数据集许可证。

### 📚 引用（BibTeX）

```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```

```bibtex
@misc{deepcoder2025,
  title={DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level},
  author={Michael Luo and Sijun Tan and Roy Huang and Ameen Patel and Alpay Ariyak and Qingyang Wu and Xiaoxiang Shi and Rachel Xin and Colin Cai and Maurice Weber and Ce Zhang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51}},
  note={Notion Blog},
  year={2025}
}
```

```bibtex
@article{li2022competition,
  title={Competition-Level Code Generation with AlphaCode},
  author={Li, Yujia and Choi, David and Chung, Junyoung and Kushman, Nate and
    Schrittwieser, Julian and Leblond, R{\'e}mi and Eccles, Tom and
    Keeling, James and Gimeno, Felix and Dal Lago, Agustin and
    Hubert, Thomas and Choy, Peter and de Masson d'Autume, Cyprien and
    Babuschkin, Igor and Chen, Xinyun and Huang, Po-Sen and Welbl, Johannes and
    Gowal, Sven and Cherepanov, Alexey and Molloy, James and
    Mankowitz, Daniel and Sutherland Robson, Esme and Kohli, Pushmeet and
    de Freitas, Nando and Kavukcuoglu, Koray and Vinyals, Oriol},
  journal={arXiv preprint arXiv:2203.07814},
  year={2022}
}
```
