<div align="center">

# 🧬 TinyLoRA-GRPO-Coder

**Fine-tune a code model with configurable tiny parameters (default: 32) — and it actually works.**

[![Paper](https://img.shields.io/badge/Paper-Learning_to_Reason_in_13_Parameters-blue)](./paper-Learning%20to%20Reason%20in%2013%20Parameters/README.md)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-green)](./LICENSE)
[![Model](https://img.shields.io/badge/Base-Qwen2.5--Coder--3B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
[![Dataset](https://img.shields.io/badge/Data-CodeContests-orange)](https://huggingface.co/datasets/deepmind/code_contests)
[![Dataset](https://img.shields.io/badge/Data-DeepCoder-blue)](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)
![Version](https://img.shields.io/badge/Version-v4.1-red)

<br>

| 🔢 Parameters | 🧠 Base Model | 🎯 Task | ⚡ Method | 💾 VRAM |
| :---: | :---: | :---: | :---: | :---: |
| **u=32 (adjustable)** | **Qwen2.5-Coder-3B(adjustable)** | C++ Code Gen | GRPO (RL) | 16GB+ |

</div>

> **v4.1** — Rename this project from `TinyLoRA-Qwen-Coder` to `TinyLoRA-GRPO-Coder`.

> **v4.0 historical runs** — `num_iterations=4` · DeepCoder-Preview-Dataset (lcbv5, 28 samples) · Pass@1 3/165 → 6/165 · Recorded training time -73%

> We adapt TinyLoRA from math reasoning to competitive programming: inject tiny shared parameters into Qwen2.5-Coder-3B, train with GRPO, and reward real `g++` compile-and-run correctness.
>
> If this project is useful to you, please give it a ⭐ Star.<br>
> 如果你觉得我的项目有意思的话，可否留下一个star呢(✿◠‿◠)?

**语言**: [English](./README.md) | 中文

---

## 中文说明

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
  2. 生成固定随机投影矩阵 $P_i \in \mathbb{R}^{r \times r}$
  3. 计算混合低秩基: $R = \sum_i v_i P_i$
  4. 计算增量: $\Delta W = U\Sigma R V^\top$
  5. 最终权重: $W' = W + \Delta W$

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


### 📈 实验结果

**v4.0: `num_iterations=4` + DeepCoder 数据集 (lcbv5)**

两个微调 checkpoint 的历史结果（报告的测试条件相同）：

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

历史记录：
- 这些历史运行同时改变数据、样本数及 `num_iterations`，新增做对 3 题（+1.82 个百分点）。两行均为微调模型，不能据此证明稳定超过基座，或单独归因于某项改变；旧评测还包含本次修复的问题，历史 adapter 实现也不同于当前共享投影的完整矩阵实现。
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
| **旧指标：零用例通过** | **7/10** | **8/10** | **+1** |
| **未提取到代码** | 0/10 | 0/10 | 持平 |

解读：

- 极小参数 RL 已在严格对照下改变模型行为；
- 旧版“部分通过”实际统计编译成功但零用例通过，其增加不能说明解题正确率提升。

### ⚡ 快速开始（安装 + 配置 + 启动）

1）环境准备：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2）下载并预处理数据：

```bash
# CodeContests 数据集（默认；train + valid + test）
python download_code_contests.py --splits train valid test

# 转换本地 HF 缓存中的 DeepCoder parquet 数据；本脚本不下载缺失文件。
# 请先准备 agentica-org/DeepCoder-Preview-Dataset 的本地缓存。
python download_DeepCoder-Preview-Dataset.py
```

3）可选流水线自检：

```bash
python verify_pipeline.py
```

使用 ModelScope 做小模型 smoke test（适合在下载/运行 3B coder 模型前先检查流程）：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
python verify_pipeline.py --model ./models/Qwen3-0.6B --no_quant --max_tokens 128
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
  --dataset NAME: 选择数据集，'code_contests'（默认）或 'deepcoder'<br>
  --model_id NAME: local model directory 缺失时使用的 ModelScope model ID<br>
  --local_model_dir PATH: 本地模型目录<br>
  --output_dir PATH: checkpoint 和 trainer 文件输出目录<br>

```bash
# 使用 CodeContests 数据集（默认）
python train_rl.py 32 2000
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
python train_rl.py 32 2000 --no_quant
python train_rl.py 32 2000 --rank 4

# 使用 DeepCoder 数据集
python train_rl.py 32 2000 --dataset deepcoder

# 使用 Qwen3-0.6B 的极小 smoke run（检查流程，不代表最终效果）
python train_rl.py 8 2 --model_id Qwen/Qwen3-0.6B --local_model_dir ./models/Qwen3-0.6B --no_quant
```

5）评估：

```bash
python validate.py --num_samples 50
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
python test.py --baseline --num_samples 50
```

### 🧭 可调控参数（按“块”组织）

下面按代码结构给出 5 个可调控块，避免“散点式参数罗列”。

#### 🎯 块 1：奖励系统（`train_rl.py` + `utils.py`）

- **入口**：`code_reward_func` + `compile_and_run`
- **当前机制**：编译失败 `0.0`，零用例通过 `0.5`，部分通过 `0.5 + 0.5 × (k/N)`，全部通过 `1.0`
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
- **实现说明**：当前代码使用完整 TinyLoRA 随机基矩阵形式
  `R = Σ_i v_i P_i`，其中每个 `P_i` 都是 `r × r`，并且所有被包装层复用同一组固定
  projection bank；旧的 diagonal checkpoint 不再兼容。
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
- 更新日志（详细版）： [docs/changelog_zh.md](./docs/changelog_zh.md)
- 已知坑点与注意事项： [docs/warning_zh.md](./docs/warning_zh.md)
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


## Star History

[![Star History Chart](https://api.star-history.com/image?repos=Chi-Shan0707/TinyLoRA-GRPO-Coder&type=date&legend=top-left)](https://www.star-history.com/?repos=Chi-Shan0707%2FTinyLoRA-GRPO-Coder&type=date&legend=top-left)
