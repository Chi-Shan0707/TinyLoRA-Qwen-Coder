

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

> **v4.0** — Increased `num_iterations=4` (making clip_high more effective) · DeepCoder-Preview-Dataset (lcbv5, 28 samples) · Pass@1 +100% · Training time -73%

> We adapt TinyLoRA from math reasoning to competitive programming: inject tiny shared parameters into Qwen2.5-Coder-3B, train with GRPO, and reward real `g++` compile-and-run correctness.
>
> If this project is useful to you, please give it a ⭐ Star.<br>

**Language**: English | [中文](./README_zh.md)

---

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
  2. Generate fixed random projection matrices $P_i \in \mathbb{R}^{r \times r}$
  3. Compute the mixed low-rank basis: $R = \sum_i v_i P_i$
  4. Compute delta: $\Delta W = U\Sigma R V^\top$
  5. Final weight: $W' = W + \Delta W$

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


### ⚡ Quick Start (Install + Configure + Run)

1) Environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Download and preprocess dataset:

```bash
# Option A: CodeContests dataset (default; train + validation + test)
python download_code_contests.py --splits train valid test

# Option B: DeepCoder dataset (from agentica-org/DeepCoder-Preview-Dataset, parquet format; requires a local HuggingFace cache)
python download_DeepCoder-Preview-Dataset.py
```

3) Optional end-to-end sanity check:

```bash
python verify_pipeline.py
```

Small-model smoke test with ModelScope (useful before downloading/running the 3B coder model):

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
python verify_pipeline.py --model ./models/Qwen3-0.6B --no_quant --max_tokens 128
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
  --model_id NAME: ModelScope model ID used if the local model directory is missing<br>
  --local_model_dir PATH: local model directory<br>
  --output_dir PATH: output directory for checkpoints and trainer files<br>

```bash
# Using CodeContests dataset (default)
python train_rl.py 32 2000
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
python train_rl.py 32 2000 --no_quant
python train_rl.py 32 2000 --rank 4

# Using DeepCoder dataset
python train_rl.py 32 2000 --dataset deepcoder

# Tiny smoke run with Qwen3-0.6B (checks wiring, not final quality)
python train_rl.py 8 2 --model_id Qwen/Qwen3-0.6B --local_model_dir ./models/Qwen3-0.6B --no_quant
```

5) Evaluate:

```bash
python validate.py --num_samples 50
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
- **Implementation note**: current code uses the full TinyLoRA random-basis matrix
  form `R = Σ_i v_i P_i`, where each `P_i` is `r × r` and the same fixed
  projection bank is reused across wrapped layers; legacy diagonal checkpoints
  are intentionally not supported.
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
