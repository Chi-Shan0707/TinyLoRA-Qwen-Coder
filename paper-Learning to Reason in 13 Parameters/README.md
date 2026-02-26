<div align="center">

# 📄 Learning to Reason in 13 Parameters

**Paper Notes & Theory Reference / 论文笔记与理论参考**

[![Paper](https://img.shields.io/badge/arXiv-2602.04118-b31b1b)](https://arxiv.org/abs/2602.04118)
[![PDF](https://img.shields.io/badge/PDF-Local_Copy-blue)](./2602.04118v1.pdf)

[⬅ Back to Main Project / 返回主项目](../README.md)

</div>

---

## What is TinyLoRA? / TinyLoRA 是什么？

TinyLoRA is an extreme parameter-efficient fine-tuning method that enables language models to learn reasoning capabilities with as few as **13 trainable parameters** (26 bytes in bf16).

TinyLoRA 是一种极端参数高效微调方法，仅用 **13 个可训练参数**（bf16 下仅 26 字节）就能让语言模型学会推理。

### Core Idea / 核心思想

$$W' = W + U \Sigma \left(\sum_{i=1}^{u} v_i P_i\right) V^\top$$

- $U, \Sigma, V$：来自原权重 SVD 分解的冻结骨架 / Frozen skeleton from SVD of original weights
- $P_i$：固定随机投影矩阵 / Fixed random projection matrices
- $v$：**唯一的可训练参数** / **The only trainable parameters**

---

## Key Findings / 关键发现

| Finding | Details |
| :--- | :--- |
| 🔢 **13 params, 91% accuracy** | GSM8K with Qwen2.5-7B-Instruct — only 13 trained parameters |
| 📉 **1000x compression** | Recovers 90% of full fine-tuning improvement with 1000x fewer params |
| 🎯 **RL >> SFT** | At <100 params, SFT completely fails; only RL (GRPO) works |
| 🧠 **Bigger = Better** | Larger models (Qwen) are more parameter-efficient than smaller ones (LLaMA) |

---

## Why RL, Not SFT? / 为什么用 RL 而不是 SFT？

| | SFT (监督微调) | RL (强化学习) |
| :--- | :--- | :--- |
| **学什么** | 模仿参考答案的格式+内容 | 只关心最终结果的对错 |
| **所需容量** | 高（需记忆格式噪声） | 低（仅编码逻辑信号） |
| **极小参数下** | 完全失效 | ✅ 依然有效 |

> SFT 强迫模型记忆 "Noise"（行文风格、格式），RL 只传递 "Signal"（对/错）。
> 所以在仅有 13 个参数时，SFT 准确率 83%，而 RL 达到 91%。

---

## Performance Comparison / 性能对比

| Method | Parameters | GSM8K Accuracy |
| :--- | :---: | :---: |
| Full Fine-Tuning | 7B+ | 95% |
| LoRA (r=1) | ~3M | 94% |
| LoRA-XS (r=1) | ~100K | 93% |
| **TinyLoRA (RL)** | **13** | **91%** |
| TinyLoRA (SFT) | 13 | 83% |

---

## How We Use It / 我们如何使用

本项目在 TinyLoRA 基础上进行了适配：

| Feature | Original Paper | Our Adaptation |
| :--- | :--- | :--- |
| **Task** | Math (GSM8K, MATH) | **Code Competitions (CodeContests)** |
| **Model** | Qwen2.5-7B / Llama-3 | **Qwen2.5-Coder-3B-Instruct** |
| **Params** | 13 ($u=13$) | **32 ($u=32$)**, adjustable |
| **Precision** | BF16 / FP32 | **4-bit NF4 + Dequant SVD** |
| **Reward** | Exact Match | **g++ Compile + Test Execution** |

---

## Further Reading / 深入阅读

- 📝 [详细理论推导与工程解析 (explain.md)](./explain.md) — 从 SVD 到 Tiling 的完整数学推导，GRPO 流程细节
- 📄 [原论文 PDF](./2602.04118v1.pdf)

---

## Citation / 引用

```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```

---

<div align="center">

[⬅ Back to Main Project / 返回主项目](../README.md)

</div>
