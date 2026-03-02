<div align="center">

# 📄 TinyLoRA Paper Hub

**Learning to Reason in 13 Parameters — Notes, adaptation map, and technical guide**

[![Paper](https://img.shields.io/badge/arXiv-2602.04118-b31b1b)](https://arxiv.org/abs/2602.04118)
[![PDF](https://img.shields.io/badge/PDF-Local_Copy-blue)](./2602.04118v1.pdf)

[⬅ Back to Main Project](../README.md)

</div>

**Language / 语言**: [English](#english) | [中文](#中文)

---

## English

### Why this folder exists

This folder bridges theory and implementation for TinyLoRA:
- what the original paper proves,
- what this repository adapts for code generation,
- and how to reproduce/debug the method in practice.

### Fast Links

- Full technical guide (English, default): [TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md)
- 技术文档（中文）: [TECHNICAL_GUIDE_CN.md](./TECHNICAL_GUIDE_CN.md)
- Original paper PDF (local): [2602.04118v1.pdf](./2602.04118v1.pdf)
- Main repository: [../README.md](../README.md)

### Core idea (one equation)

$$W' = W + U\Sigma\left(\sum_{i=1}^{u} v_i P_i\right)V^\top$$

- Freeze SVD skeleton (`U, Σ, V`)
- Keep random projection bases (`P`) fixed
- Train only tiny shared vector (`v`)

### What our repo adapts

| Dimension | Paper | This Repository |
| :--- | :--- | :--- |
| Task | Math reasoning | Competitive code generation |
| Reward | Exact match | `g++` compile + testcase execution |
| Params | 13 (example) | configurable (`u`, default often 32) |
| Precision | BF16/FP32 | 4-bit NF4 or BF16 (`--no_quant`) |

### Suggested reading order

1. Main README: project setup and runnable workflow
2. Technical Guide: mechanism and implementation details
3. Paper PDF: original experimental evidence

---

## 中文

### 这个目录的作用

本目录用于把 TinyLoRA 的**理论、工程实现、复现要点**连接起来：
- 论文到底证明了什么，
- 本仓库在代码任务上做了哪些适配，
- 实际训练/调试时该关注哪些关键机制。

### 快速入口

- 技术文档（英文默认）：[TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md)
- 技术文档（中文）：[TECHNICAL_GUIDE_CN.md](./TECHNICAL_GUIDE_CN.md)
- 论文本地 PDF：[2602.04118v1.pdf](./2602.04118v1.pdf)
- 返回主仓库：[../README.md](../README.md)

### 核心公式

$$W' = W + U\Sigma\left(\sum_{i=1}^{u} v_i P_i\right)V^\top$$

- 冻结 SVD 骨架（`U, Σ, V`）
- 固定随机投影基（`P`）
- 只训练极小共享向量（`v`）

### 本仓库适配点

| 维度 | 论文 | 本仓库 |
| :--- | :--- | :--- |
| 任务 | 数学推理 | 竞赛代码生成 |
| 奖励 | 答案匹配 | `g++` 编译 + 测试运行 |
| 参数量 | 13（示例） | 可配置（`u`，常用 32） |
| 精度 | BF16/FP32 | 4-bit NF4 或 BF16（`--no_quant`） |

### 建议阅读顺序

1. 主 README：先跑通全流程
2. 技术文档：理解机制与关键实现
3. 原论文 PDF：查看原始实验结论

---

```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```

<div align="center">

[⬅ Back to Main Project / 返回主项目](../README.md)

</div>
