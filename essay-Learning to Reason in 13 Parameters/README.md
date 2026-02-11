

# Learning to Reason in 13 Parameters

## TinyLoRA: Minimal Parameter Reasoning

[![Paper](https://img.shields.io/badge/Paper-2602.04118-brightgreen)](https://arxiv.org/abs/2602.04118)
[![GitHub](https://img.shields.io/badge/GitHub-TinyLoRA-blue)](https://github.com/your-repo)

TinyLoRA is a revolutionary parameter-efficient fine-tuning method that enables language models to learn reasoning capabilities with as few as **13 trainable parameters** (26 bytes in bf16).

## Key Findings

✅ **13 Parameters, 91% Accuracy**: Achieved 91% accuracy on GSM8K with only 13 trained parameters (Qwen2.5-7B-Instruct).

✅ **1000x Smaller Updates**: Recovered 90% of performance improvements while training **1000x fewer parameters** across challenging math reasoning benchmarks (MATH, AIME, AMC).

✅ **RL Beats SFT**: Reinforcement Learning (RL) enables much more efficient parameter usage than Supervised Fine-Tuning (SFT). SFT requires 100-1000x more parameters to achieve similar performance.

✅ **Model Size Matters**: Larger models (like Qwen) are more parameter-efficient than smaller models (like LLaMA) at tiny update sizes.

## How It Works

TinyLoRA is an ultra-low-rank variant of LoRA that scales down to a single trainable parameter by:
1. Using truncated SVD to decompose weight matrices
2. Replacing the low-rank matrix with a single trainable vector
3. Sharing this vector across multiple model modules

## Performance Comparison

| Method | Parameters | GSM8K Accuracy | 
|--------|------------|----------------|
| Full Fine-Tuning | 7B+ | 95% |
| LoRA (r=1) | ~3M | 94% |
| LoRA-XS (r=1) | ~100K | 93% |
| **TinyLoRA** | **13** | **91%** |
| SFT (13 parameters) | 13 | 83% |

## Why It Matters

- **Extreme Efficiency**: 13 parameters is 1000x smaller than typical LoRA (10K-1M parameters)
- **RL Advantage**: RL enables much more information-dense updates than SFT
- **Scalability**: Enables efficient personalization at scale (10x more LoRAs can be stored in memory)
- **Future Potential**: Suggests trillion-scale models could be trained for specific tasks with just a few parameters

## Usage

```python
# Example usage of TinyLoRA
model = TinyLoRA(model, rank=1, u=1, ntie=560)  # 560 layers × 7 modules = 1 parameter total
model.train()
```

## Paper

[Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118)  
John X. Morris, Niloofar Mireshghallah, Mark Ibrahim, Saeed Mahloujifar  
FAIR at Meta, Cornell University, Carnegie Mellon University

---

# 用13个参数学习推理

## TinyLoRA：极小参数推理

[![论文](https://img.shields.io/badge/论文-2602.04118-brightgreen)](https://arxiv.org/abs/2602.04118)
[![GitHub](https://img.shields.io/badge/GitHub-TinyLoRA-blue)](https://github.com/your-repo)

TinyLoRA是一种革命性的参数高效微调方法，使语言模型能够仅用**13个训练参数**（26字节bf16）学习推理能力。

## 关键发现

✅ **13个参数，91%准确率**：使用Qwen2.5-7B-Instruct在GSM8K数据集上仅用13个训练参数达到91%准确率。

✅ **1000倍更小的更新**：在更难的数学推理基准测试（MATH、AIME、AMC）中，仅需1000倍更少的参数就能达到90%的性能提升。

✅ **RL优于SFT**：强化学习（RL）比监督微调（SFT）在小参数规模下更高效。SFT需要100-1000倍更多的参数才能达到相同性能。

✅ **模型规模影响**：大型模型（如Qwen）在极小更新规模下比小型模型（如LLaMA）更高效。

## 工作原理

TinyLoRA是LoRA的超低秩变体，通过以下方式缩小到单个可训练参数：
1. 使用截断SVD分解权重矩阵
2. 用单个可训练向量替换低秩矩阵
3. 在多个模型模块间共享该向量

## 性能对比

| 方法 | 参数量 | GSM8K准确率 |
|------|--------|------------|
| 完整微调 | 7B+ | 95% |
| LoRA (r=1) | ~3M | 94% |
| LoRA-XS (r=1) | ~100K | 93% |
| **TinyLoRA** | **13** | **91%** |
| SFT (13参数) | 13 | 83% |

## 重要性

- **极高效**：13个参数比典型LoRA（10K-1M参数）小1000倍
- **RL优势**：RL比SFT提供更密集的信息更新
- **可扩展性**：支持高效个性化（内存中可存储10倍更多LoRA）
- **未来潜力**：表明万亿级模型可能只需几个参数就能训练特定任务

## 使用方法

```python
# TinyLoRA使用示例
model = TinyLoRA(model, rank=1, u=1, ntie=560)  # 560层 × 7个模块 = 总共1个参数
model.train()
```

## 论文

[Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118)  
John X. Morris, Niloofar Mireshghallah, Mark Ibrahim, Saeed Mahloujifar  
FAIR at Meta, Cornell University, Carnegie Mellon University