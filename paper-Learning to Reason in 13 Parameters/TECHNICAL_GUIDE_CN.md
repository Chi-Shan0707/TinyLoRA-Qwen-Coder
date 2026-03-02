# TinyLoRA 技术文档与复现指南

**语言切换**： [English](./TECHNICAL_GUIDE.md) | 中文

## Technical Documentation and Implementation Guide

本文档提供 TinyLoRA 的完整技术解析，包括数学推导、架构设计、GRPO 强化学习流程，以及基于 TRL + vLLM 的工程实现指南。

---

## 目录

1. [范式转移：从 SFT 到极低秩 RL](#1-范式转移从-sft-到极低秩-rl)
2. [TinyLoRA 架构详解](#2-tinylora-架构详解)
3. [GRPO 强化学习流程](#3-grpo-强化学习流程)
4. [混合引擎架构与数值不匹配问题](#4-混合引擎架构与数值不匹配问题)
5. [截断重要性采样修正](#5-截断重要性采样修正)
6. [完整代码实现](#6-完整代码实现)
7. [实验配置与超参数](#7-实验配置与超参数)
8. [常见问题与调试指南](#8-常见问题与调试指南)

---

## 1. 范式转移：从 SFT 到极低秩 RL

### 1.1 SFT 与 RL 的内在维度差异

在传统的大语言模型微调实践中，监督微调（Supervised Fine-Tuning, SFT）长期占据主导地位。SFT 通过最小化专家演示数据的负对数似然（Negative Log-Likelihood, NLL）来调整模型权重，其损失函数定义为：

$$\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y) \sim (X,Y)} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t}) \right]$$

然而，SFT 本质上是一个高维度的模仿学习过程。模型不仅需要学习解决问题的核心逻辑路径（Reasoning Path），还被迫记忆演示者的行文风格、词汇选择、标点习惯乃至格式噪声。这种"全模态模仿"导致了 SFT 对参数容量的巨大需求——即便是参数高效的 LoRA，通常也需要秩（Rank）为 8 或 16，涉及数百万参数的更新，才能在不破坏预训练知识的前提下有效拟合下游任务。

相比之下，强化学习（Reinforcement Learning, RL），特别是基于可验证奖励（Verifiable Rewards）的 RL（如 RLVR），对参数容量的需求呈现出截然不同的特征。RL 的目标函数是最大化期望奖励：

$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim X; y \sim \pi_\theta} \left[ \sum_{t=1}^{|y|} \nabla_\theta \log \pi_\theta(y_t | x, y_{<t}) \cdot R(y) \right]$$

在数学推理或代码生成等任务中，只要最终答案正确，模型可以自由探索任何有效的推理路径。这种稀疏但精准的信号极大地降低了任务的"内在维度"（Intrinsic Dimension）。

### 1.2 论文核心发现

TinyLoRA 论文的实验数据揭示了一个惊人的现象：

| 配置 | 可训练参数 | GSM8K 准确率 | 绝对提升 |
|:-----|:----------:|:------------:|:--------:|
| Qwen2.5-7B-Instruct (Base) | 0 | 76.0% | - |
| TinyLoRA + GRPO | **13** | **91.8%** | +15.8% |
| TinyLoRA + SFT | 13 | 83% | +7% |

这一发现彻底颠覆了"参数量即能力"的传统认知，证明了在特定任务中，RL 所需的更新量比 SFT 低 **3 个数量级**（100-1000 倍）。

### 1.3 为什么 SFT 在极低秩下失效？

当参数空间被压缩至极限（如 $10^1$ 量级）时，SFT 的表现会发生灾难性崩溃，原因如下：

**信息密度过载**：SFT 的梯度更新包含了大量与推理能力无关的"噪声信息"（如句式结构、格式细节）。当参数空间被压缩至几十个维度时，模型无法同时编码这些语法细节与逻辑模式，导致严重的欠拟合。

**损失函数的几何形态**：SFT 的损失面（Loss Surface）在参数空间中是崎岖且狭窄的，要求模型精确命中特定的 token 序列。而 RL 的损失面，特别是在 GRPO 的平滑作用下，呈现出更宽阔的"盆地"。TinyLoRA 仅有的几个自由度足以在这个平滑的势能面上找到通往高奖励区域的梯度下降路径。

---

## 2. TinyLoRA 架构详解

### 2.1 从 LoRA 到 LoRA-XS

标准 LoRA 假设权重更新 $\Delta W$ 具有低秩结构：

$$W' = W + BA$$

其中 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，$B \in \mathbb{R}^{d_{out} \times r}$，$A \in \mathbb{R}^{r \times d_{in}}$。即使在 $r=1$ 的极限情况下，对于一个隐藏层维度 $d=4096$ 的模型，单层的参数量仍为：

$$|B| + |A| = d_{out} \times 1 + 1 \times d_{in} = 8192$$

对于拥有 32 层、每层包含 7 个投影矩阵（q, k, v, o, gate, up, down）的模型，总参数量依然在百万级别。

**LoRA-XS** 引入了基于奇异值分解（SVD）的重参数化方法来突破这一限制：

$$W \approx U_r \Sigma_r V_r^\top$$

其中 $U_r \in \mathbb{R}^{d_{out} \times r}$，$\Sigma_r \in \mathbb{R}^{r \times r}$，$V_r \in \mathbb{R}^{d_{in} \times r}$。这三个矩阵在初始化后被永久冻结，更新公式变为：

$$W' = W + U_r \Sigma_r R V_r^\top$$

此时，唯一可训练的参数是中间的小矩阵 $R \in \mathbb{R}^{r \times r}$，参数量变为 $r^2$，与模型维度 $d$ 完全解耦。

### 2.2 TinyLoRA 的核心创新

TinyLoRA 在 LoRA-XS 的基础上，通过**随机投影**和**极致权重绑定**进一步压缩参数：

$$R = \sum_{i=1}^{u} v_i P_i$$

代入完整公式：

$$W' = W + U_r \Sigma_r \left( \sum_{i=1}^{u} v_i P_i \right) V_r^\top$$

**关键组件详解**：

| 组件 | 维度 | 说明 |
|:-----|:-----|:-----|
| $U_r, \Sigma_r, V_r$ | 来自 $W$ 的 SVD | 冻结的"骨架"，捕捉预训练权重的主成分子空间 |
| $P_i \in \mathbb{R}^{r \times r}$ | $u \times r \times r$ | 固定随机投影矩阵，通常从 $\mathcal{N}(0, 1/r)$ 采样 |
| $\mathbf{v} \in \mathbb{R}^u$ | $u$ | **唯一可训练参数**，学习如何重组随机基底 |

### 2.3 权重绑定策略

为了达到"13 个参数"的极致，TinyLoRA 实施了跨层、跨模块的参数共享：

**绑定因子 $n_{tie}$**：定义为共享同一个向量 $\mathbf{v}$ 的模块数量。

**总参数量计算**：

$$\text{Total Params} = \frac{n \times m \times u}{n_{tie}}$$

其中 $n$ 为层数，$m$ 为每层模块数。当 $n_{tie} = n \times m$ 时，所有模块共享同一个 $\mathbf{v}$，总参数量降至 $u$。

**共享策略对比**：

| 策略 | 描述 | 实验效果 |
|:-----|:-----|:---------|
| **Tiled Sharing** | 物理位置相邻的模块共享参数 | ✅ 更优 |
| Structured Sharing | 相同类型的模块（如所有 Q 投影）共享参数 | 较差 |
| Global Sharing | 所有模块共享同一个 $\mathbf{v}$ | 可行 |

论文实验表明，**Tiled Sharing 效果最优**，这表明参数共享不必受限于模块的功能语义，空间上的局部性在极低参数下更为鲁棒。

### 2.4 冻结秩 $r$ 的选择

论文消融实验显示，保留更多的奇异值方向并不总是更好：

| 冻结秩 $r$ | GSM8K 准确率 |
|:----------:|:------------:|
| 1 | 基准 |
| 2 | ✅ 最优 |
| 4 | 略差 |
| 8 | 更差 |

**原因分析**：过高的 $r$ 会引入更多自由度，使得优化小向量 $\mathbf{v}$ 变得困难。论文推荐 $r=2$ 作为默认配置。

---

## 3. GRPO 强化学习流程

### 3.1 GRPO 核心机制

Group Relative Policy Optimization (GRPO) 是 DeepSeekMath 提出的强化学习算法，其核心创新是**移除 Critic 模型**，通过群组相对优势来估计价值函数。

**传统 PPO 的问题**：

- 需要额外的 Value Model（Critic），显存需求翻倍
- Critic 训练往往比 Policy 更难收敛

**GRPO 的解决方案**：

对于每个输入 Prompt $q$，模型采样生成一组输出 $\{o_1, o_2, \dots, o_G\}$（通常 $G=64$），利用群组统计量计算优势：

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

这种方法利用同批次采样的其他样本作为对照，天然降低了方差，且无需额外的参数来拟合价值函数。

### 3.2 GRPO 损失函数

GRPO 的损失函数结合了策略优化和 KL 正则化：

$$\mathcal{L}_{GRPO} = -\mathbb{E} \left[ \sum_{t=1}^{|y|} \min\left( \rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref}) \right]$$

其中 $\rho_t = \frac{\pi_\theta(y_t | \cdot)}{\pi_{old}(y_t | \cdot)}$ 为重要性采样比率。

**TinyLoRA 特殊配置**：

由于 TinyLoRA 仅有 13 个参数，其表达能力本身就构成了极强的正则化，模型很难发生灾难性的"奖励欺骗"（Reward Hacking），因此：

- **KL Penalty ($\beta$)**：通常设为 0 或极小值（如 0.001）
- **Clip Range ($\epsilon$)**：可以使用标准值 0.2

### 3.3 奖励函数设计

在 GSM8K 和 MATH 任务中，奖励函数是稀疏但确定的（Verifiable）：

```python
def accuracy_reward(completions: list[str], solutions: list[str]) -> list[float]:
    """
    精确匹配奖励函数
    
    Args:
        completions: 模型生成的回答列表
        solutions: 标准答案列表
    
    Returns:
        奖励值列表（1.0 表示正确，0.0 表示错误）
    """
    rewards = []
    for completion, solution in zip(completions, solutions):
        # 提取 \boxed{...} 中的答案
        pred_answer = extract_boxed_answer(completion)
        true_answer = extract_boxed_answer(solution)
        
        # 数值比较（支持浮点数容差）
        if numerical_equal(pred_answer, true_answer, tolerance=1e-3):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards
```

**格式奖励（可选）**：为了引导模型进入推理模式，可以给予微小奖励（如 0.1）如果模型输出了 `<think >...</think >` 标签。

---

## 4. 混合引擎架构与数值不匹配问题

### 4.1 混合引擎架构

在工程实现层面，GRPO 训练涉及两个阶段：

| 阶段 | 目的 | 推荐引擎 |
|:-----|:-----|:---------|
| **Rollout** | 生成训练数据 | vLLM（高吞吐量） |
| **Training** | 梯度更新 | PyTorch FSDP（自动微分） |

**vLLM 的优势**：
- PagedAttention 优化 KV Cache
- 连续批处理（Continuous Batching）
- 高度优化的 CUDA 核函数

### 4.2 内核不兼容问题

vLLM 对 LoRA 的支持有限：
- 只支持标准的 $W + BA$ 形式
- 对 Rank 有最小限制（通常 $r \ge 4$）
- TinyLoRA 的复杂结构 $U \Sigma (\sum v P) V^\top$ 无现成支持

**解决方案：Merge-and-Unmerge Trick**

```python
def merge_tiny_lora_weights(model):
    """将 TinyLoRA 增量合并到基座权重中"""
    for name, module in model.named_modules():
        if isinstance(module, TinyLoRALayer):
            # 计算 delta_W = U @ S @ R @ V.T
            R = torch.einsum('i,irr->rr', module.v, module.P)
            delta_W = module.U @ module.S @ R @ module.V.T
            
            # 合并到基座权重
            module.base_layer.weight.data += delta_W

def unmerge_tiny_lora_weights(model):
    """从基座权重中移除 TinyLoRA 增量"""
    for name, module in model.named_modules():
        if isinstance(module, TinyLoRALayer):
            R = torch.einsum('i,irr->rr', module.v, module.P)
            delta_W = module.U @ module.S @ R @ module.V.T
            
            # 从基座权重中减去
            module.base_layer.weight.data -= delta_W
```

### 4.3 数值不匹配问题

混合架构引入了一个隐蔽但致命的问题：**Off-Policy Drift**

**精度差异**：
- vLLM 为了加速，常使用 float16 或 KV Cache Int8 量化
- PyTorch 训练通常使用 bfloat16 或 float32 累加

**算子差异**：
- FlashAttention 在 vLLM 中的实现与 PyTorch 中的 SDPA 在数值上并非逐位一致

**后果**：

$$\pi_{vllm}(a|s) \neq \pi_{train}(a|s)$$

虽然两者理论上是同一个模型，但数值误差导致它们对同一输入的概率分布存在微小偏差。在 RL 训练中，这种偏差会被累积放大，导致 KL 散度异常升高，训练曲线震荡甚至崩塌。

---

## 5. 截断重要性采样修正

### 5.1 数学原理

当样本实际来自 $\pi_{vllm}$ 而非 $\pi_\theta$ 时，需要引入重要性采样权重 $\rho$ 来校正梯度估计：

$$\nabla J(\theta) = \mathbb{E}_{s,a \sim \pi_{vllm}} \left[ \rho_t \cdot A_t \cdot \nabla \log \pi_\theta(a_t|s_t) \right]$$

其中 $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{vllm}(a_t|s_t)}$。

由于数值不稳定，$\rho_t$ 可能出现极端值，导致方差过大。**截断重要性采样（TIS）** 通过引入截断机制来平衡偏差与方差：

$$\hat{\rho}_t = \min(\rho_t, \epsilon_{cap})$$

### 5.2 TIS 实现步骤

```python
def compute_tis_weights(
    log_probs_train: torch.Tensor,  # [batch, seq_len]
    log_probs_vllm: torch.Tensor,   # [batch, seq_len]
    cap: float = 2.0
) -> torch.Tensor:
    """
    计算截断重要性采样权重
    
    Args:
        log_probs_train: PyTorch 计算的对数概率
        log_probs_vllm: vLLM 记录的对数概率
        cap: 截断阈值
    
    Returns:
        截断后的重要性采样权重
    """
    # 计算对数比率
    log_ratio = log_probs_train - log_probs_vllm
    
    # 转换为比率
    rho = torch.exp(log_ratio)
    
    # 截断
    rho_capped = torch.clamp(rho, max=cap)
    
    return rho_capped
```

### 5.3 推荐配置

| 参数 | 推荐值 | 说明 |
|:-----|:------:|:-----|
| `vllm_importance_sampling_correction` | `True` | 开启 TIS 修正 |
| `vllm_importance_sampling_mode` | `"token_truncate"` | 超出阈值时截断 |
| `vllm_importance_sampling_cap` | `2.0` | 允许的最大比率 |

---

## 6. 完整代码实现

### 6.1 TinyLoRA 层定义

```python
import torch
import torch.nn as nn
import math
from typing import Optional

class TinyLoRALayer(nn.Module):
    """
    TinyLoRA 层实现
    
    更新公式: W' = W + U @ S @ (sum_i v_i * P_i) @ V.T
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank_frozen: int = 2,
        u_dim: int = 13,
        tying_vector: Optional[nn.Parameter] = None,
        svd_dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            base_layer: 原始线性层
            rank_frozen: SVD 截断秩 (推荐 2)
            u_dim: 可训练向量维度
            tying_vector: 共享的可训练向量（用于权重绑定）
            svd_dtype: SVD 计算精度
        """
        super().__init__()
        self.base_layer = base_layer
        self.rank_frozen = rank_frozen
        self.u_dim = u_dim
        
        # 离线 SVD 分解
        with torch.no_grad():
            W = base_layer.weight.data.to(svd_dtype)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            
            # 截断
            U_r = U[:, :rank_frozen]
            S_r = S[:rank_frozen]
            V_r = Vh[:rank_frozen, :].T
            
            # 注册为 buffer（冻结）
            self.register_buffer("U", U_r)
            self.register_buffer("S", torch.diag(S_r))
            self.register_buffer("V", V_r)
        
        # 固定随机基底 P
        P = torch.randn(u_dim, rank_frozen, rank_frozen) / math.sqrt(rank_frozen)
        self.register_buffer("P", P)
        
        # 可训练向量 v
        if tying_vector is not None:
            self.v = tying_vector
        else:
            self.v = nn.Parameter(torch.zeros(u_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        利用结合律优化计算: x @ V @ R.T @ S @ U.T
        避免构建大矩阵
        """
        # 基座输出
        base_out = self.base_layer(x)
        
        # 计算 R = sum(v_i * P_i)
        R = torch.einsum('i,ijk->jk', self.v, self.P)
        
        # 计算 delta 输出（利用结合律）
        # x: [batch, in_dim]
        # V: [in_dim, r]
        # R: [r, r]
        # S: [r, r]
        # U: [out_dim, r]
        h = x @ self.V          # [batch, r]
        h = h @ R.T             # [batch, r]
        h = h @ self.S          # [batch, r]
        delta_out = h @ self.U.T  # [batch, out_dim]
        
        return base_out + delta_out
    
    def get_delta_weight(self) -> torch.Tensor:
        """获取完整的增量权重矩阵"""
        R = torch.einsum('i,ijk->jk', self.v, self.P)
        return self.U @ self.S @ R @ self.V.T


def apply_tiny_lora_to_model(
    model: nn.Module,
    rank_frozen: int = 2,
    u_dim: int = 13,
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
    global_tying: bool = True
) -> nn.Module:
    """
    将 TinyLoRA 应用到模型的所有目标模块
    
    Args:
        model: 原始模型
        rank_frozen: SVD 截断秩
        u_dim: 可训练向量维度
        target_modules: 目标模块名称列表
        global_tying: 是否全局共享 v
    
    Returns:
        包装后的模型
    """
    # 创建共享向量（如果启用全局绑定）
    shared_v = nn.Parameter(torch.zeros(u_dim)) if global_tying else None
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 获取父模块和属性名
                *path, attr = name.split('.')
                parent = model
                for p in path:
                    parent = getattr(parent, p)
                
                # 替换为 TinyLoRALayer
                tiny_lora = TinyLoRALayer(
                    base_layer=module,
                    rank_frozen=rank_frozen,
                    u_dim=u_dim,
                    tying_vector=shared_v
                )
                setattr(parent, attr, tiny_lora)
    
    return model
```

### 6.2 GRPO 训练配置

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 应用 TinyLoRA
model = apply_tiny_lora_to_model(
    model,
    rank_frozen=2,
    u_dim=13,
    global_tying=True
)

# GRPO 配置
training_args = GRPOConfig(
    output_dir="./tiny_lora_output",
    
    # 学习率（TinyLoRA 需要较大学习率）
    learning_rate=1e-4,
    
    # 批处理配置
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    
    # GRPO 特定配置
    num_generations=64,           # Group Size G
    max_completion_length=4096,   # 最大生成长度
    beta=0.001,                   # KL 惩罚系数（TinyLoRA 可设很小）
    
    # vLLM 混合引擎配置
    use_vllm=True,
    vllm_device="cuda:0",
    vllm_gpu_memory_utilization=0.5,
    vllm_importance_sampling_correction=True,
    vllm_importance_sampling_mode="token_truncate",
    vllm_importance_sampling_cap=2.0,
    
    # 日志
    logging_steps=10,
    report_to="wandb"
)

# 定义奖励函数
def accuracy_reward(completions, solutions, **kwargs):
    rewards = []
    for comp, sol in zip(completions, solutions):
        pred = extract_boxed_answer(comp)
        true = extract_boxed_answer(sol)
        rewards.append(1.0 if numerical_equal(pred, true) else 0.0)
    return rewards

# 创建 Trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[accuracy_reward]
)

# 开始训练
trainer.train()
```

---

## 7. 实验配置与超参数

### 7.1 GSM8K 实验配置

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| **模型** | Qwen2.5-7B-Instruct | 论文主实验模型 |
| **数据集** | GSM8K (7,500 训练样本) | 数学应用题 |
| **Epochs** | 3 | - |
| **Batch Size** | 64 | - |
| **Samples per Problem** | 4 | 每个 prompt 生成 4 个样本 |
| **Max Generation Length** | 4096 | - |
| **Learning Rate** | $10^{-4}$ | 扫描范围 $[10^{-7}, 2 \times 10^{-4}]$ |
| **KL Penalty** | 0 | TinyLoRA 不需要强 KL 惩罚 |

### 7.2 MATH 实验配置（SimpleRL 设置）

| 参数 | 值 |
|:-----|:---|
| **训练数据** | SimpleRL 困难子集 (8,523 问题) |
| **Max Prompt Length** | 1024 |
| **Max Response Length** | 3072 |
| **KL Coefficient** | 0.001 |
| **Temperature** | 1.0 |
| **Batch Size** | 256 |
| **Generations per Response** | 8 |

### 7.3 学习率扫描建议

由于参数量极少，TinyLoRA 通常需要比全量微调更大的学习率：

```python
learning_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 2e-4]

# 推荐起点
recommended_lr = 1e-4  # 对于 u=13
```

---

## 8. 常见问题与调试指南

### 8.1 训练不收敛

**症状**：Reward 曲线震荡或持续下降

**可能原因与解决方案**：

| 原因 | 解决方案 |
|:-----|:---------|
| 学习率过大 | 降低至 $10^{-5}$ 或 $5 \times 10^{-6}$ |
| TIS 截断阈值过小 | 增大 `vllm_importance_sampling_cap` 至 3.0 |
| vLLM 显存不足 | 降低 `vllm_gpu_memory_utilization` |
| 数值精度问题 | 尝试使用 fp32 进行 SVD 计算 |

### 8.2 KL 散度异常升高

**症状**：KL divergence 持续增长

**诊断**：检查 TIS 权重分布

```python
# 监控 TIS 权重
if trainer.state.global_step % 100 == 0:
    mean_rho = torch.mean(tis_weights).item()
    std_rho = torch.std(tis_weights).item()
    print(f"TIS weights: mean={mean_rho:.3f}, std={std_rho:.3f}")
    
    # 如果 mean 显著偏离 1.0，说明存在数值不匹配
    if abs(mean_rho - 1.0) > 0.5:
        print("Warning: Significant numerical mismatch detected!")
```

### 8.3 显存不足

**优化策略**：

1. **降低 vLLM 显存占用**：
   ```python
   vllm_gpu_memory_utilization=0.3  # 从 0.5 降低
   ```

2. **使用量化模型**：
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2.5-7B-Instruct",
       load_in_4bit=True,
       device_map="auto"
   )
   ```

3. **减少 Group Size**：
   ```python
   num_generations=32  # 从 64 降低
   ```

### 8.4 性能监控指标

训练过程中应监控以下关键指标：

| 指标 | 健康范围 | 异常信号 |
|:-----|:---------|:---------|
| **Mean Reward** | 持续上升 | 震荡或下降 |
| **KL Divergence** | < 0.1 | > 0.5 |
| **TIS Mean** | 0.8 - 1.2 | > 1.5 或 < 0.5 |
| **Response Length** | 逐渐增长 | 突然缩短或爆炸 |

---

## 参考文献

1. Morris, J. X., et al. (2026). Learning to Reason in 13 Parameters. arXiv:2602.04118.
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
3. Bałazy, K., et al. (2025). LoRA-XS: Low-rank adaptation with extremely small number of parameters. arXiv:2405.17604.
4. Shao, Z., et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning. arXiv:2402.03300.
5. Ionides, E. L. (2008). Truncated importance sampling. Journal of Computational and Graphical Statistics.
6. Schulman, J., & Lab, T. M. (2025). LoRA Without Regret. Thinking Machines Lab.

---

<div align="center">

**[⬅ 返回 README](./README.md)**

</div>
