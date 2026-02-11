# 从SFT微调迁移到TinyLoRA复现：架构修改与GRPO强化学习流程详解

## 摘要

在大语言模型（LLM）的后训练（Post-training）范式中，我们正经历一场从高秩监督微调（SFT）向极低秩强化学习（RL）适配的深刻转型。本文提供了一份详尽的研究报告，旨在深入剖析TinyLoRA——一种能够通过仅训练13个参数便在7B模型上实现复杂推理能力激增的新型参数高效微调（PEFT）方法。本报告不仅解构了TinyLoRA如何利用奇异值分解（SVD）初始化与子空间复用技术突破传统LoRA的秩限制，更详细阐述了与之配套的Group Relative Policy Optimization (GRPO) 强化学习流程。我们将重点探讨从SFT向RL迁移过程中的核心工程挑战，特别是**混合引擎架构（Hybrid Engine Architecture）带来的训练-推理数值不匹配问题，以及如何通过截断重要性采样（Truncated Importance Sampling, TIS）**来修正这一偏差。通过对架构修改、数学原理及工程实现的全面拆解，本文旨在为高阶研究人员与工程师提供一份从理论到复现的权威指南。

## 1. 范式转移：从监督微调到极低秩强化学习

### 1.1 SFT与RL的内在维度差异

在传统的LLM微调实践中，监督微调（SFT）长期占据主导地位。SFT通过最小化专家演示数据的负对数似然（Negative Log-Likelihood, NLL）来调整模型权重。然而，SFT本质上是一个高维度的模仿游戏。模型不仅需要学习解决问题的核心逻辑路径（Reasoning Path），还被迫记忆演示者的行文风格、词汇选择、标点习惯乃至格式噪声。这种“全模态模仿”导致了SFT对参数容量的巨大需求。即便是参数高效的LoRA（Low-Rank Adaptation），通常也需要秩（Rank）为8或16，涉及数百万甚至上千万个参数的更新，才能在不破坏预训练知识的前提下有效拟合下游任务 。相比之下，强化学习（RL），特别是基于可验证奖励（Verifiable Rewards）的RL（如RLVR），对参数容量的需求呈现出截然不同的特征。RL的目标函数是最大化期望奖励，而非逐字逐句的概率拟合。在数学推理或代码生成等任务中，只要最终答案正确，模型可以自由探索任何有效的推理路径。这种稀疏但精准的信号极大地降低了任务的“内在维度”（Intrinsic Dimension）。TinyLoRA的研究表明，对于Qwen2.5-7B这类经过指令微调的模型，其内部早已蕴含了强大的推理能力，仅需极微小的扰动（Perturbation）即可将其“激活”或“引导”至特定任务模式。实验数据揭示了一个惊人的现象：在GRPO强化学习框架下，仅更新13个参数（26字节）即可将GSM8K的准确率从76%提升至91% 。这一发现彻底颠覆了“参数量即能力”的传统认知，证明了在特定任务中，RL所需的更新量比SFT低3个数量级（100-1000倍）。

### 1.2 为什么SFT在极低秩下失效？

当我们试图将更新参数量压缩至极限（如$10^3$或$10^1$量级）时，SFT的表现会发生灾难性崩溃。

- **信息密度过载**： SFT的梯度更新包含了大量与推理能力无关的“噪声信息”（如句式结构）。当参数空间被压缩至几十个维度时，模型无法同时编码这些语法细节与逻辑模式，导致严重的欠拟合。
- **损失函数的几何形态**： SFT的损失面（Loss Surface）在参数空间中是崎岖且狭窄的，要求模型精确命中特定的token序列。而RL的损失面，特别是在Group Relative Policy Optimization (GRPO) 的平滑作用下，呈现出更宽阔的“盆地”。TinyLoRA仅有的几个自由度足以在这个平滑的势能面上找到通往高奖励区域的梯度下降路径，而无需关心具体的路径坐标（即具体的Token序列）。

## 2. TinyLoRA架构详解：13个参数的数学推导

TinyLoRA并非凭空而来，它是对LoRA及其变体LoRA-XS的进一步数学抽象与工程极致化。为了理解如何将70亿参数模型的控制权压缩至13个变量，我们需要逐步拆解其线性代数结构。

### 2.1 传统LoRA的秩瓶颈

标准LoRA假设权重更新 $\Delta W$ 具有低秩结构：

$$\Delta W = B A$$

其中 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，$B \in \mathbb{R}^{d_{out} \times r}$，$A \in \mathbb{R}^{r \times d_{in}}$。即使在 $r=1$ 的极限情况下，对于一个隐藏层维度 $d=4096$ 的模型，单层的参数量仍为 $4096 \times 1 + 1 \times 4096 = 8192$。对于一个拥有32层、每层包含7个投影矩阵（q, k, v, o, gate, up, down）的模型，总参数量依然在百万级别。这显然无法满足“13个参数”的目标。

### 2.2 LoRA-XS：引入SVD先验

为了摆脱模型维度 $d$ 对参数量的束缚，LoRA-XS  引入了基于奇异值分解（SVD）的重参数化方法。

首先，对冻结的预训练权重 $W$ 进行截断SVD分解：

$$W \approx U_r \Sigma_r V_r^\top$$

其中 $U_r \in \mathbb{R}^{d_{out} \times r}$，$\Sigma_r \in \mathbb{R}^{r \times r}$，$V_r \in \mathbb{R}^{d_{in} \times r}$。这三个矩阵在初始化后被永久冻结，不再更新。它们构成了预训练权重的“主成分子空间”。更新公式变为：

$$W' = W + U_r \Sigma_r R V_r^\top$$

此时，唯一可训练的参数是中间的小矩阵 $R \in \mathbb{R}^{r \times r}$。

- **突破**： 参数量变为 $r^2$，与模型维度 $d$ 完全解耦。
- **局限**： 即使 $r=1$，每个模块仍需训练1个参数。对于数百个模块，总参数量仍为数百。且 $r=1$ 时，更新被限制在单一奇异向量方向，表达能力受限。

### 2.3 TinyLoRA的核心创新：子空间投影与随机基底

TinyLoRA在LoRA-XS的基础上，为了进一步压缩 $R$ 矩阵并实现全局参数共享，引入了随机投影（Random Projection）与极致权重绑定（Extreme Weight Tying）。TinyLoRA将 $R$ 矩阵分解为一组固定随机基底的线性组合。

$$R = \sum_{i=1}^{u} v_i P_i$$

代入完整公式，TinyLoRA的更新规则为：

$$W' = W + U_r \Sigma_r \left( \sum_{i=1}^{u} v_i P_i \right) V_r^\top$$

**关键组件解析：**

- **冻结的SVD子空间 ($U_r, \Sigma_r, V_r$)**：来源：对预训练权重 $W$ 进行离线SVD分解。秩选择 ($r_{frozen}$)：论文消融实验显示，保留更多的奇异值方向并不总是更好。过高的 $r$ 会引入过多自由度，导致优化困难。经验上，$r_{frozen}$ 设为2至8即可捕捉主要的语义方向，主实验中通常设为2 。
- **固定的随机基底 ($P_i$)**：定义：一组预先生成并冻结的随机矩阵 $\{P_1, P_2, \dots, P_u\}$，每个 $P_i \in \mathbb{R}^{r \times r}$。分布：虽然论文未详述具体分布，但根据低秩适配的惯例及压缩感知理论，通常采用标准高斯分布 $\mathcal{N}(0, 1)$ 或 Rademacher 分布（$\pm 1$），并乘以缩放因子 $1/\sqrt{r}$ 以保持方差稳定 。这些矩阵充当了将低维向量 $\mathbf{v}$ 映射到 $r \times r$ 空间的“字典”。
- **可训练向量 ($\mathbf{v}$)**：定义：一个长度为 $u$ 的向量 $\mathbf{v} = [v_1, v_2, \dots, v_u]$。作用：这是整个架构中唯一需要通过梯度下降更新的参数。它学习的是如何重组（Recombine）那些固定的随机基底，从而在SVD定义的主成分子空间内产生有效的扰动。

### 2.4 极致权重绑定 ($n_{tie}$) 与平铺策略

为了达到“13个参数”的极致，TinyLoRA实施了跨层、跨模块的参数共享。

- **绑定因子 ($n_{tie}$)**：定义为共享同一个向量 $\mathbf{v}$ 的模块数量。
- **全参数共享 (Global Sharing)**：在最极端的配置下，模型中所有的线性层（无论是在第1层还是第32层，无论是Query投影还是MLP的Up投影）都共享同一个 $\mathbf{v}$。这意味着，每一层虽然拥有不同的 $U, \Sigma, V$（来自各自的权重），但它们都以完全相同的方式（由 $\mathbf{v}$ 决定）重组其主成分。这种设计基于一个强假设：推理能力的激发可能对应于某种全局的、通用的谱特征变换，而非逐层独立的微调。
- **Tiled（平铺） vs. Structured（结构化）共享策略** ：
  - **Structured Sharing**：相同类型的模块（如所有层的 $Q$ 投影）共享一组参数。
  - **Tiled Sharing**：物理位置相邻的模块（无论类型，如第1层的 $Q, K, V$）共享一组参数。
- **实验结论**：惊人的是，Tiled Sharing 的效果优于 Structured Sharing。这表明参数共享不必受限于模块的功能语义，空间上的局部性或简单的全局共享在极低参数下更为鲁棒。

## 3. GRPO强化学习流程详解

TinyLoRA提供了极简的车辆底盘，而驱动这辆车在GSM8K上达到91%准确率的引擎则是Group Relative Policy Optimization (GRPO)。从SFT迁移到GRPO，意味着训练目标从“预测下一个Token”转变为“优化生成群组的相对优势”。

### 3.1 摒弃Critic模型：内存与稳定性的双重优化

传统的PPO（Proximal Policy Optimization）算法依赖于一个Value Model（Critic）来估计状态价值 $V(s)$，进而计算优势函数 $A(s, a)$。然而，在LLM场景下：

- **内存开销**：Critic模型通常与Policy模型大小相当（如7B），这使得显存需求翻倍。
- **训练不稳**：Critic本身的训练往往比Policy更难收敛，且容易过拟合。

GRPO通过**群组相对优势（Group Relative Advantage）**彻底移除了Critic模型。

**核心机制**：对于每个输入Prompt $q$，模型采样生成一组输出 $\{o_1, o_2, \dots, o_G\}$（通常 $G=64$）。基线计算：利用这组输出的平均奖励作为基线（Baseline）。优势 $A_i$ 计算如下：

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

这种方法利用同批次采样的其他样本作为对照，天然降低了方差，且无需额外的参数来拟合价值函数 。

### 3.2 奖励函数设计：Verifiable Rewards

在GSM8K和MATH任务中，奖励函数的信号是稀疏但确定的（Verifiable）。

- **硬性奖励 (Hard Reward)**：答案正确得1分，错误得0分。这要求模型输出必须包含特定格式（如 \boxed{answer}）。
- **格式奖励 (Format Reward)**：为了引导模型进入推理模式，通常会给予微小的奖励（如0.1）如果模型输出了 <think>...</think> 标签。这在TinyLoRA的极低秩训练中尤为重要，因为初始阶段模型极易退化 。

### 3.3 训练配置与超参数

在TinyLoRA的复现中，以下GRPO超参数至关重要 ：

- **Group Size ($G$)**：64。较大的群组尺寸能提供更准确的基线估计。
- **KL Penalty ($\beta$)**：通常设为0或极小值（如0.001）。由于TinyLoRA仅有13个参数，其表达能力本身就构成了极强的正则化，模型很难发生灾难性的“奖励欺骗”（Reward Hacking）或过度偏离基座模型，因此无需强KL惩罚。
- **Learning Rate**：由于参数量极少，TinyLoRA通常需要比全量微调更大的学习率来驱动那些仅有的自由度。建议扫描范围 $\{10^{-5}, 10^{-4}, 2 \times 10^{-4}\}$ 。

## 4. 混合引擎架构与vLLM数值不匹配挑战

在工程实现层面，从SFT迁移到RL最大的障碍在于训练效率与推理效率的博弈。为了加速GRPO中的Rollout（数据生成）阶段，业界普遍采用混合引擎架构（Hybrid Engine Architecture）：

- **Rollout阶段**：使用vLLM。它利用PagedAttention、连续批处理（Continuous Batching）和高度优化的CUDA核函数，能以极高的吞吐量生成文本。
- **Training阶段**：使用PyTorch FSDP。它支持自动微分和参数分片，适合进行反向传播。

### 4.1 内核不兼容问题

vLLM虽然推理极快，但其对LoRA的支持有限，通常只支持标准的 $W + AB$ 形式，且对Rank有最小限制（如 $r \ge 4$）。TinyLoRA那种复杂的 $U \Sigma (\sum v P) V^\top$ 结构在vLLM中没有现成的算子支持 。

**解决方案：Merge-and-Unmerge Trick**

- **Merge (推理前)**：在每个训练Step开始前，手动计算TinyLoRA产生的 $\Delta W$，并将其加到基座权重中：

$$W_{inference} = W_{base} + \text{compute\_delta}(U, \Sigma, v, P, V)$$

此时，vLLM加载的是一个标准的稠密模型，无需感知TinyLoRA的存在。

- **Rollout**：vLLM使用 $W_{inference}$ 生成数据。
- **Unmerge (训练前)**：由于我们只更新 $\mathbf{v}$，基座 $W_{base}$ 保持不变。训练时，PyTorch图依然保持分离状态，梯度只回传给 $\mathbf{v}$。

### 4.2 致命的数值不匹配 (Numerical Mismatch)

这种混合架构引入了一个隐蔽但致命的问题：Off-Policy Drift。

- **精度差异**：vLLM为了加速，常使用 float16 或 KV Cache Int8 量化。而PyTorch训练通常使用 bfloat16 或 float32 累加。
- **算子差异**：FlashAttention在vLLM中的实现与PyTorch中的SDPA（Scaled Dot Product Attention）在数值上并非逐位（Bit-wise）一致。
- **后果**：

$$\pi_{vllm}(a|s) \neq \pi_{train}(a|s)$$

虽然两者理论上是同一个模型，但数值误差导致它们对同一输入的概率分布存在微小偏差。在RL训练中，这种偏差会被累积放大。GRPO假设数据是On-Policy的（即由当前策略生成），这种隐式的Off-Policy行为会导致KL散度异常升高，训练曲线震荡甚至崩塌 。

## 5. 修正方案：截断重要性采样 (Truncated Importance Sampling, TIS)

为了解决混合引擎带来的分布偏移，TinyLoRA复现中必须引入数学修正。Ionides (2008) 和 Yao et al. (2025) 提出的截断重要性采样（TIS） 成为了稳定训练的关键 。

### 5.1 数学原理

标准策略梯度（Policy Gradient）假设样本来自 $\pi_\theta$。当样本实际来自 $\pi_{old}$（在这里即 $\pi_{vllm}$）时，我们需要引入重要性采样权重（Importance Sampling Weight）$\rho$ 来校正梯度估计：

$$\nabla J(\theta) = \mathbb{E}_{s,a \sim \pi_{vllm}} \left[ \rho_t \cdot A_t \cdot \nabla \log \pi_\theta(a_t|s_t) \right]$$

其中 $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{vllm}(a_t|s_t)}$。然而，由于数值不稳定，$\rho_t$ 可能会出现极端值（爆炸或趋零），导致方差过大。TIS通过引入截断机制来平衡偏差与方差。

### 5.2 TIS在GRPO中的实现

在TinyLoRA的复现代码（基于VERL或TRL）中，TIS通常按以下步骤实施 ：

- **双重对数概率计算**：在vLLM生成阶段，记录每个token的对数概率 log_probs_vllm。在PyTorch训练前向传播时，重新计算同一序列的对数概率 log_probs_train。
- **计算比率**：

$$\text{log\_ratio} = \text{log\_probs\_train} - \text{log\_probs\_vllm}$$

$$\rho = \exp(\text{log\_ratio})$$

- **截断 (Truncation)**：设定一个阈值 $\epsilon$（如 vllm_importance_sampling_cap = 2.0），对 $\rho$ 进行截断：

$$\hat{\rho} = \min(\rho, 2.0)$$

或者采用更激进的 token_mask 模式，如果 $\rho$ 超出范围，直接将该Token的Loss置零，不进行更新 。

- **序列级 vs. Token级**：TinyLoRA复现推荐使用Token级截断，因为数值误差通常是局部且随机的。

| 参数配置 | 推荐值 | 解释 |
|----------|--------|------|
| vllm_importance_sampling_correction | True | 开启TIS修正 |
| vllm_importance_sampling_mode | "token_truncate" | 超出阈值时截断为阈值，而非丢弃 |
| vllm_importance_sampling_cap | 2.0 | 允许训练分布偏离推理分布的最大倍数 |

## 6. 复现指南：从代码到部署

本节将上述理论转化为具体的复现步骤，基于 TRL 和 vLLM 库。

### 6.1 环境准备

- **Library**: transformers >= 4.37, trl >= 0.8.0, vllm >= 0.4.0, peft (需自定义修改)。
- **Hardware**: 至少一张24GB显卡（用于7B模型+vLLM），推荐A100以支持大Batch训练。

### 6.2 步骤一：SVD初始化与TinyLoRA层定义

由于 peft 库暂不支持TinyLoRA，需要手动实现 TinyLoRALayer。

```python
import torch
import torch.nn as nn
import math

class TinyLoRALayer(nn.Module):
    def __init__(self, base_layer, rank_frozen=16, u_dim=13, tying_vector=None):
        super().__init__()
        self.base_layer = base_layer # 原始线性层
        self.rank = rank_frozen
        self.u_dim = u_dim
        
        # 1. 离线SVD (实际应在初始化前完成并传入)
        # W = U @ S @ V.T
        # 这里假设已经提取了U_r, S_r, V_r
        # self.register_buffer("U", U_r)...
        
        # 2. 固定随机基底 P
        # 初始化为高斯分布，缩放因子 1/sqrt(r)
        self.register_buffer("P", torch.randn(u_dim, rank_frozen, rank_frozen) / math.sqrt(rank_frozen))
        
        # 3. 可训练向量 v (共享)
        if tying_vector is not None:
            self.v = tying_vector
        else:
            self.v = nn.Parameter(torch.zeros(u_dim)) # 初始化为0

    def forward(self, x):
        # 计算 TinyLoRA 增量
        # R = sum(v_i * P_i)
        R = torch.einsum('i,irr->rr', self.v, self.P)
        
        # Delta W = U @ S @ R @ V.T
        # 优化：利用结合律，避免构建大矩阵
        # x @ V ->
        h = x @ self.V 
        h = h @ R.t() 
        h = h @ self.S 
        delta_out = h @ self.U.t()
        
        return self.base_layer(x) + delta_out
```

### 6.3 步骤二：GRPO Trainer配置

在 TRL 的 GRPOConfig 中，必须正确设置混合引擎参数 。

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="tiny_lora_qwen",
    learning_rate=1e-4, # 较大的LR
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16, # 增大等效Batch Size
    num_generations=64, # Group Size G=64
    max_completion_length=512,
    
    # 混合引擎关键配置
    use_vllm=True,
    vllm_device="cuda:0", # 假设单卡Colocate
    vllm_gpu_memory_utilization=0.5, # 留显存给训练
    vllm_importance_sampling_correction=True, # 开启TIS
    vllm_importance_sampling_mode="token_truncate",
    vllm_importance_sampling_cap=2.0,
    
    # 奖励函数配置
    report_to="wandb"
)

# 定义奖励函数 (Exact Match)
def accuracy_reward(completions, solution, **kwargs):
    # 解析 completions 中的 \boxed{...} 与 solution 对比
    # 返回 [1.0, 0.0,...]
    pass 

trainer = GRPOTrainer(
    model=model, # 包装了TinyLoRA的模型
    reward_funcs=[accuracy_reward],
    args=training_args,
    train_dataset=dataset,
)
```

### 6.4 步骤三：训练循环中的权重合并

由于 GRPOTrainer 内部封装了vLLM调用，若使用由于自定义的TinyLoRA层，需要重写 Trainer 的 _generate_with_vllm 方法，或者利用 PyTorch 的 Hook 机制，在调用 model.generate (vLLM) 之前，将 TinyLoRA 的权重 add 到 base_layer.weight 中，生成完后再 sub 回来。这是确保 vLLM 能正确推理的关键“黑魔法”。

## 7. 实验结果分析与数据洞察

### 7.1 精度与参数量的帕累托前沿

在GSM8K测试集上，TinyLoRA展示了极高的参数效率：

- Baseline (Qwen2.5-7B-Instruct): 76.0%
- SFT (全量): ~88% (需要大量数据)
- TinyLoRA (13 Params) + GRPO: 91.8% 
- LoRA (Rank 16, ~10M Params) + GRPO: 95.0%

**洞察**：从13参数到1000万参数，性能仅提升了3.2%。这意味着，对于数学推理任务，模型95%的潜能可以通过仅仅调整13个自由度来激发。这强烈暗示了LLM的推理能力并非是在微调中“学会”的，而是通过微调“对齐”或“解锁”的。SFT因为试图强行灌输具体的Token序列（高维噪声），反而掩盖了这种低维的推理本质。

### 7.2 训练动态监控

在复现过程中，观察WandB曲线应注意以下特征：

- **Reward上升，KL平稳**：由于参数极少，KL散度通常不会剧烈波动，这是TinyLoRA的一大优势。
- **TIS Weights分布**：监控TIS权重的均值和方差。如果均值长期显著偏离1.0（如 >1.5），说明vLLM与训练模型的数值偏差过大，需要检查量化设置或降低 vllm_gpu_memory_utilization 以避免过度压缩。

## 8. 结论与展望

TinyLoRA与GRPO的结合，标志着LLM适配技术进入了“外科手术式”精准微调的新时代。通过SVD提取骨架、随机投影构建基底、RL提供精准导航，我们得以用26字节的极小代价，撬动70亿参数模型的推理潜能。对于工程实践者而言，这一技术的复现不仅仅是参数的缩减，更是一次对混合计算架构的深刻理解。掌握vLLM与PyTorch之间的数值协调，理解TIS背后的数学修正，将是在未来Edge AI、联邦学习及千人千面模型服务中占据先机的关键。

## 参考文献

- Learning to Reason in 13 Parameters (Morris et al., 2026)
- LoRA-XS: Low-rank adaptation with extremely small number of parameters
- Implementing TinyLoRA in vLLM; Truncated Importance Sampling (Ionides, 2008; Yao et al., 2025)
- Group Relative Policy Optimization (GRPO) Mechanism
- HuggingFace TRL GRPO Trainer Documentation