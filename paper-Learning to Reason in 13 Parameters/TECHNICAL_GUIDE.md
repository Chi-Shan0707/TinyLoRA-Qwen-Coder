# TinyLoRA Technical Guide (English)

**Language**: English (default) | [中文](./TECHNICAL_GUIDE_CN.md)

This document is the English technical guide for TinyLoRA, covering theory, architecture, GRPO training logic, and practical implementation notes.

---

## Contents

1. [From SFT to Tiny-Parameter RL](#1-from-sft-to-tiny-parameter-rl)
2. [TinyLoRA Architecture](#2-tinylora-architecture)
3. [GRPO Training Mechanics](#3-grpo-training-mechanics)
4. [Hybrid Engine and Numerical Drift](#4-hybrid-engine-and-numerical-drift)
5. [Truncated Importance Sampling (TIS)](#5-truncated-importance-sampling-tis)
6. [Implementation Reference](#6-implementation-reference)
7. [Experiment Setup](#7-experiment-setup)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. From SFT to Tiny-Parameter RL

### 1.1 Why SFT and RL behave differently at low capacity

Supervised fine-tuning (SFT) optimizes token-level imitation:

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y)}\left[\sum_t \log \pi_\theta(y_t\mid x,y_{<t})\right]
$$

At extremely low parameter budgets, SFT must encode both reasoning and stylistic noise, which over-constrains the update space.

Reinforcement learning (RL) optimizes expected reward:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y\sim\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(y_t\mid\cdot)\cdot R(y)\right]
$$

For math/code tasks with verifiable outcomes, RL can focus on correctness signal, reducing the effective intrinsic dimension needed to improve behavior.

### 1.2 Key finding (paper context)

| Setup | Trainable Params | GSM8K |
| :--- | :---: | :---: |
| Base Qwen2.5-7B-Instruct | 0 | 76.0% |
| TinyLoRA + GRPO | **13** | **91.8%** |
| TinyLoRA + SFT | 13 | 83% |

This is the core motivation for using RL with extreme parameter compression.

---

## 2. TinyLoRA Architecture

### 2.1 From LoRA to LoRA-XS style reparameterization

Classic LoRA:

$$W' = W + BA$$

Even with rank 1, total trainable parameters remain large across many layers.

LoRA-XS style decomposition freezes SVD factors:

$$W \approx U_r\Sigma_rV_r^\top,\quad W' = W + U_r\Sigma_rRV_r^\top$$

Only the tiny matrix $R \in \mathbb{R}^{r\times r}$ is trainable.

### 2.2 TinyLoRA projection mixing

TinyLoRA further parameterizes $R$ by fixed random bases:

$$R = \sum_{i=1}^{u} v_iP_i$$

Then:

$$W' = W + U_r\Sigma_r\left(\sum_{i=1}^{u} v_iP_i\right)V_r^\top$$

Where:
- $U_r,\Sigma_r,V_r$: frozen SVD buffers
- $P_i$: fixed random projection matrices
- $v \in \mathbb{R}^u$: the only trainable vector

### 2.3 Parameter tying

With global tying, many layers share one vector $v$.

$$
\text{Total Params} = \frac{n\times m\times u}{n_{tie}}
$$

At full tying ($n_{tie}=n\times m$), total trainable params reduce to $u$.

### 2.4 Rank choice

Empirically, moderate rank (often `r=2`) is a good default. Larger $r$ can add flexibility but may reduce optimization stability in tiny-parameter regimes.

---

## 3. GRPO Training Mechanics

GRPO (Group Relative Policy Optimization) removes explicit critic training by using relative rewards inside a sampled group.

For one prompt, sample $G$ completions and normalize reward:

$$A_i = \frac{r_i-\mu_r}{\sigma_r+\epsilon}$$

Policy objective uses clipped ratios similar to PPO:

$$
\mathcal{L}_{GRPO} = -\mathbb{E}\left[\sum_t \min(\rho_tA_t,\text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)-\beta D_{KL}(\pi_\theta\|\pi_{ref})\right]
$$

TinyLoRA often works with very small KL penalties because capacity itself is a strong regularizer.

---

## 4. Hybrid Engine and Numerical Drift

A practical GRPO stack may use:

- Rollout engine: vLLM (high-throughput generation)
- Training engine: PyTorch/FSDP (backprop and updates)

Potential issue: off-policy numerical drift.

$$\pi_{rollout}(a\mid s)\neq\pi_{train}(a\mid s)$$

Differences in precision and kernels can accumulate into unstable KL behavior.

---

## 5. Truncated Importance Sampling (TIS)

To correct for rollout-policy mismatch, use importance ratio:

$$\rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{rollout}(a_t\mid s_t)}$$

Truncate to reduce variance:

$$\hat{\rho}_t = \min(\rho_t, c)$$

Typical practical choice: `c=2.0`.

Example:

```python
def compute_tis_weights(logp_train, logp_rollout, cap=2.0):
    ratio = torch.exp(logp_train - logp_rollout)
    return torch.clamp(ratio, max=cap)
```

---

## 6. Implementation Reference

### 6.1 TinyLoRA layer sketch

```python
class TinyLoRALayer(nn.Module):
    def __init__(self, base_layer, rank_frozen=2, u_dim=13, tying_vector=None):
        super().__init__()
        self.base_layer = base_layer
        # SVD buffers (frozen): U, S, V
        # Fixed random projection banks: P
        # Trainable vector: v

    def forward(self, x):
        base_out = self.base_layer(x)
        # R = sum(v_i * P_i)
        # delta = x @ V @ R.T @ S @ U.T
        return base_out + delta_out
```

### 6.2 Training config sketch (GRPO)

```python
training_args = GRPOConfig(
    learning_rate=1e-4,
    num_generations=64,
    max_completion_length=4096,
    beta=0.001,
    use_vllm=True,
    vllm_importance_sampling_correction=True,
    vllm_importance_sampling_cap=2.0,
)
```

### 6.3 Reward design principles

- Prefer verifiable rewards (exact answer / compile-and-run correctness).
- Keep rewards simple and stable.
- For code tasks, reward tiers (fail / compile-only / full pass) are practical and robust.

---

## 7. Experiment Setup

### 7.1 Typical ranges

| Parameter | Suggested Range |
| :--- | :--- |
| `u` | 13, 16, 32 |
| `rank_frozen` | 1, 2, 4 |
| `learning_rate` | `5e-6 ~ 2e-4` |
| `num_generations` | `4 ~ 64` |
| `max_completion_length` | `512 ~ 4096` |
| `beta` (KL) | `0 ~ 0.001` |

### 7.2 Practical default (code adaptation)

- `u=32`
- `rank=2`
- compile-and-run reward
- quantized base model for memory efficiency

---

## 8. Troubleshooting

### 8.1 No improvement / unstable reward

- Lower or tune learning rate.
- Increase sample count or training steps.
- Check reward function consistency and code extraction reliability.

### 8.2 KL spike or oscillation

- Enable TIS correction.
- Check rollout-vs-train precision mismatch.
- Reduce aggressive policy updates.

### 8.3 OOM / memory pressure

- Lower generations, batch size, completion length.
- Increase gradient accumulation.
- Prefer quantized loading where possible.

---

## References

1. Morris et al. (2026), *Learning to Reason in 13 Parameters*.
2. Hu et al. (2021), *LoRA*.
3. Bałazy et al. (2025), *LoRA-XS*.
4. Shao et al. (2024), *DeepSeekMath*.

---

[Back to paper index](./README.md) | [中文技术文档](./TECHNICAL_GUIDE_CN.md)
