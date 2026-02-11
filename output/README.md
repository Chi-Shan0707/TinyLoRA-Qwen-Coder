# tiny_lora_v.pt — TinyLoRA Checkpoint

<div align="center">

**TinyLoRA Training Artifact: Trained Global Vector & Reconstruction Metadata**

[中文版本](#中文版本) | [English Version](#english-version)

</div>

---

# 中文版本

## tiny_lora_v.pt — TinyLoRA 参数文件

本文件是通过训练得到的 LoRA 适配器检查点，包含训练后的全局共享向量和重建模型所需的全部元信息。

### 文件内容

- **文件位置**：`./output/luoguqwencoder-lora/tiny_lora_v.pt`
- **文件类型**：PyTorch 字典（dict）
- **包含字段**：
  - `global_v`：训练好的共享向量（torch.Tensor），形状例 `shape=torch.Size([32])`，即 u 维向量。
  - `u_value`：共享向量的维度（int），例如 32。
  - `rank`：TinyLoRA 的秩（int），通常为 2。
  - `seed`：用于生成固定随机投影矩阵 P 的随机种子（int），例如 42。
  - `model_id`：基座模型的 Hugging Face ID（str），例如 `qwen/Qwen2.5-Coder-3B-Instruct`。
  - `total_replaced_layers`：被替换为 TinyLoRALinear 的 Linear 层总数（int），便于记录。

### 快速恢复步骤

按以下步骤在本地恢复训练好的模型：

```python
import torch
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# 1) 加载检查点
sd = torch.load("./output/luoguqwencoder-lora/tiny_lora_v.pt", map_location="cpu")

# 2) 提取元数据
u = sd["u_value"]
seed = sd["seed"]
v = sd["global_v"]  # torch.Tensor
model_id = sd.get("model_id", "qwen/Qwen2.5-Coder-3B-Instruct")

# 3) 加载基座模型（须与训练时的量化/device_map 配置一致）
#    这里使用 4bit 量化，与训练脚本保持一致
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)

# 4) 创建 global params 容器并加载 v（使用 train_rl.py 中的 TinyLoRAGlobalParams）
#    确保使用相同 u 值、相同随机种子
from train_rl import TinyLoRAGlobalParams, apply_tiny_lora

device = model.model.layers[0].self_attn.q_proj.weight.device
global_params = TinyLoRAGlobalParams(u_dim=u, device=device, dtype=torch.bfloat16)
with torch.no_grad():
    global_params.global_v.copy_(v.to(global_params.global_v.dtype).to(device))

# 5) 固定随机种子并注入 TinyLoRA（这会生成与训练时相同的 P 矩阵）
torch.manual_seed(seed)
apply_tiny_lora(model, global_params)

# 6) 验证加载成功
print(f"成功加载 TinyLoRA 模型")
print(f"global_v 形状: {global_params.global_v.shape}")
print(f"替换的 Linear 层数: {sd.get('total_replaced_layers', 'N/A')}")
```

### 重要说明

- **量化配置一致性**：恢复时务必使用与训练相同的量化配置（4bit NF4）。
- **随机种子**：P 矩阵的生成依赖于随机种子。使用相同的 `seed` 值保证 P 矩阵完全一致。
- **SVD 决定性**：SVD 分解是确定性运算，所以 U、S、Vh 完全可复现。
- **完全复现**：结合相同的 `u_value`、`rank`、`seed` 和基座模型，可实现 100% 可复现的增量权重重建。

---

# English Version

## tiny_lora_v.pt — TinyLoRA Checkpoint

This file is the trained LoRA adapter checkpoint containing the trained global shared vector and all metadata needed to reconstruct the model.

### File Contents

- **Path**: `./output/luoguqwencoder-lora/tiny_lora_v.pt`
- **Type**: PyTorch dict
- **Fields**:
  - `global_v`: Trained shared vector (torch.Tensor), e.g., shape=[32] (u-dimensional vector).
  - `u_value`: Dimension of the shared vector (int), e.g., 32.
  - `rank`: Rank of TinyLoRA (int), typically 2.
  - `seed`: Random seed for generating fixed random projection matrices P (int), e.g., 42.
  - `model_id`: Hugging Face model ID of the base model (str), e.g., `qwen/Qwen2.5-Coder-3B-Instruct`.
  - `total_replaced_layers`: Total number of Linear layers replaced with TinyLoRALinear (int).

### Quick Restore Steps

Follow these steps to restore the trained model locally:

```python
import torch
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# 1) Load checkpoint
sd = torch.load("./output/luoguqwencoder-lora/tiny_lora_v.pt", map_location="cpu")

# 2) Extract metadata
u = sd["u_value"]
seed = sd["seed"]
v = sd["global_v"]  # torch.Tensor
model_id = sd.get("model_id", "qwen/Qwen2.5-Coder-3B-Instruct")

# 3) Load base model (must use same quantization/device_map as training)
#    Using 4-bit quantization consistent with training
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)

# 4) Create global params container and load v (using TinyLoRAGlobalParams from train_rl.py)
#    Ensure using same u value and random seed
from train_rl import TinyLoRAGlobalParams, apply_tiny_lora

device = model.model.layers[0].self_attn.q_proj.weight.device
global_params = TinyLoRAGlobalParams(u_dim=u, device=device, dtype=torch.bfloat16)
with torch.no_grad():
    global_params.global_v.copy_(v.to(global_params.global_v.dtype).to(device))

# 5) Fix random seed and inject TinyLoRA (generates identical P matrices)
torch.manual_seed(seed)
apply_tiny_lora(model, global_params)

# 6) Verify successful loading
print(f"TinyLoRA model loaded successfully")
print(f"global_v shape: {global_params.global_v.shape}")
print(f"Total replaced layers: {sd.get('total_replaced_layers', 'N/A')}")
```

### Important Notes

- **Quantization Consistency**: Always use the same quantization config (4-bit NF4) as training when restoring.
- **Random Seed**: P matrices are generated based on the random seed. Using the same `seed` ensures identical P matrices.
- **SVD Determinism**: SVD is a deterministic operation, so U, S, Vh are fully reproducible.
- **Full Reproducibility**: Combined with identical `u_value`, `rank`, `seed`, and base model, 100% reproducible delta weight reconstruction is guaranteed.