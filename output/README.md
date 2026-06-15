# Output Artifacts

**Language**: English | [中文](#中文)

This directory is intentionally kept lightweight. Large generated checkpoints should be produced locally by training and should not be committed unless there is a clear release reason.

## Current State

- `tiny_lora_v.pt` is not included in the repository.
- Older diagonal TinyLoRA checkpoints were removed because they are incompatible with the current full `r x r` TinyLoRA implementation.
- New checkpoints are produced by `train_rl.py` under `./output/luoguqwencoder-lora/` by default.

## Checkpoint Schema

A valid current checkpoint is a PyTorch dictionary with:

- `global_v`: trained shared vector.
- `u_value`: dimension of `global_v`.
- `rank`: TinyLoRA SVD rank.
- `seed`: random seed used to reproduce fixed projection matrices.
- `model_id`: base model identifier.
- `is_quantized`: whether training used 4-bit quantization.
- `total_replaced_layers`: number of replaced linear layers.

## Restore

Use the project scripts instead of loading artifacts manually:

```bash
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
python validate.py --num_samples 50
```

For compatibility, ensure that `rank`, `seed`, quantization mode, and base model match the training run. The current TinyLoRA formula is:

```text
R = sum_i v_i P_i, where P_i is r x r
Delta W = U @ diag(S) @ R @ Vh
```

---

## 中文

本目录仅保留轻量说明。大型训练产物应由本地训练生成，除非有明确发布需求，否则不应提交。

### 当前状态

- 仓库不再包含 `tiny_lora_v.pt`。
- 旧的 diagonal TinyLoRA checkpoint 已删除，因为它们不兼容当前 full `r x r` TinyLoRA 实现。
- 新 checkpoint 默认由 `train_rl.py` 生成到 `./output/luoguqwencoder-lora/`。

### Checkpoint 字段

当前有效 checkpoint 是 PyTorch 字典，包含：

- `global_v`：训练得到的共享向量。
- `u_value`：`global_v` 的维度。
- `rank`：TinyLoRA SVD rank。
- `seed`：用于复现固定投影矩阵的随机种子。
- `model_id`：基座模型标识。
- `is_quantized`：训练是否使用 4-bit 量化。
- `total_replaced_layers`：替换的线性层数量。

### 恢复方式

优先使用项目脚本：

```bash
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
python validate.py --num_samples 50
```

恢复时需要保证 `rank`、`seed`、量化模式和基座模型与训练时一致。当前 TinyLoRA 公式为：

```text
R = sum_i v_i P_i，其中 P_i 是 r x r 矩阵
Delta W = U @ diag(S) @ R @ Vh
```
