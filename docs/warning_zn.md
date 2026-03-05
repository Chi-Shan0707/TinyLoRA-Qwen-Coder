# ⚠️ 已知坑点与注意事项

开发过程中遇到并修复的问题。修改 `train_rl.py` 前请先阅读。

---

### 1. `Dataset` 的导入：用 `datasets`，不是 `transformers`

```python
# ✗ 错误 — transformers 中没有 Dataset
from transformers import Dataset

# ✓ 正确
from datasets import Dataset
```

`Dataset`（包括 `Dataset.from_list()`）属于 HuggingFace 的 `datasets` 库，从 `transformers` 导入会直接抛出 `ImportError`。

---

### 2. 奖励函数所需列在 `dataset.map()` 后必须保留

之前 `apply_chat_template` 后会删除除 `prompt` 以外的所有列：

```python
remove_columns=[c for c in rl_dataset.column_names if c != 'prompt']
```

这导致 `input_output`、`public_tests`、`private_tests` 等字段全部被丢弃。奖励函数收到的测试字段全为 `None` → **无测试用例 → reward 恒为 0 → GRPO advantage=0 → loss=0, grad_norm=0**。

**症状**：训练全程 `loss: 0.0` / `grad_norm: 0.0`，但不报任何错误。

**修复**：保留 `code_reward_func` 所需的所有列。

---

### 3. DeepCoder 数据集：超大测试用例导致 OOM

DeepCoder 的 `deepcoder_lcbv5_train.jsonl` 仅 599 条样本却有 **~5 GB**（平均 ~8.7 MB/条）。部分样本的单个测试输入超过 **20 万字符**，一条 `input_output` 字段就有 856 KB+。

加载到内存 → Arrow 序列化 → GRPO Trainer 整理数据时，会导致 Linux OOM killer 发送 `SIGKILL` —— **游戏本或 ≤32 GB 内存机器上尤为常见**。

CodeContests 不会触发此问题，因为其每条样本的测试用例规模小得多。

**修复**：在两处将测试用例截断为 **5 个**：
- **加载时**（reservoir 采样阶段）：`input_output` 截断为前 5 条。
- **奖励计算时**（`code_reward_func`）：`test_cases_list = test_cases_list[:5]`。

这是一个**游戏本友好的默认值**。5 个测试用例足以为 RL 训练提供有效的奖励信号。如果你有更大内存（如专用训练服务器），可以自行调大：
- DeepCoder 加载代码块中 `sample["input_output"]` 的截断逻辑
- `code_reward_func` 中的 `test_cases_list[:5]`

---

### 4. `load_dataset(..., streaming=True)` 读本地 JSONL 很慢

HuggingFace 的 streaming 模式在迭代本地 JSONL 时，每条样本都有 Arrow 序列化开销。599 条数据可能需要数分钟。

**修复**：用 `open()` + `json.loads()` 直接读文件，对小数据集几乎瞬间完成。
