# ⚠️ Known Pitfalls & Notes

Issues encountered and resolved during development. Read before modifying `train_rl.py`.

---

### 1. `Dataset` import: `datasets`, not `transformers`

```python
# ✗ Wrong — Dataset does not exist in transformers
from transformers import Dataset

# ✓ Correct
from datasets import Dataset
```

`Dataset` (including `Dataset.from_list()`) belongs to the HuggingFace `datasets` library. Importing from `transformers` raises `ImportError` at runtime.

---

### 2. Reward columns must survive `dataset.map()`

After applying `apply_chat_template`, all columns except `prompt` were previously removed:

```python
remove_columns=[c for c in rl_dataset.column_names if c != 'prompt']
```

This silently dropped `input_output`, `public_tests`, `private_tests`, etc. When the reward function received `None` for all test fields → **no test cases → reward=0 for every generation → GRPO advantage=0 → loss=0, grad_norm=0**.

**Symptom**: training completes with `loss: 0.0` / `grad_norm: 0.0` across all steps, but no error is raised.

**Fix**: preserve all columns needed by `code_reward_func`.

---

### 3. DeepCoder dataset: OOM from oversized test cases

The DeepCoder `deepcoder_lcbv5_train.jsonl` is **~5 GB for only 599 samples** (~8.7 MB/sample average). Some samples contain test cases with inputs exceeding **200,000 characters each**, making a single `input_output` field 856 KB+.

When loaded into memory → Arrow serialization → GRPO Trainer collation, this causes the Linux OOM killer to send `SIGKILL` — especially on **gaming laptops or machines with ≤32 GB RAM**.

CodeContests does not trigger this because its test cases are much smaller per sample.

**Fix**: test cases are trimmed to **5 per sample** at two points:
- **At load time** (reservoir sampling): `input_output` is truncated to the first 5 entries.
- **At reward time** (`code_reward_func`): `test_cases_list = test_cases_list[:5]`.

This is a **gaming-laptop-friendly default**. 5 test cases provide sufficient reward signal for RL training. If you have more memory (e.g., a dedicated training server), you can increase this limit by modifying:
- `sample["input_output"]` trimming in the DeepCoder loading block
- `test_cases_list[:5]` in `code_reward_func`

---

### 4. `load_dataset(..., streaming=True)` is slow for local JSONL

Using HuggingFace's streaming mode to iterate over local JSONL files incurs per-sample Arrow serialization overhead. For a 599-sample file, this can take minutes.

**Fix**: use `open()` + `json.loads()` for direct file I/O, which is near-instant for small datasets.
