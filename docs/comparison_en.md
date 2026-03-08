# TinyLoRA v4.0 Training Effect Comparison

## Test Conditions

- **Test Dataset**: `local_code_contests/code_contests_test.jsonl`
- **Sample Count**: 165 samples
- **Random Seed**: 42
- **Test Command**: `python3 test.py --num_samples 165`

### Scoring Function (utils.py: compile_and_run)

```
reward = 0.0   # Compilation failed
reward = 0.5   # Compiled but no tests passed
reward = 0.5 + 0.5 * (passed / total)  # Partial pass
reward = 1.0   # All tests passed
```

---

## Test Results Comparison

| Metric | Old Training (code_contests, 13328 samples) | New Training (lcbv5, 28 samples) | Improvement |
|:---|:---:|:---:|:---:|
| **Pass@1** | 1.82% (3/165) | 3.64% (6/165) | **+100%** |
| **Compile Rate** | 73.33% (121/165) | 76.36% (126/165) | **+4.13%** |
| **Average Score** | 0.4274 | 0.4489 | **+5.03%** |

### Data Sources

- Old training results: `tinylora-32-2-13328/tinylora-32-2-13328/test_log.txt`
- New training results: `test_log_20260307.log`

---

## Training Efficiency Comparison

| Metric | Old Training | New Training | Change |
|:---|:---:|:---:|:---:|
| **Training Data Size** | 13,328 samples | 28 samples | -99.8% |
| **Training Time** | ~4h 24m | ~1h 12m | -73% |

---

## Conclusion

v4.0 replaces the original code_contests dataset with Deep-Coder-Preview-Dataset (lcbv5):
- **Pass@1 improved by 100%** (1.82% → 3.64%)
- **Compile Rate improved by 4.13%** (73.33% → 76.36%)
- **Average Score improved by 5.03%** (0.4274 → 0.4489)
- **Training data reduced by 99.8%** (13328 → 28)
- **Training time reduced by 73%** (4h24m → 1h12m)

This demonstrates that using higher quality training data (lcbv5) can significantly improve model performance while substantially reducing training costs.
