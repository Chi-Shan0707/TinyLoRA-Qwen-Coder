# Usage Guide (English)

## 1) End-to-End Workflow

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare dataset:

```bash
python download_dataset.py
```

3. Optional pipeline check:

```bash
python verify_pipeline.py
```

4. Train:

```bash
python train_rl.py 32 2000
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
python train_rl.py 32 2000 --no_quant
python train_rl.py 32 2000 --rank 4
```

**Command-line arguments for training:**
- `u_value` (first positional): TinyLoRA parameter count (default: 16)
- `max_samples` (second positional): maximum training samples (default: 2000)
- `--do_validate`: enable validation during training
- `--val_steps N`: run validation every N steps (default: 100)
- `--val_samples N`: number of validation samples (default: 10)
- `--no_quant`: disable 4-bit quantization, load model in BF16
- `--rank N`: TinyLoRA SVD rank (default: 2)

5. Evaluate:

```bash
python validate.py 50
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
python test.py --baseline --num_samples 50
```

**Command-line arguments for validation:**
- `num_samples` (positional): number of samples to validate (default: 10)
- `--no_quant`: disable 4-bit quantization for validation

**Command-line arguments for testing:**
- `--checkpoint_path`: path to checkpoint .pt file (default: ./output/luoguqwencoder-lora/tiny_lora_v.pt)
- `--num_samples`: number of samples to test (default: 50)
- `--test_data`: path to test dataset (default: ./local_code_contests/code_contests_test.jsonl)
- `--baseline`: test base model without TinyLoRA
- `--test_seed`: random seed for evaluation (default: 42)
- `--use_quant`: load model with 4-bit quantization (default: True)
- `--no_quant`: disable 4-bit quantization, load in BF16

---

## 2) Data & Pipeline Details

### Dataset files

- `local_code_contests/code_contests_train.jsonl`
- `local_code_contests/code_contests_valid.jsonl`
- `local_code_contests/code_contests_test.jsonl`

### Data preprocessing entry

- script: `download_dataset.py`
- expected result: local train/valid/test jsonl files ready for loading in `train_rl.py`, `validate.py`, and `test.py`.

### Training dataset controls

Main controls are in `train_rl.py`:

- `DATASET_CONFIG`: source and filtering policy
- `MAX_SAMPLES`: hard cap for sampled training examples
- `TINYLORA_SEED`: reproducible random projection/sampling behavior
- `filter_dataset(...)`: difficulty/source filtering gate.

---

## 3) Validation and Testing

### In-training validation

```bash
python train_rl.py 32 2000 --do_validate --val_steps 100 --val_samples 10
```

- `--do_validate`: turn on periodic validation
- `--val_steps`: validation interval during training
- `--val_samples`: sampled validation size.

### Post-training validation

```bash
python validate.py 50
```

- runs validation-style evaluation on a selected sample count.

### Baseline vs checkpoint test

```bash
python test.py --baseline --num_samples 50
python test.py --checkpoint_path ./output/luoguqwencoder-lora/tiny_lora_v.pt --num_samples 50
```

- `--baseline`: evaluate base model directly
- `--checkpoint_path`: evaluate TinyLoRA checkpoint.

---

## 4) Quantization Path

### Default path (quantized)

- default loading uses 4-bit quantized model path for efficiency.

### BF16 path

```bash
python train_rl.py 32 2000 --no_quant
```

- `--no_quant` disables quantized loading and uses BF16 pipeline.

### Checkpoint metadata

- checkpoint saves `is_quantized` metadata.
- loading logic uses this flag to avoid quantization mismatch.

---

## 5) Troubleshooting

- CUDA OOM:
  - reduce batch-related pressure (smaller sample count / lighter validation frequency)
  - use quantized path by default.
- No code extracted in reward step:
  - inspect prompt/template in `apply_chat_template`
  - verify output format assumptions in reward parsing.
- Compile-run timeout issues:
  - adjust timeout in `compile_and_run(..., timeout=2)`.
