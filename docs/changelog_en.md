# Changelog (English)

## v3.5

- **Remove test case examples from prompt**: Stop concatenating `input_output`/`public_tests` to the chat template:
  - Some samples carry extremely large test cases (200K+ chars each), causing prompt bloat and OOM during training
  - Now prompt contains only problem description without examples
  - This change significantly reduces memory usage during training
- **Limit test cases during training**:
  - In `code_reward_func`, limit to first 5 test cases to avoid excessive subprocess fork pressure (each fork creates a large model process; too many forks → OOM killer)
- **DeepCoder data loading optimization**:
  - Trim `input_output` to first 5 test cases at load time
  - Some samples carry huge test cases causing multi-GB memory bloat → OOM kill

- **Clip High (from DeepCoder/DAPO paper)**: Implemented asymmetric clipping in GRPO loss:
  - Added `epsilon=0.2` (lower bound: 1 - 0.2 = 0.8)
  - Added `epsilon_high=0.5` (upper bound: 1 + 0.5 = 1.5)
  - Unlike symmetric clipping [1-ε, 1+ε], Clip High increases only the upper bound
  - This encourages more exploration and prevents premature convergence when correct solutions are found
  - Refer to [DeepCoder paper](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) for details
- **DeepCoder Dataset Support**: Added support for [DeepCoder-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset):
  - Added `download_DeepCoder-Preview-Dataset.py` script (parquet format) to download and preprocess DeepCoder dataset
  - Supports 4 configs: codeforces, lcbv5, primeintellect, taco
  - Added `--dataset` CLI argument to choose between 'code_contests' (default) and 'deepcoder'
  - Modified reward function to support both dataset formats
  - DeepCoder dataset requires at least 5 test cases per problem (filtered during download)
- Renamed `download_dataset.py` to `download_code_contests.py` for clarity
- Updated GRPOConfig to use new `epsilon` and `epsilon_high` parameters (replacing deprecated `clip_range`)
- Updated README to include DeepCoder dataset badge and usage instructions
- Added DeepCoder dataset citation to README

## v3.1.5

- Inspired by [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51): adjusted two key training hyperparameters:
  - Set KL divergence coefficient `beta=0` in `train_rl.py`, allowing the policy to explore freely without being anchored to the base model during training;
  - Increased `clip_range` from the default `0.2` to `0.3`, allowing larger policy update steps when correct solutions are found.
- Made TinyLoRA SVD rank `rank` a configurable CLI argument (`--rank N`, default: `2`), enabling fine-grained control over the capacity/stability tradeoff.
- Updated README, usage guides, and output/README to document the `--rank` argument and added DeepCoder citation.

## v3.1

- Fix issue 001. The fix ensures proper handling of dequantized weight tensors in a distributed data parallel (DDP) setting, eliminating the CUBLAS error. 
- Added explicit configurable controls section in README, organized by **five system blocks**:
  - reward system
  - data selection
  - TinyLoRA architecture
  - GRPO optimization
  - prompt construction
- Clarified quantization compatibility and checkpoint metadata behavior:
  - default training uses 4-bit quantized loading
  - `--no_quant` enables BF16 path
  - checkpoint stores `is_quantized` for safe loading
- Improved README information architecture:
  - complete quick start in homepage
  - move detailed changelog and operational details to docs

## v3.0

- Upgraded core implementation to support RL training with TinyLoRA global parameters.
- Added end-to-end path for:
  - dataset download
  - training
  - validation
  - baseline and checkpoint testing
- Introduced `verify_pipeline.py` for pipeline sanity checks.

## v2.5

- Switched to Qwen2.5-Coder series for code-generation alignment.
- Refined model wrapping behavior for TinyLoRA-injected linear layers.
- Improved default training stability under small-sample settings.

## v2.0

- Initial public structure of TinyLoRA-Qwen-Coder project.
- Added first runnable scripts for data processing and basic evaluation.
