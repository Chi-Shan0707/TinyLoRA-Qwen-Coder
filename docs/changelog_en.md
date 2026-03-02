# Changelog (English)

## v3.1

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
