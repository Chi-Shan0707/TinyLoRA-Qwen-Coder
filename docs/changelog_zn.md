# 更新日志（中文）

## v3.1.5

- 受 [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) 启发，调整了两项关键训练超参数：
  - 在 `train_rl.py` 中将 KL 散度系数 `beta` 置为 `0`，允许策略在探索阶段完全脱离基座约束，更自由地寻找正确解法；
  - 将 `clip_range` 从默认值 `0.2` 扩大至 `0.3`，允许模型在遇到正确解时以更大步长更新策略。
- 将 TinyLoRA SVD 秩 `rank` 提升为可配置 CLI 参数（`--rank N`，默认值 `2`），支持对容量/稳定性权衡进行精细调控。
- 同步更新 README、使用指南与 output/README，补充 `--rank` 参数说明及 DeepCoder 引用。

## v3.1

- 处理了issue 001。该修复方案确保在分布式数据并行（DDP，Distributed Data Parallel）环境下对反量化后的权重张量进行正确处理，从而消除CUBLAS错误。
- 在 README 中新增“可调控参数”总览，并改为 **5 个系统块**组织：
  - 奖励系统
  - 数据选择
  - TinyLoRA 架构
  - GRPO 优化配置
  - Prompt 模板系统
- 明确量化兼容与 checkpoint 元信息机制：
  - 默认训练为 4-bit 量化路径
  - `--no_quant` 可切换到 BF16 路径
  - checkpoint 中写入 `is_quantized` 用于安全加载
- 调整文档职责边界：
  - README 首页保留完整快速开始
  - 详细变更与操作细节迁移到 docs

## v3.0

- 核心实现升级为 TinyLoRA + RL（GRPO）训练路径。
- 补齐从数据下载到训练、验证、对比测试的端到端脚本链路。
- 新增 `verify_pipeline.py` 用于流水线快速自检。

## v2.5

- 基座切换为 Qwen2.5-Coder 系列，更贴合代码生成任务。
- 优化 TinyLoRA 注入线性层的封装行为。
- 提升小样本场景下训练稳定性。

## v2.0

- 项目初版公开结构落地。
- 提供最早的数据处理与基础评测脚本。
