# 更新日志（中文）

## v4.0

- **`num_iterations` 从 1 提升到 4**：使 clip_high (DeepCoder 方法) 更有效：
  - 更多迭代次数允许更好地探索策略空间
  - 结合非对称截断 (epsilon=0.2, epsilon_high=0.5)，显著提升 Pass@1
- **切换到 DeepCoder-Preview-Dataset (lcbv5, 28 samples)**：
  - 使用来自 agentica-org/DeepCoder-Preview-Dataset 的高质量 lcbv5 子集
  - 大幅减少训练数据同时改善结果
- **性能提升**：
  - Pass@1 提升 **100%** (1.82% → 3.64%)
  - 编译成功率提升 **4.13%** (73.33% → 76.36%)
  - 平均分数提升 **5.03%** (0.4274 → 0.4489)
- **效率提升**：
  - 训练数据减少 **99.8%** (13328 → 28)
  - 训练时间减少 **73%** (4小时24分 → 1小时12分)
- 更新 README，添加 v4.0 结果和详细对比链接

## v3.5

- **移除 prompt 中的测试用例样例**：停止将 `input_output`/`public_tests` 拼接到聊天模板中：
  - 某些样本携带巨大的测试用例（200K+ 字符），导致 prompt 膨胀，训练时 OOM
  - 现在 prompt 只包含问题描述，不含样例
  - 此更改显著降低了训练时的内存占用
- **训练时限制测试用例数量**：
  - 在 `code_reward_func` 中限制为前 5 个测试用例，避免过多 subprocess fork（每次 fork 会创建大模型进程；过多 fork → OOM killer）
- **DeepCoder 数据加载优化**：
  - 加载时将 `input_output` 裁剪为前 5 个测试用例
  - 某些样本携带巨大测试用例导致多 GB 内存膨胀 → OOM

- **Clip High（来自 DeepCoder/DAPO 论文）**：在 GRPO 损失中实现非对称截断：
  - 添加 `epsilon=0.2`（下界：1 - 0.2 = 0.8）
  - 添加 `epsilon_high=0.5`（上界：1 + 0.5 = 1.5）
  - 与对称截断 [1-ε, 1+ε] 不同，Clip High 仅提高上界
  - 这鼓励更多探索，防止在找到正确解法时过早收敛
  - 详见 [DeepCoder 论文](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51)
- **DeepCoder 数据集支持**：新增对 [DeepCoder-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) 的支持：
  - 新增 `download_DeepCoder-Preview-Dataset.py` 脚本（parquet格式）用于下载和预处理 DeepCoder 数据集
  - 支持4个配置：codeforces, lcbv5, primeintellect, taco
  - 新增 `--dataset` CLI 参数，可选择 'code_contests'（默认）或 'deepcoder'
  - 修改奖励函数以支持两种数据集格式
  - DeepCoder 数据集要求每个问题至少有 5 个测试用例（下载时已过滤）
- 将 `download_dataset.py` 重命名为 `download_code_contests.py` 以明确区分
- 更新 GRPOConfig 使用新的 `epsilon` 和 `epsilon_high` 参数（替换已弃用的 `clip_range`）
- 更新 README，添加 DeepCoder 数据集徽章和使用说明
- 在 README 中添加 DeepCoder 数据集引用

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
