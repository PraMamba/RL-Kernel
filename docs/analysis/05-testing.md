# 测试基础设施（Testing）

> 源码位置：`rl_engine/testing/`
> 文件数：3 个 | 总行数：368 行
> 最后更新：2026-06-13

## 1. 模块职责概述

测试基础设施层提供两个核心组件：(1) 一组纯 PyTorch 的参考算子实现（`reference_ops.py`），作为验证加速内核数值正确性的基准真值；(2) 确定性合成 RL 批次生成器（`rl_batch.py`），为内核测试和性能基准提供标准化的输入数据。

这不是一个仅用于测试的模块——它同时被生产代码依赖。`alignment/model_wrappers.py` 使用 `selected_logprobs_reference` 计算 logprobs；`executors/training_contract.py` 和 `executors/deepspeed_trainer.py` 使用 `SyntheticRLKernelBatch` 和参考算子。

上游消费者：无内部依赖（纯 PyTorch 实现）。

下游消费者：`alignment/model_wrappers.py`、`executors/training_contract.py`、`executors/deepspeed_trainer.py`、所有 benchmark 文件、大部分测试文件。被 9 个外部文件导入，是第二被依赖的模块。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 27 | 聚合导出：`__all__` 包含 9 个公共符号 |
| `reference_ops.py` | 142 | 7 个参考算子函数 |
| `rl_batch.py` | 199 | SyntheticRLKernelBatch dataclass 和工厂函数 |

## 3. 核心数据结构与接口

### SyntheticRLKernelBatch

- **类型**：frozen dataclass
- **字段数**：11 个
- **关键字段**：
  - `input_ids`: `Tensor` `(B, total_seq_len)` — 完整序列（prompt + completion）
  - `attention_mask`: `Tensor` `(B, total_seq_len)` — 注意力掩码
  - `prompt_mask`: `Tensor` `(B, total_seq_len)` — prompt 区域掩码
  - `completion_mask`: `Tensor` `(B, completion_len)` — completion 有效 token 掩码
  - `token_ids`: `Tensor` `(B, completion_len)` — completion 区域的 token IDs
  - `rewards`: `Tensor` `(B,)` — 每序列奖励
  - `advantages`: `Tensor` `(B, completion_len)` — 每 token 优势
  - `old_logps`: `Tensor` `(B, completion_len)` — 旧策略 log-probabilities
  - `ref_logps`: `Tensor` `(B, completion_len)` — 参考策略 log-probabilities
  - `valid_indices`: `Tensor | None` — completion_mask 中非零元素的 flat 索引
  - `metadata`: `dict` — 批次生成参数快照
- **关键方法**：
  - `batch_size` / `total_seq_len` / `prompt_len` / `completion_len` — 形状属性
  - `flat_completion_mask` / `flat_token_ids` — 展平视图
  - `dense_completion_values(values)` → `Tensor` — 验证 values 的前两维与批次匹配
  - `compact_completion_values(values)` → `Tensor` — 仅提取 active token 的 values
  - `compact_token_ids()` → `Tensor` — 仅提取 active token 的 IDs
  - `benchmark_metadata()` → `dict` — 返回可序列化的基准测试元数据

### 参考算子函数

| 函数 | 签名 | 职责 |
|------|------|------|
| `selected_logprobs_reference` | `(logits, token_ids, mask?, temperature?, output_dtype?)` → `Tensor` | 计算 selected-token log-probabilities，支持掩码和温度缩放 |
| `masked_sum` | `(values, mask?)` → `Tensor` | 掩码求和 |
| `active_token_count` | `(mask?, values?)` → `Tensor` | 活跃 token 计数 |
| `masked_mean` | `(values, mask?, eps?)` → `Tensor` | 掩码均值 |
| `compute_policy_ratio` | `(current_logps, old_logps, mask?)` → `Tensor` | 策略比率 exp(current - old) |
| `compute_reference_kl` | `(current_logps, ref_logps, mask?)` → `Tensor` | k3 KL 估计器 exp(d) - d - 1 |
| `summarize_kernel_drift` | `(candidate, reference, mask?)` → `dict` | 候选与参考的最大/平均绝对误差 |

## 4. 算法与逻辑详解

### selected_logprobs_reference 算法

1. **输入验证** (`reference_ops.py:24-32`): 验证 temperature > 0，logits 前导形状匹配 token_ids 形状
2. **掩码处理** (`reference_ops.py:34-38`): 若有 mask，将 inactive 位置的 token_ids 替换为 0（避免越界）
3. **温度缩放** (`reference_ops.py:40`): `logits.float() / temperature`
4. **LogSoftmax + Gather** (`reference_ops.py:41-42`): `log_softmax` 后 `gather` 选择目标 token 的 logp
5. **掩码清零** (`reference_ops.py:44-45`): inactive 位置的 logp 设为 0.0

### make_synthetic_rl_kernel_batch 数据生成

1. **参数验证** (`rl_batch.py:100-112`): 验证所有参数的有效范围
2. **确定性生成** (`rl_batch.py:118-119`): 使用固定 seed 的 `torch.Generator`
3. **共享 prompt** (`rl_batch.py:129-139`): 每个 prompt 生成一次，通过 `repeat_interleave` 复制到同组的 samples_per_prompt 个序列
4. **稀疏 completion mask** (`rl_batch.py:146-153`): 按 `valid_density` 比例随机选择 active token 位置
5. **valid_indices** (`rl_batch.py:170`): 预计算 flat completion_mask 的非零索引

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `eps` (masked_mean) | `1e-8` | `reference_ops.py:74` | 分母钳位避免除零 |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  调用方参数   │────▶│make_synthetic_rl │────▶│SyntheticRLKernel │
│              │     │_kernel_batch     │     │Batch             │
│ num_prompts, │     │                  │     │                  │
│ samples_per_ │     │ 确定性生成      │     │ input_ids,       │
│ prompt, ...  │     │ 共享 prompt     │     │ completion_mask, │
│              │     │ 稀疏 mask       │     │ token_ids, ...   │
└──────────────┘     └──────────────────┘     └──────────────────┘
```

### 输入契约

- `selected_logprobs_reference` 要求 `logits.shape[:-1] == token_ids.shape`
- `make_synthetic_rl_kernel_batch` 要求 `num_prompts > 0`、`vocab_size > 1`、`0 <= valid_density <= 1`

### 输出保证

- `SyntheticRLKernelBatch` 对相同参数（含 `seed`）产生完全相同的数据
- 同组序列（同一 prompt）的 `input_ids[:, :prompt_len]` 完全一致

## 6. 关键设计决策与不变量

1. **参考算子强制 FP32 计算** — `selected_logprobs_reference` 在 log_softmax 前将 logits 转为 `float()`。原因：作为数值基准，必须使用最高精度避免累积误差影响正确性判断。

2. **生产代码直接依赖 testing 模块** — `alignment/model_wrappers.py` 的 `selected_logprobs` 方法直接调用 `selected_logprobs_reference`。原因：参考实现同时是数值正确的生产回退路径。

3. **batch_size = num_prompts × samples_per_prompt** — 显式建模 GRPO 的 prompt-group 结构。原因：GRPO 训练需要同一 prompt 的多个 completion 共享 prompt prefix 并在组内归一化奖励。

### 不变量

- `SyntheticRLKernelBatch.metadata["batch_size"]` 始终等于 `num_prompts * samples_per_prompt`
- `valid_indices` 是 `flat_completion_mask` 的精确非零索引

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 无已知问题 | — | — | — |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_reference_ops.py` | 7 | 参考算子正确性、掩码行为、温度缩放、错误拒绝 |
| `tests/test_rl_batch_fixture.py` | 6 | 合成批次形状、确定性、共享 prompt、compact values、CUDA smoke |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `tests/test_grpo_loss.py` | 多个 | 使用 `make_synthetic_rl_kernel_batch` 和 `selected_logprobs_reference` |
| `tests/test_rl_kernel_loss_step.py` | 3 | 使用完整的 testing 模块验证端到端 loss 步骤 |
| `tests/test_alignment_model_wrappers.py` | 11 | 通过 `PolicyModelWrapper` 间接使用 `selected_logprobs_reference` |
