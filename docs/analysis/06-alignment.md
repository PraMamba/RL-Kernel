# 对齐层（Alignment）

> 源码位置：`rl_engine/alignment/`
> 文件数：2 个 | 总行数：174 行
> 最后更新：2026-06-13

## 1. 模块职责概述

对齐层提供 RLHF/GRPO 训练中策略模型（Policy）和参考模型（Reference）的标准化适配器。核心功能是将任意 HuggingFace 风格的语言模型包装为统一接口，提供 logits 提取和 selected-token logprobs 计算。

在整体流水线中，对齐层将原始模型输出（可能是 Tensor、dict、ModelOutput 对象、tuple 等多种格式）规范化为 GRPO loss 所需的 logprobs 输入。

上游：依赖 `testing/reference_ops.py`（`selected_logprobs_reference`）。

下游：被训练循环和 `tests/test_alignment_model_wrappers.py` 消费。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 14 | 导出 `PolicyModelWrapper`、`ReferenceModelWrapper`、`extract_logits` |
| `model_wrappers.py` | 160 | 模型适配器和 logits 提取函数 |

## 3. 核心数据结构与接口

### extract_logits

- **类型**：function
- **签名**：`extract_logits(model_output: Any) -> torch.Tensor`
- **职责**：从多种模型输出格式中提取 logits 张量
- **支持格式**：
  - `torch.Tensor` → 直接返回
  - `Mapping`（含 `"logits"` 键）→ 返回 `mapping["logits"]`
  - 含 `.logits` 属性的对象 → 返回 `.logits`
  - `tuple` / `list` → 递归搜索第一个 ndim ≥ 2 的 Tensor 元素

### PolicyModelWrapper

- **类型**：class（`torch.nn.Module` 子类）
- **字段数**：1 个
- **关键字段**：
  - `model`: `torch.nn.Module` — 被包装的原始模型
- **关键方法**：
  - `forward(input_ids, **model_kwargs)` → `Any` — 直接委托给内部模型
  - `forward_logits(input_ids, **model_kwargs)` → `Tensor` — `forward` + `extract_logits`
  - `selected_logprobs(input_ids, token_ids, *, mask, logits_start, logits_end, temperature, output_dtype, **model_kwargs)` → `Tensor` — 前向 + logits 切片 + `selected_logprobs_reference`

### ReferenceModelWrapper

- **类型**：class（`PolicyModelWrapper` 子类）
- **字段数**：继承 `model` 字段
- **关键方法**：
  - `__init__(model, *, freeze=True, eval_mode=True)` — 构造时冻结参数并切换到 eval 模式
  - `freeze()` → `Self` — 将所有参数的 `requires_grad` 设为 `False`
  - `forward_logits(input_ids, **model_kwargs)` → `Tensor` — 在 `torch.no_grad()` 上下文中执行
  - `selected_logprobs(...)` → `Tensor` — 在 `torch.no_grad()` 上下文中执行

## 4. 算法与逻辑详解

### extract_logits 搜索策略

1. **直接张量** (`model_wrappers.py:43-44`): 若 `model_output` 本身是 Tensor，直接返回
2. **Mapping 查找** (`model_wrappers.py:46-49`): 检查 `"logits"` 键
3. **属性查找** (`model_wrappers.py:51-53`): 检查 `.logits` 属性
4. **序列递归** (`model_wrappers.py:55-72`): 遍历 tuple/list 的每个元素递归调用，要求候选张量 ndim ≥ 2（排除 loss 等标量张量）

### _slice_logits 位置切片

1. **入口** (`model_wrappers.py:24-37`): 当提供 `logits_start` 或 `logits_end` 时，在倒数第二维（token 位置维度）执行切片
2. **用途**：支持 causal LM 中将 logits 按 completion 范围裁剪

### 关键常量与阈值

无硬编码常量。

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  原始 LLM    │────▶│PolicyModel / │────▶│  GRPO Loss Op    │
│              │     │ReferenceModel│     │                  │
│ 输出：Any    │     │Wrapper       │     │ 接收：per-token  │
│ (Tensor/dict/│     │              │     │ logprobs (fp32)  │
│ ModelOutput) │     │ extract_logits│    │                  │
│              │     │ selected_logp│     │                  │
└──────────────┘     └──────────────┘     └──────────────────┘
```

### 输入契约

- `extract_logits` 接受任意模型输出格式，但至少一个路径必须产生 ndim ≥ 2 的 Tensor
- `selected_logprobs` 要求 logits 的 token 维度与 `token_ids` 长度匹配

### 输出保证

- `PolicyModelWrapper` 保持模型的 `training` 状态和梯度跟踪
- `ReferenceModelWrapper` 的所有输出不跟踪梯度

## 6. 关键设计决策与不变量

1. **继承 + 覆盖而非组合** — `ReferenceModelWrapper` 继承 `PolicyModelWrapper`，覆盖 `forward_logits` 和 `selected_logprobs` 添加 `torch.no_grad()`。原因：两种包装器共享大部分逻辑（logits 提取、切片、logprobs 计算），仅在梯度行为上不同。

2. **selected_logprobs 委托给 testing 模块** — 直接调用 `selected_logprobs_reference` 而非内联实现。原因：参考实现是经过充分测试的数值基准，复用避免了重复代码和潜在的实现差异。

### 不变量

- `ReferenceModelWrapper` 构造后，`model.parameters()` 的所有元素的 `requires_grad` 为 `False`（除非 `freeze=False`）
- `extract_logits` 对无法提取 logits 的输入始终抛出 `TypeError`

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 无已知问题 | — | — | — |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_alignment_model_wrappers.py` | 11 | extract_logits 多格式支持、Policy/Reference 冻结行为、logprobs 与参考一致性、梯度跟踪 |

### 间接测试

无间接测试。
