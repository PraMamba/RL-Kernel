# 算子实现层（Kernels Ops）

> 源码位置：`rl_engine/kernels/ops/`
> 文件数：19 个（含 10 个空 `__init__.py`） | 总行数：1,465 行
> 最后更新：2026-06-13

## 1. 模块职责概述

算子实现层提供三个算子族（LogP、Attention、GRPO Loss）的多后端实现。每个算子族对应三个层级：CUDA 编译内核（`ops/cuda/`）、Triton JIT 内核（`ops/triton/`）、纯 PyTorch 回退（`ops/pytorch/`）。ROCm 后端（`ops/rocm/`）目前为占位目录。

在整体流水线中，算子层是被 `KernelRegistry` 动态加载的实现层——注册表根据硬件选择优先级实例化对应的 Op 类，上层代码通过统一接口调用。

上游消费者：`ops/base.py` 提供编译扩展 `_C` 的加载入口，被 `cuda/*` 各 Op 导入。

下游消费者：`kernels/registry.py` 通过 `importlib.import_module` 按需加载各 Op 类。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `base.py` | 17 | 加载编译的 C++ 扩展 `_C`，暴露 `_EXT_AVAILABLE` 标志 |
| `cuda/attention/__init__.py` | 9 | 导出 FlashAttentionOp、PrefixSharedAttentionOp |
| `cuda/attention/flash_attn.py` | 57 | FlashAttention CUDA 包装器 |
| `cuda/attention/prefix_shared_attn.py` | 70 | GRPO 前缀共享注意力 CUDA 内核包装器 |
| `cuda/loss/logp.py` | 142 | Fused LogP CUDA 内核包装器（SM90 + Generic 两个类） |
| `triton/triton_attn.py` | 495 | Triton JIT FlashAttention 前后向实现 |
| `triton/triton_grpo_loss.py` | 378 | Triton JIT GRPO Loss 前后向实现（含 autograd.Function） |
| `pytorch/loss/grpo_loss.py` | 194 | 纯 PyTorch GRPO Loss 参考实现 |
| `pytorch/loss/logp.py` | 103 | 纯 PyTorch LogP 参考实现 |
| 空 `__init__.py` | 0 × 10 | 包标记 |

## 3. 核心数据结构与接口

### NativeLogpOp（PyTorch 回退）

- **类型**：class
- **字段数**：0 个
- **关键方法**：
  - `apply(logits, token_ids)` → `Tensor` — 基准 selected-token LogP 提取
  - `apply_fp32(logits, token_ids)` → `Tensor` — 强制 FP32 输出
  - `out(logits, token_ids, output)` → `Tensor` — 写入预分配缓冲区
  - `indexed_out(logits, token_ids, row_indices, output)` → `Tensor` — 仅更新指定行
  - `indexed_fp32(logits, token_ids, row_indices)` → `Tensor` — 索引 + FP32
  - `online_out` / `online_fp32` / `online_indexed_out` / `online_indexed_fp32` — 与非 online 版本行为相同（PyTorch 回退中无区别）

### FusedLogpGenericOp（CUDA 通用）

- **类型**：class
- **字段数**：2 个（`_backend`: `_C` 模块引用，`op`: `_C.fused_logp` 函数）
- **关键方法**：与 `NativeLogpOp` 接口一致（`apply`、`apply_fp32`、`out`、`indexed_out` 等），但委托给 `_C` 编译内核
- **关键差异**：`_prepare_inputs` 将 logits 展平为 2D、token_ids 展平为 1D

### FusedLogpSM90Op（CUDA SM90+ TMA）

- **类型**：class
- **字段数**：1 个（`op`: `_C.fused_logp_sm90` 函数）
- **关键方法**：
  - `__call__(logits, labels)` → `Tensor` — 仅支持 BF16 连续张量

### FlashAttentionOp

- **类型**：class
- **字段数**：1 个（`op`: 外部 `flash_attn_func` 或 `_C.flash_attn_forward`）
- **关键方法**：
  - `__call__(q, k, v, dropout_p, softmax_scale, causal)` → `Tensor`
- **输入形状**：`(batch, seqlen, nheads, headdim)`

### PrefixSharedAttentionOp

- **类型**：class
- **字段数**：2 个（`has_hardware_op`: `bool`，`op`: 可选的 `_C.prefix_shared_attention`）
- **关键方法**：
  - `__call__(q, k, v)` → `Tensor`
- **输入形状**：q `[bs, G, seq_len_q, head_dim]`，k/v `[bs, seq_len_kv, head_dim]`
- **关键行为**：硬件路径仅在 `head_dim == 128` 且有 `_C` 扩展时启用，否则回退到 `F.scaled_dot_product_attention`

### NativeGRPOLossOp（PyTorch 回退）

- **类型**：class
- **字段数**：0 个
- **关键方法**：
  - `forward(current_logps, old_logps, ref_logps, rewards, completion_mask, *, clip_eps, beta, samples_per_prompt, group_boundaries, eps)` → `(loss, policy_loss, kl)` — 完整前向
  - `group_advantages(rewards, *, samples_per_prompt|group_boundaries, eps)` → `Tensor` — 组内奖励归一化
  - `expand_advantages(sample_advantages, completion_mask)` → `Tensor` — 序列级优势→token 级
  - `apply(current_logps, old_logps, ref_logps, advantages, completion_mask, *, clip_eps, beta)` → `(loss, policy_loss, kl)` — 从 per-token 优势计算

### TritonGRPOLossOp

- **类型**：class
- **字段数**：0 个
- **关键方法**：与 `NativeGRPOLossOp` 接口一致（`forward`、`group_advantages`、`apply`）
- **关键差异**：`group_advantages` 调用 Triton JIT 内核 `_group_norm_kernel`；`apply` 通过 `_GRPOLossFunction`（`torch.autograd.Function`）支持反向传播

### _GRPOLossFunction

- **类型**：`torch.autograd.Function`
- **关键方法**：
  - `forward(ctx, current_logps, old_logps, ref_logps, adv_seq, mask, completion_len, clip_eps, beta)` → `(loss, policy_loss, kl)` — 调用 `_grpo_fwd_kernel`
  - `backward(ctx, grad_loss, grad_policy, grad_kl)` → `(grad_cur, None×7)` — 调用 `_grpo_bwd_kernel`

## 4. 算法与逻辑详解

### GRPO Loss 前向算法（Triton）

1. **组内奖励归一化** (`triton_grpo_loss.py:30-55`): `_group_norm_kernel` 在每个 group（由 CSR 偏移 `bounds_ptr` 定义）内计算 mean 和 population std，输出 z-score 归一化的优势
2. **Token 并行前向** (`triton_grpo_loss.py:58-97`): `_grpo_fwd_kernel` 按 token 粒度并行计算 policy loss 和 KL 散度
   - `ratio = exp(current - old)`
   - `unclipped = ratio * advantage`
   - `clipped = clamp(ratio, 1-eps, 1+eps) * advantage`
   - `policy_term = -min(unclipped, clipped)`
   - `kl_term = exp(ref - current) - (ref - current) - 1`（k3 估计器）
3. **Block 归约** (`triton_grpo_loss.py:96-97`): 每个 Triton block 将 policy 和 kl 的 block-sum 写入 `partials` 缓冲区
4. **全局归约** (`triton_grpo_loss.py:180-183`): Python 侧对 partials 求和后除以 `num_active`

### GRPO Loss 反向算法（Triton）

1. **梯度计算** (`triton_grpo_loss.py:100-146`): `_grpo_bwd_kernel` 计算 `d(loss)/d(current_logps)`
   - Policy 梯度：选择未裁剪/裁剪分支的梯度（裁剪分支在 ratio 超出 (lo, hi) 范围时梯度为 0）
   - KL 梯度：`d(kl)/d(current) = 1 - exp(ref - current)`
   - 总梯度：`scale * (d_policy + beta * d_kl)`

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `_BLOCK` | `1024` | `triton_grpo_loss.py:11` | Triton GRPO 前后向内核的 block 大小 |
| `_MAX_GROUP_SIZE` | `1024` | `triton_grpo_loss.py:18` | 组内归一化的最大序列数上限 |
| `_EXT_AVAILABLE` | 运行时确定 | `base.py:13` | `_C` 编译扩展是否可用 |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ KernelRegistry│────▶│    Op 实例       │────▶│  调用方      │
│              │     │                  │     │ (训练循环)   │
│ 选择最优后端 │     │ CUDA / Triton /  │     │              │
│              │     │ PyTorch          │     │ 接收：logps, │
│              │     │                  │     │ loss 标量    │
│              │     │ 处理：logits→logp│     │              │
│              │     │       →loss      │     │              │
└──────────────┘     └──────────────────┘     └──────────────┘
```

### 输入契约

- LogP ops：`logits` 为 `(batch, seq, vocab)` 或 `(N, vocab)` 的浮点张量；`token_ids` 为 long 张量
- GRPO Loss ops：`current_logps`/`old_logps`/`ref_logps` 为 `(B, T)` 浮点张量；`completion_mask` 为 `(B, T)` bool/float
- Attention ops：CUDA 路径要求 FP16 或 BF16；PrefixSharedAttention CUDA 路径要求 BF16 且 head_dim=128

### 输出保证

- LogP ops 返回与 `logits.shape[:-1]` 相同形状的 log-probability 张量
- GRPO Loss ops 返回 `(loss, policy_loss, kl)` 三元组标量
- Triton GRPO Loss 的 `loss` 支持 autograd 反向传播；`policy_loss` 和 `kl` 标记为 non-differentiable

## 6. 关键设计决策与不变量

1. **统一的 API surface** — 所有后端的同一算子族暴露完全相同的方法签名（如 `NativeLogpOp.apply` 和 `FusedLogpGenericOp.apply` 参数一致）。原因：`KernelRegistry.get_op` 返回的实例可互换使用，调用方无需知道底层后端。

2. **PyTorch 回退的 online_* 方法直接委托非 online 版本** — `NativeLogpOp.online_out` 直接调用 `self.out`。原因：PyTorch 实现中 two-pass 和 online 无计算差异，但保留方法确保 API 兼容。

3. **Triton GRPO Loss 使用 autograd.Function 而非 torch.compile** — 手写前后向内核而非依赖编译器融合。原因：GRPO 的 clipped surrogate objective 涉及 `min(unclipped, clipped)` 的条件分支梯度，需要精确控制哪个分支的梯度传播。

4. **CSR 偏移格式的 group_boundaries** — Triton 归一化内核接受 CSR 风格的 `bounds_ptr[num_groups + 1]`，而非 group_id per-sequence。原因：CSR 格式允许每个 program_id 直接索引到自己的 group 范围，无需全局扫描。

### 不变量

- 所有 `_EXT_AVAILABLE` 为 `False` 时，CUDA Op 的 `__init__` 必须抛出 `RuntimeError`
- `TritonGRPOLossOp` 的所有方法要求 CUDA 张量（`is_cuda` 检查在入口处）

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | `_MAX_GROUP_SIZE = 1024` 硬限制 | `triton_grpo_loss.py:18` | samples_per_prompt > 1024 时无法使用 Triton GRPO Loss |
| 2 | `FlashAttentionOp` 尝试外部 `flash_attn` 库失败后再尝试 `_C`，但 `_EXT_AVAILABLE` 检查在两者之前 | `flash_attn.py:17-34` | 如果 `_C` 不可用但外部 `flash_attn` 可用，会被错误拒绝 |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_grpo_loss.py` | 19 | NativeGRPOLossOp 和 TritonGRPOLossOp 的前向/反向一致性 |
| `tests/test_op_accuracy.py` | 15 | NativeLogpOp 和 FusedLogpGenericOp 的精度和 API 变体覆盖 |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `tests/test_rl_kernel_loss_step.py` | 3 | 通过 `kernel_registry.get_op("logp")` 间接使用 logp ops |
| `tests/test_cpu_hal.py` | 1 | 验证 CPU 路径的 PyTorch 回退 |
