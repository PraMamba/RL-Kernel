# CUDA/C++ 内核（csrc）

> 源码位置：`csrc/`
> 文件数：5 个 | 总行数：1,219 行
> 最后更新：2026-06-13

## 1. 模块职责概述

csrc 目录包含 RL-Kernel 的原生 CUDA/C++ 内核实现，通过 PyTorch 的 `CppExtension` / `CUDAExtension` 编译为 Python 扩展模块 `rl_engine._C`。核心提供两个算子族的 GPU 加速实现：(1) Fused LogP — 将 log_softmax + gather 融合为单一内核，支持 SM90 TMA 加速变体；(2) Prefix Shared Attention — 针对 GRPO 场景优化的前缀共享注意力内核。

在整体流水线中，csrc 是性能关键路径——当 `_EXT_AVAILABLE` 为 `True` 时，`KernelRegistry` 会优先使用这些编译内核替代纯 PyTorch/Triton 实现。

上游：无 Python 层依赖。通过 `setup.py` 编译为 `rl_engine._C`。

下游：被 `rl_engine/kernels/ops/base.py` 加载，被 `ops/cuda/` 各包装器类调用。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `ops.cpp` | 93 | pybind11 模块注册、前向声明、attention 包装函数 |
| `fused_logp_kernel.cu` | 588 | 通用 Fused LogP 内核（two-pass + online 变体） |
| `cuda/fused_logp_sm90.cu` | 124 | SM90 TMA 加速 Fused LogP 内核 |
| `cuda/attention/prefix_shared_attention.cu` | 331 | GRPO 前缀共享注意力 CUDA 内核 |
| `utils/tma_utils.cuh` | 83 | TMA（Tensor Memory Accelerator）工具函数头文件 |

## 3. 核心数据结构与接口

### pybind11 导出函数（ops.cpp）

| 函数名 | 签名概要 | 用途 |
|--------|---------|------|
| `fused_logp` | `(logits_2d, token_ids_1d)` → `Tensor` | 基础 Fused LogP |
| `fused_logp_sm90` | `(logits, labels)` → `Tensor` | SM90 TMA LogP |
| `fused_logp_forward_out` | `(logits, tokens, output)` → `Tensor` | 写入预分配输出 |
| `fused_logp_forward_fp32` | `(logits, tokens)` → `Tensor` | FP32 输出 |
| `fused_logp_forward_indexed_out` | `(logits, tokens, indices, output)` → `Tensor` | 仅更新指定行 |
| `fused_logp_forward_indexed_fp32` | `(logits, tokens, indices)` → `Tensor` | 索引 + FP32 |
| `fused_logp_forward_online_out` | `(logits, tokens, output)` → `Tensor` | Online（单趟）变体 |
| `fused_logp_forward_online_fp32` | `(logits, tokens)` → `Tensor` | Online + FP32 |
| `fused_logp_forward_online_indexed_out` | `(logits, tokens, indices, output)` → `Tensor` | Online + 索引 |
| `fused_logp_forward_online_indexed_fp32` | `(logits, tokens, indices)` → `Tensor` | Online + 索引 + FP32 |
| `prefix_shared_attention` | `(q, k, v)` → `Tensor` | 前缀共享注意力 |

### LogSumExpState（fused_logp_kernel.cu）

- **类型**：struct
- **字段数**：2 个
- **关键字段**：`max_val: float`、`sum_exp: float`
- **用途**：online LogSumExp 的中间状态，支持数值稳定的单趟归约

### FusedLogpOnlineLaunchVariant（fused_logp_kernel.cu）

- **类型**：enum class
- **值**：`kDefault`、`kSparseLargeRow`
- **用途**：online 内核在运行时根据行大小和稀疏度选择的启动策略

## 4. 算法与逻辑详解

### Fused LogP Two-Pass 算法（fused_logp_forward_kernel）

1. **第一趟 — Max 归约** (`fused_logp_kernel.cu:~60-80`): 每个 block 处理一行 logits，通过 `blockReduceMax` 找到行最大值
2. **第二趟 — LogSumExp + Gather** (`fused_logp_kernel.cu:~80-100`): 计算 `sum(exp(logits - max))`，然后 `logits[token_id] - max - log(sum_exp)`

### Fused LogP Online 算法（fused_logp_forward_online_kernel）

1. **单趟融合** (`fused_logp_kernel.cu:~120-180`): 使用 `LogSumExpState` 在单次遍历中同时维护 max 和 sum_exp
2. **Block 归约** (`fused_logp_kernel.cu:~180-200`): `blockReduceLogSumExp` 将多个 warp 的 state 合并
3. **选择行大小策略** (`fused_logp_kernel.cu:~230-280`): `select_fused_logp_online_launch_variant` 根据行字节数和稀疏密度选择 block size

### SM90 TMA Fused LogP 算法（fused_logp_sm90.cu）

1. **TMA 异步加载** (`fused_logp_sm90.cu:~20-50`): 使用 `tma_2d_g2s`（来自 `tma_utils.cuh`）将 logits tile 从全局内存异步加载到共享内存
2. **mbarrier 同步** (`fused_logp_sm90.cu:~50-60`): 使用 SM90 的 `mbarrier` 原语同步 TMA 加载完成
3. **CUB BlockReduce** (`fused_logp_sm90.cu:~60-80`): 使用 CUB 库的 BlockReduce 计算 max 和 sum
4. **LogP 输出** (`fused_logp_sm90.cu:~80-100`): `logits[token_id] - max - log(sum)` 写入输出

### Prefix Shared Attention 算法

1. **Swizzled 加载** (`prefix_shared_attention.cu:~100-150`): 使用 swizzle 模式将 K/V 从全局内存加载到共享内存，利用 bank 冲突优化
2. **WGMMA/MMA 计算** (`prefix_shared_attention.cu:~150-250`): 使用 `mma_m16n8k16` PTX 指令计算 Q·K^T 注意力分数
3. **在线 Softmax** (`prefix_shared_attention.cu:~250-300`): 使用 online softmax（每个 KV tile 更新 max 和 sum，不需要两趟）
4. **前缀广播** (`prefix_shared_attention.cu:~300-330`): K/V 在 `G` 个 query group 间共享加载，避免重复读取

### TMA 工具函数（tma_utils.cuh）

| 函数 | 用途 |
|------|------|
| `init_tensor_map<InType>` | 创建 CUTensorMap 描述符 |
| `mbarrier_init` / `mbarrier_arrive` / `mbarrier_arrive_expect_tx` / `mbarrier_wait` | mbarrier 同步原语 |
| `tma_2d_g2s` | 2D TMA 全局→共享内存异步拷贝 |

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `FUSED_LOGP_TWOPASS_BLOCK_SIZE` | 256（默认） | `fused_logp_kernel.cu` 宏 | Two-pass 内核的 block 大小 |
| `FUSED_LOGP_ONLINE_BLOCK_SIZE` | 128（默认） | `fused_logp_kernel.cu` 宏 | Online 内核的 block 大小 |
| `FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE` | 256（默认） | `fused_logp_kernel.cu` 宏 | 大词表稀疏场景的 block 大小 |
| `FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD` | 65536（默认） | `fused_logp_kernel.cu` 宏 | online 变体切换阈值（行字节数） |
| `TILE_V` | 4096 | `fused_logp_sm90.cu` 宏 | SM90 TMA tile 大小 |
| `WARP_SIZE` | 32 | `prefix_shared_attention.cu` | CUDA warp 大小 |
| `BLOCK_Q=64, BLOCK_KV=64, DIM=128, NUM_WARPS=4` | 硬编码 | `prefix_shared_attention.cu` | 注意力内核的 tile 尺寸 |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Python 层   │────▶│   _C 扩展    │────▶│  CUDA GPU    │
│              │     │ (pybind11)   │     │              │
│ logits_2d,   │     │              │     │ CUDA 内核    │
│ token_ids_1d │     │ 输入验证     │     │ 执行         │
│              │     │ 内核启动     │     │              │
│ 输出：logps  │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 输入契约

- Fused LogP: `logits` 为 contiguous 2D 张量 `(N, V)`；`token_ids` 为 1D long 张量 `(N,)`
- SM90 Fused LogP: `logits` 必须为 BF16 且 contiguous
- Prefix Shared Attention: `q` 为 BF16 `[bs, G, seq_q, 128]`；`k`/`v` 为 BF16 `[bs, seq_kv, 128]`

### 输出保证

- Fused LogP 输出形状为 `(N,)` 或与输入 logits 的前导维度一致
- SM90 变体使用 TMA 异步加载但输出同步可用
- Prefix Shared Attention 输出形状为 `[bs, G, seq_q, 128]`

## 6. 关键设计决策与不变量

1. **Two-pass 与 Online 双算法** — 提供两种 LogP 计算策略而非仅一种。原因：小词表（<65K token）下 two-pass 的 memory bandwidth 开销可接受且更简单；大词表下 online 变体减少一次完整的行扫描。

2. **编译期可调的 block size** — 通过预处理器宏（`FUSED_LOGP_TWOPASS_BLOCK_SIZE` 等）允许编译时自定义。原因：不同 GPU 架构的 SM 寄存器数量不同，block size 直接影响 occupancy。

3. **Prefix Shared Attention 仅支持 DIM=128** — head_dim 硬编码为 128。原因：MMA 指令（`mma_m16n8k16`）的 tile 尺寸与 DIM=128 完美匹配，支持其他尺寸需要额外的 tile 策略。

4. **SM90 条件编译** — `fused_logp_sm90.cu` 通过 `KERNEL_ALIGN_WITH_SM90` 宏和 `KERNEL_ALIGN_FORCE_SM90=1` 环境变量条件编译。原因：TMA 和 mbarrier 是 SM90+ 专有指令，编译到低版本架构会失败。

### 不变量

- 所有内核的 `token_ids` 值必须在 `[0, vocab_size)` 范围内
- SM90 内核仅在 `KERNEL_ALIGN_WITH_SM90` 宏定义时编译
- Prefix Shared Attention 仅在 `KERNEL_ALIGN_WITH_CUDA` 宏定义时编译

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | `prefix_shared_attention` 仅支持 BF16 + DIM=128 | `prefix_shared_attention.cu` | 其他精度/维度配置回退到 Python 层 |
| 2 | 无 ROCm/HIP 变体的内核实现 | `csrc/` | AMD GPU 回退到 Triton 或 PyTorch |
| 3 | SM90 TMA 内核需要显式设置 `KERNEL_ALIGN_FORCE_SM90=1` 编译 | `setup.py` | 默认构建不包含 TMA 内核 |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_op_accuracy.py` | 15 | Fused LogP 所有变体（out/fp32/indexed/online）的精度验证 |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `tests/test_rl_kernel_loss_step.py` | 1 (`test_minimal_rl_loss_step_fused_logp_candidate_cuda`) | 通过 kernel_registry 调度到 fused logp |
| `benchmarks/benchmark_rl_kernels.py` | 多个 | 性能基准使用 fused logp 内核 |
| `benchmarks/benchmark_attention.py` | 1 | PrefixSharedAttentionOp 性能基准 |
