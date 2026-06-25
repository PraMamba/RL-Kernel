# 平台层（Platforms）

> 源码位置：`rl_engine/platforms/`
> 文件数：3 个 | 总行数：167 行
> 最后更新：2026-06-13

## 1. 模块职责概述

平台层是 RL-Kernel 的最底层基础设施，提供两个核心功能：(1) 定义跨模块共享的硬件类型与配置枚举常量；(2) 在引擎加载时检测当前硬件后端（NVIDIA CUDA / AMD ROCm / CPU），构建全局设备上下文单例。

在整体流水线中，平台层处于依赖树的叶子位置——被 `utils/logger`、`kernels/registry`、`kernels/sampling`、`executors` 等上层模块广泛导入，但自身仅依赖标准库和 PyTorch。

上游消费者：无（叶子模块）。

下游消费者：`utils/logger.py` 导入 `DeviceType`；`kernels/registry.py` 导入 `device_ctx`；`kernels/sampling.py` 导入 `constants`；所有 benchmark 和测试文件导入 `device_ctx`。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 0 | 空包标记 |
| `constants.py` | 112 | 定义 12 个枚举类和 `Constants` 聚合单例 |
| `device.py` | 55 | 硬件检测与全局设备上下文 `device_ctx` |

## 3. 核心数据结构与接口

### DeviceType

- **类型**：enum
- **字段数**：6 个
- **关键字段**：
  - `CUDA`: `"cuda"` — NVIDIA GPU
  - `ROCM`: `"rocm"` — AMD GPU
  - `NPU`: `"npu"` — 华为昇腾
  - `TPU`: `"tpu"` — Google TPU
  - `XPU`: `"xpu"` — Intel GPU
  - `CPU`: `"cpu"` — CPU 回退

### BackendLib

- **类型**：enum
- **字段数**：5 个
- **关键字段**：
  - `FLASHINFER`: `"flashinfer"` — NVIDIA 高性能采样
  - `AITER`: `"aiter"` — AMD ROCm 算子
  - `TRITON`: `"triton"` — Triton JIT 编译
  - `CANN`: `"cann"` — 华为 NPU
  - `NATIVE`: `"native"` — PyTorch 原生回退

### PrecisionType / SamplingMethod / MemoryFormat / OperatorFusionLevel / KernelOptimizationLevel / LoggingLevel / ProfilingMode / DistributedStrategy / CheckpointFormat / ActivationFunction

- **类型**：enum（共 10 个附加枚举）
- **用途**：分别定义数值精度、采样策略、内存格式、算子融合级别、内核优化级别、日志级别、性能分析模式、分布式策略、检查点格式、激活函数

### Constants

- **类型**：class
- **字段数**：12 个
- **关键字段**：每个字段对应一个枚举类的引用（如 `self.DeviceType = DeviceType`）
- **关键方法**：无（纯属性容器）

### DeviceContext

- **类型**：class
- **字段数**：4 个
- **关键字段**：
  - `device`: `torch.device` — 检测到的 PyTorch 设备
  - `is_rocm`: `bool` — 是否为 AMD ROCm 环境
  - `backend_version`: `str` — CUDA/HIP 版本字符串
  - `device_type`: `str` — 设备类型字符串值
- **关键方法**：
  - `get_preferred_dtype()` → `torch.dtype` — 根据硬件返回最优数据类型（ROCm 返回 bfloat16，CUDA 返回 float16）

## 4. 算法与逻辑详解

### 硬件检测流程

1. **入口** (`device.py:18`): `DeviceContext.__init__` 通过 `torch.cuda.is_available()` 判断是否有 GPU
2. **CUDA 分支** (`device.py:26-42`): 若 GPU 可用，检查 `torch.version.hip` 区分 AMD ROCm 与 NVIDIA CUDA
3. **ROCm 路径** (`device.py:28-34`): 设置 `is_rocm=True`，`device_type="rocm"`，版本取自 `torch.version.hip`
4. **CUDA 路径** (`device.py:35-42`): 设置 `is_rocm=False`，`device_type="cuda"`，版本取自 `torch.version.cuda`
5. **CPU 回退** (`device.py:43-45`): 设置 `device_type="cpu"`，发出警告

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `constants` | `Constants()` 单例 | `constants.py:112` | 全局枚举常量访问点 |
| `device_ctx` | `DeviceContext()` 单例 | `device.py:55` | 全局设备上下文 |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  PyTorch     │────▶│ DeviceContext│────▶│ KernelRegistry /     │
│  Runtime     │     │              │     │ SamplerBackend /     │
│              │     │ 检测：CUDA/  │     │ RolloutExecutor      │
│ torch.cuda.  │     │ ROCm/CPU    │     │                      │
│ is_available │     │ 输出：device │     │ 接收：device_type,   │
│ torch.version│     │   device_type│     │   is_rocm, device    │
└──────────────┘     └──────────────┘     └──────────────────────┘
```

### 输入契约

- 依赖 `torch.cuda.is_available()` 和 `torch.version.hip/cuda` 正确反映硬件状态

### 输出保证

- `device_ctx.device` 始终是有效的 `torch.device`
- `device_ctx.device_type` 值域为 `{"cuda", "rocm", "cpu"}`
- `device_ctx` 在模块导入时构造，进程生命周期内不变

## 6. 关键设计决策与不变量

1. **模块级单例初始化** — `device_ctx` 和 `constants` 在各自模块导入时立即构造，不使用延迟初始化。原因：设备检测结果在进程生命周期内不变，尽早初始化避免首次调用时的不确定延迟。

2. **Constants 聚合类** — 使用一个 `Constants` 实例聚合所有枚举类型，而非直接导入各枚举。原因：提供类似命名空间的访问方式（`constants.BackendLib.FLASHINFER`），减少导入语句。

3. **ROCm 通过 HIP 版本检测** — 使用 `torch.version.hip is not None` 而非 `rocm` 关键字检测。原因：PyTorch 的 ROCm 构建通过 HIP 层实现，`torch.version.hip` 是官方的区分方式。

### 不变量

- `device_ctx.device_type` 的值必须是 `DeviceType` 枚举的 `.value` 之一
- 当 `device_ctx.is_rocm` 为 `True` 时，`device_ctx.device.type` 仍为 `"cuda"`（PyTorch 的 ROCm 后端共享 CUDA 设备类型）

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | NPU/TPU/XPU 枚举已定义但无对应的 `DeviceContext` 检测逻辑 | `constants.py:10-12` | 这些平台需要额外实现检测路径 |
| 2 | `get_preferred_dtype()` 硬编码 ROCm→BF16、CUDA→FP16 | `device.py:52` | 未考虑 Ampere+ 架构对 BF16 的原生支持 |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_cpu_hal.py` | 1 | HAL 路由和回退（验证无 GPU 时注册表能否工作） |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `rl_engine/tests/test_dispatch.py` | 1 (`test_device_and_registry`) | 通过 `device_ctx` 验证设备检测与注册表联动 |
| `tests/test_op_accuracy.py` | 多个 | 通过 `device_ctx` 检查 CUDA 可用性后路由到 GPU 测试 |
