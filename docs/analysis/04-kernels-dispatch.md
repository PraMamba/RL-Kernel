# 内核调度层（Kernels Dispatch）

> 源码位置：`rl_engine/kernels/`
> 文件数：3 个（含 1 个空 `__init__.py`） | 总行数：278 行
> 最后更新：2026-06-13

## 1. 模块职责概述

内核调度层是算子实现层与上层执行器之间的中间层，提供两个核心组件：(1) `KernelRegistry` — 根据硬件平台和算子类型自动选择最优后端实现的中央调度器；(2) `SamplerBackend` — 跨硬件的统一采样接口（FlashInfer / AITER / PyTorch 回退）。

在整体流水线中，内核调度层是上层代码访问高性能算子的唯一入口。`RolloutExecutor._prepare_kernels` 和训练循环通过 `kernel_registry.get_op("logp")` 获取最优算子实例。

上游：依赖 `platforms/device`（`device_ctx`）、`utils/logger`、`ops/base`（`_C`, `_EXT_AVAILABLE`）。

下游：被 `executors/rollout.py`、所有 benchmark 文件、示例代码和测试文件消费。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 0 | 空包标记 |
| `registry.py` | 176 | 中央内核注册表、硬件感知优先级路由 |
| `sampling.py` | 102 | 统一采样后端（FlashInfer/AITER/PyTorch） |

## 3. 核心数据结构与接口

### _KernelEnumMeta

- **类型**：metaclass（`EnumMeta` 子类）
- **职责**：为 `OpBackend` 枚举提供友好的 KeyError 消息

### OpBackend

- **类型**：enum（使用 `_KernelEnumMeta` 元类）
- **字段数**：10 个
- **关键字段**：
  - `CUDA_FUSED_LOGP_SM90`: `"rl_engine.kernels.ops.cuda.loss.logp.FusedLogpSM90Op"` — TMA 加速 LogP
  - `CUDA_FUSED_LOGP_GENERIC`: `"rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp"` — 通用 CUDA LogP
  - `TRITON_GRPO_LOSS`: `"rl_engine.kernels.ops.triton.triton_grpo_loss.TritonGRPOLossOp"` — Triton GRPO Loss
  - `PYTORCH_NATIVE`: `"rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp"` — PyTorch 回退
  - `FLASH_ATTN`: 外部 FlashAttention 库包装器
  - `FLASHINFER`: 指向 `rl_engine.kernels.ops.cuda.flashinfer.FlashInferOp`（**未实现占位符，加载将失败回退**）
  - `TRITON_GENERIC`: 指向 `rl_engine.kernels.ops.triton.generic.TritonOp`（**未实现占位符，加载将失败回退**）
  - `ROCM_AITER` / `ROCM_CK` / `PYTORCH_GRPO_LOSS`
- **值格式**：每个值是完整的 Python 模块路径 + 类名，用于 `importlib` 动态加载

### KernelRegistry

- **类型**：class
- **字段数**：3 个
- **关键字段**：
  - `_instance_cache`: `Dict[str, Any]` — 已实例化的后端缓存（按 `OpBackend.name` 索引）
  - `_failed_backends`: `Set[str]` — 加载失败的后端名称集合（避免重复尝试）
  - `_priority_map`: `Dict[str, Dict[str, List[OpBackend]]]` — 三层嵌套映射 `{platform: {op_type: [backends...]}}`
- **关键方法**：
  - `get_op(op_type: str)` → `Any` — 核心调度逻辑，按优先级尝试加载后端
  - `_load_backend(backend: OpBackend)` → `Optional[Type]` — 动态导入并返回 Op 类
  - `_adjust_priority_for_hardware()` → `None` — SM90+ 设备自动提升 TMA 内核优先级

### SamplerBackend

- **类型**：class（`torch.nn.Module` 子类）
- **字段数**：2 个
- **关键字段**：
  - `backend`: `str` — 检测到的后端名称（`"flashinfer"` 或 `"aiter"`）
  - `flashinfer`: 运行时加载的 FlashInfer 模块引用
- **关键方法**：
  - `sample(logits, top_k, top_p, temperature, deterministic)` → `Tensor` — 统一采样接口
  - `compute_logp(logits, token_ids)` → `Tensor` — 分块 LogP 计算

## 4. 算法与逻辑详解

### KernelRegistry.get_op 调度流程

1. **平台选择** (`registry.py:127-132`): 根据 `device_ctx.is_rocm`/`device_ctx.device_type` 确定平台键（`"rocm"` / `"cuda"` / `"cpu"`）
2. **候选获取** (`registry.py:133`): 从 `_priority_map[platform][op_type]` 获取有序后端列表，未知 op_type 默认为 `[PYTORCH_NATIVE]`
3. **缓存命中** (`registry.py:136-137`): 若后端已在 `_instance_cache` 中，直接返回
4. **失败跳过** (`registry.py:139-140`): 若后端在 `_failed_backends` 中，跳过
5. **动态加载** (`registry.py:142-152`): 调用 `_load_backend` 进行 `importlib.import_module`，成功则实例化并缓存，失败则加入 `_failed_backends`
6. **全部失败** (`registry.py:154`): 抛出 `RuntimeError`

### _adjust_priority_for_hardware 流程

1. **设备检测** (`registry.py:96-104`): 仅 CUDA 平台执行；读取 `torch.cuda.get_device_capability()`
2. **TMA 编译检查** (`registry.py:105`): 检查 `_C` 扩展是否编译了 `fused_logp_sm90` 函数
3. **SM90/100/120 提升** (`registry.py:107-114`): 若 TMA 已编译且 CC major 为 9/10/12（Hopper/Blackwell/...），将 `CUDA_FUSED_LOGP_SM90` 插入 `logp` 优先级列表首位

### SamplerBackend.sample 采样流程

1. **温度缩放** (`sampling.py:47-48`): `logits /= temperature`
2. **FlashInfer 路径** (`sampling.py:50-65`): 调用 FlashInfer 的 `top_k_renorm_probs` 和 `top_p_sampling_from_probs`
3. **AITER 路径** (`sampling.py:67-70`): 占位 `pass`（未实现）
4. **PyTorch 回退** (`sampling.py:72-78`): top-k 过滤 → softmax → multinomial

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `kernel_registry` | `KernelRegistry()` 单例 | `registry.py:176` | 全局内核注册表 |
| `chunk_size` | `4096` | `sampling.py:90` | SamplerBackend.compute_logp 的分块大小 |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  调用方      │────▶│KernelRegistry│────▶│  Op 实例     │
│ (RolloutExec,│     │              │     │              │
│  训练循环)   │     │ get_op("logp")│    │ 已缓存的后端 │
│              │     │ get_op("attn")│    │ 实例         │
│ 输入：op_type│     │ get_op       │     │              │
│ 输出：Op实例 │     │("grpo_loss") │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 输入契约

- `op_type` 必须是 `_priority_map` 中定义的键之一：`"logp"`, `"logp_indexed"`, `"logp_online"`, `"logp_online_indexed"`, `"attn"`, `"grpo_loss"`
- 未知 `op_type` 回退到 `[PYTORCH_NATIVE]`

### 输出保证

- 返回的 Op 实例在注册表生命周期内被缓存，相同 `op_type` 的多次调用返回相同实例
- 后端一旦加载失败，不会在同一注册表实例中重试

## 6. 关键设计决策与不变量

1. **每个 OpBackend 枚举值是完整的 Python 导入路径** — 而非类引用或工厂函数。原因：避免在注册表定义时触发所有后端模块的导入；`importlib.import_module` 仅在首次 `get_op` 时按需加载。

2. **实例缓存 + 失败后端集合** — 分开维护，而非统一的状态映射。原因：缓存存放成功实例用于快速返回，失败集合用于避免重复的 `importlib` 调用（失败 import 有 I/O 开销）。

3. **_load_backend 区分内部错误和后端缺失** — 当 `ImportError.name` 包含 `rl_engine` 但不是目标模块自身时，视为内部实现 bug 并抛出而非回退。原因：防止 rl_engine 内部的拼写错误被静默吞掉。

### 不变量

- `_priority_map` 的每个列表的最后一个元素必须是可在当前平台上始终加载成功的后端（通常是 `PYTORCH_NATIVE` 或 `PYTORCH_GRPO_LOSS`）
- `kernel_registry` 在模块导入时构造，内部调用 `_adjust_priority_for_hardware`

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | `SamplerBackend._detect_backend` 使用 `torch.version.hip` 而非 `device_ctx.is_rocm` | `sampling.py:18` | 两处硬件检测逻辑未统一 |
| 2 | `compute_logp` 的 `chunk_size=4096` 为硬编码常量 | `sampling.py:90` | 无法根据 GPU 显存动态调整 |
| 3 | `OpBackend.FLASHINFER` 和 `OpBackend.TRITON_GENERIC` 指向不存在的模块路径 | `registry.py:26,41` | 这些后端在优先级列表中占位但永远加载失败，回退到下一候选 |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_cpu_hal.py` | 1 | CPU 平台下注册表路由到 PyTorch 回退 |
| `tests/test_sampler_temperature.py` | 2 | FlashInfer 温度应用和幂等性 |
| `rl_engine/tests/test_dispatch.py` | 1 (`test_device_and_registry`) | 注册表初始化和 get_op 基本功能 |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `tests/test_op_accuracy.py` | 多个 | 通过 `kernel_registry.get_op("logp")` 间接测试注册表调度 |
| `tests/test_grpo_loss.py` | 1 (`test_registry_dispatches_grpo_loss`) | 验证注册表能调度到 GRPO Loss 算子 |
