# 权重同步桥（Executors Bridge）

> 源码位置：`rl_engine/executors/bridge.py`
> 文件数：1 个 | 总行数：2,684 行
> 最后更新：2026-06-13

## 1. 模块职责概述

权重同步桥是 RL-Kernel 中规模最大的单一模块，负责在 GRPO 训练循环的训练 worker 和 rollout worker 之间实现安全、高效的模型权重传输。核心功能是定义一套形式化的 publish → import → acknowledge/reject → release 四阶段权重更新生命周期协议，并提供四种传输后端实现。

在整体流水线中，权重同步桥是连接训练阶段（产出新权重）和 rollout 阶段（消费权重进行推理采样）的关键基础设施。训练 worker 通过 `WeightPublisher.publish` 发布权重清单，rollout worker 通过 `WeightConsumer.import_update` 导入并安装。

上游：仅依赖 `utils/logger`。

下游：被 `executors/rollout.py`、`executors/deepspeed_trainer.py`、`executors/ray_actor_manager.py`、所有 vLLM 适配器类消费。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `bridge.py` | 2,684 | 完整的权重同步协议、4 种传输后端、5 种 vLLM 适配器、工具函数 |

## 3. 核心数据结构与接口

### TensorDescriptor

- **类型**：frozen dataclass
- **字段数**：8 个
- **关键字段**：`name: str`、`shape: tuple[int, ...]`、`dtype: str`、`stride: tuple[int, ...]`、`device: str`、`numel: int`、`nbytes: int`、`sha256: str`
- **关键方法**：`from_tensor(name, tensor)` → `TensorDescriptor` — 从实际张量提取元数据

### WeightUpdateManifest

- **类型**：frozen dataclass
- **字段数**：8 个
- **关键字段**：
  - `update_id: str` — UUID 唯一标识
  - `source_worker: str` — 发布方 worker 名称
  - `source_rank: int` — 发布方 rank
  - `weight_version: int` — 单调递增的版本号
  - `transport: str` — 传输协议名称
  - `tensors: Mapping[str, TensorDescriptor]` — 张量清单
  - `created_at: float` — 发布时间戳（`time.perf_counter()` 生成）
  - `metadata: Mapping[str, Any]` — 传输特定元数据

### WeightLayout

- **类型**：frozen dataclass
- **字段数**：8 个
- **关键字段**：`kind`（"full-state" 或 "replicated"）、`world_size`、`rank`、`tensor_parallel_size`、`data_parallel_size`（验证 ≥ 1）、`zero_stage`、`node_count`、`rdma_enabled`
- **关键方法**：`validate_supported()` — 检查布局是否被当前传输支持（tensor_parallel != 1、multi-node、data_parallel_size < 1 会被阻断）

### WeightPublisher / WeightConsumer / WeightBridge / WeightInstallAdapter（Protocols）

- `WeightPublisher`: `publish(model, *, weight_version, metadata)` → `WeightUpdateManifest` + `release(update_id)`
- `WeightConsumer`: `import_update(manifest)` → `Mapping[str, Tensor]` + `acknowledge(update_id)` + `reject(update_id, reason)` + `release(update_id)`
- `WeightBridge`: 同时实现 Publisher + Consumer
- `WeightInstallAdapter`: `install(manifest, tensors)` + `release(update_id)` — vLLM 侧的张量安装适配器

### 错误层级

- `WeightBridgeError` — 基类
  - `WeightBridgeUnavailableError` — 传输不可用（运行时环境不支持）
  - `WeightManifestValidationError` — 清单校验失败（元数据不一致）
  - `WeightUpdateRejectedError` — 更新被拒绝

### 传输后端

| 类 | transport 值 | 机制 | 零拷贝 |
|----|-------------|------|--------|
| `LocalTensorCopyBridge` | `"local-clone"` | `state_dict.clone()` | 否 |
| `SharedMemoryTensorBridge` | `"shared-memory"` | Python `multiprocessing.SharedMemory` | 是（CPU） |
| `CUDAVMMTensorBridge` | `"cuda-vmm"` | CUDA VMM + POSIX fd + DLPack | 是（GPU） |
| `IPCWeightBridge` | `"cuda-ipc"` | PyTorch `reduce_tensor` + CUDA IPC | 是（GPU） |

### vLLM 适配器

| 类 | 安装路径 | 零拷贝 |
|----|---------|--------|
| `VLLMWeightInstallAdapter` | 能力协商（install_callable / request_builder / 属性探测） | 取决于后端 |
| `VLLMIPCWeightUpdateRequestBuilder` | 构建 `update_weights({"update_info": ...})` 请求 | CUDA IPC |
| `VLLMInProcessWeightReloadAdapter` | `reload_weights` / `collective_rpc("reload_weights")` | 否 |
| `VLLMCheckpointWeightReloadAdapter` | `reload_weights(weights_path=...)` | 否 |
| `VLLMCUDAVMMExternalStorageAdapter` | `apply_model` + parameter.data 替换 | 是（GPU） |

## 4. 算法与逻辑详解

### 权重更新生命周期（以 LocalTensorCopyBridge 为例）

1. **publish** (`bridge.py:1527-1571`): 克隆 model.state_dict()，计算每个张量的 SHA-256，生成 UUID，创建 `WeightUpdateManifest`，存入 `_updates` 字典
2. **import_update** (`bridge.py:1573-1590`): 验证 manifest 一致性，克隆存储的张量返回给消费方
3. **acknowledge** (`bridge.py:1592-1607`): 验证 import 已成功完成，重新验证张量校验和，更新状态为 "acknowledged"
4. **reject** (`bridge.py:1609-1614`): 标记更新为 "rejected"
5. **release** (`bridge.py:1616-1623`): 清除张量数据，释放内存

### CUDAVMMTensorBridge 发布流程

1. **张量打包** (`bridge.py:1477-1505`): `_pack_tensors_for_cuda_vmm` 将所有 CUDA 张量按 256 字节对齐排列，计算总字节数
2. **VMM 分配** (`bridge.py:2055-2056`): 通过 `_CUDAVMMDriverBackend.create_allocation` 调用 cuMemCreate + cuMemAddressReserve + cuMemMap + cuMemSetAccess + cuMemExportToShareableHandle
3. **数据复制** (`bridge.py:2060-2069`): 通过 cuMemcpyDtoDAsync 将每个张量异步复制到 VMM 分配的地址空间
4. **同步事件** (`bridge.py:2070-2072`): 创建 IPC 可用的 CUDA Event 并录制到当前 stream
5. **fd 代理** (`bridge.py:2074`): 启动 Unix Domain Socket 监听线程，通过 SCM_RIGHTS 向消费方发送 POSIX fd

### CUDAVMMTensorBridge 导入流程

1. **连接代理** (`bridge.py:2310-2313`): 连接 publisher 的 Unix Domain Socket，通过 SCM_RIGHTS 接收 POSIX fd
2. **VMM 导入** (`bridge.py:2314`): 通过 cuMemImportFromShareableHandle + cuMemMap 映射到本地地址空间
3. **DLPack 张量** (`bridge.py:2317-2327`): 通过 `_DLPackOwner` 构建 DLPack capsule，`torch.utils.dlpack.from_dlpack` 转为 PyTorch 张量
4. **校验验证** (`bridge.py:2328-2338`): 比较每个张量的 TensorDescriptor（含 SHA-256）

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `_BRIDGE_METADATA_KEY` | `"weight_bridge"` | `bridge.py:24` | manifest.metadata 中桥接专用键 |
| `_SHARED_MEMORY_FORMAT` | `"python-multiprocessing-shared-memory-v1"` | `bridge.py:25` | 共享内存元数据格式标识 |
| `_CUDA_IPC_FORMAT` | `"pytorch-cuda-ipc-reduce-tensor-v1"` | `bridge.py:26` | CUDA IPC 元数据格式标识 |
| `_CUDA_VMM_FORMAT` | `"cuda-vmm-posix-fd-v1"` | `bridge.py:27` | CUDA VMM 元数据格式标识 |
| `_CUDA_VMM_TENSOR_ALIGNMENT` | `256` | `bridge.py:28` | VMM 张量地址对齐字节数 |
| `_SUPPORTED_LAYOUT_KINDS` | `{"full-state", "replicated"}` | `bridge.py:29` | 支持的权重布局类型 |

## 5. 数据流（输入/输出）

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Training Worker  │────▶│  WeightBridge    │────▶│ Rollout Worker   │
│                  │     │                  │     │                  │
│ model.state_dict │     │ publish:         │     │ import_update:   │
│ → publish()     │     │  clone/VMM/IPC   │     │  attach/DLPack   │
│                  │     │  SHA-256 签名   │     │  SHA-256 验证    │
│ 输出：manifest  │     │  fd broker      │     │  acknowledge()   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                         │
                                                  ┌──────▼─────────┐
                                                  │ vLLM Adapter   │
                                                  │                │
                                                  │ install tensors│
                                                  │ into engine    │
                                                  └────────────────┘
```

### 输入契约

- `publish` 要求 `weight_version` 单调递增
- `publish` 要求 `model.state_dict()` 产出至少一个张量
- `import_update` 的 manifest 必须与已发布的完全一致（包括张量 SHA-256）

### 输出保证

- `import_update` 返回的张量名称集合与 manifest.tensors 完全一致
- `acknowledge` 在 `import_update` 成功前调用会抛出异常
- `release` 后张量数据被清除，内存可被回收

## 6. 关键设计决策与不变量

1. **SHA-256 端到端校验** — 每个张量在 publish 时计算 SHA-256，import 方可在 acknowledge 前重新验证。原因：CUDA VMM 和共享内存传输中的位翻转或竞态条件会导致静默的权重损坏。

2. **POSIX fd 而非 CUDA IPC handles** — CUDA VMM 使用 cuMemExportToShareableHandle + Unix Domain Socket 的 SCM_RIGHTS 传递 fd，而非 cudaIpcGetMemHandle。原因：WSL2 环境下 CUDA IPC handles 不可靠。

3. **继承链：Local → SharedMemory → CUDAVMM** — `SharedMemoryTensorBridge` 和 `CUDAVMMTensorBridge` 继承 `LocalTensorCopyBridge`。原因：共享 `_validate_manifest`、状态管理和 `acknowledge`/`reject`/`release` 逻辑。

4. **vLLM 适配器的能力协商** — 不反射探测 vLLM 私有 API，而是要求调用方传入显式的 callable。原因：vLLM 的热权重更新接口跨版本不稳定。

### 不变量

- `weight_version` 在同一 bridge 实例中严格单调递增
- 每个 `update_id` 的状态转换只能沿 `published → imported → acknowledged → released` 或 `published → rejected` 方向进行
- `_BRIDGE_METADATA_KEY` 在用户元数据中被保留，不允许用户写入

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | SharedMemoryTensorBridge 仅支持 CPU 张量 | `bridge.py:1786-1789` | CUDA 场景需使用 CUDAVMMTensorBridge |
| 2 | 多节点/RDMA 传输被显式阻断 | `bridge.py:2654-2658` | 跨节点部署不可用 |
| 3 | IPCWeightBridge 的 CUDA IPC handle 重建在 WSL2 下可能失败 | `bridge.py:2607-2613` | WSL2 用户应使用 cuda-vmm 传输 |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_weight_sync_bridge.py` | 41 | 所有 4 种传输后端的生命周期、校验和验证、错误处理、vLLM 适配器 |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `tests/test_deepspeed_training_worker.py` | 多个 | 通过 DeepSpeedTrainingWorker 间接使用 LocalTensorCopyBridge |
| `tests/test_vllm_rollout_sampler.py` | 多个 | 通过 RolloutExecutor 间接使用 make_weight_bridge |
| `benchmarks/benchmark_weight_sync_bridge.py` | 多个 | 性能基准覆盖所有传输后端 |
