# 执行编排层（Executors Orchestration）

> 源码位置：`rl_engine/executors/rollout.py` + `deepspeed_trainer.py` + `ray_actor_manager.py`
> 文件数：3 个 | 总行数：834 行
> 最后更新：2026-06-13

## 1. 模块职责概述

执行编排层负责 GRPO 训练循环的三个运行时组件：(1) `RolloutExecutor` — 统一的 rollout 执行引擎，管理权重导入和 vLLM 采样调度；(2) `DeepSpeedTrainingWorker` — 基于 DeepSpeed 引擎的训练 worker 实现；(3) `RayActorManager` — Ray 分布式 actor 的生命周期管理。

在整体流水线中，编排层是 GRPO 训练循环的实际运行时——RolloutExecutor 在推理侧执行采样，DeepSpeedTrainingWorker 在训练侧执行梯度更新，RayActorManager 将两者封装为 Ray actor 进行分布式部署。

上游：`rollout.py` 依赖 `bridge`、`vllm_sampler`、`kernels/registry`、`utils/logger`；`deepspeed_trainer.py` 依赖 `bridge`、`training_contract`、`testing`；`ray_actor_manager.py` 依赖 `bridge`（仅类型）。

下游：被外部训练脚本、Ray 调度器和测试文件消费。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `rollout.py` | 187 | RolloutExecutor — 权重导入 + 算子准备 + vLLM 采样 |
| `deepspeed_trainer.py` | 414 | DeepSpeedTrainingWorker — DeepSpeed 引擎初始化 + 训练步骤 + 权重发布 |
| `ray_actor_manager.py` | 233 | RayActorManager — Ray actor 创建 + worker handle + 生命周期管理 |

## 3. 核心数据结构与接口

### RolloutExecutor

- **类型**：class
- **字段数**：10 个
- **关键字段**：
  - `config`: `dict` — 模型配置
  - `bridge`: `WeightConsumer` — 权重消费桥（默认 cuda-vmm）
  - `shared_weights`: `dict[str, Tensor]` — 当前导入的权重
  - `weight_install_adapter`: `Optional[WeightInstallAdapter]` — vLLM 适配器
  - `active_weight_version` / `active_weight_update_id` — 当前活跃权重版本
  - `logp_op` / `attn_op` — 懒加载的算子实例
  - `sampler_config` / `sampler` — 懒加载的 vLLM 采样器配置与实例
- **关键方法**：
  - `update_weights(manifest)` → `Mapping[str, Tensor]` — 导入权重、安装到引擎、acknowledge
  - `release_weights()` → `None` — 释放当前权重
  - `generate_candidates(prompts, *, num_generations, sampling_params)` → `dict` — vLLM 候选生成
  - `execute_rollout(input_ids)` → `dict` — 占位的内核级 rollout（尚未完成）
  - `_prepare_kernels()` — 通过 `kernel_registry.get_op` 懒加载 logp 和 attn 算子

### DeepSpeedTrainingConfig

- **类型**：frozen dataclass（继承 `TorchRLTrainingConfig`）
- **字段数**：15 个（继承 12 个 + 3 个）
- **额外字段**：`zero_stage=0`、`deepspeed_config: Mapping`、`initialize_kwargs: Mapping`

### DeepSpeedTrainingWorker

- **类型**：class（混入 `RolloutBatchMixin`）
- **字段数**：8 个
- **关键字段**：`config`、`device`、`weight_bridge`（`WeightPublisher`）、`_latest_published_weight_version`（版本计数器）、`_deepspeed`（模块引用）、`model`、`optimizer`、`engine`（DeepSpeed 引擎）
- **关键方法**：
  - `train(rollout: RolloutStageResult)` → `TrainingStageResult` — 完整训练步骤
  - `publish_weights(*, weight_version, metadata)` → `WeightUpdateManifest` — 发布权重（ZeRO-3 感知）
  - `release_weights(update_id)` — 释放已发布的权重

### RayActorManager

- **类型**：class（支持 context manager）
- **字段数**：3 个
- **关键字段**：`runtime_config: RayRuntimeConfig`、`_ray` 模块引用、`_actors` 列表
- **关键方法**：
  - `create_worker_actor(spec: RayWorkerSpec)` → `actor` — 创建 Ray remote actor
  - `create_rollout_worker(spec)` → `RayRolloutWorkerHandle` — 包装 rollout actor
  - `create_training_worker(spec)` → `RayTrainingWorkerHandle` — 包装 training actor
  - `health_check()` → `list[dict]` — 批量健康检查
  - `shutdown()` → `None` — 终止所有 actor

### RayRolloutWorkerHandle / RayTrainingWorkerHandle

- **类型**：class（同步适配器）
- **职责**：将异步的 `actor.method.remote()` 调用包装为同步的 `ray.get()` 调用

### RayRuntimeConfig / RayActorOptions / RayWorkerSpec

- **类型**：frozen dataclass
- **职责**：分别配置 Ray 运行时初始化参数、actor 资源选项、worker 工厂规格

## 4. 算法与逻辑详解

### RolloutExecutor.update_weights 流程

1. **导入** (`rollout.py:67`): `self.bridge.import_update(manifest)` — 从桥接获取张量
2. **安装** (`rollout.py:68-69`): 若有 `weight_install_adapter`，调用 `adapter.install(manifest, imported)` 安装到 vLLM 引擎
3. **确认** (`rollout.py:70`): `self.bridge.acknowledge(manifest.update_id)`
4. **错误处理** (`rollout.py:72-76`): 安装失败时调用 `bridge.reject`，不更新 active 版本
5. **版本切换** (`rollout.py:77-82`): 更新 `active_weight_version`，释放旧版本

### DeepSpeedTrainingWorker.train 流程

1. **批次准备** (`deepspeed_trainer.py:106`): `_batch_from_rollout_or_synthetic` 从 rollout payload 或合成数据获取批次
2. **前向传播** (`deepspeed_trainer.py:108-114`): 模型前向 → logits → `selected_logprobs_reference` → current_logps
3. **Loss 计算** (`deepspeed_trainer.py:115-122`): 手动实现 PPO-clip loss + KL penalty（使用 `testing` 模块的参考算子）
4. **反向传播** (`deepspeed_trainer.py:124-132`): `engine.backward(loss)` + `engine.step()`
5. **权重发布** (`deepspeed_trainer.py:135`): `_next_published_weight_version` 递增版本号

### DeepSpeedTrainingWorker._export_zero3_full_state_model

1. **Rank 检查** (`deepspeed_trainer.py:201-204`): 仅 rank 0 可发布
2. **GatheredParameters 路径** (`deepspeed_trainer.py:213-216`): 使用 `deepspeed.zero.GatheredParameters` 上下文管理器收集所有分片参数
3. **单 rank 回退** (`deepspeed_trainer.py:217-219`): 若 world_size=1 且无 gather API，直接使用 state_dict
4. **多 rank 阻断** (`deepspeed_trainer.py:220-225`): 若 world_size>1 且无 gather API，显式报错

### _configure_cuda_home_from_python_packages

1. **显式 CUDA_HOME** (`deepspeed_trainer.py:282-286`): 若 `CUDA_HOME` 已设置，同步到 torch
2. **Python 包搜索** (`deepspeed_trainer.py:287-296`): 扫描 `site-packages/nvidia/cu*` 目录寻找 nvcc 和 cuda.h
3. **环境变量设置** (`deepspeed_trainer.py:288-296`): 找到后设置 `CUDA_HOME`、`PATH`、`LD_LIBRARY_PATH`

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| 默认 `weight_transport` | `"cuda-vmm"` | `rollout.py:37` | RolloutExecutor 默认的权重传输协议 |
| 默认 `weight_transport` | `"local-clone"` | `deepspeed_trainer.py:71` | DeepSpeedTrainingWorker 默认传输 |
| clip 范围 | `[0.8, 1.2]` | `deepspeed_trainer.py:119` | PPO clipping 参数 |
| KL 系数 | `0.01` | `deepspeed_trainer.py:122` | KL penalty 权重 |

## 5. 数据流（输入/输出）

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Training Worker  │────▶│   Weight Bridge  │────▶│ Rollout Executor │
│ (DeepSpeed)      │     │                  │     │                  │
│                  │     │ publish manifest │     │ import + install  │
│ train() → loss   │     │ ←──────────────  │     │ → generate_      │
│ publish_weights()│     │                  │     │   candidates()   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         ▲                                                │
         │                ┌──────────────────┐             │
         └────────────────│ RayActorManager  │◀────────────┘
                          │                  │
                          │ create/dispatch  │
                          │ /shutdown actors │
                          └──────────────────┘
```

### 输入契约

- `RolloutExecutor.update_weights` 要求 manifest.transport 与 bridge 类型一致
- `DeepSpeedTrainingWorker.__init__` 在 CUDA 请求时要求 `torch.cuda.is_available()`
- `RayActorManager.create_worker_actor` 要求 `spec.worker_factory` 可调用

### 输出保证

- `DeepSpeedTrainingWorker.train` 始终返回有效的 `TrainingStageResult`（含 metrics）
- `RayActorManager.shutdown` 终止所有已创建的 actor

## 6. 关键设计决策与不变量

1. **DeepSpeed 延迟导入** — `_load_deepspeed` 在 worker 构造时才导入 DeepSpeed，且先调用 `_configure_cuda_home_from_python_packages` 确保 CUDA 工具链可发现。原因：DeepSpeed 的 import 路径依赖 CUDA_HOME 和 nvcc，Python NVIDIA wheel 安装方式不设置这些环境变量。

2. **RolloutExecutor 默认 cuda-vmm 传输** — 而非 local-clone 或 shared-memory。原因：GRPO rollout 在 GPU 上运行 vLLM，cuda-vmm 避免了 GPU→CPU→GPU 的额外拷贝。

3. **Ray 模块延迟导入** — `_load_ray` 在 `ensure_runtime` 时才导入。原因：Ray 是可选依赖，不应在不使用 Ray 的场景中被强制加载。

4. **_RayWorkerActor 通用 shim** — 单一 actor 类通过 `worker_factory` 可构造任意 worker。原因：避免为每种 worker 类型维护独立的 Ray remote 类。

### 不变量

- `DeepSpeedTrainingWorker._latest_published_weight_version` 严格单调递增
- `RolloutExecutor.active_weight_version` 在 `update_weights` 成功后才更新

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | `RolloutExecutor.execute_rollout` 仅返回占位结果，且 CPU 回退错误返回 `device: "rocm"` | `rollout.py:172-187` | 内核级 rollout 路径未完成端到端集成；CPU 环境下 device 值应为 `"cpu"` |
| 2 | `DeepSpeedTrainingWorker.train` 使用硬编码的 clip=0.8/1.2 和 beta=0.01 | `deepspeed_trainer.py:119-122` | 这些超参数应从 config 读取 |
| 3 | `update_weights_via_ipc` 被显式阻断 | `rollout.py:105-119` | 旧式 CUDA IPC 入口不可用 |
| 4 | `train()` 的 `old_logps`/`ref_logps` 为硬编码偏移占位符（`current - 0.01`/`current - 0.02`） | `deepspeed_trainer.py:115-116` | 完整 RLHF 需从 rollout payload 或独立参考模型获取这两个值 |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_deepspeed_training_worker.py` | 11 | DeepSpeed 延迟导入、CUDA_HOME 配置、训练步骤、权重发布、ZeRO-3 |
| `tests/test_ray_actor_manager.py` | 5 | Ray 延迟导入、actor 创建/调度/清理、权重传递 |
| `tests/test_vllm_rollout_sampler.py` | 多个 | RolloutExecutor 的权重更新、vLLM 配置默认值 |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `rl_engine/tests/test_dispatch.py` | 1 (`test_executor_flow`) | 通过 RolloutExecutor 验证完整的内核调度路径 |
