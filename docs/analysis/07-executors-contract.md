# 执行合约层（Executors Contract）

> 源码位置：`rl_engine/executors/training_contract.py` + `rl_engine/executors/vllm_sampler.py`
> 文件数：2 个 | 总行数：588 行
> 最后更新：2026-06-13

## 1. 模块职责概述

执行合约层定义了 GRPO 训练循环中 rollout 阶段与 training 阶段之间的数据合约（`training_contract.py`），以及 vLLM 推理引擎的采样器封装（`vllm_sampler.py`）。

`training_contract.py` 定义了阶段结果（`RolloutStageResult`、`TrainingStageResult`）、训练 worker 协议（`TrainingWorker`）、配置（`TorchRLTrainingConfig`）和 rollout payload 到 RL 批次的转换逻辑（`RolloutBatchMixin`）。

`vllm_sampler.py` 封装 vLLM 的 LLM 引擎，提供 prefix-caching 感知的批量候选生成和输出规范化。

上游：`training_contract.py` 依赖 `testing`；`vllm_sampler.py` 无内部依赖。

下游：被 `executors/rollout.py`、`executors/deepspeed_trainer.py`、`executors/ray_actor_manager.py`、测试文件消费。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `training_contract.py` | 286 | 阶段数据合约、rollout payload 解析、合成批次回退 |
| `vllm_sampler.py` | 302 | vLLM 采样配置、懒引擎构造、输出规范化 |

## 3. 核心数据结构与接口

### RolloutStageResult

- **类型**：frozen dataclass
- **字段数**：6 个
- **关键字段**：
  - `iteration`: `int` — 训练迭代编号
  - `weight_version`: `int` — 使用的权重版本
  - `payload`: `Any` — rollout 产出的原始数据
  - `started_at` / `finished_at`: `float` — 时间戳
  - `metrics`: `Mapping[str, Any]` — 可选的 rollout 指标

### TrainingStageResult

- **类型**：frozen dataclass
- **字段数**：6 个
- **关键字段**：
  - `iteration`: `int`
  - `consumed_weight_version`: `int` — 消费的 rollout 权重版本
  - `published_weight_version`: `Optional[int]` — 训练后发布的新权重版本
  - `metrics`: `Mapping[str, Any]` — 训练指标（loss、active_tokens 等）

### TrainingWorker（Protocol）

- **类型**：Protocol
- **关键方法**：
  - `train(rollout: RolloutStageResult)` → `TrainingStageResult`

### TorchRLTrainingConfig

- **类型**：frozen dataclass
- **字段数**：12 个
- **关键字段**：`num_prompts=1`、`samples_per_prompt=2`、`prompt_len=4`、`completion_len=8`、`vocab_size=64`、`hidden_dim=32`、`valid_density=0.75`、`lr=1e-3`、`device="cpu"`

### RolloutBatchMixin

- **类型**：class（mixin，需要 `config` 和 `device` 属性）
- **关键方法**：
  - `_batch_from_rollout_or_synthetic(rollout)` → `(SyntheticRLKernelBatch, dict)` — 尝试从 payload 提取 token groups，失败则合成
  - `_batch_from_token_groups(token_groups, rollout)` → `SyntheticRLKernelBatch` — 将 token groups 填充到固定长度批次

### VLLMSamplerConfig

- **类型**：frozen dataclass
- **字段数**：6 个
- **关键字段**：`model`、`num_generations=1`、`enable_prefix_caching=True`、`sampling_params`、`engine_kwargs`、`backend="vllm"`

### VLLMSharedPrefixSampler

- **类型**：class
- **字段数**：4 个
- **关键字段**：
  - `config`: `VLLMSamplerConfig` — 采样配置
  - `_engine`: `Optional[Any]` — 懒构造的 vLLM LLM 实例
  - `_llm_cls`: `Optional[type]` — 可注入的 LLM 类（用于测试）
  - `_sampling_params_cls`: `Optional[type]` — 可注入的 SamplingParams 类
- **关键方法**：
  - `generate(prompts, *, num_generations, sampling_params)` → `dict` — 生成带 prefix caching 的 GRPO 候选
  - `engine` → `Any` — 懒构造的 vLLM LLM 实例

### NormalizedRolloutCandidate

- **类型**：frozen dataclass
- **字段数**：10 个
- **关键字段**：
  - `prompt_index: int` — 所属 prompt 的索引
  - `candidate_index: int` — 候选在组内的索引
  - `request_id: Optional[str]` — vLLM 请求 ID
  - `prompt_token_ids: Optional[list[int]]` — 原始 prompt 的 token IDs
  - `token_ids: list[int]` — 生成的 token IDs
  - `text: str` — 生成的文本
  - `finish_reason: Optional[str]` — 停止原因
  - `cumulative_logprob: Optional[float]` — 累积对数概率
  - `logprobs: Optional[Any]` — 逐 token 对数概率
  - `raw_output: Optional[Any]` — 原始 vLLM 输出（`repr=False`, `compare=False`）

## 4. 算法与逻辑详解

### extract_rollout_token_groups 解析策略

1. **normalized_outputs 路径** (`training_contract.py:229-240`): 优先从 `payload["normalized_outputs"]` 提取（RL-Kernel 规范化格式）
2. **outputs 路径** (`training_contract.py:242-253`): 回退到 `payload["outputs"]`（原始 vLLM 格式）
3. **候选解析** (`training_contract.py:256-278`): `_candidate_token_ids` 递归搜索 `token_ids`（支持 Mapping、属性对象、嵌套 outputs）

### VLLMSharedPrefixSampler.generate 流程

1. **Prompt 规范化** (`vllm_sampler.py:107`): 单字符串 → 列表，Mapping → 字典列表
2. **Prompt 扩展** (`vllm_sampler.py:112`): 每个 prompt 复制 `num_generations` 次，保持字节一致性以利用 prefix cache
3. **引擎调用** (`vllm_sampler.py:114`): `self.engine.generate(expanded_prompts, params)`
4. **输出分组** (`vllm_sampler.py:115`): 按 `num_generations` 切片回 prompt-group 结构
5. **规范化** (`vllm_sampler.py:123`): `normalize_grouped_outputs` 将原始 vLLM 输出转为 `NormalizedRolloutCandidate`

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `min_completion_len` | `1`（默认） | `training_contract.py:66` | `_batch_from_token_groups` 的最小 completion 长度 |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Prompts    │────▶│VLLMSharedPrefix  │────▶│RolloutStageResult│
│              │     │Sampler           │     │                  │
│ 文本/token   │     │                  │     │ payload:         │
│ prompt 列表  │     │ vLLM + prefix    │     │  normalized_     │
│              │     │ caching          │     │  outputs         │
└──────────────┘     └──────────────────┘     └──────────────────┘
                                                     │
                                              ┌──────▼─────────┐
                                              │RolloutBatchMixin│
                                              │                │
                                              │ extract_token_ │
                                              │ groups → batch │
                                              └────────────────┘
```

### 输入契约

- `VLLMSharedPrefixSampler.generate` 的 prompts 必须是字符串或 Mapping
- `RolloutBatchMixin._batch_from_rollout_or_synthetic` 的 rollout.payload 应为 Mapping（否则回退合成）

### 输出保证

- `VLLMSharedPrefixSampler.generate` 返回的 `outputs` 列表长度等于 `num_prompts`，每个子列表长度等于 `num_generations`
- `_batch_from_rollout_or_synthetic` 始终返回有效的 `SyntheticRLKernelBatch`（合成回退保底）

## 6. 关键设计决策与不变量

1. **Prompt 复制而非 n= 参数** — 通过重复 prompt 列表实现多候选采样，而非使用 vLLM 的 `n` 参数。原因：确保每个候选独立生成，共享完全一致的 prompt prefix，最大化 vLLM prefix cache 命中率。

2. **合成数据回退** — 当 rollout payload 无法解析为 token groups 时，自动生成合成批次而非报错。原因：解耦训练 worker 对 rollout worker 的依赖，便于独立测试和渐进集成。

3. **vllm_sampler.py 零内部依赖** — 不导入 rl_engine 的任何模块。原因：vLLM 采样器可能在独立的 Ray actor 中运行，保持自包含减少序列化/反序列化的依赖链。

### 不变量

- `VLLMSamplerConfig.num_generations >= 1`
- `TorchRLTrainingConfig.samples_per_prompt >= 2`（用于组内归一化）

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 无已知问题 | — | — | — |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/test_vllm_rollout_sampler.py` | 14 | VLLMSamplerConfig、prefix cache 标志、多 prompt 分组、输出规范化 |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `tests/test_deepspeed_training_worker.py` | 多个 | 通过 `DeepSpeedTrainingWorker.train` 间接使用 `RolloutStageResult` 和 `_batch_from_rollout_or_synthetic` |
| `tests/test_ray_actor_manager.py` | 多个 | 使用 `RolloutStageResult`、`TrainingStageResult` 验证 Ray 调度 |
