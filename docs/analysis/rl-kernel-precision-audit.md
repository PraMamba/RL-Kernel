# RL-Kernel 训练精度对齐体系源码与 Commit History 深度分析

## 0. Executive Summary

### 一页结论

- **项目名称**: RL-Kernel (rl-engine)
- **分析日期**: 2026-06-13
- **仓库状态**: 154 commits, main branch + 多条 feature branches
- **总体判断**: **局部具备** — 内核算子层精度验证达到体系化水准 (L4), 但训练循环层 (checkpoint/resume, 分布式同步, NaN 监控) 基本缺失
- **最强能力**: 多后端算子精度交叉验证 (PyTorch reference → CUDA/Triton fused), 以及确定性 CUDA LogP 内核的 batch-invariance 保证
- **最大短板**: 无 checkpoint resume 一致性验证; 无分布式梯度同步正确性测试; 无 NaN/Inf 系统化检测; 主 CI 不运行任何精度测试
- **最值得借鉴的源码模块**: `rl_engine/testing/reference_ops.py`, `tests/test_grpo_loss.py`, `csrc/deterministic_logp_kernel.cu` (feat/deterministic-logp 分支)
- **最值得研究的 commits**: `6579d1ab` (测试基础设施建立), `3f44526` (确定性 CUDA LogP), `f1bc93a8` (sqrt(0) + atomic_add 非确定性修复), `f28e638a` (temperature 双除 bug)
- **是否适合作为训练精度对齐基础设施参考**: **Yes, 但限于算子层** — 内核精度验证模式 (reference op + drift summary + dtype-adaptive tolerance) 是工业级水准, 可直接迁移; 训练循环和分布式层尚不具备参考价值

---

## 1. 项目训练流程与精度相关架构总览

### 1.1 训练主入口

- **单 GPU 示例**: `examples/grpo_single_gpu.py` — 完整的 GRPO 训练循环 (logp → ratio → KL → clipped surrogate → loss → backward → step)
- **DeepSpeed 训练 Worker**: `rl_engine/executors/deepspeed_trainer.py:104` — `DeepSpeedTrainingWorker.train()` 方法
- **Trainer 类**: `DeepSpeedTrainingWorker` (line 55), 使用 `RolloutBatchMixin` (定义在 `training_contract.py:69`)

### 1.2 配置系统

- **训练配置**: `rl_engine/executors/training_contract.py:52` — `TorchRLTrainingConfig` dataclass, 包含 `dtype` (默认 fp32), `seed` (默认 0), `device` (默认 cpu), `vocab_size`, `hidden_dim`, `lr` 等
- **DeepSpeed 配置**: `rl_engine/executors/deepspeed_trainer.py:43` — `DeepSpeedTrainingConfig`, 扩展 `zero_stage`, `deepspeed_config` 字段
- **DeepSpeed JSON config 构建**: `deepspeed_trainer.py:252-261` — `_resolved_deepspeed_config()` 将 `config.dtype` 映射为 `fp16.enabled` / `bf16.enabled`
- **精度配置流**: dtype 从 `TorchRLTrainingConfig.dtype` → DeepSpeed JSON config → DeepSpeed 引擎内部 loss scaling. **无全局精度策略对象, 无 autocast context**

### 1.3 训练 Step 流程

- **Forward**: `deepspeed_trainer.py:108` — `self.engine(batch.token_ids.long())` → `_extract_logits()`
- **Loss 计算**: `deepspeed_trainer.py:109-122` — `selected_logprobs_reference()` (fp32) → `compute_policy_ratio()` → `compute_reference_kl()` → `masked_mean()`
- **Backward**: `deepspeed_trainer.py:131` — `self.engine.backward(loss)`
- **Optimizer 更新**: `deepspeed_trainer.py:132` — `self.engine.step()`
- **无梯度裁剪, 无显式 loss scaling (依赖 DeepSpeed 内部管理)**

### 1.4 随机性控制

- **CPU 种子**: `deepspeed_trainer.py:85` — `torch.manual_seed(self.config.seed)`. **仅设置 CPU RNG, 未调用 `torch.cuda.manual_seed_all()`**
- **示例脚本**: `examples/grpo_single_gpu.py:167` — `torch.manual_seed(args.seed)`, 同样无 CUDA seed
- **测试用**: `rl_engine/testing/rl_batch.py` — `torch.Generator.manual_seed(seed)` 用于确定性 batch 生成
- **确定性标志**: **未发现** `torch.backends.cudnn.deterministic`, `torch.use_deterministic_algorithms()`, 或 `CUBLAS_WORKSPACE_CONFIG` 的任何使用

### 1.5 Mixed Precision 逻辑

- **AMP/autocast**: **未使用**. 整个代码库无 `torch.cuda.amp.autocast` 或 `torch.amp.autocast` 调用
- **Loss scaling**: **无显式实现**. 完全委托给 DeepSpeed 的 `fp16.enabled` / `bf16.enabled` 配置
- **Master weights**: **未发现**. 无 fp32 master weight 管理
- **内核内部精度**: 所有 CUDA/Triton 内核内部计算均使用 float32 (verified: `csrc/fused_logp_kernel.cu` line 129: `static_cast<float>`, `triton_grpo_loss.py` line 45: `.to(tl.float32)`)

### 1.6 Distributed Parallel 逻辑

- **DP**: DeepSpeed 数据并行 (通过 `deepspeed.initialize`)
- **TP**: `rl_engine/executors/bridge.py:32` — `WeightLayout` 中定义 `tensor_parallel_size` 字段, 但无实际 TP 实现
- **PP**: **未实现**
- **SP**: **未实现**
- **EP/MoE**: **未实现**
- **ZeRO/FSDP**: DeepSpeed ZeRO (stage 0-3), 通过 `DeepSpeedTrainingConfig.zero_stage` 控制

### 1.7 Checkpoint Save/Load

- **Save**: `deepspeed_trainer.py:198-236` — `_export_zero3_full_state_model()` 使用 `deepspeed.zero.GatheredParameters` 收集 ZeRO-3 分片参数, 然后 `_clone_state_dict()` (line 399-404) 使用 `tensor.detach().clone(memory_format=torch.preserve_format)`
- **Load**: **无显式 checkpoint 加载/恢复逻辑**
- **状态组件**: 仅模型权重 state_dict. **无 optimizer state, 无 scheduler state, 无 RNG state 保存**
- **Weight Bridge**: `rl_engine/executors/bridge.py` — 4 种传输后端 (LocalTensorCopy, SharedMemory, CUDA VMM, CUDA IPC), 使用 SHA256 校验保证 bit-exact 传输

### 1.8 测试系统组织

- **Test 目录**: `tests/` (14 个测试文件 + conftest.py), `rl_engine/tests/` (1 个测试文件)
- **精度测试**: `test_op_accuracy.py` (15 test functions, 16 cases with parameterization), `test_grpo_loss.py` (19 tests), `test_rl_kernel_loss_step.py` (pipeline 验证), `test_sampler_temperature.py` (temperature 回归), `test_reference_ops.py` (参考实现自检)
- **基础设施测试**: `test_weight_sync_bridge.py`, `test_ray_actor_manager.py`, `test_vllm_rollout_sampler.py`, `test_deepspeed_training_worker.py`
- **CI 工作流**:
  - `.github/workflows/ci.yml` — **仅运行 `rl_engine/tests/test_dispatch.py`**, 无 GPU, 无精度测试
  - `.github/workflows/gpu-ci.yml` — 在 RunPod 云 GPU (2x A4000 或 1x A40) 上运行 `tests/` 全套. 需要 `needs-gpu-ci` 标签触发

---

## 2. 精度对齐能力矩阵

### A. Pre-Training Foundation

| # | Capability | Present? | Source Evidence | Commit Evidence | Maturity | Notes |
|---|-----------|----------|----------------|-----------------|----------|-------|
| A1 | 配置一致性扫描 | Partial | `training_contract.py:52` — `TorchRLTrainingConfig` dataclass 统一配置, 但无跨 rank 一致性校验 | 未发现 | 1 | 配置对象存在但无 validation |
| A2 | 随机种子/RNG 控制 | Partial | `deepspeed_trainer.py:85` — `torch.manual_seed(seed)`, `rl_batch.py` — seeded Generator | `6579d1ab` — 建立 seeded batch fixture | 2 | 仅 CPU RNG, 无 CUDA seed, 无 cudnn.deterministic |
| A3 | 数据加载顺序确定性 | Partial | `rl_engine/testing/rl_batch.py` — `make_synthetic_rl_kernel_batch()` seeded 生成 | `6579d1ab` | 2 | 合成数据确定性, 但无真实 DataLoader/DistributedSampler |
| A4 | 初始权重一致性 | No | `deepspeed_trainer.py:86-89` — 直接构建模型, 无 broadcast from rank 0 | 未发现 | 0 | DeepSpeed 可能内部处理, 但项目无显式验证 |

### B. Single-Step Alignment

| # | Capability | Present? | Source Evidence | Commit Evidence | Maturity | Notes |
|---|-----------|----------|----------------|-----------------|----------|-------|
| B1 | 单步 forward loss 对齐 | Yes | `test_grpo_loss.py:111-141` — native vs reference (atol=1e-6); `test_rl_kernel_loss_step.py:180-187` — pipeline drift < 2e-2 | `25e19bf` — GRPO loss tests; `6579d1ab` — loss step tests | 4 | 完整的 reference → native → fused 三级验证 |
| B2 | Activation dump/compare | Partial | `reference_ops.py:109-142` — `summarize_kernel_drift()` 计算 max/mean abs error | `6579d1ab` | 2 | 仅 drift 统计, 无逐层 activation dump |
| B3 | Gradient dump/compare | Yes | `test_grpo_loss.py:144-164` — gradient flow; `test_grpo_loss.py:195-213` — Triton vs native backward (atol=1e-4) | `25e19bf` | 3 | GRPO 梯度交叉验证完善; CUDA LogP 无梯度测试 |
| B4 | Optimizer state 对齐 | No | 未发现 optimizer state 比对 | 未发现 | 0 | |
| B5 | Scheduler/LR curve 对齐 | No | 未发现 learning rate scheduler | 未发现 | 0 | |

### C. Long-Horizon & Regression

| # | Capability | Present? | Source Evidence | Commit Evidence | Maturity | Notes |
|---|-----------|----------|----------------|-----------------|----------|-------|
| C1 | Loss curve golden regression | No | `test_grpo_loss.py:399-410` — SGD 5 步 loss 下降检查, 但非 golden baseline | `25e19bf` | 1 | 基本收敛测试, 非 golden loss 回归 |
| C2 | Mixed precision 对齐 | Partial | `test_op_accuracy.py:122-166` — dtype-adaptive 阈值 (fp16: 1e-2, fp32: 1e-5) | `0da66070` — fused logp variants; `3315bc16` — native API parity | 3 | LogP 算子有 fp16/bf16/fp32 对齐; GRPO loss 仅 fp32 |
| C3 | FP16/BF16/FP8 数值稳定性 | Partial | 所有内核内部 fp32 计算; `grpo_loss.py:73` — `variance.clamp_min(eps**2).sqrt()`; `reference_ops.py:40` — `logits.float()` | `f1bc93a8` — sqrt(0) 修复; `9a804949` — softmax float 修复 | 3 | 系统化 upcast-to-fp32 策略, 但无 fp8 支持 |
| C4 | TF32 控制 | No | 未发现 `allow_tf32` 或 `matmul.allow_tf32` 的任何使用 | 未发现 | 0 | |
| C5 | NaN/Inf/overflow 检测 | Partial | `examples/grpo_single_gpu.py:260` — `torch.isfinite(loss)` check; `test_grpo_loss.py:162,320` — `isfinite(grad).all()` | 未发现专门 commit | 1 | 零散检查, 无系统化检测框架 |
| C6 | Checkpoint resume 一致性 | No | 无 checkpoint resume 测试 | 未发现 | 0 | |

### D. Distributed Correctness

| # | Capability | Present? | Source Evidence | Commit Evidence | Maturity | Notes |
|---|-----------|----------|----------------|-----------------|----------|-------|
| D1 | Data parallel correctness | No | DeepSpeed DP 配置存在但无 DP 正确性测试 | 未发现 | 0 | |
| D2 | Tensor parallel correctness | No | `WeightLayout` 有 `tensor_parallel_size` 字段但无 TP 实现 | 未发现 | 0 | |
| D3 | Pipeline parallel correctness | No | 未实现 | 未发现 | 0 | |
| D4 | Sequence parallel correctness | No | 未实现 | 未发现 | 0 | |
| D5 | Expert parallel / MoE correctness | No | 未实现 | 未发现 | 0 | |
| D6 | Collective communication correctness | No | Weight bridge 使用 SHA256 校验传输完整性, 但无 NCCL/collective 测试 | 未发现 | 0 | |

### E. Infrastructure & Tooling

| # | Capability | Present? | Source Evidence | Commit Evidence | Maturity | Notes |
|---|-----------|----------|----------------|-----------------|----------|-------|
| E1 | CI 精度回归测试 | Partial | `.github/workflows/gpu-ci.yml` — GPU CI 执行精度测试, 但需人工触发 (`needs-gpu-ci` label) | `5868a47` 等 — GPU CI 系列 | 2 | GPU CI 存在但非自动触发; 主 CI 不运行精度测试 |
| E2 | 自动化二分定位 | No | 未发现 bisect 工具 | 未发现 | 0 | |
| E3 | 跨硬件/跨后端对齐 | Partial | `rl_engine/kernels/registry.py` — 多后端 dispatch (CUDA/Triton/PyTorch/ROCm); `test_op_accuracy.py` — CPU/GPU 分别测试 | `1c3e388e` — Blackwell SM120 兼容修复 | 2 | 多后端架构存在, 但无跨硬件精度对比测试 |

### Summary Statistics

```
Total capabilities assessed: 24
Maturity 4 (systematic):     1  (B1: forward loss alignment)
Maturity 3 (CI-integrated):  3  (B3, C2, C3)
Maturity 2 (has tests):      5  (A2, A3, B2, E1, E3)
Maturity 1 (scattered):      3  (A1, C1, C5)
Maturity 0 (not found):     12  (A4, B4, B5, C4, C6, D1-D6, E2)

Average maturity: 1.08
Strongest domain: B (Single-Step Alignment) — avg 1.8
Weakest domain:   D (Distributed Correctness) — avg 0.0
```

---

## 3. 源码证据地图

### 3.1 Reference Implementation 体系

**File**: `rl_engine/testing/reference_ops.py`
**Location**: 模块级函数 (line 15-142)
**Capability**: 算子精度对齐的 ground truth oracle
**Evidence type**: 明确存在
**Description**: 提供 6 个纯 PyTorch fp32 参考实现: `selected_logprobs_reference` (line 15), `masked_sum` (line 50), `active_token_count` (line 59), `masked_mean` (line 71), `compute_policy_ratio` (line 82), `compute_reference_kl` (line 95), 加上 `summarize_kernel_drift` (line 109) 用于量化误差.

```python
# reference_ops.py:40-41 — 所有 logp 计算强制 upcast 到 fp32
scaled_logits = logits.float() / float(temperature)
log_probs = torch.log_softmax(scaled_logits, dim=-1)
```

### 3.2 多后端算子交叉验证

**File**: `rl_engine/kernels/ops/pytorch/loss/logp.py` (NativeLogpOp), `rl_engine/kernels/ops/cuda/loss/logp.py` (FusedLogpGenericOp), `rl_engine/kernels/ops/triton/triton_grpo_loss.py` (TritonGRPOLossOp)
**Location**: 各 Op 类
**Capability**: 同一算子的多个实现, 可互相验证
**Evidence type**: 明确存在
**Description**: 三级验证链: Reference (纯 PyTorch fp32) → Native Op (PyTorch 生产) → Fused Op (CUDA/Triton). 每层容差递增: reference↔native (1e-6), native↔Triton (1e-4), pipeline E2E (2e-2).

### 3.3 CUDA LogP 内核 fp32 内部计算

**File**: `csrc/fused_logp_kernel.cu`
**Location**: `fused_logp_forward_kernel` (line 112), `fused_logp_forward_online_kernel` (line 158)
**Capability**: Mixed precision 数值稳定性
**Evidence type**: 明确存在
**Description**: 所有 block reduction 均使用 `static_cast<float>` (line 129, 139, 150). Online kernel 使用 `LogSumExpState` 结构 (line 52) 实现单 pass 数值稳定 log-sum-exp.

### 3.4 GRPO Loss 数值稳定性修复

**File**: `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py`
**Location**: `NativeGRPOLossOp.group_advantages` (line 73), `NativeGRPOLossOp.apply` (line 101-102)
**Capability**: 数值稳定性
**Evidence type**: 明确存在
**Description**: 方差 clamp 策略 `variance.clamp_min(eps**2).sqrt()` (line 73) — 在 `sqrt` 前 clamp 在 `eps^2`, 而非 `sqrt` 后 clamp 在 `eps`. 所有 logp 差值在 `exp()` 前通过 `masked_fill(~bool_mask, 0.0)` 清零 (line 101-102), 防止 `exp(garbage)` 产生 NaN.

### 3.5 Temperature 精度回归测试

**File**: `tests/test_sampler_temperature.py`
**Location**: 全文件
**Capability**: 采样精度回归保护
**Evidence type**: 明确存在
**Description**: 使用 stub FlashInfer 模块捕获实际 softmax 概率, 验证 `probs == softmax(logits/T)` 而非 `softmax(logits/T^2)`. Atol=1e-6.

### 3.6 `--use_fast_math` 编译标志

**File**: `setup.py`
**Location**: line 64
**Capability**: **精度风险**
**Evidence type**: 明确存在
**Description**: NVCC 编译使用 `--use_fast_math`, 启用 `__fmul_rz` (round-to-zero 而非 round-to-nearest), fast transcendentals (`__expf`, `__logf` ~2 ULP 误差). 影响所有 CUDA kernel 中的 `expf`/`logf` 精度.

```python
# setup.py:64
nvcc_flags = ["-O3", "--use_fast_math", "-Xfatbin", "-compress-all"]
```

### 3.7 SamplerBackend.compute_logp 缺少 fp32 upcast

**File**: `rl_engine/kernels/sampling.py`
**Location**: `SamplerBackend.compute_logp` (line 97)
**Capability**: **精度风险**
**Evidence type**: 明确存在
**Description**: `torch.log_softmax(c_logits, dim=-1)` 直接在输入 dtype 上计算, 当输入为 fp16/bf16 时会损失精度. 对比 `sample()` 方法 (line 53) 已正确 upcast: `logits = logits.float().contiguous()`.

---

## 4. Commit / PR 历史演进时间线

### Entry 1

- **时间**: 2026-05-26
- **Commit**: `ec042acd`
- **涉及文件**: `csrc/cuda/fused_logp_sm90.cu`, `csrc/utils/tma_utils.cuh`, `tests/test_op_accuracy.py`
- **问题背景**: 需要 Hopper GPU 上的 TMA 加速 LogP 内核
- **修改内容**: 引入 SM90 warp-specialized LogP 内核, 使用 TMA 批量内存加载, 内部 fp32 累加
- **新增测试**: Partial — 修改了 `test_op_accuracy.py`
- **影响范围**: C2 (mixed precision), E3 (跨硬件)
- **启发**: 实验性硬件内核需要功能 gating (后来发现 TMA box 宽度超限和 warp deadlock)

### Entry 2

- **时间**: 2026-05-29
- **Commit**: `0da66070`
- **涉及文件**: `csrc/fused_logp_kernel.cu`, `tests/test_op_accuracy.py` (5 files, +887 lines)
- **问题背景**: 需要多种 logp 内核变体: online (单 pass), indexed (稀疏), fp32 output
- **修改内容**: 实现 online log-sum-exp 算法, `LogSumExpState` merge, 运行时 launch variant 选择
- **新增测试**: Yes — 扩展 `test_op_accuracy.py`
- **影响范围**: B1, C2, C3
- **启发**: Online log-sum-exp (单 pass) 避免 two-pass max-then-sum, 数值等价但更高效. `merge_logsumexp_state` 是跨 warp 稳定归约的关键

### Entry 3

- **时间**: 2026-05-30
- **Commit**: `6579d1ab`
- **涉及文件**: `rl_engine/testing/reference_ops.py`, `rl_engine/testing/rl_batch.py`, `tests/test_reference_ops.py`, `tests/test_rl_kernel_loss_step.py` (7 files, +814 lines)
- **问题背景**: 缺少系统化的精度测试基础设施
- **修改内容**: **建立了整个精度验证基础设施**: reference ops, deterministic batch factory, drift summary, loss-step validation
- **新增测试**: Yes — 3 个测试文件
- **影响范围**: B1, B2, B3, A2, A3
- **启发**: **项目中最重要的精度 commit.** 先建立 reference op 库和确定性 fixture, 再写内核 — 正确的工程顺序

### Entry 4

- **时间**: 2026-06-02
- **Commit**: `3315bc16`
- **涉及文件**: `rl_engine/kernels/ops/pytorch/loss/logp.py`, `tests/test_op_accuracy.py` (3 files)
- **问题背景**: NativeLogpOp 缺少 `indexed_out`, `online_out` 等变体, 无法作为所有 CUDA 变体的正确性 oracle
- **修改内容**: 统一 `_selected_logps()` 内部方法, 所有入口点共享 fp32 log_softmax
- **新增测试**: Yes — 7 个新测试函数
- **影响范围**: B1, C2
- **启发**: Reference/fallback op 必须与优化 op 有完全相同的 API surface, 否则无法作为替代测试 oracle

### Entry 5

- **时间**: 2026-06-02
- **Commit**: `cddd5b00`
- **涉及文件**: `rl_engine/executors/bridge.py`, `tests/test_weight_sync_bridge.py` (9 files, +4511 lines)
- **问题背景**: 多进程 RL 训练需要零拷贝权重同步 (training actor → rollout actor)
- **修改内容**: 4 种传输后端 (LocalCopy, SharedMemory, CUDA VMM, CUDA IPC), SHA256 完整性校验
- **新增测试**: Yes — 947 行 bridge 测试
- **影响范围**: D6 (通信正确性)
- **启发**: 权重同步正确性是 RL 训练精度的前提. SHA256 校验保证 bit-exact 传输

### Entry 6 (issue-38-rms-norm-fusion-kernel 分支, 未合并到 main)

- **时间**: 2026-06-03
- **Commit**: `17e88148`
- **涉及文件**: `rl_engine/kernels/ops/pytorch/norm/rms_norm.py`, `rl_engine/testing/reference_ops.py` (4 files, +175 lines)
- **问题背景**: 需要 RMSNorm 参考实现作为未来 fused 内核的 baseline
- **修改内容**: `rms_norm_reference()` 使用 `torch.promote_types(x.dtype, torch.float32)`, eps 在 sqrt 内: `rsqrt(variance + eps)`
- **新增测试**: Yes — fp32 accumulation 验证 + `gradcheck` with fp64
- **影响范围**: B1, C3
- **启发**: Norm 层 reference 必须明确 eps 应用位置 (sqrt 内 vs sqrt 外) 和 promotion 规则
- **注意**: 此 commit 位于 `issue-38-rms-norm-fusion-kernel` feature 分支, 尚未合并到 main

### Entry 7

- **时间**: 2026-06-08
- **Commit**: `9a804949`
- **涉及文件**: `rl_engine/kernels/sampling.py` (1 file, +1 line)
- **问题背景**: FlashInfer sampling 路径对 bf16/fp16 logits 直接计算 softmax, 精度损失
- **修改内容**: 添加 `logits = logits.float().contiguous()` (line 53)
- **新增测试**: No
- **影响范围**: C3
- **启发**: **每个 softmax 都必须在 fp32 中计算.** 一行修复, 防止所有下游采样的静默精度退化

### Entry 8

- **时间**: 2026-06-08
- **Commit**: `f28e638a`
- **涉及文件**: `rl_engine/kernels/sampling.py`, `tests/test_sampler_temperature.py` (2 files)
- **问题背景**: FlashInfer 路径 temperature 被除了两次: 函数顶部一次, FlashInfer 分支内又一次
- **修改内容**: 移除重复除法. 添加 stub-based 回归测试
- **新增测试**: Yes — `test_sampler_temperature.py`
- **影响范围**: C3
- **启发**: **Temperature bug 在 T=1.0 时不可检测** (1.0/1.0/1.0 = 1.0). 必须用 T != 1.0 测试. Stub 注入是测试硬件特定路径的有效技术

### Entry 9

- **时间**: 2026-06-09
- **Commit**: `13195f53` + `25e19bf1`
- **涉及文件**: `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py`, `rl_engine/kernels/ops/triton/triton_grpo_loss.py`, `tests/test_grpo_loss.py` (5 files, +1149 lines)
- **问题背景**: 需要 GRPO loss 的 fused 实现
- **修改内容**: NativeGRPOLossOp (PyTorch 正确性 oracle) + TritonGRPOLossOp (fused kernel). 19 个测试用例
- **新增测试**: Yes — 19 tests with tolerance hierarchy: 1e-6 (native↔ref) → 1e-4 (Triton↔native)
- **影响范围**: B1, B3, C2
- **启发**: 显式标注 PyTorch op 为 "correctness oracle" 是优秀实践. 容差阶梯 (1e-6 → 1e-4) 正确反映了 Triton reduction order 差异

### Entry 10

- **时间**: 2026-06-09
- **Commit**: `a4ca4535`
- **涉及文件**: `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py` (1 file, +3/-2)
- **问题背景**: Masked position 的 logp 值可能是 NaN 或大数, `exp()` 前未清零导致 NaN 传播
- **修改内容**: 在 `exp()` 前对 delta 和 diff 执行 `masked_fill(~bool_mask, 0.0)`
- **新增测试**: No (但已有 `valid_density < 1.0` 测试现在正确通过)
- **影响范围**: C5
- **启发**: **Mask before exp, not after.** `exp(garbage) * 0` 不是 `0` 而是 `NaN`. 必须在 exp 前将 masked position 填零

### Entry 11

- **时间**: 2026-06-10
- **Commit**: `f1bc93a8`
- **涉及文件**: `grpo_loss.py`, `triton_grpo_loss.py` (2 files, +27/-8)
- **问题背景**: 三个数值问题: (1) `sqrt(0)=0` 后 clamp 不如 `clamp(eps^2).sqrt()` 稳定; (2) Triton `tl.atomic_add` 跨 block 非确定性; (3) group size 超限
- **修改内容**: 方差 clamp 改为 `eps^2`; Triton 改用 per-block partial sums + host 归约; 添加 `_MAX_GROUP_SIZE=1024`
- **新增测试**: No
- **影响范围**: C3, C5, B1
- **启发**: **(a)** `clamp_min(eps^2).sqrt()` 是标准差 epsilon 的正确模式. **(b)** **Triton `tl.atomic_add` 非确定性** — 精度敏感场景必须用 per-block partial sums + host 归约

### Entry 12

- **时间**: 2026-06-08
- **Commit**: `1c3e388e`
- **涉及文件**: `rl_engine/kernels/registry.py`, `setup.py` (2 files)
- **问题背景**: TMA 内核在 Blackwell SM120/SM100 上编译失败 (TMA box 宽度超限, warp deadlock)
- **修改内容**: TMA 内核仅在 `KERNEL_ALIGN_FORCE_SM90=1` 时编译; 根据实际 GPU 架构设置 gencode
- **新增测试**: No (但 commit message 报告 RTX PRO 6000 (SM120) 验证: `kernel_max_abs_error 4.77e-07`)
- **影响范围**: E3
- **启发**: 架构特定内核选择必须同时检查编译时符号和运行时设备能力

### Entry 13

- **时间**: 2026-06-08
- **Commit**: `5304c1f0`
- **涉及文件**: `rl_engine/alignment/model_wrappers.py`, `rl_engine/testing/reference_ops.py`, `tests/test_alignment_model_wrappers.py` (4 files)
- **问题背景**: `selected_logprobs_reference()` 对 ignore-index (-100) token 会越界索引
- **修改内容**: `gather_token_ids = gather_token_ids.masked_fill(~active_mask, 0)` — mask 前用 0 替换 ignore-index
- **新增测试**: Yes — ignore-index 处理, tuple 输出, logits slicing
- **影响范围**: B1
- **启发**: Reference op 必须处理所有生产数据模式, 包括 ignore-index token

### Entry 14 (feat/deterministic-logp 分支, 未合并到 main)

- **时间**: 2026-06-11
- **Commit**: `3f44526`
- **涉及文件**: `csrc/deterministic_logp_kernel.cu`, `tests/test_deterministic_logp.py` (14 files, +804 lines)
- **问题背景**: 现有 fused logp 内核虽然数值接近, 但同一行在不同 batch 中可能产生不同 bits (共享内存/grid scheduling 变化)
- **修改内容**: 全新确定性 CUDA logp 内核 — 固定 block size, 无 atomicAdd, 纯 warp-shuffle + shared memory reduction, 一个 block 处理一行
- **新增测试**: Yes — 211 行, 包括: 10 次运行 bitwise 一致, batch-size invariance, batch-position invariance, **源码合约锁** (assert CUDA 源码不包含 `atomicAdd`/`cub::BlockReduce`)
- **影响范围**: A2, B1, C2
- **启发**: **项目中最精密的精度对齐 commit.** batch-invariance 属性对 RL 训练可复现性至关重要. "源码合约锁" 测试模式 (assert 源码内容) 是创新性的回归防护

### Entry 15 (feat/deterministic-logp 分支)

- **时间**: 2026-06-12
- **Commit**: `302386d` + `08c7369` + `7db5247`
- **涉及文件**: `csrc/deterministic_logp_kernel.cu`, `tests/test_deterministic_logp.py`
- **问题背景**: 确定性内核需要更广泛的 shape/dtype/edge-case 覆盖和性能优化 (bucket block size)
- **修改内容**: 10 种 shape, 3 种 dtype, output dtype 矩阵, noise-injection batch invariance, 极端 logit 值测试, 3-bucket block size 系统
- **新增测试**: Yes — 大幅扩展覆盖范围
- **影响范围**: B1, C2, C3
- **启发**: noise-injection 测试 (向非目标行注入噪声, 目标行必须 bitwise 不变) 是检测 shared-state 泄漏的强力测试. 3-bucket block size 在确定性框架内实现性能调优

---

## 5. 典型精度问题案例复盘

### Case 1: Temperature 双除 Bug (Case Pattern 9 变体)

- **现象**: FlashInfer sampling 路径的 temperature 被应用了两次, 产生 `softmax(logits / T^2)` 而非 `softmax(logits / T)`
- **根因**: `sample()` 函数顶部统一除 T, FlashInfer 分支内再除一次. 因 `if temperature != 1.0` 守卫, T=1.0 测试检测不到
- **定位方法**: 手动代码审查
- **修复方案**: 移除 FlashInfer 分支内的重复除法 (`f28e638a`)
- **是否新增测试**: Yes — `test_sampler_temperature.py`, 使用 stub FlashInfer 模块
- **借鉴**: 默认配置值 (T=1.0) 是 false negative 陷阱. 精度测试必须使用非默认参数

### Case 2: Masked Position exp(garbage) → NaN (Case Pattern 7 变体)

- **现象**: GRPO loss 在 `valid_density < 1.0` 时产生 NaN
- **根因**: Masked position 的 logp 值未定义 (随机内存), `exp(current - old)` 在 mask 前执行, 产生 Inf; `Inf * 0 = NaN`
- **定位方法**: 推测 (通过分析数据流)
- **修复方案**: 在 `exp()` 前 `masked_fill(~bool_mask, 0.0)` (`a4ca4535`)
- **是否新增测试**: No (但已有 partial-mask 测试现在通过)
- **借鉴**: **Mask before exp, not after.** 这是 RL loss 计算的通用规则

### Case 3: sqrt(0) + Triton atomic_add 非确定性 (Case Pattern 8 变体)

- **现象**: GRPO group normalization 在某些 reward 分布下数值不稳定; Triton forward 在不同运行间可能产生不同结果
- **根因**: (1) `sqrt(0)=0` 后 `clamp_min(eps)` 来不及防护除零; (2) `tl.atomic_add` 跨 block 非确定性 (浮点加法非结合性)
- **定位方法**: 数值异常观察
- **修复方案**: `variance.clamp_min(eps**2).sqrt()`; per-block partial sums + host 归约 (`f1bc93a8`)
- **是否新增测试**: No
- **借鉴**: (a) Epsilon 应用于方差 (eps^2) 再 sqrt, 而非 sqrt 后 clamp. (b) Triton `atomic_add` 非确定性 — 精度敏感场景改用 per-block partials

### Case 4: Softmax 未 upcast 到 fp32 (Case Pattern 2 变体)

- **现象**: FlashInfer sampling 路径对 bf16/fp16 logits 直接计算 softmax, 精度不足
- **根因**: `torch.softmax(logits, dim=-1)` 未先 `.float()`
- **定位方法**: 代码审查
- **修复方案**: 添加 `logits = logits.float().contiguous()` (`9a804949`)
- **是否新增测试**: No
- **借鉴**: 所有 softmax/log_softmax 必须在 fp32 中计算

### 未发现的 Case Pattern:

| Pattern | Status | Reason |
|---------|--------|--------|
| 1. Checkpoint Resume Loss Mismatch | 未发现 | 项目无 checkpoint resume 功能 |
| 4. Pipeline Parallel Micro-Batch Error | 未发现 | 项目无 PP 实现 |
| 5. Distributed Sampler Inconsistency | 未发现 | 项目无 DistributedSampler |
| 6. Optimizer State Mismatch | 未发现 | 项目无 optimizer state 保存/恢复 |
| 10. CI Golden Loss Regression | 未发现 | 项目无 golden loss baseline |

---

## 6. 三阶段精度对齐流程映射

### 阶段一: 训练前准备与基础对齐

| Check Item | Project Coverage | Evidence | Gap? |
|-----------|-----------------|----------|------|
| 配置一致性 | Partial | `TorchRLTrainingConfig` dataclass (training_contract.py:52) | 无跨 rank 校验 |
| 环境一致性 | No | 未发现 | GPU CI 在不同硬件间切换 (A4000/A40) 无精度对比 |
| Seed/RNG | Partial | `torch.manual_seed(seed)` (deepspeed_trainer.py:85) | 无 CUDA seed, 无 cudnn.deterministic |
| 数据顺序 | Partial | Seeded `torch.Generator` in `rl_batch.py` | 仅合成数据, 无真实 DataLoader |
| 模型结构 | No | 未发现跨 rank 结构验证 | |
| 初始化权重 | No | 直接构建模型, 无 broadcast | |
| Dropout/正则 | N/A | 项目模型无 dropout | |
| Deterministic flags | No | 未发现 | |

**Stage 1 verdict**: Mostly missing

### 阶段二: 单卡/单步对齐

| Check Item | Project Coverage | Evidence | Gap? |
|-----------|-----------------|----------|------|
| Forward loss | Well-covered | `test_grpo_loss.py:111-141` (atol=1e-6); `test_rl_kernel_loss_step.py` (drift <2e-2) | |
| Activation | Partial | `summarize_kernel_drift()` | 无逐层 dump |
| Backward gradient | Covered | `test_grpo_loss.py:195-213` Triton vs native (atol=1e-4) | CUDA logp 无梯度测试 |
| Optimizer update | Not covered | | |
| Scheduler | Not covered | | |
| Loss scaling | N/A | 无显式 loss scaling | |
| Tensor dump | Partial | `summarize_kernel_drift()` 仅统计, 不保存值 | |
| Operator-level compare | Well-covered | `test_op_accuracy.py` — 22 tests, dtype-adaptive | Attention 算子无精度测试 |

**Stage 2 verdict**: Partially covered (logp/loss 算子强, attention/optimizer/scheduler 缺失)

### 阶段三: 多步/分布式/长稳对齐

| Check Item | Project Coverage | Evidence | Gap? |
|-----------|-----------------|----------|------|
| Loss curve | Minimal | 5-step SGD descent check | 无 golden baseline |
| Checkpoint resume | Not covered | | |
| DP correctness | Not covered | | |
| TP correctness | Not covered | | |
| PP correctness | N/A | 未实现 | |
| SP correctness | N/A | 未实现 | |
| EP correctness | N/A | 未实现 | |
| Gradient accumulation | Not covered | | |
| Communication collectives | Not covered | | |
| Mixed precision stability | Partial | fp32 internal computation policy | 无长期稳定性测试 |
| NaN/Inf monitoring | Minimal | 散落 `isfinite()` check | |
| CI regression | Partial | GPU CI 存在但需手动触发 | |

**Stage 3 verdict**: Mostly missing

### Cross-Stage Analysis

```
Stage 1 (Foundation):     [██________] 2/8 covered
Stage 2 (Single-Step):    [██████____] 5/8 covered
Stage 3 (Multi/Dist):     [█_________] 1/12 covered
```

**最强阶段**: Stage 2 — 算子级精度验证深度达到 L4
**最弱阶段**: Stage 3 — 分布式和长稳对齐几乎空白
**阶段间风险**: Stage 1 seed/RNG 不完整 → Stage 2 forward loss 对比可能不可复现 (CUDA 非确定性); Stage 2 optimizer 未验证 → Stage 3 checkpoint resume 无法信任

---

## 7. 可复用设计模式

### Pattern 1: Reference Op + Drift Summary 交叉验证

- **设计目标**: 为每个算子提供 fp32 ground truth, 量化优化版本的精度偏移
- **源码位置**: `rl_engine/testing/reference_ops.py` (reference ops), 各 `test_*.py` (cross-validation)
- **工作流程**: (1) 用纯 PyTorch fp32 实现 reference op; (2) 被测内核对同一输入计算; (3) `summarize_kernel_drift()` 报告 max/mean abs error; (4) 按 dtype-adaptive tolerance 判定 PASS/FAIL
- **优点**: 全自动, 无 golden file 依赖; reference op 与 fused op API 完全对齐; tolerance 按 (op, dtype) 分级
- **局限**: 不保存 tensor 值, 无法事后 debug; 无法检测跨 commit 的精度漂移 (因为 reference 每次重新计算)
- **迁移指南**: 为每个精度敏感算子编写 `_reference()` 函数; 统一 API surface; 使用 dtype-dependent threshold dict

### Pattern 2: Source Contract Lock (源码合约锁)

- **设计目标**: 防止 CUDA 内核意外引入非确定性操作 (如 `atomicAdd`)
- **源码位置**: `tests/test_deterministic_logp.py` (feat/deterministic-logp 分支)
- **工作流程**: 测试读取 CUDA `.cu` 源码, assert 包含特定拓扑常量, assert 不包含 `atomicAdd`/`cub::BlockReduce`/`select_deterministic`
- **优点**: 零运行时开销; 捕获编译时引入的非确定性; 比运行时多次对比更早发现问题
- **局限**: 仅适用于项目控制的 CUDA 源码, 不适用于外部库; 正则匹配可能 false positive
- **迁移指南**: 识别影响确定性的关键原语 (atomicAdd, threadfence, __shfl_xor_sync 的特定模式), 编写 source-content assertion

### Pattern 3: Stub-Based Hardware Path Testing

- **设计目标**: 在 CPU 上测试 GPU-specific 代码路径的数值正确性
- **源码位置**: `tests/test_sampler_temperature.py`
- **工作流程**: 创建 stub 模块 (`types.ModuleType("flashinfer")`) 替换硬件库, 捕获中间值 (如 softmax 概率), 与 fp32 reference 对比
- **优点**: 无需 GPU 即可测试 GPU 路径; 能精确定位哪些参数被传给硬件库
- **局限**: 只测试数据流正确性, 不测试硬件执行精度
- **迁移指南**: 识别硬件 dispatch 分支点; 创建 stub 模块记录输入; 验证输入精度和正确性

### Pattern 4: Deterministic Batch Factory (确定性批次工厂)

- **设计目标**: 确保测试用 RL 数据 (logits, token_ids, rewards, masks) 跨运行 bitwise 一致
- **源码位置**: `rl_engine/testing/rl_batch.py` — `make_synthetic_rl_kernel_batch()`
- **工作流程**: 使用 `torch.Generator.manual_seed(seed)` 生成所有 tensor; `SyntheticRLKernelBatch` dataclass 封装; 测试验证 `torch.equal(batch1, batch2)`
- **优点**: 完全可复现; 支持可配置参数 (shape, dtype, density, seed)
- **局限**: 仅覆盖合成数据, 不覆盖真实数据路径
- **迁移指南**: 为训练系统的每种 batch 类型建立 seeded factory; 使用 `torch.Generator` (非全局 seed) 避免干扰

---

## 8. 缺口分析与改造建议

### P0: 必须补齐

| # | Problem | Why It Matters | Partial? | Suggested Design | Modules | Expected Benefit |
|---|---------|---------------|----------|-----------------|---------|-----------------|
| 1 | 主 CI 不运行精度测试 | 精度回归可能在合并后才被发现 | GPU CI 存在但需 label 触发 | 将 CPU-runnable 精度测试 (test_grpo_loss.py, test_reference_ops.py 等) 加入 `ci.yml`; GPU 测试保持 label-triggered | `.github/workflows/ci.yml` | 每次 PR 自动捕获纯 PyTorch 层的精度回归 |
| 2 | 无 CUDA RNG seed 管理 | 含 CUDA random ops 的训练不可复现 | `torch.manual_seed` 已设 | 添加 `torch.cuda.manual_seed_all(seed)`; 设置 `torch.backends.cudnn.deterministic = True`; 配置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `deepspeed_trainer.py`, `training_contract.py` | 训练运行可复现性 |
| 3 | `SamplerBackend.compute_logp` 未 upcast fp32 | 大词表 fp16/bf16 log_softmax 精度损失 | `sample()` 已修复 | 改为 `torch.log_softmax(c_logits.float(), dim=-1).to(c_logits.dtype)` | `rl_engine/kernels/sampling.py:97` | 消除采样精度风险 |

### P1: 强烈建议补齐

| # | Problem | Why It Matters | Partial? | Suggested Design | Modules | Expected Benefit |
|---|---------|---------------|----------|-----------------|---------|-----------------|
| 4 | 无 attention 算子精度测试 | FlashAttention/Triton Attention/PrefixShared Attention 无数值验证 | 算子代码存在 | 为每个 attention op 编写 reference `F.scaled_dot_product_attention` 对比测试 | `tests/test_attention_accuracy.py` (新) | 覆盖 3 个 attention 后端的精度 |
| 5 | 无 checkpoint resume 一致性测试 | 训练中断恢复后精度无法保证 | Weight bridge 有完整性校验 | 实现 save/load checkpoint (model + optimizer + scheduler + RNG state); 添加 resume-after-save 对比测试 | `deepspeed_trainer.py`, `tests/` | 训练可恢复性 |
| 6 | Tolerance 散布无中心管理 | 维护困难, 新测试可能用错阈值 | 无 | 创建 `PRECISION_TOLERANCES = {(op, dtype, backend): (rtol, atol)}` 中心注册表 | `rl_engine/testing/tolerances.py` (新) | 一致的精度标准 |
| 7 | 无系统化 NaN/Inf 检测 | 训练中 NaN 传播可能静默 | 散落 `isfinite()` | 添加 `torch.autograd.set_detect_anomaly(True)` 模式; 在训练循环中添加 NaN 检测 hook | `deepspeed_trainer.py`, training loop | 快速定位 NaN 源头 |

### P2: 长期优化

| # | Problem | Why It Matters | Partial? | Suggested Design | Modules | Expected Benefit |
|---|---------|---------------|----------|-----------------|---------|-----------------|
| 8 | `--use_fast_math` 编译标志 | 影响 CUDA 内核 exp/log 精度 (~2 ULP) | 无 | 评估移除 `--use_fast_math` 对性能的影响; 或至少为精度敏感路径编译不带 fast_math 的版本 | `setup.py:64` | 消除编译器级精度风险 |
| 9 | 无 golden loss baseline 回归 | 跨 commit 精度漂移不可检测 | 基本收敛测试存在 | 建立 golden loss JSON (op, dtype, input_hash) → expected_output; CI 中对比 | `tests/golden/`, CI workflow | 自动检测精度回归 |
| 10 | 无分布式梯度同步测试 | DP/ZeRO 梯度聚合正确性无验证 | DeepSpeed 集成存在 | 实现 single-GPU vs multi-GPU loss/gradient 对比测试 | `tests/test_distributed_precision.py` (新) | 分布式训练正确性保证 |

---

## 9. 推荐学习路线

### 第 1 步: 读文档
- `README.md` — 项目概述和架构
- `docs/operators/` — 算子文档
- `docs/design/` — 设计决策

### 第 2 步: 跑 examples/tests
1. `python examples/grpo_single_gpu.py --steps 5` — 观察 drift summary 输出
2. `pytest tests/test_reference_ops.py -v` — 理解 reference op 体系
3. `pytest tests/test_grpo_loss.py -v` — 理解 native↔Triton 交叉验证
4. `pytest tests/test_op_accuracy.py -v` — 理解 CPU/GPU 精度对比

### 第 3 步: 读源码
1. `rl_engine/testing/reference_ops.py` — 精度验证的核心: 理解 fp32 reference 设计
2. `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py` — 理解 GRPO loss 数值稳定性策略
3. `csrc/fused_logp_kernel.cu` — 理解 online log-sum-exp 和 fp32 内部计算
4. `rl_engine/kernels/registry.py` — 理解多后端 dispatch 机制
5. `rl_engine/executors/bridge.py` — 理解权重同步和 SHA256 完整性校验

### 第 4 步: 复现 commit 中的问题
1. `f28e638a` — 复现 temperature 双除 bug (改回旧代码, 观察 T=0.5 时的输出差异)
2. `a4ca4535` — 复现 masked position NaN (移除 masked_fill, 用 valid_density=0.5 观察 NaN)
3. `f1bc93a8` — 复现 sqrt(0) 和 atomic_add 非确定性 (改回旧 clamp 策略; 多次运行 Triton forward 观察差异)

### 第 5 步: 抽象设计模式
1. Reference Op + Drift Summary 交叉验证 → 可迁移到任何算子库
2. Source Contract Lock → 可迁移到任何包含 CUDA/OpenCL 源码的项目
3. Stub-Based Hardware Path Testing → 可迁移到任何多硬件后端项目
4. Deterministic Batch Factory → 可迁移到任何训练系统

### 第 6 步: 迁移到自己的训练系统
- 优先实现 Pattern 1 (Reference Op + Drift Summary) — 投入产出比最高
- 然后补齐 P0 缺口 (CUDA seed, CI 精度测试, softmax fp32)
- 最后引入 golden loss regression 和分布式测试

---

## 10. 对自研分布式训练系统的迁移建议

1. **直接借鉴**: `rl_engine/testing/reference_ops.py` 的设计可以 1:1 复制. 为每个精度敏感算子编写 fp32 reference, 使用 `summarize_kernel_drift()` 量化误差. 这是投入产出比最高的精度基础设施

2. **避免重复**: RL-Kernel 的以下问题在自研系统中应提前解决:
   - 从 Day 1 设置 `torch.cuda.manual_seed_all(seed)` + `cudnn.deterministic`
   - 从 Day 1 在 CI 中运行精度测试 (不要等到 GPU CI 可用)
   - 从 Day 1 为 softmax/log_softmax 建立 fp32 upcast 规范

3. **Tolerance 管理**: 建立中心化 tolerance 注册表: `{(op="logp", dtype=torch.float16, backend="cuda"): (rtol=1e-3, atol=1e-3)}`. RL-Kernel 的散布式 tolerance 是维护成本来源

4. **确定性优先**: 如果训练可复现性是目标, 研究 `csrc/deterministic_logp_kernel.cu` 的设计: 固定 block size, 无 atomicAdd, 一行一 block. 性能会有 10-20% 下降, 但获得 bitwise 确定性

5. **Triton 注意事项**: `tl.atomic_add` 在精度敏感场景不可用. 改用 per-block partial sums + host 归约. 所有内部计算 cast 到 `tl.float32`

---

## Appendix A. 检索关键词与命令记录

### 源码搜索关键词

| Domain | Keywords Searched |
|--------|------------------|
| Numerical Comparison | allclose, rtol, atol, tolerance, isclose, assert_close, mismatch, drift |
| Determinism | deterministic, reproducible, seed, manual_seed, cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG |
| Golden Values | golden, baseline, expected, reference, ground_truth, regression |
| Loss | loss, loss_curve, train_loss, loss_nan, loss_inf |
| Gradient | gradient, grad, backward, grad_check, grad_norm, grad_clip, register_hook |
| Tensor Dump | dump, tensor_dump, save_tensor, torch.save, compare_tensor |
| Mixed Precision | fp32, fp16, bf16, fp8, tf32, amp, autocast, GradScaler, loss_scale, master_weight |
| Numerical Stability | nan, inf, overflow, isnan, isinf, detect_anomaly, clamp, eps |
| Optimizer | optimizer, adam, adamw, state_dict, scheduler, lr_scheduler |
| Checkpoint | checkpoint, save, load, resume, rng_state |
| Distributed | distributed, ddp, tensor_parallel, pipeline_parallel, deepspeed, zero, fsdp, world_size, rank |
| Collectives | all_reduce, reduce_scatter, all_gather, broadcast, nccl |
| Data Loading | dataloader, sampler, shuffle, drop_last |
| Testing | pytest, assert, ci, regression, correctness_test |

### Git Log 命令

执行了 50+ 条 `git log --oneline --all --grep=<keyword>` 命令, 覆盖所有上述 domain. 详见 Phase 3 subagent 报告.

## Appendix B. 关键文件清单

| File | Precision Relevance |
|------|-------------------|
| `rl_engine/testing/reference_ops.py` | **核心**: 6 个 fp32 reference 实现 + drift summary |
| `rl_engine/testing/rl_batch.py` | 确定性 batch 工厂 (seeded Generator) |
| `rl_engine/kernels/ops/pytorch/loss/grpo_loss.py` | GRPO loss PyTorch oracle — eps^2 clamp, mask-before-exp |
| `rl_engine/kernels/ops/pytorch/loss/logp.py` | NativeLogpOp — fp32 log_softmax fallback |
| `rl_engine/kernels/ops/cuda/loss/logp.py` | FusedLogpGenericOp — 10 CUDA 内核变体 wrapper |
| `rl_engine/kernels/ops/triton/triton_grpo_loss.py` | Triton GRPO loss — fp32 内部计算, per-block partials |
| `rl_engine/kernels/ops/triton/triton_attn.py` | Triton FlashAttention — fp32 accumulator |
| `rl_engine/kernels/sampling.py` | Sampling — compute_logp 无 fp32 upcast (P0 风险) |
| `rl_engine/kernels/registry.py` | 多后端 dispatch — 无全局精度策略 |
| `rl_engine/executors/deepspeed_trainer.py` | DeepSpeed 训练 — 仅 CPU seed, 无梯度裁剪 |
| `rl_engine/executors/bridge.py` | 权重传输 — SHA256 完整性, 4 种后端 |
| `rl_engine/executors/training_contract.py` | 训练配置 — dtype/seed 定义 |
| `csrc/fused_logp_kernel.cu` | CUDA fused logp — online log-sum-exp, fp32 reduction |
| `csrc/cuda/fused_logp_sm90.cu` | SM90 TMA logp — bf16 only, fp32 accumulation |
| `csrc/cuda/attention/prefix_shared_attention.cu` | Prefix attention — bf16 MMA, `__expf` (精度风险) |
| `setup.py` | `--use_fast_math` flag (精度风险) |
| `tests/test_op_accuracy.py` | 22 logp 精度测试 — dtype-adaptive tolerance |
| `tests/test_grpo_loss.py` | 20+ GRPO 精度测试 — 三级容差阶梯 |
| `tests/test_rl_kernel_loss_step.py` | E2E pipeline 精度 — drift threshold |
| `tests/test_sampler_temperature.py` | Temperature 回归 — stub-based |
| `tests/test_reference_ops.py` | Reference op 自检 |
| `tests/test_rl_batch_fixture.py` | Batch 确定性验证 |
| `tests/test_weight_sync_bridge.py` | 权重传输完整性 — SHA256 + torch.equal |
| `.github/workflows/ci.yml` | 主 CI — 不运行精度测试 (P0 缺口) |
| `.github/workflows/gpu-ci.yml` | GPU CI — 运行全套但需手动触发 |
| `ci/run_gpu_ci.sh` | GPU CI 脚本 — RunPod 云 GPU |

## Appendix C. 关键 commits / PR / issues 清单

| Hash | Date | Summary |
|------|------|---------|
| `ec042acd` | 2026-05-26 | TMA SM90 fused logp kernel (实验性) |
| `0da66070` | 2026-05-29 | Native fused logp 多变体 + online log-sum-exp |
| `409a32ed` | 2026-05-29 | Fused logp registry 重构 |
| `6579d1ab` | 2026-05-30 | **精度测试基础设施建立** (reference ops, batch factory, drift summary) |
| `3315bc16` | 2026-06-02 | Native logp API parity — fp32 一致性 |
| `cddd5b00` | 2026-06-02 | 零拷贝权重同步 bridge + SHA256 |
| `ecb73244` | 2026-06-02 | Weight sync validation 加固 |
| `17e88148` | 2026-06-03 | RMSNorm reference + fp32 promotion test [issue-38-rms-norm-fusion-kernel] |
| `9a804949` | 2026-06-08 | FlashInfer sampling fp32 upcast |
| `f28e638a` | 2026-06-08 | **Temperature 双除 bug fix** + 回归测试 |
| `1c3e388e` | 2026-06-08 | Blackwell SM120 兼容修复 |
| `5304c1f0` | 2026-06-08 | Alignment wrapper ignore-index fix |
| `13195f53` | 2026-06-09 | GRPO loss op (PyTorch oracle + Triton kernel) |
| `25e19bf1` | 2026-06-09 | GRPO loss tests — 三级容差验证 |
| `a4ca4535` | 2026-06-09 | **Masked position NaN fix** (mask before exp) |
| `f1bc93a8` | 2026-06-10 | **sqrt(0) + atomic_add 非确定性 fix** |
| `3f44526` | 2026-06-11 | **确定性 CUDA logp** (batch-invariance, source contract lock) [feat/deterministic-logp] |
| `302386d` | 2026-06-12 | 确定性 logp 扩展覆盖 [feat/deterministic-logp] |
| `08c7369` | 2026-06-12 | 确定性 logp 极端值测试 [feat/deterministic-logp] |
| `7db5247` | 2026-06-12 | 确定性 logp bucket block size [feat/deterministic-logp] |

**注意**: 上表中标记了分支名的 commits 位于 feature branches, 尚未合并到 main. 能力矩阵评分反映 main + feature branches 的综合状态; 仅基于 main 分支的评分会有差异 (主要影响确定性 LogP 和 RMSNorm 相关能力).

---

## Appendix D. 审计验证记录

### 审计轮次: 1
### 最终判定: PASS

#### Architect-Reviewer 审计结果

- **轮次 1**: PASS (with minor corrections)
  - ~~Critical 1~~: Entry 6 (commit `17e88148` / RMSNorm) 未标注为未合并分支 → **已修复**: 添加 "(issue-38-rms-norm-fusion-kernel 分支, 未合并到 main)" 注解
  - ~~Critical 2~~: `test_op_accuracy.py` 测试数量标注为 22, 实际为 16 (15 函数 + 1 参数化) → **已修复**: 改为 "15 test functions, 16 cases with parameterization"
  - Minor 1: `tests/` 目录文件数从 16 改为 14 → **已修复**
  - Minor 2: `test_grpo_loss.py` 测试数量从 "20+" 改为 19 → **已修复**
  - Minor 3: `fused_logp_forward_online_kernel` line 158→159 — 可接受 (装饰器 vs 函数名行)
  - Minor 4: `DeepSpeedTrainingConfig` line 42→43 → **已修复**
  - 验证项: B1 Maturity 4 ✓, C2 Maturity 3 ✓, D1-D6 Maturity 0 ✓, E1 Maturity 2 ✓, 4 个设计模式 ✓, 15 个 commit hash ✓, 三阶段映射 ✓

#### Code-Reviewer 审计结果

- **轮次 1**: PASS (0 critical issues)
  - Minor 1: 测试文件数 16→14 → **已修复** (与 architect-reviewer 重复)
  - Minor 2: `test_op_accuracy.py` 测试数量 22→16 → **已修复** (与 architect-reviewer 重复)
  - Minor 3: `test_grpo_loss.py` 测试数量 "20+"→19 → **已修复** (与 architect-reviewer 重复)
  - Minor 4: Entry 6 未标注分支 → **已修复** (与 architect-reviewer 重复)
  - Minor 5: `DeepSpeedTrainingConfig` line 42→43 → **已修复**
  - Minor 6: `WeightLayout` line 32→33 — 装饰器行 vs class 行, 可接受
  - Minor 7: online kernel line 158→159 — 可接受
  - 验证项: 全部 20 个 commit hash ✓, 所有文件路径 ✓, 所有函数/类名 ✓, CI workflow claims ✓, 无幻觉证据

#### 未通过项
无
