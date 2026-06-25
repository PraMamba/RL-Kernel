# KernelWiki Training Library Expansion Report: RL-Kernel

**Library:** RL-Kernel  
**GitHub URL:** RL-Align/RL-Kernel  
**Local Source Path:** /root/RL-Kernel  
**Analysis Date:** 2026-06-12  
**Library Type Classification:** Training-Compute (RLHF/GRPO-specialized kernel library with orchestration layer)

---

## Dimension 1: Compute Kernels

### Kernel File Census

Total kernel files: **7**
- CUDA/C++: **4** files in `csrc/`, `csrc/cuda/`, `csrc/cuda/attention/`, `csrc/utils/`
  - `csrc/fused_logp_kernel.cu` (588 lines)
  - `csrc/cuda/fused_logp_sm90.cu` (124 lines)
  - `csrc/cuda/attention/prefix_shared_attention.cu` (340 lines)
  - `csrc/utils/tma_utils.cuh` (84 lines)
- Triton: **2** files in `rl_engine/kernels/ops/triton/`
  - `rl_engine/kernels/ops/triton/triton_attn.py`
  - `rl_engine/kernels/ops/triton/triton_grpo_loss.py`
- Extension entry point: `csrc/ops.cpp` (PYBIND11_MODULE `rl_engine._C`)

### Training-Specific / RL-Specific Kernels

| Kernel | File Path | Proposed Tag | Description |
|--------|-----------|--------------|-------------|
| `fused_logp_forward_kernel` (two-pass) | `csrc/fused_logp_kernel.cu:L112` | fused-logp | Two-pass numerically stable log-probability extraction for policy gradient training |
| `fused_logp_forward_online_kernel` (one-pass) | `csrc/fused_logp_kernel.cu:L158` | fused-logp | Single-pass online log-sum-exp variant; halves vocabulary memory reads |
| `fused_logp_online_tma_kernel` (SM90) | `csrc/cuda/fused_logp_sm90.cu:L11` | fused-logp | SM90/Hopper TMA-accelerated variant with warp specialization and mbarrier producer-consumer pipeline |
| `_bwd_preprocess` | `rl_engine/kernels/ops/triton/triton_attn.py:L134` | attention-backward | Backward preprocessing: computes Delta = sum(O * dO) per row |
| `_bwd_kernel` | `rl_engine/kernels/ops/triton/triton_attn.py:L185` | attention-backward | Full dQ/dK/dV backward with softmax recompute from saved M/L statistics |
| `_group_norm_kernel` | `rl_engine/kernels/ops/triton/triton_grpo_loss.py:L30` | grpo-advantage | Per-group reward normalization for GRPO advantage computation; CSR-indexed groups |
| `_grpo_fwd_kernel` | `rl_engine/kernels/ops/triton/triton_grpo_loss.py:L58` | grpo-loss | Fused GRPO surrogate loss: clipped PPO policy term + K3 KL estimator in one kernel |
| `_grpo_bwd_kernel` | `rl_engine/kernels/ops/triton/triton_grpo_loss.py:L100` | grpo-loss | Backward for GRPO loss; recomputes from saved tensors (memory-efficient checkpointing) |

### Inference-Shared Kernels (with training behavior differences)

| Kernel | File Path | Training Behavior Difference |
|--------|-----------|------------------------------|
| `prefix_shared_attention_kernel` | `csrc/cuda/attention/prefix_shared_attention.cu:L85` | GRPO-optimized: K loaded once into shared memory and reused by all G groups; layout `Q:[bs,G,len_q,DIM]`, K/V:`[bs,len_kv,DIM]` |
| `_fwd_kernel` (Triton FlashAttention) | `rl_engine/kernels/ops/triton/triton_attn.py:L6` | Saves L (sum) and M (max) for backward pass; used for training-time attention |
| `SamplerBackend.sample` | `rl_engine/kernels/sampling.py:L41` | Delegates to FlashInfer top-k/top-p for rollout sampling in RLHF pipeline |

### Fused Kernels

| Kernel | Operations Fused | Estimated Memory Savings |
|--------|-----------------|------------------------|
| `fused_logp_forward_kernel` | log_softmax + gather (log-probability extraction) | Eliminates O(G·L·V) intermediate softmax tensor via chunking + pre-allocation |
| `fused_logp_forward_online_kernel` | max-reduction + sum-exp in single pass (online log-sum-exp) | Halves vocabulary memory reads vs. two-pass variant |
| `fused_logp_online_tma_kernel` | TMA bulk load + online log-sum-exp with warp specialization | TMA frees compute warps from load duty; 4096-element tiles avoid full vocabulary in registers |
| `prefix_shared_attention_kernel` | QK^T + softmax + PV (FlashAttention-style) with prefix sharing | K loaded once, shared across G groups; online softmax avoids full attention matrix |
| `_grpo_fwd_kernel` | Clipped PPO surrogate + K3 KL estimator | Single kernel handles policy loss + KL; avoids two separate reduction passes |

### Kernel Dependency Graph

RL-Kernel provides its own CUDA/Triton kernels for core operations but orchestrates upstream libraries for attention and sampling:

| Provider Library | Kernel Types Provided | Import/Include Evidence |
|-----------------|----------------------|------------------------|
| FlashAttention | `flash_attn_func` (forward attention) | `rl_engine/kernels/ops/cuda/attention/flash_attn.py:L14` — `from flash_attn import flash_attn_func` |
| FlashInfer | `top_k_renorm_probs`, `top_p_sampling_from_probs` | `rl_engine/kernels/sampling.py:L6` — `from flashinfer.sampling import ...` |
| AITER (AMD, stub) | Planned: ROCm attention and logp ops | `rl_engine/kernels/ops/rocm/__init__.py` — empty; registry references `rocm.aiter.AiterOp` |
| CUB | `cub::BlockReduce` | `csrc/cuda/fused_logp_sm90.cu:L6` — `#include <cub/block/block_reduce.cuh>` |
| CUDA Driver API | `cuTensorMapEncodeTiled` | `csrc/utils/tma_utils.cuh:L22` — TMA descriptor initialization |

### Proposed New kernel_types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| fused-logp | `csrc/fused_logp_kernel.cu` | Fused log-softmax + gather for per-token log-probability extraction in policy gradient training |
| grpo-loss | `rl_engine/kernels/ops/triton/triton_grpo_loss.py` | Fused GRPO surrogate loss (clipped PPO + K3 KL estimator) |
| grpo-advantage | `rl_engine/kernels/ops/triton/triton_grpo_loss.py:L30` | Per-group reward normalization for GRPO advantage computation |
| prefix-shared-attention | `csrc/cuda/attention/prefix_shared_attention.cu` | GRPO-optimized attention with shared-prefix K/V reuse across generation groups |
| attention-backward | `rl_engine/kernels/ops/triton/triton_attn.py:L134` | Training-time attention backward (dQ/dK/dV) with softmax recompute |

---

## Dimension 2: Communication Kernels and Strategies

### Collective Operations

RL-Kernel contains **no direct distributed collective operations** (no `torch.distributed` calls, no explicit NCCL usage). All gradient communication is delegated to DeepSpeed internals.

| Operation | Present | Mechanism | File Path |
|-----------|---------|-----------|-----------|
| AllReduce (gradient sync) | Indirect | DeepSpeed engine internals (ZeRO-0/1/2) | `rl_engine/executors/deepspeed_trainer.py:L131` |
| AllGather (ZeRO-3 weight export) | Indirect | `deepspeed.zero.GatheredParameters` | `rl_engine/executors/deepspeed_trainer.py:L207` |
| ReduceScatter | Indirect | DeepSpeed ZeRO-3 internals | Delegated |
| Send/Recv | Not present | — | — |
| AllToAll | Not present | — | — |

### Weight Synchronization Bridge (Primary Communication System)

RL-Kernel's primary communication innovation is a **multi-transport weight synchronization bridge** for trainer-to-rollout weight transfer.

| Transport | Class | Mechanism | SM Usage | File Path |
|-----------|-------|-----------|----------|-----------|
| `local-clone` | `LocalTensorCopyBridge` | `tensor.detach().clone()` — CPU/GPU tensor copy | None (CPU) | `bridge.py:L1508` |
| `shared-memory` | `SharedMemoryTensorBridge` | POSIX `multiprocessing.shared_memory` zero-copy alias | None (CPU) | `bridge.py:L1696` |
| `cuda-vmm` (default) | `CUDAVMMTensorBridge` | CUDA VMM `cuMemCreate` + POSIX-fd + DLPack alias | Zero SM (Copy Engine D2D) | `bridge.py:L1993` |
| `cuda-ipc` | `IPCWeightBridge` | PyTorch `reduce_tensor` IPC handles | None | `bridge.py:L2371` |
| `multi-node/rdma/nccl-rdma` | Blocked | `WeightBridgeUnavailableError` | — | `bridge.py:L2654` |

### Communication-Compute Overlap Patterns

| Pattern | Mechanism | Evidence |
|---------|-----------|----------|
| TMA + mbarrier double-buffering (intra-kernel) | Warp 0 drives TMA loads while warps 1..N compute online log-sum-exp on current tile | `csrc/cuda/fused_logp_sm90.cu:L46-68` |
| CUDA VMM async D2D copy | `cuMemcpyDtoDAsync_v2` runs on Copy Engine concurrently with compute | `bridge.py:L1325` |
| cp.async K prefetch in attention | K tile N+1 prefetched via `cp.async.cg.shared.global` while computing QK^T for tile N | `csrc/cuda/attention/prefix_shared_attention.cu:L168-210` |
| DeepSpeed gradient overlap | Delegated to DeepSpeed internals | Implicit in `engine.backward()` |

### Advanced Communication Features Checklist

- [ ] Symmetric memory support (NCCL 2.27+): **No**
- [ ] Device API support (NCCL 2.28+): LSA **No**, Multimem **No**, GIN **No**
- [ ] Copy Engine zero-SM collectives: **No** (but intra-kernel TMA Copy Engine usage present)
- [ ] NCCL Inspector integration: **No**
- [ ] PyTorch SymmetricMemory: **No**
- [ ] Alternative backend support (MSCCL++): **No**
- [x] CUDA VMM zero-copy cross-process weight transfer: **Yes** — `cuMemCreate` + `cuMemExportToShareableHandle` + POSIX-fd + `SCM_RIGHTS` broker
- [x] CUDA IPC event synchronization: **Yes** — `torch.cuda.Event(interprocess=True)` for producer-consumer stream ordering

### Proposed New Communication kernel_types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| weight-sync-bridge | `rl_engine/executors/bridge.py` | Multi-transport weight synchronization for RLHF trainer-to-rollout communication |

### Proposed New Communication techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| cuda-vmm-zero-copy | `bridge.py:L1993` | CUDA Virtual Memory Management with POSIX-fd export for zero-copy cross-process GPU tensor sharing |
| ipc-event-sync | `bridge.py:L2070` | Interprocess CUDA event for stream-ordered memory visibility without full device sync |
| fd-broker | `bridge.py:L2243` | Unix domain socket `SCM_RIGHTS` fd passing for CUDA VMM allocation handle transfer |

---

## Dimension 3: Parallelism Strategies

### Supported Parallelism Dimensions

| Dimension | Supported | Implementation File | Communication Pattern Triggered |
|-----------|-----------|--------------------|---------------------------------|
| Data Parallel (DeepSpeed) | **Yes** | `deepspeed_trainer.py:L131` | AllReduce via DeepSpeed internals |
| ZeRO-1/2 (optimizer/gradient sharding) | **Yes** | `deepspeed_trainer.py:L255` | Implicit in DeepSpeed engine |
| ZeRO-3 (parameter sharding) | **Yes** | `deepspeed_trainer.py:L178-236` | AllGather via `GatheredParameters` on weight publish |
| Tensor Parallel | **Blocked** | `bridge.py:L91-95` | `WeightBridgeUnavailableError` when `tensor_parallel_size != 1` |
| Pipeline Parallel | **Enum only** | `constants.py:L77` | Not implemented |
| Context Parallel | **Not present** | — | — |
| Expert Parallel | **Not present** | — | — |
| Sequence Parallel | **Not present** | — | — |
| FSDP / DeviceMesh | **Not present** | — | — |
| **RLHF Rollout-Training Split** | **Yes (primary)** | `ray_actor_manager.py`, `bridge.py`, `rollout.py` | Weight bridge (CUDA VMM / shared-memory / IPC) + Ray RPC |
| Multi-node / RDMA | **Blocked** | `bridge.py:L96-100` | `WeightBridgeUnavailableError` |

### RLHF-Specific Parallelism: Rollout-Training Protocol

The central parallelism strategy is a **process-level separation** between training and rollout (inference), mediated by a versioned weight bridge:

1. **Training step** (`DeepSpeedTrainingWorker.train(rollout)`) — Consumes a `RolloutStageResult`, computes PPO/GRPO loss, runs `engine.backward()` + `engine.step()`. Returns `TrainingStageResult` with `published_weight_version = consumed_weight_version + 1`.

2. **Weight publication** — Gathers ZeRO-3 shards if needed (AllGather), then pushes all tensors into CUDA VMM allocation. Starts fd-broker thread.

3. **Weight delivery** — Ray orchestrator delivers `WeightUpdateManifest` to rollout actor. Rollout executor imports via bridge (connects to fd-broker, receives POSIX-fd via `SCM_RIGHTS`, maps CUDA VMM zero-copy, waits on IPC event).

4. **Rollout execution** — vLLM generates completions under new weights. `VLLMSharedPrefixSampler` expands prompts × G for GRPO, preserving shared prefix for KV-cache hits.

5. **Weight release** — Previous version's CUDA VMM allocation freed via `cuMemUnmap` + `cuMemRelease`.

The training and rollout phases are **on-policy and synchronous**: rollout always uses the weights from the most recent training step. No experience buffer or off-policy replay.

### Ray Actor Orchestration

- `RayActorManager` creates `_RayWorkerActor` remote actors for both training and rollout workers
- All inter-worker calls use `ray.get(actor.method.remote(...))` — fully synchronous from the caller's perspective
- `WeightUpdateManifest` is serialized through Ray's object store (Arrow/Plasma) for manifest metadata; actual tensor data goes through the weight bridge transport

### vLLM Weight Install Adapters

| Adapter | Mechanism | Use Case |
|---------|-----------|----------|
| `VLLMInProcessWeightReloadAdapter` | `engine.reload_weights()` / `collective_rpc("reload_weights")` | Single-process vLLM (V1) |
| `VLLMCheckpointWeightReloadAdapter` | `engine.reload_weights(weights_path=...)` | Non-IPC environments |
| `VLLMCUDAVMMExternalStorageAdapter` | `engine.apply_model(fn)` with DLPack `parameter.data` rebinding | Zero-copy production path |
| `VLLMIPCWeightUpdateRequestBuilder` | `LLM.update_weights({"update_info": ...})` | vLLM 0.18+ IPC backend |

### Proposed New Parallelism techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| rlhf-rollout-training-split | `ray_actor_manager.py`, `bridge.py`, `rollout.py` | Process-level separation of training and rollout with versioned weight bridge |
| on-policy-weight-sync | `bridge.py:L1993` | Synchronous on-policy weight delivery via CUDA VMM zero-copy |
| zero3-gather-on-publish | `deepspeed_trainer.py:L207` | ZeRO-3 AllGather triggered at weight publication boundary, not at every forward pass |

---

## Dimension 4: Memory Management

### Memory Component Analysis

| Component | Storage Format | Sharding Strategy | Communication Kernel Triggered |
|-----------|---------------|-------------------|---------------------------------|
| Parameters | FP16/BF16 | DeepSpeed ZeRO-0/1/2/3 | AllGather for ZeRO-3 publish |
| Gradients | FP16/BF16 | DeepSpeed ZeRO-2/3 | AllReduce (ZeRO-0/1) / ReduceScatter (ZeRO-2/3) |
| Optimizer States | FP32 | DeepSpeed ZeRO-1/2/3 | None (local optimizer step) |
| Logit intermediates | FP32 (compute) | **Chunked** (chunk_size=4096) | None (local) |
| Rollout weights (bridge) | FP16/BF16 | CUDA VMM zero-copy alias | Copy Engine D2D (async) |

### Core Memory Strategy: Chunked Pre-allocated Log-Probability Computation

This is RL-Kernel's signature technique for solving the O(G·L·V) memory explosion in GRPO:

| Strategy | File | Lines | Memory Savings |
|----------|------|-------|----------------|
| Chunked logp (chunk_size=4096) | `rl_engine/kernels/sampling.py` | 86-102 | O(G·L·V) → O(4096·L·V); ~31x for G=128 |
| Pre-allocated output tensors | `csrc/fused_logp_kernel.cu` | 452-587 | Zero temporary allocation per kernel call |
| Online log-sum-exp (single-pass) | `csrc/fused_logp_kernel.cu` | 158-206 | Halves vocabulary memory reads |
| Sparse indexed dispatch | `csrc/fused_logp_kernel.cu` | 475-552 | ~50% compute/memory for padded batches |
| CUDA VMM zero-copy bridge | `rl_engine/executors/bridge.py` | 1993-2368 | Eliminates full model weight duplication (~14 GB for 7B) |
| Shared memory tiling + double-buffering | `csrc/cuda/attention/prefix_shared_attention.cu` | 83-311 | ~10x bandwidth reduction (SRAM vs. HBM) |
| TMA + mbarrier async DMA (SM90) | `csrc/cuda/fused_logp_sm90.cu` | 1-124 | 4096-element tiles; frees compute warps from load duty |
| DeepSpeed ZeRO-stage sharding | `rl_engine/executors/deepspeed_trainer.py` | 252-261 | Up to N× per-GPU reduction (N=world_size) at ZeRO-3 |
| Gradient accumulation | `rl_engine/executors/deepspeed_trainer.py` | 256 | 1/N activation footprint |
| `set_to_none=True` gradient free | `rl_engine/executors/deepspeed_trainer.py` | 124-128 | Frees gradient buffers between steps |
| Triton block-partial reduction | `rl_engine/kernels/ops/triton/triton_grpo_loss.py` | 149-218 | Avoids saving intermediate activations in autograd |

### Activation Checkpointing Strategies

| Strategy | Supported | Evidence |
|----------|-----------|----------|
| Full recompute | **Not implemented** | No `torch.utils.checkpoint` usage found |
| Selective (attention only) | **Not implemented** | — |
| No recompute | Default | All activations stored (standard PyTorch autograd) |

RL-Kernel does not implement activation checkpointing. Memory savings come from the chunked/fused kernel approach rather than recomputation.

### Memory Estimation (reference: Llama-3-8B, 8B parameters)

| Component | Per-GPU Memory (no optimization, single GPU) | Per-GPU Memory (with ZeRO-3 on 8 GPUs) |
|-----------|------------------------------------------------|------------------------------------------|
| Parameters (2×PSI, BF16) | 16 GB | 2 GB |
| Gradients (2×PSI, BF16) | 16 GB | 2 GB |
| Optimizer (12×PSI, FP32 Adam) | 96 GB | 12 GB |
| Logit intermediates (naive G=128) | 16.8 GB | 16.8 GB (not sharded) |
| Logit intermediates (chunked, chunk=4096) | 0.5 GB | 0.5 GB |
| **Total** | **144.8 GB (naive)** / **128.5 GB (chunked)** | **33.3 GB (chunked + ZeRO-3)** |

---

## Dimension 5: Precision Management

### FP8 Scaling Strategies Found

**None.** FP8 is entirely absent. All searches for E4M3, E5M2, fp8, FP8, float8, DelayedScaling, MXFP8, amax_history returned zero results. The precision surface is strictly FP16/BF16/FP32.

### Supported Precision Formats

| Format | Used In | Evidence |
|--------|---------|----------|
| FP32 | Accumulation everywhere; optimizer states; reference ops | Universal across all kernels |
| BF16 | SM90 TMA kernel input; attention Tensor Core MMA input; ROCm preferred dtype | `fused_logp_sm90.cu`, `prefix_shared_attention.cu`, `device.py:L47` |
| FP16 | NVIDIA CUDA preferred dtype; FlashAttention input | `device.py:L52`, `flash_attn.py:L52` |
| INT8/INT4 | Declared in `PrecisionType` enum | `constants.py:L28-29` — not implemented |

### Precision per Training Component

| Component | Storage Format | Compute Accumulation | Output | Evidence |
|-----------|---------------|---------------------|--------|----------|
| CUDA fused logp (two-pass) | FP16/BF16/FP32 (polymorphic) | FP32 registers | Configurable (FP16/BF16/FP32) | `csrc/fused_logp_kernel.cu:L127-151` |
| CUDA fused logp fp32 variants | FP16/BF16 input | FP32 registers | FP32 forced | `csrc/fused_logp_kernel.cu:L560-588` |
| SM90 TMA fused logp | BF16 only | FP32 (explicit `__bfloat162float`) | FP32 forced | `csrc/cuda/fused_logp_sm90.cu:L72,82,103` |
| Prefix-shared attention | BF16 only | FP32 MMA accumulators (`f32.bf16.bf16.f32`) | BF16 (downcast via `__float22bfloat162_rn`) | `csrc/cuda/attention/prefix_shared_attention.cu:L62-72` |
| Triton FlashAttention | FP16/BF16 (polymorphic) | FP32 (`tl.float32`) | Input dtype (downcast) | `triton_attn.py:L72-74,124` |
| Triton GRPO loss | FP16/BF16 input | FP32 (`.to(tl.float32)`) | FP32 grads → downcast to input dtype | `triton_grpo_loss.py:L76-80,197,216` |
| PyTorch native logp | FP16/BF16/FP32 | FP32 (`.float()`) | Configurable | `pytorch/loss/logp.py:L30` |
| PyTorch GRPO loss | FP16/BF16/FP32 | FP32 (`.float()`) | FP32 | `pytorch/loss/grpo_loss.py:L101-104` |
| DeepSpeed trainer | FP16 or BF16 (config) | FP32 for loss ops | N/A | `deepspeed_trainer.py:L258-261` |
| Weight bridge transport | Any dtype | N/A | Configurable via `target_dtype` | `bridge.py:L385-386` |

### Numerical Stability Techniques

| Technique | Location | Description |
|-----------|----------|-------------|
| Two-pass log-softmax with warp reduction | `fused_logp_kernel.cu:L112-156` | Separate max-reduction pass prevents exp overflow |
| Online log-sum-exp state machine | `fused_logp_kernel.cu:L52-108` | `LogSumExpState` tracks running `(max_val, sum_exp)` with proper rescaling |
| FP32 accumulation in all Tensor Core ops | `prefix_shared_attention.cu:L62` | `mma.sync f32.bf16.bf16.f32` — prevents BF16 accumulation drift |
| Epsilon guard in group normalization | `triton_grpo_loss.py:L52` | `tl.maximum(std, eps)` with `eps=1e-6` prevents zero-std division |
| Active token clamping | `triton_grpo_loss.py:L163` | `mask_f.sum().clamp_min(1e-8)` prevents zero-division on empty batches |
| Explicit `.float()` upcast before log_softmax | `pytorch/loss/logp.py:L30` | Forces FP32 softmax regardless of input dtype |
| Accuracy thresholds by dtype | `tests/test_op_accuracy.py:L152` | BF16/FP16: 1e-2 tolerance; FP32: 1e-5 tolerance |

### FP8 Communication Integration

- [ ] FP8 AllGather in FSDP2: **Not applicable** (no FP8 support)
- [ ] FP8 ReduceScatter: **Not applicable**
- [ ] NVLink-SHARP FP8 in-switch reduction: **Not applicable**

### Proposed New Precision techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| online-logsumexp | `csrc/fused_logp_kernel.cu:L52` | Single-pass online log-sum-exp with running state machine for numerically stable softmax |
| fp32-accumulation | `csrc/cuda/attention/prefix_shared_attention.cu:L62` | FP32 accumulation in Tensor Core MMA despite BF16 inputs |

---

## Dimension 6: Profiling and Observability

### Built-in Profiling Capabilities

| Feature | Supported | Integration Method | Evidence |
|---------|-----------|-------------------|----------|
| NVTX annotations for nsys | **No** | — | No `torch.cuda.nvtx` or `record_function` usage |
| NCCL Inspector plugin | **No** | — | — |
| Flight Recorder | **No** | — | — |
| Structured logging | **Partial** | `rl_engine/utils/logger.py` with `info_once`, `warn_once`, `info_on_rank` | Only text format; no JSON/structured output |
| MFU metrics | **No** | TFLOPS estimated but not divided by hardware peak | `benchmarks/profiler.py:L227` |
| Memory profiling | **Yes** | `torch.cuda.max_memory_allocated` with baseline subtraction | `benchmarks/profiler.py:L177-224` |
| CUDA Event timing | **Yes** | `torch.cuda.Event(enable_timing=True)` median latency | `benchmarks/profiler.py:L187-214` |
| NCU line-info build flag | **Yes** | `-lineinfo` via `KERNEL_ALIGN_NCU_LINEINFO=1` env var | `setup.py:L107-108` |

### Benchmark Infrastructure

| Benchmark | What It Measures | Output Format | File |
|-----------|-----------------|---------------|------|
| `benchmark_rl_kernels.py` | Accuracy + latency for all FusedLogp variants | CSV + console table | `benchmarks/benchmark_rl_kernels.py` |
| `benchmark_grpo_op.py` | VRAM overhead: RL-Kernel vs. native PyTorch | Console table | `benchmarks/benchmark_grpo_op.py` |
| `benchmark_grpo_loss.py` | Forward + forward+backward latency for GRPO loss | Console table | `benchmarks/benchmark_grpo_loss.py` |
| `benchmark_sampling.py` | Sampling latency: native vs. FlashInfer-backed | Console table | `benchmarks/benchmark_sampling.py` |
| `benchmark_attention.py` | Attention TFLOPS via Triton `do_bench` | Console table | `benchmarks/benchmark_attention.py` |
| `benchmark_weight_sync_bridge.py` | Phase-level timing for all transport modes | CSV + JSON stdout | `benchmarks/benchmark_weight_sync_bridge.py` |
| `PerformanceProfiler` suite | Unified logp + sampling benchmark with JSON/CSV report | Timestamped JSON + CSV | `benchmarks/profiler.py:L156-481` |

### Training Step Observability

| Metric | Source | Evidence |
|--------|--------|----------|
| `started_at` / `finished_at` / `duration_seconds` | `TrainingStageResult` | `training_contract.py:L31-44` |
| `loss` | Training step output | `deepspeed_trainer.py:L136` |
| `active_tokens` | Training step output | `deepspeed_trainer.py:L138` |
| `training_backend` | Training step metadata | `deepspeed_trainer.py:L139` |
| `deepspeed_zero_stage` | Training step metadata | `deepspeed_trainer.py:L140` |

### Recommended Profiling Dimensions for KernelWiki

| Dimension | What It Measures | Tool | Key Metrics |
|-----------|-----------------|------|-------------|
| Logp kernel latency | Fused logp kernel execution time | CUDA Events via `PerformanceProfiler` | median_latency_ms, tokens_per_sec |
| VRAM overhead | Extra memory beyond input tensors | `torch.cuda.max_memory_allocated` baseline subtraction | peak_vram_reduction_gb |
| Weight sync latency | End-to-end trainer→rollout weight delivery | `time.perf_counter` phase timing | publish_ms, import_ms, ack_ms, release_ms |
| Kernel numerical accuracy | Drift from reference PyTorch implementation | Max absolute error + ratio/KL drift | max_error, ratio_drift, kl_drift |
| GRPO loss throughput | Forward + backward throughput for GRPO surrogate loss | CUDA Events | fwd_ms, fwd+bwd_ms, peak_vram_mb |
| Attention TFLOPS | Prefix-shared attention throughput | Triton `do_bench` | tflops, speedup_vs_native |

---

## Synthesis: Expansion Decision Summary

### S.1 Library Classification

| Property | Value |
|----------|-------|
| Library | RL-Kernel |
| GitHub URL | RL-Align/RL-Kernel |
| Type | **training-compute** (RLHF/GRPO-specialized kernel library with orchestration layer) |
| Contains CUDA Kernels | Yes (4 CUDA files + 2 Triton files) |
| Primary Knowledge Dimensions | Dim 1 (Compute Kernels), Dim 4 (Memory Management), Dim 3 (RLHF Parallelism) |
| Recommended KernelWiki Priority | **P1** (important RLHF-specific kernel provider; niche but growing domain) |

### S.2 Proposed Tags (for controlled vocabulary YAML)

```yaml
kernel_types:
  # New from RL-Kernel
  - fused-logp           # Fused log-softmax + gather for per-token log-probability extraction
  - grpo-loss            # Fused GRPO surrogate loss (clipped PPO + KL estimator)
  - grpo-advantage       # Per-group reward normalization for GRPO advantage computation
  - prefix-shared-attention  # GRPO-optimized attention with shared-prefix K/V reuse
  - attention-backward   # Training-time attention backward (dQ/dK/dV) with softmax recompute
  - weight-sync-bridge   # Multi-transport weight synchronization for RLHF pipeline

techniques:
  # New from RL-Kernel
  - online-logsumexp         # Single-pass online log-sum-exp with running state machine
  - pre-allocated-chunking   # Pre-allocated output buffers with fixed-size micro-chunking to cap memory
  - sparse-indexed-dispatch  # Kernel launch only for valid (non-padding) token indices
  - cuda-vmm-zero-copy      # CUDA VMM with POSIX-fd export for zero-copy cross-process GPU sharing
  - ipc-event-sync           # Interprocess CUDA event for stream-ordered memory visibility
  - fd-broker                # Unix domain socket SCM_RIGHTS fd passing for allocation handle transfer
  - rlhf-rollout-training-split  # Process-level separation of training and rollout with versioned weight bridge
  - on-policy-weight-sync   # Synchronous on-policy weight delivery via zero-copy transport
  - zero3-gather-on-publish  # ZeRO-3 AllGather at weight publication boundary
  - prefix-kv-sharing       # Shared-prefix K/V reuse across GRPO generation groups (load K once, reuse G times)
  - fp32-accumulation       # FP32 accumulation in Tensor Core MMA despite lower-precision inputs

hardware_features:
  # New from RL-Kernel (already proposed by inference libraries, confirmed training usage)
  - tma                  # Tensor Memory Accelerator (SM90+) for async bulk shared memory loads
  - mbarrier             # SM90 named barrier for producer-consumer warp synchronization

source_categories:
  # New
  - rlhf-training-framework  # RLHF/alignment post-training framework with specialized kernels
```

### S.3 Wiki Page Topics

| # | Wiki Subdirectory | Proposed Page ID | Title | Source Evidence | Related Existing KernelWiki Pages |
|---|-------------------|------------------|-------|----------------|-----------------------------------|
| 1 | training/ | training-fused-logp | Fused Log-Probability Kernels for RLHF | `csrc/fused_logp_kernel.cu`, `csrc/cuda/fused_logp_sm90.cu` | fused-softmax, flash-attention |
| 2 | training/ | training-grpo-loss | GRPO Surrogate Loss: Fused Policy Gradient + KL | `rl_engine/kernels/ops/triton/triton_grpo_loss.py` | — |
| 3 | training/ | training-rlhf-memory | Memory-Efficient RLHF: Solving O(G·L·V) | `rl_engine/kernels/sampling.py`, benchmarks | — |
| 4 | communication/ | comm-weight-sync-bridge | Weight Synchronization Bridge for RLHF Pipelines | `rl_engine/executors/bridge.py` | — |
| 5 | techniques/ | tech-online-logsumexp | Online Log-Sum-Exp: Single-Pass Numerically Stable Softmax | `csrc/fused_logp_kernel.cu:L52-108` | — |
| 6 | techniques/ | tech-prefix-kv-sharing | Prefix-Shared Attention for Group RL | `csrc/cuda/attention/prefix_shared_attention.cu` | flash-attention |
| 7 | parallelism/ | parallel-rlhf-rollout-training | RLHF Rollout-Training Parallelism with Ray | `rl_engine/executors/ray_actor_manager.py`, `rollout.py` | — |

### S.4 Repository Mappings (slug -> org/repo)

```python
# For the PR candidate search script
"rl-kernel": "RL-Align/RL-Kernel",

# For the PR page generation script
"rl-kernel": "RL-Align/RL-Kernel",
```

### S.5 Keyword-to-Tag Mappings (for automated PR tagger)

```python
# keyword -> kernel_type tag
"fused_logp": "fused-logp",
"fused logp": "fused-logp",
"log_softmax": "fused-logp",
"logprob": "fused-logp",
"log_prob": "fused-logp",
"grpo_loss": "grpo-loss",
"grpo loss": "grpo-loss",
"grpo_fwd": "grpo-loss",
"grpo_bwd": "grpo-loss",
"group_norm_kernel": "grpo-advantage",
"group_advantages": "grpo-advantage",
"prefix_shared_attention": "prefix-shared-attention",
"prefix_shared_attn": "prefix-shared-attention",
"shared_prefix": "prefix-shared-attention",
"attn_bwd": "attention-backward",
"attention_backward": "attention-backward",
"dgrad": "attention-backward",
"weight_sync": "weight-sync-bridge",
"weight_bridge": "weight-sync-bridge",
"WeightBridge": "weight-sync-bridge",
"CUDAVMMTensorBridge": "weight-sync-bridge",

# keyword -> technique tag
"online_logsumexp": "online-logsumexp",
"LogSumExpState": "online-logsumexp",
"online_softmax": "online-logsumexp",
"chunking": "pre-allocated-chunking",
"chunk_size": "pre-allocated-chunking",
"pre_alloc": "pre-allocated-chunking",
"preallocated": "pre-allocated-chunking",
"sparse_indexed": "sparse-indexed-dispatch",
"row_indices": "sparse-indexed-dispatch",
"indexed_out": "sparse-indexed-dispatch",
"cuda_vmm": "cuda-vmm-zero-copy",
"CUDAVMMTensorBridge": "cuda-vmm-zero-copy",
"cuMemCreate": "cuda-vmm-zero-copy",
"cuMemExportToShareableHandle": "cuda-vmm-zero-copy",
"ipc_event": "ipc-event-sync",
"interprocess": "ipc-event-sync",
"fd_broker": "fd-broker",
"SCM_RIGHTS": "fd-broker",
"rollout_training": "rlhf-rollout-training-split",
"RolloutExecutor": "rlhf-rollout-training-split",
"weight_version": "on-policy-weight-sync",
"GatheredParameters": "zero3-gather-on-publish",
"prefix_shared": "prefix-kv-sharing",
"fp32_accum": "fp32-accumulation",

# keyword -> hardware_feature tag
"tma": "tma",
"TMA": "tma",
"cp.async.bulk": "tma",
"CUtensorMap": "tma",
"cuTensorMapEncodeTiled": "tma",
"mbarrier": "mbarrier",
"mbarrier.init": "mbarrier",
"mbarrier.arrive": "mbarrier",
```

### S.6 PR Search Keywords (for candidate ledger)

```yaml
keywords_used:
  - fused_logp
  - grpo_loss
  - grpo
  - logprob
  - log_softmax
  - prefix_shared_attention
  - weight_sync
  - weight_bridge
  - CUDAVMMTensorBridge
  - online_logsumexp
  - chunking
  - sparse_indexed
  - sampling
  - FlashInfer
  - DeepSpeed
  - ZeRO
  - vllm
  - Ray
  - rollout
  - RLHF
  - PPO
  - DPO
  - TMA
  - SM90
  - mbarrier
  - attention_backward
```

### S.7 Inclusion Policy Lane

```yaml
rlhf-training-compute:
  description: |
    Captures PRs touching CUDA/Triton kernel implementations, fused operators,
    weight synchronization bridge, and RLHF executor modules. Skips pure
    documentation, CI configuration, and test-only changes.
  capture_criteria:
    - changed_paths_match:
        - "csrc/**/*.cu"
        - "csrc/**/*.cuh"
        - "csrc/**/*.cpp"
        - "rl_engine/kernels/**/*.py"
        - "rl_engine/executors/**/*.py"
        - "rl_engine/alignment/**/*.py"
        - "benchmarks/**/*.py"
    - title_contains_any:
        - kernel
        - fused
        - logp
        - grpo
        - attention
        - sampling
        - bridge
        - weight_sync
        - cuda_vmm
        - TMA
        - SM90
        - triton
        - DeepSpeed
        - ZeRO
        - rollout
        - RLHF
        - PPO
        - DPO
        - precision
        - bf16
        - fp32
        - chunking
  skip_criteria:
    - changed_paths_match_only:
        - "docs/**"
        - ".github/**"
        - "*.md"
        - "mkdocs.yaml"
        - "requirements-docs.txt"
    - pure_config_only: true
```

### S.8 Schema Extensions (if any)

New optional frontmatter fields for Wiki pages from this library:

- `scope: rlhf-training` — distinguishes RLHF training kernels from generic training or inference kernels
- `rl_algorithm: grpo | ppo | dpo` — which RL algorithm the kernel targets
- `memory_pattern: chunked | pre-allocated | zero-copy | sparse` — memory management strategy used
- `weight_transport: cuda-vmm | cuda-ipc | shared-memory | local-clone` — for communication/bridge pages
- `precision_accumulation: fp32 | bf16 | fp16` — accumulation precision in compute kernels

### S.9 Hardware Features Relevant to This Library's Training Workloads

| Hardware Feature | Inference Relevance | Training Relevance | Specific Impact on This Library |
|-----------------|--------------------|--------------------|--------------------------------|
| TMA (SM90+) | Yes | Core | SM90 fused logp kernel uses TMA for async vocabulary tile loads, freeing compute warps |
| mbarrier (SM90+) | Yes | Core | Producer-consumer warp synchronization in TMA kernel; phase-flipping double-buffer protocol |
| BF16 Tensor Core MMA (SM80+) | Yes | Core | Prefix-shared attention uses `mma.sync f32.bf16.bf16.f32` for all QK^T and PV |
| cp.async (SM80+) | Yes | Core | Attention kernel double-buffers K tiles via `cp.async.cg.shared.global` |
| NVLink 5 (1.8 TB/s) | Partial | Beneficial | Would improve DeepSpeed gradient AllReduce and ZeRO-3 AllGather bandwidth |
| NVSwitch 4 (NVL72) | Partial | Beneficial | Multi-node expansion (currently blocked; would benefit future RDMA transport) |
| NVLink-SHARP FP8 | No | Not applicable | No FP8 support in this library |
| Symmetric Memory | Partial | Not applicable | No symmetric memory usage |
| Copy Engine (zero-SM) | Partial | Beneficial | CUDA VMM `cuMemcpyDtoDAsync_v2` already uses Copy Engine for weight transfer |
| MXFP8 hardware (Blackwell) | Yes | Not applicable | No FP8 precision in any kernel |
| 192 MB L2 Cache (Blackwell) | Beneficial | Beneficial | Would improve online logsumexp performance for large vocabularies |
| 192 GB HBM3e @ 8 TB/s | Core | Core | Larger VRAM enables larger GRPO group sizes; higher bandwidth improves logp throughput |

### S.10 Upstream/Downstream Dependencies to Also Track

| Slug | GitHub URL | Relationship | Justification |
|------|-----------|-------------|---------------|
| flashinfer | flashinfer-ai/flashinfer | kernel-provider | Provides fused top-k/top-p sampling kernels for the NVIDIA sampling backend |
| flash-attn | Dao-AILab/flash-attention | kernel-provider | Provides FlashAttention forward kernel as primary attention backend |
| deepspeed | microsoft/DeepSpeed | runtime-dependency | Provides ZeRO optimizer, gradient communication, and training engine |
| vllm | vllm-project/vllm | runtime-dependency | Provides inference engine for rollout phase; weight install adapters tightly coupled |
| ray | ray-project/ray | runtime-dependency | Provides actor-based process orchestration for RLHF pipeline |

---

## Library Type Adaptation Rationale

RL-Kernel is classified as a **training-compute** library with an orchestration layer:

- **Dimension 1 (Compute Kernels):** Deep analysis — 7 kernel files with custom CUDA, Triton, and SM90-specific implementations. The fused logp kernels and GRPO loss kernels are novel contributions not found in other training frameworks.

- **Dimension 2 (Communication):** Light analysis — No direct collective operations; all gradient communication delegated to DeepSpeed. However, the CUDA VMM weight synchronization bridge is a significant communication innovation specific to RLHF pipelines.

- **Dimension 3 (Parallelism):** Moderate analysis — Standard DeepSpeed DP/ZeRO; TP/PP/CP not implemented. The RLHF-specific rollout-training split with versioned weight bridge is the primary parallelism contribution.

- **Dimension 4 (Memory Management):** Deep analysis — This is RL-Kernel's primary innovation. The chunked pre-allocated approach solves the O(G·L·V) memory explosion, and the CUDA VMM zero-copy bridge eliminates weight duplication.

- **Dimension 5 (Precision):** Light analysis — No FP8; standard FP16/BF16/FP32 with consistent FP32 accumulation. Well-engineered numerical stability but no novel precision techniques.

- **Dimension 6 (Profiling):** Moderate analysis — Rich benchmark infrastructure for performance measurement, but no runtime observability integration (no NVTX, no torch.profiler, no flight recorder).
