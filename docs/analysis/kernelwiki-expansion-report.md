# KernelWiki Source Analysis: RL-Kernel (RL-Align/RL-Kernel)

**Analysis date**: 2026-06-12
**Library**: RL-Align/RL-Kernel
**Scope**: training (RL post-training GPU kernel library)
**Analyzed path**: /root/RL-Kernel

---

## 1. Kernel File Census

**Total kernel files: 7** (~2,092 lines of GPU kernel code)

- **CUDA/C++**: 3 `.cu` files + 1 `.cuh` header (1,126 lines) in `csrc/`
- **Triton**: 2 `.py` files (873 lines) in `rl_engine/kernels/ops/triton/`
- **Other DSL**: 0 files (no TileLang, Pallas, or JAX)
- **Extension entry points**: 1 `.cpp` file (93 lines) — `csrc/ops.cpp` with PYBIND11_MODULE

### Directory Heatmap

| Directory | `.cu` | `.cuh` | `.py` (Triton) | `.cpp` (Entry) | Total Lines |
|-----------|-------|--------|----------------|-----------------|-------------|
| `csrc/` | 1 | 0 | 0 | 1 | 681 |
| `csrc/cuda/` | 1 | 0 | 0 | 0 | 124 |
| `csrc/cuda/attention/` | 1 | 0 | 0 | 0 | 331 |
| `csrc/utils/` | 0 | 1 | 0 | 0 | 83 |
| `rl_engine/kernels/ops/triton/` | 0 | 0 | 2 | 0 | 873 |
| **Total** | **3** | **1** | **2** | **1** | **2,092** |

### Category Summary

| Category | File Count | Total Lines | Avg Lines/File |
|----------|-----------|-------------|----------------|
| CUDA `.cu` | 3 | 1,043 | 348 |
| CUDA Header `.cuh` | 1 | 83 | 83 |
| Triton `.py` | 2 | 873 | 437 |
| Extension `.cpp` | 1 | 93 | 93 |

### Kernel File Detail

| File | Lines | Contents |
|------|-------|---------|
| `csrc/fused_logp_kernel.cu` | 588 | `fused_logp_forward_kernel` (two-pass), `fused_logp_forward_online_kernel` (single-pass), `blockReduceMax`, `blockReduceSum`, `blockReduceLogSumExp` — numerically stable log-prob extraction from logits |
| `csrc/cuda/attention/prefix_shared_attention.cu` | 331 | `prefix_shared_attention_kernel` — custom GRPO attention with inline PTX (`ldmatrix.x4`, `mma.m16n8k16`), shared-memory swizzling, double-buffered `cp.async` for shared KV prefix broadcast across G query groups |
| `csrc/cuda/fused_logp_sm90.cu` | 124 | `fused_logp_online_tma_kernel` — SM90 TMA-accelerated variant with producer/consumer warp specialization, `mbarrier`, `CUB::BlockReduce` |
| `csrc/utils/tma_utils.cuh` | 83 | Header-only TMA descriptor utilities: `init_tensor_map` with auto-swizzle selection, `tma_2d_g2s` bulk async copy |
| `rl_engine/kernels/ops/triton/triton_attn.py` | 495 | `_fwd_kernel`, `_bwd_preprocess`, `_bwd_kernel` — Triton flash-attention (causal/non-causal) with `tl.make_block_ptr` + `tl.advance` |
| `rl_engine/kernels/ops/triton/triton_grpo_loss.py` | 378 | `_group_norm_kernel`, `_grpo_fwd_kernel`, `_grpo_bwd_kernel` — fused GRPO loss with CSR-group normalization, clipped PPO surrogate, and K3-KL |
| `csrc/ops.cpp` | 93 | PYBIND11_MODULE with 10 bindings: `fused_logp`, `fused_logp_sm90`, 8 variant entry points, `prefix_shared_attention` |

### Key Observation

**RL-Kernel is a custom kernel library that writes its own kernels from scratch.** The core deliverables are original CUDA and Triton implementations for RL training primitives: fused log-probability computation (with online softmax), prefix-shared attention for GRPO, and the GRPO loss. External libraries (`flash_attn`, FlashInfer) appear only as optional fallback accelerators in the `KernelRegistry` priority chain. The ROCm path (`ROCM_AITER`, `ROCM_CK`) is architecturally registered but not yet implemented (empty stub).

---

## 2. Kernel Type Discovery

### Existing types found (already in tags.yaml)

- **flash-attention** / **attention** / **attention-backward**: `rl_engine/kernels/ops/triton/triton_attn.py` (`_fwd_kernel`, `_bwd_kernel`)
- **attention**: `csrc/cuda/attention/prefix_shared_attention.cu` (`prefix_shared_attention_kernel`)
- **reduction**: `csrc/fused_logp_kernel.cu` (`blockReduceMax`, `blockReduceSum`, `blockReduceLogSumExp`)
- **fused-loss-grpo**: `rl_engine/kernels/ops/triton/triton_grpo_loss.py` (`_grpo_fwd_kernel`, `_grpo_bwd_kernel`)
- **groupnorm**: `rl_engine/kernels/ops/triton/triton_grpo_loss.py` (`_group_norm_kernel` — reward group normalization)
- **loss-computation**: `csrc/fused_logp_kernel.cu`, `csrc/cuda/fused_logp_sm90.cu` (fused log-softmax + token-index gather)

### Proposed new kernel types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| `fused-logp` | `csrc/fused_logp_kernel.cu` → `fused_logp_forward_kernel`, `fused_logp_forward_online_kernel` | Fuses online-softmax (log-sum-exp reduction) with sparse token-index gather to extract per-token log-probabilities from logits in a single kernel pass, central to RL policy gradient pipelines |
| `tma-fused-logp` | `csrc/cuda/fused_logp_sm90.cu` → `fused_logp_online_tma_kernel` | SM90+ warp-specialized variant that uses TMA bulk async loads with mbarrier synchronization to pipeline tile fetches against the online-softmax consumer warps |
| `prefix-shared-attention` | `csrc/cuda/attention/prefix_shared_attention.cu` → `prefix_shared_attention_kernel` | Attention kernel loading shared KV prefix once into SMEM and broadcasting across G independent query groups (GRPO generation heads), exploiting shared-prompt structure |
| `grpo-reward-normalization` | `rl_engine/kernels/ops/triton/triton_grpo_loss.py` → `_group_norm_kernel` | Within-group reward standardization (mean/std) over variable-size groups encoded as CSR boundary offsets, producing per-sequence advantages for GRPO |

### Classification rationale

- **`fused-logp`** differs from `selective-logsoftmax` (which implies masking of the softmax) and from `loss-computation` (which implies a terminal training objective scalar). This is a standalone primitive invoked *before* any loss function.
- **`tma-fused-logp`** is a hardware-tier specialization of `fused-logp` for SM90+ (H100/B100/B200), analogous to how `orchestrated-flash-attn-v4` is a hardware-specific attention specialization.
- **`prefix-shared-attention`** has a structurally non-square KV layout: K,V = `[bs, len_kv, DIM]` (no head dim) vs Q = `[bs, G, len_q, DIM]`. No existing tag captures this prompt-sharing motif.
- **`grpo-reward-normalization`** differs from `groupnorm` (deep learning feature-channel normalization). It normalizes rewards over sample groups with variable sizes.

---

## 3. Dependency Graph

### Kernel Providers

| Library | Import/Include | Kernel Types Provided |
|---------|---------------|----------------------|
| **CUDA Runtime** | `#include <cuda_runtime.h>`, `cuda.h` | All `.cu` kernel launch/sync infrastructure |
| **CUDA Driver API** | `cudaTypedefs.h`, `cuTensorMapEncodeTiled` | TMA descriptor creation for SM90 hardware |
| **CUB** | `#include <cub/cub.cuh>` | `cub::BlockReduce` in SM90 kernel |
| **PyTorch / LibTorch** | `torch/extension.h`, ATen, c10 | Build glue, CUDA stream, dtype dispatch; cuBLAS/cuDNN indirectly |
| **Triton** | `import triton`, `triton.language as tl` | JIT compiler for 2 kernel files (GRPO loss, attention) |
| **FlashAttention** | `from flash_attn import flash_attn_func` | Optional: fused multi-head attention CUDA kernels |
| **FlashInfer** | `from flashinfer import top_k_renorm_probs, top_p_sampling_from_probs` | Optional: sampling and decode-phase attention (≥0.1.6) |

### Runtime Dependencies

| Library | Role |
|---------|------|
| **vLLM** (≥0.6.0) | Inference engine; CUDA VMM weight transfer via `bridge.py` |
| **DeepSpeed** | ZeRO optimizer, full-state gather for weight publishing |
| **Ray** | Actor lifecycle for distributed training/inference coordination |
| **HuggingFace Transformers** | Model loading |
| **Accelerate** | Device placement wrapper |

### Build Dependencies

| Library | Role |
|---------|------|
| **setuptools** (≥64) | Build system, `torch.utils.cpp_extension` |
| **nvidia-ml-py** | NVML monitoring (cuda extra) |

### Planned/Stub Dependencies (ROCm)

| Library | Status |
|---------|--------|
| **AITER** (`aiter`) | Registered as `OpBackend.ROCM_AITER` — stub, not implemented |
| **Composable Kernel** | Registered as `OpBackend.ROCM_CK` — stub, not implemented |

### Architectural Notes

- **No CUTLASS, NCCL, cuBLAS, or cuDNN direct headers** — the `mma.m16n8k16` Tensor Core instructions are invoked via raw inline PTX.
- **TMA kernel gated by env var** — `KERNEL_ALIGN_FORCE_SM90=1` at build time; absent from standard CI builds.
- **Triton is first-class**, not just fallback — the GRPO loss production path is Triton.

### Repositories to Track

| Slug | GitHub URL | Justification |
|------|-----------|---------------|
| `rl-kernel` | `RL-Align/RL-Kernel` | This library — primary analysis target |
| `pytorch` | `pytorch/pytorch` | ATen/c10 build infrastructure, CUDA stream APIs, bundled Triton |
| `triton` | `triton-lang/triton` | JIT compiler for 2 production kernel files; API changes break compilation |
| `flash-attention` | `Dao-AILab/flash-attention` | First-priority attention backend on CUDA |
| `flashinfer` | `flashinfer-ai/flashinfer` | Hard dependency for NVIDIA sampling (≥0.1.6) |
| `aiter` | `ROCm/aiter` | Planned ROCm kernel provider |
| `composable-kernel` | `ROCm/composable_kernel` | Planned ROCm attention path |
| `vllm` | `vllm-project/vllm` | vLLM-internal APIs consumed by weight-transfer bridge |

---

## 4. Technique and Hardware Feature Discovery

### Existing techniques found

| Technique | Evidence |
|-----------|---------|
| **online-softmax** | `fused_logp_kernel.cu:73–109`, `fused_logp_sm90.cu:65–98`, `prefix_shared_attention.cu:212–278`, `triton_attn.py:95–110` |
| **warp-specialization** | `fused_logp_sm90.cu:31–106` — warp 0 = TMA producer, warps 1..N−1 = softmax consumers |
| **kernel-fusion** | `fused_logp_kernel.cu` — fuses softmax + logsumexp + logp extraction in single pass |
| **swizzling** | `prefix_shared_attention.cu:27–47` — `swizzle<STRIDE>()` XOR remap; `tma_utils.cuh:36–38` — `CUtensorMapSwizzle` |
| **double-buffering** | `prefix_shared_attention.cu:170,199` — `(kv_id % 2)` ping-pong between K-tile SMEM slots |
| **compile-time-specialization** | `fused_logp_kernel.cu:75` — `static_assert`; `prefix_shared_attention.cu:30` — `if constexpr` |
| **jit-compilation** | `triton_attn.py`, `triton_grpo_loss.py` — `@triton.jit` |
| **multi-backend-portability** | `registry.py` — cuda/rocm/cpu priority maps; `setup.py` — ROCMExtension |
| **log-prob-chunked-softmax** | `fused_logp_kernel.cu` — two-pass and online-pass chunked softmax variants |
| **varlen-sequence-packing** | `fused_logp_kernel.cu:120` — `row_indices` optional sparse indexing |
| **occupancy-tuning** | `fused_logp_kernel.cu:234–249` — `FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM` |
| **conditional-rescaling** | `prefix_shared_attention.cu:237–244` — per-row O rescale when running max updates |
| **seqlen-adaptive-launch** | `triton_attn.py:353–354` — `BLOCK_N = 64 if head_dim > 64 else 128` |
| **kernel-autotune** | `triton_attn.py` — `num_stages`/`num_warps` tuning |
| **pipeline-stages** | `triton_attn.py:389,469` — `num_stages=2` |

### Proposed new techniques

| Tag | Evidence File | Description |
|-----|--------------|-------------|
| `twopass-softmax` | `csrc/fused_logp_kernel.cu:112–156` | Explicit two-pass (pass 1: global max; pass 2: sum of exp) softmax for log-prob, distinct from online-softmax; both coexist and are runtime-selectable |
| `sparse-indexed-kernel-dispatch` | `csrc/fused_logp_kernel.cu:256–272` | Runtime branch selects distinct block-size variant when sparse `row_indices` covers ≤50% rows and each row ≥64 KB; density-ratio heuristic dispatch |
| `logsumexp-state-merge` | `csrc/fused_logp_kernel.cu:52–109` | Numerically-stable pairwise merger of `(max_val, sum_exp)` state across warp/block reductions, enabling single-pass online logsumexp without recomputing exponentials |
| `csr-group-normalization` | `rl_engine/kernels/ops/triton/triton_grpo_loss.py:30–56` | Reward normalization over variable-size groups encoded as CSR boundary offsets; one Triton block per group with masked load/store |
| `token-parallel-rl-loss` | `rl_engine/kernels/ops/triton/triton_grpo_loss.py:58–98` | Each Triton block processes a flat token slice, computing clipped-PPO surrogate + KL in one pass using `seq_id = offs // T` to broadcast per-sequence advantages |
| `partial-block-reduction` | `rl_engine/kernels/ops/triton/triton_grpo_loss.py:64–66,179–181` | GRPO forward writes per-block `(policy_sum, kl_sum)` partials to scratch tensor; host completes reduction with `.sum(dim=0)` — avoids global atomic or second kernel |
| `prefix-shared-kv-broadcast` | `csrc/cuda/attention/prefix_shared_attention.cu:83–111` | K/V loaded once from shared prompt prefix into SMEM and reused by all G query heads; eliminates G−1 redundant global reads |
| `env-var-macro-tuning` | `setup.py:24–106` | Build-time kernel parameter injection via environment variables → `nvcc -D` defines for block sizes and density thresholds |
| `tma-swizzle-selection` | `csrc/utils/tma_utils.cuh:34–45` | Auto-selects `CU_TENSOR_MAP_SWIZZLE_32B/64B/128B` based on shared-memory stride width at TMA descriptor creation |
| `triton-block-ptr-advance` | `rl_engine/kernels/ops/triton/triton_attn.py:44–67` | Uses `tl.make_block_ptr` + `tl.advance` for structured 2D block iteration over K/V tiles |
| `ptx-named-barrier` | `csrc/cuda/fused_logp_sm90.cu:78,88,95` | `bar.sync 1` synchronizes only consumer warps independently of producer, avoiding full `__syncthreads()` |

### Existing hardware features found

| Feature | Evidence |
|---------|---------|
| **warp-shuffle-sync** | `fused_logp_kernel.cu:17,24,39,47` — `__shfl_down_sync` |
| **ptx-ldmatrix-x4** | `prefix_shared_attention.cu:50–53` |
| **mma-m16n8k16** | `prefix_shared_attention.cu:62–72` |
| **mbarrier** | `tma_utils.cuh:54–77` |
| **tma** | `tma_utils.cuh:79–83`, `fused_logp_sm90.cu:49` |
| **cp-async** | `prefix_shared_attention.cu:46,153–154,174,183,194,280` |
| **cub-block-reduce** | `fused_logp_sm90.cu:59` |
| **dynamic-shared-memory** | `prefix_shared_attention.cu:110`, `fused_logp_sm90.cu:24` |
| **launch-bounds** | `prefix_shared_attention.cu:86`, `fused_logp_kernel.cu:159` |
| **cuda-arch-guards** | `setup.py:63,113–119`, `registry.py:102–119` |
| **fp32-accumulation** | `prefix_shared_attention.cu:125` — fp32 accum from bf16 inputs |
| **sm80-bf16-conversion** | `prefix_shared_attention.cu:308`, `fused_logp_sm90.cu:75` |
| **rocm-hip-compat** | `setup.py:42–55` — ROCMExtension build path |

### Proposed new hardware features

| Tag | Evidence File | Description |
|-----|--------------|-------------|
| `ptx-ldmatrix-x4-trans` | `csrc/cuda/attention/prefix_shared_attention.cu:56–59` | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` — transposed variant loading V tiles in column-major layout for MMA operand |
| `grid-constant-tma-descriptor` | `csrc/cuda/fused_logp_sm90.cu:13` | `const __grid_constant__ CUtensorMap` — CUDA 12 storage class placing 128-byte TMA descriptor in constant memory across grid |
| `ptx-bar-sync-named` | `csrc/cuda/fused_logp_sm90.cu:78,88,95` | `bar.sync 1, %0` — numbered barrier for sub-warp-group synchronization without blocking producer warp |
| `ptx-fence-mbarrier-init` | `csrc/cuda/fused_logp_sm90.cu:35` | `fence.mbarrier_init.release.cluster` — SM90 cluster-scope fence after mbarrier initialization |
| `cutensormap-encode-tiled` | `csrc/utils/tma_utils.cuh:40–45` | `cuTensorMapEncodeTiled()` driver API for host-side TMA descriptor construction with rank, swizzle, and L2 promotion |

---

## 5. Wiki Page Topics

| # | Category | Proposed ID | Title | Source Files/PRs | Related Existing Pages |
|---|----------|-------------|-------|-----------------|----------------------|
| 1 | `kernels/` | `kernel-rl-fused-logp` | Fused Selective Log-Probability Kernel: Two-Pass vs Online-Softmax Variants | `csrc/fused_logp_kernel.cu`, `rl_engine/kernels/ops/cuda/loss/logp.py`, `csrc/ops.cpp`, `benchmarks/benchmark_grpo_op.py` | `kernels/grpo-rl-loss`, `kernels/fused-linear-cross-entropy-chunked`, `techniques/dual-chunked-selective-logprob` |
| 2 | `kernels/` | `kernel-rl-fused-logp-sm90` | TMA-Pipelined Fused LogP Kernel for SM90 (Warp-Specialization + mbarrier) | `csrc/cuda/fused_logp_sm90.cu`, `csrc/utils/tma_utils.cuh`, `csrc/ops.cpp`, `rl_engine/kernels/registry.py` | `hardware/tma`, `techniques/warp-specialization`, `kernels/flash-attention-4` |
| 3 | `kernels/` | `kernel-rl-prefix-shared-attention` | Prefix-Shared Fused Attention: Batched Q over a Shared KV Prefix | `csrc/cuda/attention/prefix_shared_attention.cu`, `rl_engine/kernels/ops/cuda/attention/prefix_shared_attn.py`, `benchmarks/benchmark_attention.py` | `kernels/flash-attention-4`, `kernels/flashmla`, `techniques/double-buffering` |
| 4 | `kernels/` | `kernel-rl-triton-grpo-loss` | Triton GRPO Loss with CSR-Group Normalization and Fused Token-Parallel KL | `rl_engine/kernels/ops/triton/triton_grpo_loss.py`, `benchmarks/benchmark_grpo_loss.py`, `tests/test_grpo_loss.py` | `kernels/grpo-rl-loss`, `techniques/chunked-preference-loss` |
| 5 | `kernels/` | `kernel-rl-triton-attention-fwdbwd` | Triton Flash-Attention Forward/Backward with Block-Pointer Advance | `rl_engine/kernels/ops/triton/triton_attn.py` | `kernels/flash-attention-4`, `languages/triton-blackwell` |
| 6 | `techniques/` | `technique-logsumexp-state-merge` | Online LogSumExp State Merge: Single-Pass Numerically Stable Softmax Reduction | `csrc/fused_logp_kernel.cu` (`LogSumExpState`, `merge_logsumexp_state`), `csrc/cuda/fused_logp_sm90.cu` | `techniques/dual-chunked-selective-logprob`, `kernels/kernel-cce-fused-cross-entropy` |
| 7 | `techniques/` | `technique-sparse-indexed-kernel-dispatch` | Sparse-Indexed Dispatch: Row-Subset Block-Size Selection via Density Heuristic | `csrc/fused_logp_kernel.cu` (`select_fused_logp_online_launch_variant`) | `techniques/cache-policy` |
| 8 | `techniques/` | `technique-tma-swizzle-auto-selection` | TMA Swizzle Mode Auto-Selection from Shared-Memory Stride Width | `csrc/utils/tma_utils.cuh` | `hardware/tma`, `techniques/swizzling` |
| 9 | `techniques/` | `technique-csr-group-reward-normalization` | CSR-Format Group Reward Normalization for Variable-Length GRPO Groups | `rl_engine/kernels/ops/triton/triton_grpo_loss.py` | `kernels/grpo-rl-loss` |
| 10 | `techniques/` | `technique-env-var-macro-tuning` | Compile-Time Block-Size Tuning via Preprocessor Macros with Env-Var Injection | `csrc/fused_logp_kernel.cu`, `setup.py` | `techniques/cache-policy` |
| 11 | `patterns/` | `pattern-rl-multi-backend-dispatch` | RL Kernel Registry: Priority-Ordered Multi-Backend Dispatch with SM-Capability Promotion | `rl_engine/kernels/registry.py`, `rl_engine/platforms/device.py` | `patterns/bnb-multi-backend-dispatch` |
| 12 | `migration/` | `migration-rl-kernel-cuda-to-rocm` | CUDA-to-ROCm Migration Roadmap: Stub Architecture and AITER/CK Integration Path | `rl_engine/kernels/ops/rocm/`, `registry.py` (ROCm map) | `migration/uccl-cuda-to-rocm` |
| 13 | `training/` | `training-rl-weight-sync-bridge` | Versioned Weight Synchronization Bridge: Manifest-Driven Transport Negotiation | `rl_engine/executors/bridge.py`, `docs/usage/weight-sync-bridge.md` | `training/comm-weight-sync-strategies`, `training/comm-delta-weight-sync` |
| 14 | `migration/` | `migration-rl-fused-logp-sm90-to-sm100` | SM90 TMA LogP Kernel Migration to SM100: tcgen05 and wgmma Considerations | `csrc/cuda/fused_logp_sm90.cu`, `tma_utils.cuh`, `registry.py` | `migration/mkernel-sm90-sm100` |

---

## 6. External Knowledge

### Official Documentation

- **RL-Kernel GitHub**: https://github.com/RL-Align/RL-Kernel — Apache-2.0, benchmark tables (A100 80GB, Llama-3-8B, Qwen3-30B-A3B MoE), architecture diagrams. Key claim: only solution scaling G=256 on single A100 via ~0.5 GB constant extra VRAM.
- **RL-Kernel Docs Site**: https://rl-align.github.io/RL-Kernel/ — MkDocs hosted site. Operator reference (Fused LogP, Sampling, GRPO Loss), design docs (runtime dispatch), API, CLI.
- **In-repo docs**: `docs/operators/fused-logp.md`, `docs/operators/grpo-loss.md`, `docs/operators/sampling.md`, `docs/design/runtime-dispatch.md`, `docs/usage/weight-sync-bridge.md`

### Papers

| Citation | Relevance to KernelWiki |
|----------|------------------------|
| Shengding Hu et al., "DeepSeekMath", arXiv:2402.03300, 2024 | **Originating paper for GRPO** — defines group-relative advantage estimation that motivates the O(G·L·V) memory problem RL-Kernel solves |
| DeepSeek-AI, "DeepSeek-R1", arXiv:2501.12948, 2025 | Scales GRPO to 671B-class model; defines production hyperparameters (ε=10, KL=0.001) |
| Zihao Ye et al., "FlashInfer", arXiv:2501.01005, 2025 (MLSys 2025 Best Paper) | RL-Kernel integrates FlashInfer as primary NVIDIA sampling backend (163x speedup at bs=32) |
| Buitrago et al., "FlashAttention-2 on Hopper using CUTLASS", arXiv:2312.11918, 2023 | TMA + WGMMA fusion patterns directly relevant to `fused_logp_sm90.cu` |
| Caged et al., "Prefix Grouper", arXiv:2506.05433, 2025 | Closest published algorithm to RL-Kernel's prefix-shared attention — decomposes attention for shared prefix across G responses |
| Wang et al., "DPO Prefix Sharing", arXiv:2410.20305, 2024 (NeurIPS 2024 Workshop) | Block-sparse attention masks for paired preference training; architectural precursor to GRPO prefix sharing |
| Apple, "Cut Cross-Entropy (CCE)", arXiv:2411.09009, 2024 (ICLR 2025) | Closest published technique to fused LogP — avoids materializing full [N,V] logit matrix; 24 GB → 1 MB for Gemma-2B |
| LinkedIn, "Liger Kernel", arXiv:2410.10989, 2024 | Most directly comparable GRPO loss implementation (chunked Triton); 80% memory reduction |

### Blog Posts

- **FlashAttention-4 Blog** (Tri Dao): https://tridao.me/blog/2026/flash4/ — ping-pong scheduling, software exp2, conditional rescaling on SM100; directly relevant to SM90 online softmax migration path
- **FlashInfer Introduction**: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html — cascade attention and prefix-sharing mechanism overview
- **How DeepSeek R1 was Trained**: https://www.philschmid.de/deepseek-r1 — GRPO training walkthrough, memory implications of large G
- **GRPO on AMD MI300X** (AMD ROCm Blog): https://rocm.blogs.amd.com/software-tools-optimization/llm-grpo-rocm/README.html — relevant to RL-Kernel's planned ROCm backend
- **ROCm Miles** (LMSYS Blog): https://www.lmsys.org/blog/2026-03-17-rocm-miles-rl-amd/ — timing breakdown: rollout=152.79s, logprob=31.53s — quantifies exactly the bottleneck RL-Kernel targets
- **TMA Supercharging AI Kernels**: https://medium.com/the-synaptic-stack/tma-how-tensor-memory-accelerator-is-supercharging-ai-kernels-2ffbc3fb5e63 — background on Hopper TMA hardware
- **Efficient GRPO Loss Kernel** (open-r1 #615): https://github.com/huggingface/open-r1/issues/615 — community demonstration of 46 GB VRAM reduction with Triton GRPO loss

### Design Discussions (GitHub Issues/PRs)

- **RL-Kernel PR #92**: GPU CI integration — active design focus
- **RL-Kernel PR #95**: Documentation wiki links update
- **Colfax TMA Tutorial**: https://research.colfax-intl.com/tutorial-hopper-tma/ — TMA requires contiguous layouts, constraining `fused_logp_sm90.cu` tensor contract
- **veRL PR #1026**: FSDP2 GRPO actor integration — context for RL-Kernel as drop-in replacement
- **Liger-Kernel PR #939**: DAPO as default GRPO loss — near-term divergence point for RL-Kernel
- **Liger-Kernel PR #993**: Multi-loss-type Triton operator — ecosystem trend toward unified operators

### Already in KernelWiki

**No existing source directly covers RL-Kernel.** Related sources present:
- `pr-Liger-Kernel-939`, `pr-Liger-Kernel-993` — GRPO/DAPO Triton losses
- 80+ FlashInfer PR sources — sampling, attention, prefix sharing
- 30+ CUTLASS PR sources — TMA, WGMMA, Hopper patterns
- `blog-flash-attention-4` — SM90/SM100 online softmax
- `pr-verl-*` — GRPO training orchestration
- `pr-ms-swift-*` — GRPO/logprob production usage

---

## 7. Expansion Decision Summary

### 7.1 tags.yaml additions

```yaml
kernel_types:
  # New from RL-Kernel
  - fused-logp
  - tma-fused-logp
  - prefix-shared-attention
  - grpo-reward-normalization

techniques:
  # New from RL-Kernel
  - twopass-softmax
  - sparse-indexed-kernel-dispatch
  - logsumexp-state-merge
  - csr-group-normalization
  - token-parallel-rl-loss
  - partial-block-reduction
  - prefix-shared-kv-broadcast
  - env-var-macro-tuning
  - tma-swizzle-selection
  - triton-block-ptr-advance
  - ptx-named-barrier

hardware_features:
  # New from RL-Kernel
  - ptx-ldmatrix-x4-trans
  - grid-constant-tma-descriptor
  - ptx-bar-sync-named
  - ptx-fence-mbarrier-init
  - cutensormap-encode-tiled

source_categories:
  # (no new categories needed — rl-training-framework already exists)
```

### 7.2 Repository mappings

```python
# refresh_candidate_ledger.py — REPO_SLUG_TO_FULL
"rl-kernel": "RL-Align/RL-Kernel",

# generate-pr-pages.py — repo_map
"rl-kernel": "RL-Align/RL-Kernel",
```

### 7.3 Keyword mappings for auto_tag()

```python
# generate-pr-pages.py — KW_TO_KT additions
"fused_logp": "fused-logp",
"fused logp": "fused-logp",
"log probability": "fused-logp",
"logprob": "fused-logp",
"log_prob": "fused-logp",
"tma_fused_logp": "tma-fused-logp",
"fused_logp_sm90": "tma-fused-logp",
"prefix_shared_attention": "prefix-shared-attention",
"prefix shared attention": "prefix-shared-attention",
"grpo_reward": "grpo-reward-normalization",
"group_norm_kernel": "grpo-reward-normalization",
"group advantages": "grpo-reward-normalization",

# generate-pr-pages.py — KW_TO_TECH additions
"twopass_softmax": "twopass-softmax",
"two-pass softmax": "twopass-softmax",
"two pass softmax": "twopass-softmax",
"row_indices": "sparse-indexed-kernel-dispatch",
"sparse indexed": "sparse-indexed-kernel-dispatch",
"LogSumExpState": "logsumexp-state-merge",
"merge_logsumexp": "logsumexp-state-merge",
"csr_group": "csr-group-normalization",
"bounds_ptr": "csr-group-normalization",
"token_parallel_rl": "token-parallel-rl-loss",
"partial_block_reduction": "partial-block-reduction",
"prefix_shared_kv": "prefix-shared-kv-broadcast",
"env_var_macro": "env-var-macro-tuning",
"FUSED_LOGP_TWOPASS_BLOCK_SIZE": "env-var-macro-tuning",
"FUSED_LOGP_ONLINE_BLOCK_SIZE": "env-var-macro-tuning",
"tma_swizzle": "tma-swizzle-selection",
"CU_TENSOR_MAP_SWIZZLE": "tma-swizzle-selection",
"make_block_ptr": "triton-block-ptr-advance",
"tl.advance": "triton-block-ptr-advance",
"bar.sync 1": "ptx-named-barrier",

# generate-pr-pages.py — KW_TO_TAGS additions (hardware features)
"ldmatrix.sync.aligned.m8n8.x4.trans": "ptx-ldmatrix-x4-trans",
"ldmatrix_x4_trans": "ptx-ldmatrix-x4-trans",
"__grid_constant__": "grid-constant-tma-descriptor",
"grid_constant": "grid-constant-tma-descriptor",
"bar.sync 1": "ptx-bar-sync-named",
"fence.mbarrier_init": "ptx-fence-mbarrier-init",
"cuTensorMapEncodeTiled": "cutensormap-encode-tiled",
"CUtensorMap": "cutensormap-encode-tiled",
```

### 7.4 Candidate ledger keywords

```yaml
# candidates/rl-kernel.yaml — keywords_used
keywords_used:
  - fused_logp
  - fused logp
  - logprob
  - tma_fused_logp
  - fused_logp_sm90
  - prefix_shared_attention
  - prefix shared attention
  - grpo
  - grpo_loss
  - grpo_reward
  - group_norm_kernel
  - online softmax
  - warp specialization
  - TMA
  - mbarrier
  - cp.async
  - ldmatrix
  - mma.m16n8k16
  - KernelRegistry
  - OpBackend
  - triton_attn
  - triton_grpo_loss
  - weight sync bridge
  - ROCm
  - AITER
```

### 7.5 Schema extensions (if any)

No new schema extensions required. The existing schema supports all proposed page types (`kernels/`, `techniques/`, `patterns/`, `migration/`, `training/`). The `source_categories` value `rl-training-framework` already exists and covers this library.

### 7.6 Scope rule changes (if any)

No scope rule changes needed. RL-Kernel falls squarely within KernelWiki's existing inclusion policy:
- Contains original GPU kernel implementations (CUDA + Triton)
- Targets high-performance training workloads
- Has SM90/Blackwell relevance via TMA kernels and migration path

**Recommended addition to README.md inclusion examples**: Add "RL-Kernel" alongside torchtitan and Megatron-LM as an example of a tracked training framework with native GPU kernels.

### 7.7 Repositories to also track

| Slug | GitHub URL | Justification |
|------|-----------|---------------|
| `rl-kernel` | `RL-Align/RL-Kernel` | Primary target — custom CUDA/Triton kernels for RL training (fused logp, prefix-shared attention, GRPO loss) |
| `flash-attention` | `Dao-AILab/flash-attention` | First-priority attention backend; API changes break RL-Kernel's wrapper |
| `flashinfer` | `flashinfer-ai/flashinfer` | Hard sampling dependency (≥0.1.6); `top_k_renorm_probs`, `top_p_sampling_from_probs` |
| `aiter` | `ROCm/aiter` | Planned ROCm kernel provider for `logp` and sampling |
| `composable-kernel` | `ROCm/composable_kernel` | Planned ROCm attention backend |
| `vllm` | `vllm-project/vllm` | Weight-transfer bridge depends on vLLM-internal APIs |

**Note**: `pytorch/pytorch` and `triton-lang/triton` are likely already tracked in KernelWiki. If not, they should be added — RL-Kernel depends critically on both.
