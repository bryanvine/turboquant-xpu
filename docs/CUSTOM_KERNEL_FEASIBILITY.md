# Native SYCL Kernel for TurboQuant + Speculative Decoding — Feasibility Study

**Date:** 2026-04-14
**Question:** Can we build a custom XPU kernel that significantly speeds up TurboQuant *specifically* in the speculative-decoding verification path?
**Short answer:** Yes, and this is actually where custom kernel work makes the most sense. Realistic target is **2.5–4× over current tuned Triton** on Qwen3-30B+TQ+spec, which would beat FP16+EAGLE3 production throughput on memory-compressed KV. Effort: 4–5 months of focused work.

## Why spec decoding changes the calculus

The decode-kernel architectural detail that matters: the current TurboQuant kernel's grid is `(batch, query_head, kv_split)` with **one query per work-group**. This is the canonical "decode" assumption — M=1 in GEMM terms.

### The M=1 problem

DPAS/XMX on Xe2 is a 2D systolic array with a native tile shape around M=8, N=16, K=16 for fp16. When your problem has M=1, you're using 1/8 of the systolic lanes — the rest are idle. That's why my initial SYCL design brief hedged on "should we use DPAS at all for decode."

For **standard** autoregressive decode (1 token out per step), M=1 is baked in by the algorithm. No way around it.

For **speculative decode verification**, the picture changes completely. When you verify N draft tokens:
- N queries all attend to the same KV cache
- All N queries fit in one forward pass
- If you batch them into the M dimension, **M = N_spec_tokens ≈ 8**

**Suddenly DPAS fits perfectly.** The exact case where systolic matrix units shine. And this is also the compute-heaviest step in spec-decoded inference (verification dominates when acceptance is good).

### Evidence from current benchmarks

I didn't benchmark TQ+spec on Qwen3-30B earlier, but we have TQ+suffix data on Gemma4:

| Config | tok/s @ C=8 | Notes |
|---|---:|---|
| FP16 baseline, no spec | 51.2 | Standard attention, single query |
| FP16 + suffix decoding | 121.5 | 2.37× from spec |
| TQ k3v4_nc, no spec | 27.0 | 0.53× vs FP16 baseline |
| TQ k3v4_nc + suffix | 36.3 @ C=16 | ~1.34× from spec — spec helped much less |
| TQ k8v4, no spec | 82.9 | 1.62× vs FP16 baseline via k8v4 preset win |

The numbers tell a story: **speculative decoding helps TurboQuant much less than it helps FP16 (1.34× vs 2.37×).** Spec amplifies the relative slowdown of TQ from 0.53× to 0.30×.

That's the gap a custom kernel would close. TQ's current decode kernel isn't a batched multi-query kernel — it's a single-query kernel dispatched multiple times. Each spec token ends up paying the full kernel-launch overhead, and the M=1 problem means none of the verification queries share DPAS resources.

## Where the speedup comes from

Four compounding effects:

### 1. M=N_spec DPAS utilization (~1.5–2×)

Issue multi-query verification as a single DPAS-based kernel with M=N_spec_tokens. For N=8 (typical suffix/EAGLE3 config), this matches Xe2's native tile shape exactly. The scoring GEMM `Q @ K^T` becomes an 8×16×128 systolic op instead of 8 separate 1×16×128 scalar dot products.

### 2. KV prefetch amortization (~1.2–1.5×)

All N verification queries read the same KV cache slots. Load the packed keys/values once into SLM (shared local memory), dequant once, reuse 8 times. Current Triton kernel reloads on every single-query dispatch.

Mechanically: a work-group handles one (batch, kv_tile) pair, iterates across N queries in registers or shared, and broadcasts the dequantized K into each query's score computation. Standard flash-attention tiling, adapted to the quantized cache.

### 3. Launch overhead elimination (~1.3–1.5×)

Current per-attention-call overhead on XPU (Python → Triton → Level-Zero dispatch) is measurable. A single SYCL kernel launch for the whole spec batch removes N-1 of these overheads.

Evidence: my quick-wins experiment showed k8v4 tuning gives **zero** improvement from larger tiles and more warps — strongly suggesting the bottleneck is launch/dispatch overhead or memory latency, not compute. Moving the whole spec batch into one launch directly attacks this.

### 4. Fused quantize-on-load (~1.1–1.2×)

Intel's `vllm-xpu-kernels` flash-attention already does fused FP8 descale inside the K operand pipeline for `joint_matrix`. Applying the same pattern to TurboQuant:

- For k8v4: fuse `tl.float8e4nv` bitcast into the DPAS K fragment load
- For k3v4_nc: fuse bit-unpack + centroid LUT into the K fragment load, identical to how flash-attn handles `k_descale`

This eliminates the intermediate "reconstruct full fp16 K tensor, then do GEMM" pattern in the current Triton code.

### Compounded estimate

| Factor | k8v4 | k3v4_nc |
|---|---:|---:|
| Native SYCL vs tuned Triton (base) | 1.2× | 1.5× |
| M=N_spec DPAS utilization | 1.8× | 1.8× |
| KV prefetch amortization | 1.3× | 1.3× |
| Fused quantize-on-load | 1.1× | 1.2× |
| **Total for spec verification** | **~3.1×** | **~4.2×** |

These are *compounded* so they shouldn't be taken as additive. Realistic expectation: **2.5–4× speedup on the TQ+spec path**.

Important: this is specifically the verification step. The draft generation step (which is small — 8 tokens on suffix decoding) isn't accelerated by this kernel. But verification dominates total time when acceptance is good, so the end-to-end throughput improvement would still be close to the verification improvement.

## What this means in absolute numbers

Extrapolating from current benchmarks:

### Gemma4-31B (dense, head_dim=256/512)

| Config | Current tok/s | Projected with custom kernel |
|---|---:|---:|
| FP16 + suffix (production) | 134.3 @ C=12 | 134.3 (unchanged) |
| TQ k8v4 (no spec measured) | ~83 @ C=8 | ~170 |
| TQ k8v4 + suffix (estimated) | ~100–110 | **~250–300** |

Would make TQ+k8v4+spec competitive with or faster than FP16+suffix on dense models — while still delivering 2.2× KV capacity.

### Qwen3-30B-A3B (MoE, head_dim=128) — the sweet spot

| Config | Current tok/s | Projected with custom kernel |
|---|---:|---:|
| FP16 + EAGLE3 (production) | 298.5 @ C=20 | 298.5 (unchanged) |
| TQ k3v4_nc (no spec tested) | 141 @ C=20 | ~220 |
| TQ k3v4_nc + spec | ~180 (est.) | **~450–600** |
| TQ k8v4 + spec | ~280 (est.) | **~550–700** |

**This is the upside case that justifies the engineering work.** Qwen3-30B with TQ + spec + custom kernel would significantly beat the current production config — with 8.5× KV capacity on top.

The MoE architecture is especially favorable: attention is a larger fraction of per-token compute, so attention speedups flow more directly to end-to-end throughput.

## Why the current Triton path can't reach this

I have direct evidence that the Triton path has hit its ceiling on k8v4: zero improvement from `BLOCK_KV=16 + num_warps=4` tuning in my quick-wins experiment. The classic tuning levers gave nothing. That means:

1. The kernel isn't tile-size-bound (more tile would help)
2. The kernel isn't EU-occupancy-bound (more warps would help)
3. Something structural is the bottleneck — most likely launch/dispatch overhead, SPIRV codegen quality for bit-unpack patterns, or lack of DPAS utilization

Triton-side tuning can't fix any of these. Native SYCL can.

## Engineering plan

### Scope for a spec-decoding-focused TurboQuant SYCL kernel

**In scope:**
- `_tq_decode_stage1_spec` — multi-query variant with M=N_spec DPAS, fused K-dequant
- `_tq_decode_stage2` — unchanged, already efficient
- `_tq_fused_store_batch` — batch-store N tokens per attention call
- Dispatcher logic in Python to route spec calls to the new kernel, regular decode to existing kernel

**Out of scope:**
- Tree attention (EAGLE3 with tree structure) — more complex, do after basic linear verification works
- The `_tq_fused_store_mse` kernel — already fast enough, no reason to rewrite
- Custom WHT butterfly in-kernel — stick with external GEMM for now, fuse later if profiling shows it matters

### Milestones

| Week | Deliverable |
|---|---|
| 1–2 | SYCL dev environment set up locally (oneAPI 2025.3 on host), first `joint_matrix` hello-world on B70 |
| 3–4 | `_tq_decode_stage1_spec` bring-up without DPAS — use XVE FMAs for correctness baseline; validate against Triton reference |
| 5–6 | Add DPAS path with M=N_spec; profile with VTune + unitrace to confirm we're actually using XMX |
| 7–8 | Add SLM-staged K prefetch + fused K-dequant; tune BLOCK_KV and WG shape |
| 9–10 | Integrate into vllm-xpu-kernels-style custom op (`torch.ops._xpu_C.tq_decode_spec`) with pybind wrapper |
| 11–12 | End-to-end benchmark vs tuned Triton; submit as PR to vllm-xpu-kernels with the design doc from this project |
| 13–16 | Tree attention support for EAGLE3 compatibility |
| 17–20 | Polish, upstream review iterations, documentation |

**Total: ~4–5 months of focused solo work.** The 12-week baseline from my earlier SYCL design brief assumed single-query decode; adding the spec-verification path is roughly +4 weeks for the DPAS path and +4 more weeks for tree attention.

## Risk register

**Risk 1: DPAS tile shapes may not match head_dim evenly.** For Qwen3-30B head_dim=128, a 16-lane sub-group × 8 elements per lane = 128, so one DPAS op covers a full dim. For Gemma4 head_dim=512, we need 4 DPAS ops per query-head pair, with extra register pressure. Mitigation: measure per model; acceptable slowdown on Gemma4 if Qwen3-30B hits the target.

**Risk 2: Correctness bugs in custom quant kernels are silent.** TurboQuant produces "plausible but wrong" output under subtle kernel bugs — not a crash, just slightly degraded quality. Mitigation: bit-exact comparison against the Triton reference for every test, plus PPL measurement on a held-out validation set before accepting any change.

**Risk 3: Intel's XMX programming is less documented than CUDA Tensor Cores.** The `joint_matrix` API exists, but edge cases (BF16 accumulation, mixed-precision fragments, reduced-precision operations) have thinner docs. Mitigation: hew close to the patterns already working in vllm-xpu-kernels' flash attention; that code has been debugged against real hardware.

**Risk 4: "Tree attention" for EAGLE3 is structurally different from suffix decoding.** Suffix decoding is linear (verify N tokens in sequence); EAGLE3 verifies a tree of candidate tokens with partial orderings. Might need two separate kernels — one linear, one tree. Mitigation: start with linear (suffix), that alone covers the simpler case and validates the architecture.

**Risk 5: Getting upstreamed into `vllm-xpu-kernels` isn't guaranteed.** Intel's team might prefer to write their own. Mitigation: maintain parallel repo with mount-patch integration (same pattern I'm using now); keep the PR offer open but don't block on acceptance.

## Comparison to alternatives

**Option A: Wait for Intel.** They filed [issue #172](https://github.com/vllm-project/vllm-xpu-kernels/issues/172) on Qwen3.5 optimization in 2026-Q1. No public TurboQuant roadmap. Best-case timing: 6–12 months. We'd ship nothing to the community in the meantime.

**Option B: Tune Triton harder.** Empirically impossible on k8v4 (zero headroom). Some room on k3v4_nc (maybe another 1.5× with autotune configs). Cap: roughly what Intel's own Triton team could achieve in a focused month.

**Option C: Custom SYCL as described.** 2.5–4× speedup on the spec verification path. 4–5 months of focused work. Likely to produce a first-of-its-kind result (no SYCL TurboQuant exists publicly).

**Option D: Don't do it.** TurboQuant on XPU is already useful as is. k8v4 at 83 tok/s on Gemma4 and k3v4_nc at 141 tok/s on Qwen3-30B are deployable today for memory-constrained workloads.

## Recommendation

**Option C is the right investment if the goal is R&D portfolio quality and first-mover position in a niche.** No public SYCL TurboQuant kernel exists. A working implementation that beats current Triton by 2.5–4× on the spec-decoding path would be a genuinely novel contribution. It would also produce the most compelling performance number of this entire project (Qwen3-30B TQ+spec on a single B70 beating FP16+EAGLE3).

**Option D is the right call if the goal is immediate production value.** Current TQ is already functional; further speedup has diminishing returns for single-user deployment.

Given your stated goal (showcase R&D skills through open-source work), **Option C is what I'd recommend.** Specifically:

1. Start with the Week 1–2 milestone: local oneAPI dev environment, hello-world, confirm B70 hardware access outside the vllm container
2. **Don't commit to the full 20-week plan up front** — the first 4 weeks are discovery. If DPAS at M=8 doesn't produce a clear win in the proof-of-concept, pivot to Option D and consolidate the research.
3. Document every failure. Negative results (like the Gemma4 heterogeneous head_dim penalty, or the k8v4 tuning-insensitivity) have been some of the most interesting findings of this project. Keep that bar high for the SYCL work.

## What I'd build first

Concretely, the minimal viable proof-of-concept:

```
turboquant-xpu/
└── sycl/
    ├── CMakeLists.txt
    ├── tq_decode_spec.cpp     # SYCL kernel
    ├── tq_decode_spec_py.cpp  # pybind11 wrapper
    ├── bench/
    │   └── bench_decode_spec.py  # compare vs Triton on synthetic data
    └── reference/
        └── tq_decode_reference.py  # numpy reference for correctness
```

Goal of the PoC: on synthetic decode data with N_spec=8, B=4, Hq=32, Hk=4, head_dim=128, seqlen=8192, measure:

- Correctness: bit-exact (or within float rounding) vs Triton reference
- Throughput: latency per spec-batch call, compared to Triton-with-N-launches

If SYCL-with-DPAS beats Triton-with-N-launches by 2× or more at this micro-benchmark level, proceed to Weeks 5+. If not, document why and fall back to Option D.

This is about 3 weeks of work to a go/no-go data point, not a 5-month commitment.
