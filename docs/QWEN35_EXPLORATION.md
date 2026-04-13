# Qwen3.5 on Intel XPU — Can TurboQuant Help?

**Date:** 2026-04-13
**Context:** Qwen3.5-35B-A3B is the most capable open MoE model as of early 2026. It uses a hybrid **Gated Delta Network (GDN)** linear attention architecture (30 linear + 10 full attention layers out of 40). It currently does **not run on Intel XPU** with vLLM 0.19. This doc explores whether our TurboQuant work opens a path to enabling it.

## TL;DR

**No, TurboQuant does not unblock Qwen3.5.** The three independent blockers for Qwen3.5 are (1) vLLM's `ChunkGatedDeltaRule` doesn't dispatch to Intel's native SYCL GDN kernel, (2) a KV cache page-size unification regression in vLLM 0.19 breaks hybrid model boot, and (3) a Triton/FLA layout bug produces garbage output. None of these touch KV cache quantization, so TurboQuant neither helps nor hurts Qwen3.5.

**However,** the SYCL kernel research we're doing for TurboQuant (see [SYCL_KERNEL_DESIGN.md](SYCL_KERNEL_DESIGN.md)) is a direct template for the work Intel/vLLM needs to do to enable Qwen3.5. The same toolchain (oneAPI 2025.3, `joint_matrix` extension, CUTLASS-SYCL patterns, AOT targets for BMG-G31) that we'd use to write a faster TurboQuant stage-1 kernel is the toolchain needed to write the missing XPU GDN dispatcher.

## Background: why Qwen3.5 is blocked on XPU

Three stacked problems, none of them about KV cache quantization:

### Blocker 1: No XPU GDN dispatcher in vLLM

A native SYCL GDN kernel exists today in `vllm-xpu-kernels 0.1.5`:

```
torch.ops._xpu_C.gdn_attention(core_attn_out, z, projected_states_qkvz,
    projected_states_ba, num_k_heads, num_v_heads, head_k_dim, head_v_dim, ...)
```

It's compiled, linked, and registered — but **nothing in vLLM 0.19 calls it for Qwen3.5**. The dispatcher in `vllm/model_executor/layers/mamba/gdn_linear_attn.py` is:

```python
supports_flashinfer = current_platform.is_cuda() and current_platform.is_device_capability(90)
self._forward_method = self.forward_cuda if use_flashinfer else self.forward_native
```

There is no `forward_xpu` branch. Non-Hopper-CUDA (and any XPU) falls through to `forward_native`, which runs the FLA (Flash Linear Attention) Triton kernels in an eager fallback at roughly CPU speed — about 1 tok/s on B70.

### Blocker 2: KV cache page-size unification regression

vLLM logs `Padding mamba page size by 0.76%` at config time, but after the profile forward pass, the padding is lost/recomputed and `unify_kv_cache_spec_page_size` raises:

```
NotImplementedError: The page size of the layer is not divisible by the
maximum page size. Cannot unify by adjusting block_size.
```

Tested 5 different combinations of `max_model_len`, `gpu_memory_utilization`, `enforce_eager`, `max_num_seqs`, `num_gpu_blocks_override` — all hit this same wall. This is a regression specific to hybrid models (mamba + attention layers in the same network).

### Blocker 3: FLA Triton layout bug (affects CUDA too)

vLLM issue [#38643](https://github.com/vllm-project/vllm/issues/38643) documents that Qwen3.5 produces gibberish output when forced onto the Triton/FLA path, even on CUDA hardware (RTX 5090 / SM89 Ada). The fix has to happen in `flash-linear-attention` upstream or in vLLM's wrapper — not in the KV cache subsystem.

## Does TurboQuant change any of this?

**No.** TurboQuant replaces the KV cache **storage** and the **attention score compute** in the standard full-attention path. It has:
- Its own attention backend (`TurboQuantAttentionBackend`)
- Its own Triton kernels for store/decode on packed cache
- Its own KV cache spec (`FullAttentionSpec` with effective_head_size = slot_size_aligned // 2)

None of that touches the `ChunkGatedDeltaRule` code path used by Qwen3.5's linear-attention layers. Those layers don't have a KV cache in the conventional sense — they maintain an SSM state (`conv_state`, `ssm_state`) which is fundamentally different.

The 10 full-attention layers of Qwen3.5 *could* theoretically use TurboQuant, but:
1. The other 30 layers would still need a working GDN dispatcher, which we don't have
2. TurboQuant attaches to the standard attention backend selection pipeline, which is shared across layers — mixing TQ for some layers and standard for others isn't wired up in vLLM today

## What IS transferable from our TurboQuant work

### 1. The vLLM integration pattern

We've documented and proven a recipe for patching vLLM's core files via Docker mount to add new KV cache dtypes and attention backends. The same recipe would apply for anyone adding a `forward_xpu` method to `ChunkGatedDeltaRule`:

```
patches/vllm_mounts/
├── cache.py                  # register new dtype
├── torch_utils.py            # dtype → torch mapping
├── xpu.py                    # XPU backend routing
├── attention.py              # layer-level buffer init
├── attention_config.py       # config flags
├── registry.py               # attention backend enum
└── backends/turboquant_attn.py  # new backend impl
```

An XPU GDN dispatcher patch would be much simpler — just one file (`gdn_linear_attn.py`) with a platform-check branch and the `_xpu_C.gdn_attention` call.

### 2. The SYCL kernel toolchain

Everything in [SYCL_KERNEL_DESIGN.md](SYCL_KERNEL_DESIGN.md) applies:
- oneAPI 2025.3 compiler + Level-Zero v2 runtime
- `joint_matrix` extension with `bmg` AOT target
- Sub-group size 16 for 2D block I/O
- SLM staging of hot data
- vllm-xpu-kernels CMake build + custom op registration

The GDN kernel already exists in SYCL (`_xpu_C::gdn_attention`), so for that specific path the kernel work is done. What's missing is the Python-side dispatcher. For a proper XPU-tuned TurboQuant, we'd be writing new SYCL kernels. For enabling Qwen3.5, we'd be writing Python glue.

### 3. The diagnostic methodology

Our approach for TurboQuant — identify integration points, patch incrementally, verify each layer works before moving up — is directly applicable. The Qwen3.5 attempt chronology in `QWEN35_FAILURE_ANALYSIS.md` shows 5 config variations, each exposing a different blocker. A methodical patch loop would:

1. Write a `forward_xpu` method in `ChunkGatedDeltaRule` that calls `_xpu_C.gdn_attention`
2. Debug the tensor layout mismatch between the Qwen3.5 code path and the kernel signature (the hard part)
3. Verify output correctness end-to-end on a short prompt
4. File fixes upstream for the KV page-size regression if still present
5. Benchmark against Qwen3-30B-A3B baseline

## Hybrid TurboQuant + GDN: a speculative long-term design

If we had:
- A working XPU GDN dispatcher (Blocker 1 fixed)
- Patched KV cache unification (Blocker 2 fixed)
- Output correctness restored (Blocker 3 fixed)

...then on the 10 full-attention layers of Qwen3.5, TurboQuant would work exactly as it does for Qwen3-30B. Those layers use standard softmax attention with a regular KV cache, just surrounded by 3× GDN layers on either side.

Expected gain from TQ on the 10 full layers:
- Full-attention layers are 25% of the total layer count, but they carry the long-context reasoning load (the GDN layers are parallelizable but stateful)
- On a model with 262K+ context, the full-attention KV cache is the memory bottleneck
- TQ k3v4_nc at 8.5× compression (same ratio we measured on Qwen3-30B) on those 10 layers would enable 500K+ context without hitting memory limits

Plumbing this would require vLLM's `kv_cache_dtype_skip_layers` mechanism to only apply TQ to the full-attention layers, leaving GDN layers with their native SSM state storage. That feature already exists in PR #38479's design and is used for "skip first/last 2 layers" for quality preservation.

## Honest assessment

**For Qwen3.5 specifically, our TurboQuant work is orthogonal.** It doesn't help enable the model. But:

1. **The SYCL research we did** is the right foundation for when someone (Intel, the community, or me) wires up the XPU GDN path
2. **The mount-patch pattern** we established for TurboQuant integration could be applied to a single-file GDN dispatcher patch
3. **Once Qwen3.5 runs**, TurboQuant immediately becomes useful on its 10 full-attention layers for 3-5× KV compression there

**Recommendation:** File a separate GitHub issue on `vllm-xpu-kernels` specifically asking for the GDN dispatcher wire-up (Issue [#189](https://github.com/vllm-project/vllm-xpu-kernels/issues/189) is the existing closest match — it's asking to expose `gdn::causal_conv1d` and `gdn::gated_delta_rule`). Comment on that issue with our TurboQuant integration experience as evidence that the vLLM-side plumbing is tractable.

## What Intel is working on

From browsing issues on `vllm-xpu-kernels`:

- Issue [#172](https://github.com/vllm-project/vllm-xpu-kernels/issues/172) "Qwen3.5 support and optimization plan" — Intel is aware, has L2norm optimization in PR #222 and fp32 ssm_state in PR #220
- PR #264 is the precision fix at 8K+ tokens (referenced but URL not yet explored)
- Nothing public yet on the vLLM-side dispatcher wire-up

When these land and someone writes the `forward_xpu` method, we can test Qwen3.5 with TurboQuant on its full-attention layers and report back. Until then, Qwen3-30B-A3B remains the best available MoE on this hardware and TurboQuant extends its usefulness to 500K+ context scenarios.

## References

- Qwen3.5 failure analysis: `/apps/b70-vllm/QWEN35_FAILURE_ANALYSIS.md` (on the deployment host, not in this repo)
- Intel issue on Qwen3.5 support plan: https://github.com/vllm-project/vllm-xpu-kernels/issues/172
- Request to expose GDN ops: https://github.com/vllm-project/vllm-xpu-kernels/issues/189
- FLA layout bug affecting CUDA: https://github.com/vllm-project/vllm/issues/38643
- Our TurboQuant SYCL design: [SYCL_KERNEL_DESIGN.md](SYCL_KERNEL_DESIGN.md)
