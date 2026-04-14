# Fused-N_spec Triton kernel results

**Headline:** Fused-N_spec Triton kernel reduces PoC-shape decode wall time by 63–76% (from 9.0–16.0 ms to 3.3–3.8 ms) over the N_spec=8 looped baseline.

---

## Results table

| Preset               | looped_ms | fused_ms | Speedup |
|----------------------|----------:|---------:|--------:|
| turboquant_k8v4      |     8.955 |    3.305 |   2.71x |
| turboquant_k3v4_nc   |    16.005 |    3.795 |   4.22x |

PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, NUM_KV_SPLITS=32.

---

## Methodology

- Platform: Intel Arc Pro B70 (Xe2, 160 EU, ~560 GB/s GDDR6X)
- Python: `.venv-sycl/bin/python` (torch-XPU 2.8.0+xpu, Triton XPU bundled)
- Warmup: 5 full passes (JIT compilation excluded from timing)
- Timed iterations: N=20, wall-clock via `time.perf_counter()` + `torch.xpu.synchronize()` after each pass
- Looped baseline: `triton_turboquant_decode_attention_xpu` called 8× in Python
- Fused kernel: `triton_turboquant_decode_attention_spec_xpu` called once

---

## What materialised

**Launch overhead savings (yes, dominates):** The pre-profiling analysis estimated ~24% of wall time as LZ submission overhead across 8×2=16 dispatches. The fused kernel reduces this to 2 dispatches (stage1+stage2), eliminating ~14 kernel launches per decode step. At the PoC shape the speedup is 2.71–4.22x — substantially exceeding the projected 35–55%, which underestimated this because the looped baseline also pays Python-loop overhead (~262 µs/call × 8 = 2.1 ms).

**K/V dequant sharing (yes, measurable):** k3v4_nc achieves 4.22x vs k8v4's 2.71x. The MSE unpack (3 loads + centroid gather + norm correction per token) is considerably more expensive than the FP8 single-load path. Sharing that work across 8 queries amplifies the savings for the MSE preset by an additional ~1.5x beyond what pure launch overhead explains.

**Grid occupancy improvement (minor):** Both kernels use grid (B, Hq, NUM_KV_SPLITS) — the fused kernel does not increase grid size, so EU occupancy does not increase. The benefit comes entirely from the two wins above.

---

## Register pressure

`q_all[8, 128]` = 4KB, `acc[8, 128]` = 4KB, `scores[8, 16]` = 512 B — approximately 8.5KB total state per program. The B70 XPU Triton compiler did not report spills (kernel compiled and ran correctly at N_spec=8 with `num_warps=1`). No spill-to-scratch was observed in practice.

---

## Correctness

2/2 parametrizations pass the `test_fused_nspec.py` suite:
- turboquant_k8v4 (FP8 keys): identical NaN mask as looped baseline, max non-NaN abs error < 5e-3
- turboquant_k3v4_nc (3-bit MSE + NC): same

NaN appearance in tests is an artefact of random uint8 KV data producing FP8 NaN bit patterns (0x7F in E4M3); the fused and looped paths produce NaN in precisely the same locations.

---

## Next steps

**Worth upstreaming to vLLM PR #38479?** Yes with caveats:
- The fused kernel is orthogonal to Intel's internal SYCL plans (issue #271) — it is pure Triton Python and would benefit any speculative-decoding integration.
- The 2.71x k8v4 speedup is compelling; k3v4_nc at 4.22x is even more so.
- Upstream path requires: (a) adding N_spec as a first-class launcher argument in the vLLM decode attention API, (b) fallback to looped when N_spec=1 to avoid regression on standard decode.
- Register pressure at N_spec=16 (16KB) may hit the spill threshold on non-B70 hardware — would need a tiled-N_spec fallback (process 4 queries at a time, 2 passes).
