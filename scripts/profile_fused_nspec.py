#!/usr/bin/env python3
"""Profile fused-N_spec Triton decode kernel vs un-fused baseline.

Mirrors scripts/profile_triton_decode.py but calls
`triton_turboquant_decode_attention_spec_xpu` (commit 425fc5c) once per
outer iteration instead of looping N_spec times internally.

Runs 50 outer loops at PoC shape for each of two presets:
  - turboquant_k8v4   (KEY_FP8=True,  NORM_CORRECTION=False)
  - turboquant_k3v4_nc (KEY_FP8=False, NORM_CORRECTION=True)

Collects:
  - Wall-clock (with/without sync)
  - Per-kernel LZ dispatch count (should be 2, not 16)
  - CPU dispatch vs GPU time ratio
  - Bandwidth estimate (bytes touched / GPU time)
  - Compute utilization estimate (FLOP / GPU time)
  - torch.profiler XPU activity table (PTI-backed)

Shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192.
"""
import os
import sys
import math
import time
import json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
import torch.profiler as tp

from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"

# PoC shape
N_SPEC  = 8
B       = 4
HQ      = 32
HK      = 4
D       = 128
SEQLEN  = 8192
BLOCK_SIZE = 16
MAX_NUM_KV_SPLITS = 32

WARMUP  = 5
N_TIMED = 50   # outer loops for timing
N_NOSYNC = 50  # outer loops for CPU dispatch measurement

# Hardware constants (Arc Pro B70 / Xe2 Battlemage)
BW_PEAK_GBS   = 608.0   # GB/s peak HBM bandwidth
FLOPS_PEAK_FP32 = 8e12  # 8 TFLOPS FP32 peak

# Bytes touched per fused kernel call:
#   KV cache read: num_blocks * block_size * Hk * slot_size_aligned
# We compute slot_size_aligned per preset from cfg.
# FLOP per fused call (all N_spec in one shot):
#   N_spec * (QK matmul + AV matmul) = N_spec * 2 * B * Hq * seqlen * D * 2
FLOP_FUSED = N_SPEC * 2 * B * HQ * SEQLEN * D * 2  # float ops


def build_hadamard(d: int) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(DEVICE)


def setup(preset: str):
    cfg = TurboQuantConfig.from_cache_dtype(preset, D)
    num_blocks = math.ceil(SEQLEN / BLOCK_SIZE) * B
    rng = np.random.default_rng(2026)
    q_np = rng.standard_normal((N_SPEC, B, HQ, D)).astype(np.float32)
    q_spec = torch.tensor(q_np, device=DEVICE)

    kv_cache = torch.randint(
        0, 255,
        (num_blocks, BLOCK_SIZE, HK, cfg.slot_size_aligned),
        dtype=torch.uint8,
        device=DEVICE,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    seq_lens    = torch.full((B,), SEQLEN, dtype=torch.int32, device=DEVICE)

    PiT = build_hadamard(D)
    Pi  = PiT.T.contiguous()
    cents = torch.randn(cfg.n_centroids, device=DEVICE) * 0.3
    scale = 1.0 / math.sqrt(D)
    return cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale


def call_fused(cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale,
               key_fp8: bool, nc: bool):
    return triton_turboquant_decode_attention_spec_xpu(
        query=q_spec,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=Pi,
        centroids=cents,
        scale=scale,
        mse_bits=cfg.mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.value_quant_bits,
        key_fp8=key_fp8,
        norm_correction=nc,
        PiT=PiT,
        max_num_kv_splits=MAX_NUM_KV_SPLITS,
    )


def call_looped(cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale,
                key_fp8: bool, nc: bool):
    for n in range(N_SPEC):
        triton_turboquant_decode_attention_xpu(
            query=q_spec[n],
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=cents,
            scale=scale,
            mse_bits=cfg.mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.value_quant_bits,
            key_fp8=key_fp8,
            norm_correction=nc,
            PiT=PiT,
            max_num_kv_splits=MAX_NUM_KV_SPLITS,
        )


def profile_preset(preset: str, key_fp8: bool, nc: bool, prof_dir: str):
    print(f"\n{'='*70}")
    print(f"PRESET: {preset}  (key_fp8={key_fp8}, norm_correction={nc})")
    print(f"{'='*70}")

    cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale = setup(preset)

    # Bytes touched per call: full KV cache read
    num_blocks = math.ceil(SEQLEN / BLOCK_SIZE) * B
    kv_bytes = num_blocks * BLOCK_SIZE * HK * cfg.slot_size_aligned
    print(f"KV bytes/call: {kv_bytes / 1e6:.2f} MB  (slot_size_aligned={cfg.slot_size_aligned})")

    def fused():
        return call_fused(cfg, q_spec, kv_cache, block_table, seq_lens,
                          Pi, PiT, cents, scale, key_fp8=key_fp8, nc=nc)

    def looped():
        call_looped(cfg, q_spec, kv_cache, block_table, seq_lens,
                    Pi, PiT, cents, scale, key_fp8=key_fp8, nc=nc)

    # ------------------------------------------------------------------
    # Warmup: triggers Triton JIT for both paths
    # ------------------------------------------------------------------
    print("[profile] warming up (JIT compilation, both paths)...")
    for _ in range(WARMUP):
        looped()
    torch.xpu.synchronize()
    for _ in range(WARMUP):
        fused()
    torch.xpu.synchronize()
    print("[profile] warmup done")

    # ------------------------------------------------------------------
    # SECTION 1: Wall-clock timing (synced) — fused path
    # ------------------------------------------------------------------
    times_fused_ms = []
    for _ in range(N_TIMED):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fused()
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        times_fused_ms.append((t1 - t0) * 1000.0)

    arr_f = sorted(times_fused_ms)
    mean_fused = sum(times_fused_ms) / len(times_fused_ms)
    med_fused  = arr_f[len(arr_f) // 2]
    p5_fused   = arr_f[int(len(arr_f) * 0.05)]
    p95_fused  = arr_f[int(len(arr_f) * 0.95)]

    # ------------------------------------------------------------------
    # SECTION 1b: Wall-clock timing (synced) — looped baseline
    # ------------------------------------------------------------------
    times_loop_ms = []
    for _ in range(N_TIMED):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        looped()
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        times_loop_ms.append((t1 - t0) * 1000.0)

    arr_l = sorted(times_loop_ms)
    mean_loop = sum(times_loop_ms) / len(times_loop_ms)
    med_loop  = arr_l[len(arr_l) // 2]

    speedup = mean_loop / mean_fused if mean_fused > 0 else float("nan")

    print(f"\n[timing] Wall-clock (synced):")
    print(f"  LOOPED  mean={mean_loop:.3f} ms  median={med_loop:.3f} ms")
    print(f"  FUSED   mean={mean_fused:.3f} ms  median={med_fused:.3f} ms  p5={p5_fused:.3f}  p95={p95_fused:.3f}")
    print(f"  SPEEDUP: {speedup:.2f}x")

    # ------------------------------------------------------------------
    # SECTION 2: CPU dispatch overhead (no-sync)
    # ------------------------------------------------------------------
    cpu_times_fused_ms = []
    torch.xpu.synchronize()
    for _ in range(N_NOSYNC):
        t0 = time.perf_counter()
        fused()
        t1 = time.perf_counter()
        cpu_times_fused_ms.append((t1 - t0) * 1000.0)
    torch.xpu.synchronize()

    cpu_times_loop_ms = []
    torch.xpu.synchronize()
    for _ in range(N_NOSYNC):
        t0 = time.perf_counter()
        looped()
        t1 = time.perf_counter()
        cpu_times_loop_ms.append((t1 - t0) * 1000.0)
    torch.xpu.synchronize()

    cpu_mean_fused = sum(cpu_times_fused_ms) / len(cpu_times_fused_ms)
    cpu_mean_loop  = sum(cpu_times_loop_ms) / len(cpu_times_loop_ms)

    gpu_compute_fused = mean_fused - cpu_mean_fused
    gpu_fraction_fused = gpu_compute_fused / mean_fused if mean_fused > 0 else 0.0

    # Per-launch CPU overhead: fused has 2 LZ dispatches (stage1+stage2)
    cpu_per_launch_fused_us = cpu_mean_fused / 2 * 1000.0
    cpu_per_launch_loop_us  = cpu_mean_loop  / (N_SPEC * 2) * 1000.0

    print(f"\n[dispatch] CPU dispatch (no-sync):")
    print(f"  LOOPED   mean={cpu_mean_loop:.3f} ms  per-launch={cpu_per_launch_loop_us:.1f} us  (x{N_SPEC*2} dispatches)")
    print(f"  FUSED    mean={cpu_mean_fused:.3f} ms  per-launch={cpu_per_launch_fused_us:.1f} us  (x2 dispatches)")
    print(f"  FUSED    GPU compute (wall-dispatch): {gpu_compute_fused:.3f} ms  ({gpu_fraction_fused*100:.1f}% of wall)")

    # ------------------------------------------------------------------
    # SECTION 3: Bandwidth + compute utilization estimates
    # ------------------------------------------------------------------
    # GPU time estimate: use wall-clock mean as upper bound;
    # stage2 is negligible so essentially all time is stage1
    gpu_time_s = mean_fused / 1000.0  # seconds (upper bound, includes some CPU)
    bw_actual_gbs = (kv_bytes / 1e9) / gpu_time_s
    bw_util_pct   = bw_actual_gbs / BW_PEAK_GBS * 100.0

    flops_actual = FLOP_FUSED / gpu_time_s
    compute_util_pct = flops_actual / FLOPS_PEAK_FP32 * 100.0
    arith_intensity  = FLOP_FUSED / kv_bytes

    print(f"\n[utilization] Hardware utilization (fused, wall-clock upper bound):")
    print(f"  KV read: {kv_bytes/1e6:.2f} MB  |  wall: {mean_fused:.3f} ms")
    print(f"  BW actual: {bw_actual_gbs:.1f} GB/s  ({bw_util_pct:.1f}% of {BW_PEAK_GBS:.0f} GB/s peak)")
    print(f"  FLOP: {FLOP_FUSED/1e9:.2f} GFLOP  |  actual: {flops_actual/1e12:.3f} TFLOPS")
    print(f"  Compute util: {compute_util_pct:.1f}% of {FLOPS_PEAK_FP32/1e12:.0f} TFLOPS (FP32)")
    print(f"  Arithmetic intensity: {arith_intensity:.1f} FLOP/byte")
    print(f"  Ridge point (FP32): {FLOPS_PEAK_FP32/1e9 / BW_PEAK_GBS:.1f} FLOP/byte")

    bw_classification = "memory-bound" if arith_intensity < (FLOPS_PEAK_FP32/1e9 / BW_PEAK_GBS) else "compute-bound (roofline)"
    print(f"  => Roofline classification: {bw_classification}")

    # ------------------------------------------------------------------
    # SECTION 4: torch.profiler XPU activity (3 active steps)
    # ------------------------------------------------------------------
    print(f"\n[profiler] running torch.profiler (XPU activity)...")
    preset_tag = preset.replace("turboquant_", "")
    prof_subdir = os.path.join(prof_dir, f"fused_{preset_tag}")
    os.makedirs(prof_subdir, exist_ok=True)

    with tp.profile(
        activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.XPU],
        record_shapes=False,
        with_stack=False,
        schedule=tp.schedule(wait=0, warmup=1, active=3, repeat=1),
        on_trace_ready=tp.tensorboard_trace_handler(prof_subdir),
    ) as prof:
        for step in range(4):  # wait=0, warmup=1, active=3
            fused()
            torch.xpu.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=25))

    # Extract per-kernel GPU times
    avgs = prof.key_averages()
    rows = []
    stage1_gpu_us = 0.0
    stage2_gpu_us = 0.0
    dispatch_count_s1 = 0
    dispatch_count_s2 = 0
    for evt in avgs:
        xpu_total = (evt.device_time_total if hasattr(evt, "device_time_total")
                     else getattr(evt, "xpu_time_total", 0))
        xpu_avg   = (evt.device_time if hasattr(evt, "device_time")
                     else getattr(evt, "xpu_time", 0))
        rows.append({
            "name":           evt.key,
            "cpu_time_us":    evt.cpu_time,
            "xpu_time_us":    xpu_avg,
            "count":          evt.count,
            "cpu_total_us":   evt.cpu_time_total,
            "xpu_total_us":   xpu_total,
        })
        if "stage1" in evt.key.lower():
            stage1_gpu_us      = xpu_avg
            dispatch_count_s1  = evt.count
        elif "stage2" in evt.key.lower():
            stage2_gpu_us      = xpu_avg
            dispatch_count_s2  = evt.count

    json_path = os.path.join(prof_subdir, "key_averages.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[profiler] key_averages saved to {json_path}")

    total_gpu_us = stage1_gpu_us + stage2_gpu_us
    if dispatch_count_s1 or dispatch_count_s2:
        print(f"\n[profiler] Stage GPU times (avg per call, from XPU activity):")
        print(f"  stage1_spec: {stage1_gpu_us:.1f} us  (x{dispatch_count_s1} calls)")
        print(f"  stage2_spec: {stage2_gpu_us:.1f} us  (x{dispatch_count_s2} calls)")
        print(f"  total GPU:   {total_gpu_us:.1f} us")
        # LZ dispatch count: should be 2 per outer iter (stage1 + stage2)
        # profiler captures N_ACTIVE_STEPS=3 outer iters
        lz_per_iter_s1 = dispatch_count_s1 / 3 if dispatch_count_s1 else 0
        lz_per_iter_s2 = dispatch_count_s2 / 3 if dispatch_count_s2 else 0
        print(f"  LZ dispatches per outer iter: stage1={lz_per_iter_s1:.0f}, stage2={lz_per_iter_s2:.0f}")
        print(f"  (Expected 1+1=2 vs un-fused 8+8=16)")

    # ------------------------------------------------------------------
    # SECTION 5: Summary block
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"SUMMARY — {preset}")
    print(f"{'─'*70}")
    print(f"  Wall (fused, synced):   mean={mean_fused:.3f} ms  p5={p5_fused:.3f}  p95={p95_fused:.3f}")
    print(f"  Wall (looped, synced):  mean={mean_loop:.3f} ms")
    print(f"  Speedup (fused/loop):   {speedup:.2f}x")
    print(f"  CPU dispatch (fused):   {cpu_mean_fused:.3f} ms  ({cpu_mean_fused/mean_fused*100:.1f}% of wall)")
    print(f"  LZ dispatches/iter:     2 (expected)  vs 16 (looped)")
    print(f"  BW utilization:         {bw_util_pct:.1f}%  ({bw_actual_gbs:.1f} GB/s / {BW_PEAK_GBS:.0f} GB/s)")
    print(f"  Compute utilization:    {compute_util_pct:.1f}%  ({flops_actual/1e12:.3f} TFLOPS / {FLOPS_PEAK_FP32/1e12:.0f} TFLOPS)")
    print(f"  Arith intensity:        {arith_intensity:.1f} FLOP/byte  (ridge={FLOPS_PEAK_FP32/1e9/BW_PEAK_GBS:.1f})")

    # Bottleneck classification
    if cpu_mean_fused / mean_fused > 0.4:
        classification = "LAUNCH-OVERHEAD-DOMINATED (still >40% dispatch)"
    elif bw_util_pct > compute_util_pct * 1.5:
        classification = "MEMORY-BANDWIDTH-BOUND"
    elif compute_util_pct > 20.0:
        classification = "COMPUTE-BOUND (EU limited)"
    else:
        classification = f"COMPUTE-UNDERUTILIZED (BW={bw_util_pct:.1f}%, FP32={compute_util_pct:.1f}%)"

    print(f"\n  BOTTLENECK: {classification}")

    return {
        "preset":           preset,
        "mean_fused_ms":    mean_fused,
        "mean_loop_ms":     mean_loop,
        "speedup":          speedup,
        "cpu_dispatch_fused_ms": cpu_mean_fused,
        "cpu_dispatch_loop_ms":  cpu_mean_loop,
        "cpu_pct_fused":    cpu_mean_fused / mean_fused * 100.0,
        "bw_util_pct":      bw_util_pct,
        "bw_actual_gbs":    bw_actual_gbs,
        "compute_util_pct": compute_util_pct,
        "arith_intensity":  arith_intensity,
        "stage1_gpu_us":    stage1_gpu_us,
        "stage2_gpu_us":    stage2_gpu_us,
        "classification":   classification,
    }


def main():
    print(f"[profile] device: {torch.xpu.get_device_name(0)}")
    print(f"[profile] shape: N_spec={N_SPEC}, B={B}, Hq={HQ}, Hk={HK}, D={D}, seqlen={SEQLEN}")
    print(f"[profile] kernel: triton_turboquant_decode_attention_spec_xpu (fused-N_spec, commit 425fc5c)")
    print(f"[profile] outer loops: {N_TIMED} timed, {N_NOSYNC} no-sync, {WARMUP} warmup")

    prof_dir = os.path.join(REPO, "vt-triton-torchprof")
    os.makedirs(prof_dir, exist_ok=True)

    presets = [
        ("turboquant_k8v4",    True,  False),
        ("turboquant_k3v4_nc", False, True),
    ]

    results = []
    for preset, key_fp8, nc in presets:
        r = profile_preset(preset, key_fp8, nc, prof_dir)
        results.append(r)

    # ------------------------------------------------------------------
    # Cross-preset comparison table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("CROSS-PRESET COMPARISON (fused-N_spec vs un-fused baseline)")
    print(f"{'='*80}")
    print(f"{'Metric':<35s}  {'k8v4 (unfused)':>15s}  {'k8v4 (fused)':>13s}  {'k3v4_nc (fused)':>16s}")
    print(f"{'─'*80}")

    # Reference numbers from TRITON_PROFILE_ANALYSIS.md (k8v4 unfused)
    unfused_wall_ms = 8.922
    unfused_cpu_ms  = 2.100
    unfused_bw_pct  = 3.9
    unfused_comp_pct = 6.2
    unfused_dispatches = 16

    k8 = results[0]
    k3 = results[1]

    rows_table = [
        ("Wall time (mean ms)",
            f"{unfused_wall_ms:.3f}", f"{k8['mean_fused_ms']:.3f}", f"{k3['mean_fused_ms']:.3f}"),
        ("CPU dispatch (ms)",
            f"{unfused_cpu_ms:.3f}", f"{k8['cpu_dispatch_fused_ms']:.3f}", f"{k3['cpu_dispatch_fused_ms']:.3f}"),
        ("CPU dispatch (% wall)",
            f"{unfused_cpu_ms/unfused_wall_ms*100:.1f}%",
            f"{k8['cpu_pct_fused']:.1f}%",
            f"{k3['cpu_pct_fused']:.1f}%"),
        ("LZ dispatches/iter",
            f"{unfused_dispatches}", "2", "2"),
        ("BW utilization",
            f"{unfused_bw_pct:.1f}%", f"{k8['bw_util_pct']:.1f}%", f"{k3['bw_util_pct']:.1f}%"),
        ("Compute utilization (FP32)",
            f"{unfused_comp_pct:.1f}%", f"{k8['compute_util_pct']:.1f}%", f"{k3['compute_util_pct']:.1f}%"),
        ("Speedup vs looped",
            "1.00x", f"{k8['speedup']:.2f}x", f"{k3['speedup']:.2f}x"),
    ]

    for label, unfused_val, k8_val, k3_val in rows_table:
        print(f"{label:<35s}  {unfused_val:>15s}  {k8_val:>13s}  {k3_val:>16s}")

    print(f"\n[profile] run complete. Profile traces in {prof_dir}/")

    # Save structured results
    all_data = {
        "shape": {"N_spec": N_SPEC, "B": B, "Hq": HQ, "Hk": HK, "D": D, "seqlen": SEQLEN},
        "unfused_reference": {
            "wall_ms": unfused_wall_ms,
            "cpu_dispatch_ms": unfused_cpu_ms,
            "bw_util_pct": unfused_bw_pct,
            "compute_util_pct": unfused_comp_pct,
            "lz_dispatches": unfused_dispatches,
        },
        "fused_results": results,
    }
    json_out = os.path.join(REPO, "sycl", "build", "fused_profile.json")
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"[profile] structured data saved to {json_out}")


if __name__ == "__main__":
    main()
