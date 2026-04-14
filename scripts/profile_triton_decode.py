#!/usr/bin/env python3
"""Minimal workload for VTune/Proton profiling of Triton TQ decode.

Runs `triton_turboquant_decode_attention_xpu` a fixed number of times at the
PoC shape. Uses torch.profiler (XPU activity) + Triton Proton for GPU-side
kernel timing, and wall-clock timing for CPU-side dispatch overhead.

Shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, preset=k8v4.
"""
import os, sys, math, time, json
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
import torch.profiler as tp

from turboquant_xpu.kernels.xpu_decode import triton_turboquant_decode_attention_xpu
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"


def build_hadamard(d: int) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(DEVICE)


def make_inputs():
    N_spec, B, Hq, Hk, D, seqlen = 8, 4, 32, 4, 128, 8192
    rng = np.random.default_rng(2025)
    q_np = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    q_spec = torch.tensor(q_np, device=DEVICE)

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", D)
    num_blocks = math.ceil(seqlen / 16) * B
    kv_cache = torch.zeros(num_blocks, 16, Hk, cfg.slot_size_aligned,
                           dtype=torch.uint8, device=DEVICE)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    seq_lens = torch.full((B,), seqlen, dtype=torch.int32, device=DEVICE)

    PiT = build_hadamard(D)
    Pi = PiT.T.contiguous()
    cents = torch.zeros(8, dtype=torch.float32, device=DEVICE)
    scale = 1.0 / math.sqrt(D)
    return q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale, cfg, N_spec


def call_kernel(q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale, cfg, N_spec):
    for n in range(N_spec):
        triton_turboquant_decode_attention_xpu(
            query=q_spec[n], kv_cache=kv_cache, block_table=block_table,
            seq_lens=seq_lens, Pi=Pi, centroids=cents, scale=scale,
            mse_bits=cfg.mse_bits, key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.value_quant_bits,
            key_fp8=True, norm_correction=False, PiT=PiT)


def main():
    print(f"[profile] device: {torch.xpu.get_device_name(0)}")

    inputs = make_inputs()
    q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale, cfg, N_spec = inputs

    # --- Warmup: triggers Triton JIT compilation ---
    print("[profile] warming up (JIT compilation)...")
    for _ in range(5):
        call_kernel(*inputs)
    torch.xpu.synchronize()
    print("[profile] warmup done")

    # -----------------------------------------------------------------------
    # SECTION 1: Raw wall-clock timing — measure N_spec loop overhead
    # -----------------------------------------------------------------------
    N_TIMING = 20
    times_ms = []
    for i in range(N_TIMING):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        call_kernel(*inputs)
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = sorted(times_ms)
    mean_ms = sum(times_ms) / len(times_ms)
    med_ms = arr[len(arr) // 2]
    p5_ms = arr[int(len(arr) * 0.05)]
    p95_ms = arr[int(len(arr) * 0.95)]
    per_kernel_ms = mean_ms / N_spec

    print(f"\n[timing] N_spec={N_spec} loop (wall-clock, includes sync):")
    print(f"  mean={mean_ms:.3f} ms  median={med_ms:.3f} ms  p5={p5_ms:.3f} ms  p95={p95_ms:.3f} ms")
    print(f"  => per-kernel estimate: {per_kernel_ms:.3f} ms  ({per_kernel_ms*1000:.1f} us)")

    # -----------------------------------------------------------------------
    # SECTION 2: No-sync CPU dispatch timing — isolate Python/LZ overhead
    # -----------------------------------------------------------------------
    N_NOSYNC = 20
    cpu_times_ms = []
    torch.xpu.synchronize()
    for _ in range(N_NOSYNC):
        t0 = time.perf_counter()
        call_kernel(*inputs)  # no sync — measures CPU dispatch only
        t1 = time.perf_counter()
        cpu_times_ms.append((t1 - t0) * 1000.0)
    torch.xpu.synchronize()

    cpu_mean_ms = sum(cpu_times_ms) / len(cpu_times_ms)
    cpu_per_launch_us = cpu_mean_ms / N_spec * 1000.0
    gpu_compute_ms = mean_ms - cpu_mean_ms  # rough subtraction
    gpu_fraction = gpu_compute_ms / mean_ms if mean_ms > 0 else 0.0

    print(f"\n[dispatch] CPU dispatch overhead (no sync):")
    print(f"  mean per N_spec loop: {cpu_mean_ms:.3f} ms")
    print(f"  mean per kernel launch: {cpu_per_launch_us:.1f} us")
    print(f"  GPU compute (wall - dispatch): {gpu_compute_ms:.3f} ms  ({gpu_fraction*100:.1f}% of wall)")

    # -----------------------------------------------------------------------
    # SECTION 3: torch.profiler with XPU activity
    # -----------------------------------------------------------------------
    print("\n[profile] running torch.profiler (XPU activity, 3 outer iters)...")
    prof_dir = os.path.join(REPO, "vt-triton-torchprof")
    os.makedirs(prof_dir, exist_ok=True)

    with tp.profile(
        activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.XPU],
        record_shapes=False,
        with_stack=False,
        schedule=tp.schedule(wait=0, warmup=1, active=3, repeat=1),
        on_trace_ready=tp.tensorboard_trace_handler(prof_dir),
    ) as prof:
        for step in range(4):  # wait=0, warmup=1, active=3
            call_kernel(*inputs)
            torch.xpu.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=20))

    # Export key_averages as JSON for parsing
    avgs = prof.key_averages()
    rows = []
    for evt in avgs:
        rows.append({
            "name": evt.key,
            "cpu_time_us": evt.cpu_time,
            "xpu_time_us": evt.device_time if hasattr(evt, "device_time") else getattr(evt, "xpu_time", 0),
            "count": evt.count,
            "cpu_total_us": evt.cpu_time_total,
            "xpu_total_us": evt.device_time_total if hasattr(evt, "device_time_total") else getattr(evt, "xpu_time_total", 0),
        })
    json_path = os.path.join(REPO, "vt-triton-torchprof", "key_averages.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[profile] key_averages saved to {json_path}")

    # -----------------------------------------------------------------------
    # SECTION 4: Triton Proton xpupti profiling
    # -----------------------------------------------------------------------
    print("\n[profile] running Triton Proton (xpupti backend)...")
    try:
        import triton.profiler as proton
        proton_path = os.path.join(REPO, "vt-triton-torchprof", "proton_tq")
        proton.start(proton_path, backend="xpupti", hook="triton")
        N_PROTON = 10
        for _ in range(N_PROTON):
            call_kernel(*inputs)
        torch.xpu.synchronize()
        proton.finalize()
        print(f"[profile] Proton result at {proton_path}.hatchet")
    except Exception as e:
        print(f"[profile] Proton failed: {e}")

    # -----------------------------------------------------------------------
    # SECTION 5: Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    print(f"Shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192")
    print(f"Wall (N_spec loop, synced):  {mean_ms:.3f} ms  (={per_kernel_ms*1000:.0f} us/kernel)")
    print(f"CPU dispatch (no sync):       {cpu_mean_ms:.3f} ms  (={cpu_per_launch_us:.0f} us/launch)")
    print(f"GPU compute (wall-dispatch):  {gpu_compute_ms:.3f} ms  ({gpu_fraction*100:.0f}% of total)")

    if cpu_per_launch_us > per_kernel_ms * 1000 * 0.5:
        print("\nCLASSIFICATION: LAUNCH-OVERHEAD-DOMINATED")
        print("=> CPU dispatch is >=50% of per-kernel wall time. Fusing N_spec saves real time.")
    elif gpu_fraction > 0.8:
        print("\nCLASSIFICATION: GPU COMPUTE/BANDWIDTH BOUND")
        print("=> GPU is doing meaningful work per launch. Need EU stall analysis to distinguish.")
    else:
        print("\nCLASSIFICATION: MIXED")

    print(f"\n[profile] done. {N_TIMING * N_spec} total kernel launches timed.")


if __name__ == "__main__":
    main()
