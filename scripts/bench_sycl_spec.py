"""Go/no-go micro-benchmark: SYCL scalar + DPAS vs Triton-looped-N.

Synthetic PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192.
Prints a table + decision line, writes docs/SYCL_POC_RESULTS.md.

TOOLCHAIN NOTE: turboquant_xpu_sycl (nightly, libsycl.so.9/libur_loader 0.12)
and torch-XPU (venv, libsycl.so.8/libur_loader 0.11) require different,
incompatible SO versions of libur_loader and cannot load in the same Python
process.  This script therefore runs three sub-benchmarks as child processes
with distinct LD_LIBRARY_PATH settings:

  - SYCL leg (scalar + DPAS):
      LD_LIBRARY_PATH = /tmp/intel-llvm-nightly/lib:...
      Runs _bench_sycl_inner() and prints JSON to stdout.

  - Triton leg:
      LD_LIBRARY_PATH = .venv-sycl/lib:...     (libur_loader 0.11)
      Runs _bench_triton_inner() and prints JSON to stdout.

The orchestrator (main()) spawns both, captures JSON, and writes the
results table + narrative stub to docs/SYCL_POC_RESULTS.md.
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
SYCL_BUILD = REPO / "sycl" / "build"
SRC = REPO / "src"
VENV_PYTHON = REPO / ".venv-sycl" / "bin" / "python"
VENV_LIB = REPO / ".venv-sycl" / "lib"
NIGHTLY_LIB = Path("/tmp/intel-llvm-nightly/lib")
ONEAPI_LIB = Path("/opt/intel/oneapi/compiler/2025.3/lib")

# ---------------------------------------------------------------------------
# Inner helpers (called as __main__ with --_run-sycl or --_run-triton)
# ---------------------------------------------------------------------------

def _bench_sycl_inner():
    """Run SYCL scalar + DPAS timing.  Invoked in a child process."""
    import os
    import sys
    import numpy as np
    sys.path.insert(0, str(SYCL_BUILD))
    sys.path.insert(0, str(REPO))

    import turboquant_xpu_sycl as tq_sycl
    from sycl.reference.tq_decode_reference import pack_cache_for_kernel
    from tests.sycl.conftest import _make_case, SHAPES
    import time

    shape = SHAPES["poc"]
    results = {}

    for preset, preset_id in (("k8v4", tq_sycl.PRESET_K8V4),
                               ("k3v4_nc", tq_sycl.PRESET_K3V4NC)):
        case = _make_case(shape, preset=preset, seed=2025)
        packed = pack_cache_for_kernel(case["cache"])
        q = case["q"]

        for fn_name, key in (("tq_decode_spec_scalar", "scalar"),
                              ("tq_decode_spec_dpas",   "dpas")):
            fn = getattr(tq_sycl, fn_name)

            def call():
                return fn(q,
                          packed["k_idx"], packed["k_norm"], packed["k_fp8"],
                          packed["v_idx"], packed["v_scale"], packed["v_zero"],
                          packed["centroids"], preset_id)

            # warmup
            for _ in range(3):
                call()

            N = 10
            t0 = time.perf_counter()
            for _ in range(N):
                call()
            ms = (time.perf_counter() - t0) / N * 1000.0
            results[f"{preset}_{key}"] = ms

    print(json.dumps(results))


def _bench_triton_inner():
    """Run Triton-looped-N timing (N=N_spec calls).  Invoked in a child process."""
    import math
    import sys
    import time
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(REPO))

    import torch
    import numpy as np
    from sycl.reference.tq_decode_reference import _build_hadamard, _LLOYD_MAX_3BIT
    from tests.sycl.conftest import SHAPES
    from turboquant_xpu.kernels.xpu_decode import triton_turboquant_decode_attention_xpu
    from turboquant_xpu.quantizer.config import TurboQuantConfig

    DEVICE = "xpu"
    shape = SHAPES["poc"]
    N_spec = shape["N_spec"]
    B      = shape["B"]
    Hq     = shape["Hq"]
    Hk     = shape["Hk"]
    D      = shape["D"]
    seqlen = shape["seqlen"]

    Pi_np = _build_hadamard(D)
    Pi  = torch.tensor(Pi_np, dtype=torch.float32, device=DEVICE)
    PiT = Pi.T.contiguous()
    centroids_np = _LLOYD_MAX_3BIT
    centroids = torch.tensor(centroids_np, dtype=torch.float32, device=DEVICE)
    scale = 1.0 / math.sqrt(D)
    seq_lens = torch.full((B,), seqlen, dtype=torch.int32, device=DEVICE)

    results = {}

    for preset in ("k8v4", "k3v4_nc"):
        cfg = TurboQuantConfig.from_cache_dtype(
            "turboquant_k8v4" if preset == "k8v4" else "turboquant_k3v4_nc",
            D,
        )
        num_blocks = (seqlen // 16) * B
        kv_cache = torch.zeros(
            num_blocks, 16, Hk, cfg.slot_size_aligned,
            dtype=torch.uint8, device=DEVICE,
        )
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
        # One query per speculative position, shape [B, Hq, D]
        q_single = torch.randn(B, Hq, D, dtype=torch.float32, device=DEVICE)

        def one_triton_call():
            return triton_turboquant_decode_attention_xpu(
                query=q_single,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=scale,
                mse_bits=cfg.mse_bits,
                key_packed_size=cfg.key_packed_size,
                value_quant_bits=cfg.value_quant_bits,
                key_fp8=cfg.key_fp8,
                norm_correction=cfg.norm_correction,
                PiT=PiT,
                max_num_kv_splits=32,
            )

        # warmup — triggers JIT compilation
        for _ in range(3):
            one_triton_call()
            torch.xpu.synchronize()

        # Timed: loop over N_spec queries (what DPAS replaces)
        N = 10
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            for _ in range(N_spec):
                one_triton_call()
        torch.xpu.synchronize()
        total_ms = (time.perf_counter() - t0) / N * 1000.0
        results[f"{preset}_triton"] = total_ms

    print(json.dumps(results))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _run_child(extra_env: dict, arg: str, timeout: int = 300) -> dict:
    """Spawn a child Python process with this script and a special flag."""
    env = os.environ.copy()
    env.update(extra_env)
    cmd = [str(VENV_PYTHON), __file__, arg]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=str(REPO), timeout=timeout,
    )
    if result.returncode != 0:
        print(f"[CHILD {arg}] stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(
            f"Child process {arg} failed (rc={result.returncode}).\n"
            f"stderr tail: {result.stderr[-2000:]}"
        )
    # Last non-empty line should be JSON
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(f"Child {arg} produced no output.\nstdout: {result.stdout[:2000]}")
    return json.loads(lines[-1])


def main():
    # LD_LIBRARY_PATH for each leg
    existing_ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    nightly_ldpath = f"{NIGHTLY_LIB}:{ONEAPI_LIB}:{existing_ldpath}"
    venv_ldpath    = f"{VENV_LIB}:{ONEAPI_LIB}:{existing_ldpath}"

    nightly_path = f"/tmp/intel-llvm-nightly/bin:{os.environ.get('PATH', '')}"

    print("Running SYCL leg (scalar + DPAS) …")
    sycl_env = {
        "PATH": nightly_path,
        "LD_LIBRARY_PATH": nightly_ldpath,
        "PYTHONPATH": f"{SYCL_BUILD}:{REPO}:{os.environ.get('PYTHONPATH', '')}",
    }
    sycl_data = _run_child(sycl_env, "--_run-sycl")
    print(f"  SYCL data: {sycl_data}")

    print("Running Triton leg …")
    triton_env = {
        "LD_LIBRARY_PATH": venv_ldpath,
        "PYTHONPATH": f"{SRC}:{REPO}:{os.environ.get('PYTHONPATH', '')}",
    }
    triton_data = _run_child(triton_env, "--_run-triton", timeout=600)
    print(f"  Triton data: {triton_data}")

    # Assemble rows
    rows = []
    for preset in ("k8v4", "k3v4_nc"):
        t_triton = triton_data[f"{preset}_triton"]
        t_scalar = sycl_data[f"{preset}_scalar"]
        t_dpas   = sycl_data[f"{preset}_dpas"]
        dpas_sp  = t_triton / t_dpas
        scal_sp  = t_triton / t_scalar
        rows.append((preset, t_triton, t_scalar, t_dpas, dpas_sp, scal_sp))

    print()
    print(f"{'preset':10} {'triton×N (ms)':>14} {'scalar (ms)':>12} {'dpas (ms)':>10} {'dpas_sp':>10} {'scalar_sp':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r[0]:10} {r[1]:14.3f} {r[2]:12.3f} {r[3]:10.3f} {r[4]:10.2f}× {r[5]:10.2f}×")

    go = any(r[4] >= 2.0 for r in rows)
    print()
    print("DECISION:", "GO" if go else "NO-GO")

    # Write results doc
    out = REPO / "docs" / "SYCL_POC_RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        f.write("# SYCL TurboQuant Spec-Decode PoC Results\n\n")
        f.write(f"_Generated: {time.strftime('%Y-%m-%d %H:%M')}_\n\n")
        f.write("## Benchmark Table\n\n")
        f.write("| preset | triton×N (ms) | SYCL scalar (ms) | SYCL DPAS (ms) | DPAS speedup | scalar speedup |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.2f}× | {r[5]:.2f}× |\n")
        f.write(f"\n**Decision:** {'GO' if go else 'NO-GO'}\n\n")
        f.write("[Narrative to be filled in after benchmark runs.]\n")

    print(f"\nResults written to {out}")


# ---------------------------------------------------------------------------
# Entry point dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_run-sycl":
        _bench_sycl_inner()
    elif len(sys.argv) > 1 and sys.argv[1] == "--_run-triton":
        _bench_triton_inner()
    else:
        main()
