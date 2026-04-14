"""NumPy TurboQuant decode reference.

Readable port of `src/turboquant_xpu/kernels/triton_decode.py::_tq_decode_stage1`
that serves as the correctness ground truth for the SYCL kernel. Two code
paths matter here: k8v4 (FP8 keys + 4-bit values) and k3v4_nc (3-bit MSE
keys + 4-bit values + norm correction). Supports a synthetic 'identity_fp32'
preset for smoke-testing — there the "quantized" cache is just fp32 passthrough.

Produces output with shape [B, Hq, D] for single-query decode and
[N_spec, B, Hq, D] for spec-batched decode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Lloyd-Max centroids for 3-bit MSE quantization of standard-normal keys.
# Values imported from upstream TurboQuant (Lloyd-Max optimal for N(0,1)).
# Kept short and in-file so the reference is self-contained.
# ---------------------------------------------------------------------------
_LLOYD_MAX_3BIT = np.array(
    [-1.7479, -1.0500, -0.5005, 0.0000, 0.5005, 1.0500, 1.7479, 0.0000],
    dtype=np.float32,
)
# Index 7 is the "unused" bucket for 7-centroid 3-bit coding; reference uses 8.


def _build_hadamard(d: int) -> np.ndarray:
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(d)


@dataclass
class TQCache:
    """Plain container for reference TQ cache tensors.

    For each preset we store exactly what the kernel needs to reconstruct
    the fp32 K and V. Layout is "unpacked" for readability — no bit packing
    happens at this layer. The SYCL kernel will use the real packed layout.
    """
    preset: str
    k_idx: np.ndarray | None       # [B, seqlen, Hk, D] int, centroid index (k3v4_nc)
    k_norm: np.ndarray | None      # [B, seqlen, Hk]    fp32, per-vector norm after WHT
    k_fp8: np.ndarray | None       # [B, seqlen, Hk, D] fp32 (dequantized fp8) for k8v4
    v_idx: np.ndarray              # [B, seqlen, Hk, D] int4 values
    v_scale: np.ndarray            # [B, seqlen, Hk]    fp32
    v_zero: np.ndarray             # [B, seqlen, Hk]    fp32
    PiT: np.ndarray                # [D, D] fp32, WHT rotation
    centroids: np.ndarray          # [8] fp32, Lloyd-Max centroids (k3v4_nc only)
    v_fp32: np.ndarray | None = None  # [B, seqlen, Hk, D] fp32 passthrough (identity_fp32 only)


def make_synthetic_tq_cache(k, v, *, preset: str, D: int, Hk: int) -> TQCache:
    """Build a readable (not bit-packed) TQ cache from full-precision K, V.

    preset:
      "identity_fp32" — no quant at all, used as a correctness anchor
      "k8v4"          — FP8 E4M3 keys, 4-bit uniform values
      "k3v4_nc"       — 3-bit Lloyd-Max keys + norm correction, 4-bit values
    """
    B, seqlen = k.shape[0], k.shape[1]
    PiT = _build_hadamard(D).astype(np.float32)

    # V uniform 4-bit quant: v_q = round((v - zero) / scale); v_d = v_q*scale + zero
    v_min = v.min(axis=-1, keepdims=True)
    v_max = v.max(axis=-1, keepdims=True)
    v_range = np.maximum(v_max - v_min, 1e-6)
    v_scale = (v_range / 15.0).astype(np.float32)
    v_zero = v_min.astype(np.float32)
    v_idx = np.clip(np.round((v - v_zero) / v_scale), 0, 15).astype(np.int32)
    # Reduce per-vector scalars to [B, seqlen, Hk]
    v_scale_pv = v_scale.squeeze(-1)
    v_zero_pv = v_zero.squeeze(-1)

    if preset == "identity_fp32":
        # No quantization at all: store raw fp32 K and V for exact round-trip.
        return TQCache(
            preset=preset, k_idx=None, k_norm=None, k_fp8=k.astype(np.float32),
            v_idx=v_idx, v_scale=v_scale_pv, v_zero=v_zero_pv, PiT=PiT,
            centroids=_LLOYD_MAX_3BIT, v_fp32=v.astype(np.float32),
        )

    if preset == "k8v4":
        # Rough FP8 E4M3 simulation: clip to ±448, quantize in log-space to 4-bit exp + 3-bit mantissa.
        # For the reference we use torch if available; otherwise a simple round-to-nearest in the
        # 256-value FP8 E4M3 grid.
        try:
            import torch
            k_t = torch.from_numpy(k)
            k_fp8 = k_t.to(torch.float8_e4m3fn).to(torch.float32).numpy()
        except Exception:
            # Fallback: round to the representable FP8 E4M3 grid manually (sparse but correct enough)
            sign = np.sign(k)
            mag = np.clip(np.abs(k), 1e-8, 448.0)
            exp = np.floor(np.log2(mag))
            mant = mag / (2.0 ** exp)
            mant_q = np.round(mant * 8) / 8  # 3-bit mantissa
            k_fp8 = (sign * mant_q * (2.0 ** exp)).astype(np.float32)
        return TQCache(
            preset=preset, k_idx=None, k_norm=None, k_fp8=k_fp8,
            v_idx=v_idx, v_scale=v_scale_pv, v_zero=v_zero_pv, PiT=PiT,
            centroids=_LLOYD_MAX_3BIT,
        )

    if preset == "k3v4_nc":
        # WHT-rotate K, then per-element assign to nearest of 8 centroids; store per-vector norm.
        k_rot = k @ PiT
        k_norm = np.linalg.norm(k_rot, axis=-1, keepdims=True)
        k_hat = k_rot / np.maximum(k_norm, 1e-6)
        # Nearest centroid
        cents = _LLOYD_MAX_3BIT.reshape(1, 1, 1, 1, -1)
        dist = (k_hat[..., None] - cents) ** 2
        k_idx = dist.argmin(axis=-1).astype(np.int32)
        return TQCache(
            preset=preset, k_idx=k_idx, k_norm=k_norm.squeeze(-1).astype(np.float32),
            k_fp8=None, v_idx=v_idx, v_scale=v_scale_pv, v_zero=v_zero_pv,
            PiT=PiT, centroids=_LLOYD_MAX_3BIT,
        )
    raise ValueError(f"unknown preset {preset!r}")


def _dequant_k(cache: TQCache, b: int, h_k: int) -> np.ndarray:
    if cache.preset == "identity_fp32":
        return cache.k_fp8[b, :, h_k]
    if cache.preset == "k8v4":
        return cache.k_fp8[b, :, h_k]
    if cache.preset == "k3v4_nc":
        cents = cache.centroids[cache.k_idx[b, :, h_k]]        # [seqlen, D]
        return cents * cache.k_norm[b, :, h_k, None]           # [seqlen, D]
    raise ValueError(cache.preset)


def _dequant_v(cache: TQCache, b: int, h_k: int) -> np.ndarray:
    if cache.v_fp32 is not None:
        return cache.v_fp32[b, :, h_k]                         # exact passthrough
    # v = v_idx * v_scale + v_zero
    vi = cache.v_idx[b, :, h_k].astype(np.float32)             # [seqlen, D]
    return vi * cache.v_scale[b, :, h_k, None] + cache.v_zero[b, :, h_k, None]


def ref_decode_single_query(q: np.ndarray, cache: TQCache, *, preset: str) -> np.ndarray:
    """Reference single-query decode. q: [B, Hq, D] fp32, returns [B, Hq, D] fp32.

    For preset "k3v4_nc" q is expected pre-rotated (q_rot = q @ PiT). For
    other presets q is in the original space. The SYCL kernel harness must
    match this expectation.
    """
    assert preset == cache.preset
    B, Hq, D = q.shape
    Hk = cache.v_scale.shape[-1]
    kv_group = Hq // Hk
    out = np.zeros_like(q)
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        for h in range(Hq):
            h_k = h // kv_group
            k = _dequant_k(cache, b, h_k)                      # [seqlen, D]
            v = _dequant_v(cache, b, h_k)                      # [seqlen, D]
            scores = (q[b, h] @ k.T) * scale                   # [seqlen]
            m = scores.max()
            p = np.exp(scores - m); p /= p.sum()
            out[b, h] = p @ v
    return out


def ref_decode_spec_batch(q_spec: np.ndarray, cache: TQCache, *, preset: str) -> np.ndarray:
    """Reference spec-batch decode. q_spec: [N_spec, B, Hq, D] fp32.

    All N_spec queries attend to the SAME KV cache — this is exactly the
    verification step of speculative decoding. Returns [N_spec, B, Hq, D].
    """
    N_spec = q_spec.shape[0]
    outs = [ref_decode_single_query(q_spec[n], cache, preset=preset) for n in range(N_spec)]
    return np.stack(outs, axis=0)


def pack_cache_for_kernel(cache: TQCache):
    """Flatten a reference TQCache into the raw arrays the SYCL kernel expects.

    PoC layout is unpacked for clarity — 1 uint8 per centroid/value index. The
    production SYCL kernel will take bit-packed bytes; we bridge that later.
    """
    k_idx = cache.k_idx.astype(np.uint8) if cache.k_idx is not None else np.zeros(1, dtype=np.uint8)
    k_norm = cache.k_norm.astype(np.float32) if cache.k_norm is not None else np.zeros(1, dtype=np.float32)
    k_fp8 = cache.k_fp8.astype(np.float32) if cache.k_fp8 is not None else np.zeros(1, dtype=np.float32)
    v_idx = cache.v_idx.astype(np.uint8)
    v_scale = cache.v_scale.astype(np.float32)
    v_zero = cache.v_zero.astype(np.float32)
    centroids = cache.centroids.astype(np.float32)
    return dict(k_idx=k_idx, k_norm=k_norm, k_fp8=k_fp8, v_idx=v_idx,
                v_scale=v_scale, v_zero=v_zero, centroids=centroids)
