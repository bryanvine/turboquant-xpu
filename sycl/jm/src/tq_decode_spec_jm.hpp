#pragma once
#include <cstdint>

namespace turboquant_xpu_sycl_jm {

// Stage 1: per-split partial attention + log-sum-exp. Uses joint_matrix DPAS.
// Writes partial_out[N_SPLITS, N_spec, B, Hq, D] fp32 and
// partial_lse[N_SPLITS, N_spec, B, Hq] fp32 to caller-allocated XPU USM.
void tq_decode_spec_jm_stage1(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len);

// Stage 2: log-sum-exp reduce over NUM_KV_SPLITS partials.
// Writes final out[N_spec, B, Hq, D] fp32.
void tq_decode_spec_jm_stage2(
    uintptr_t partial_out, uintptr_t partial_lse,
    uintptr_t out,
    int N_spec, int B, int Hq, int D);

// Convenience wrapper: launches stage 1 then stage 2.
void tq_decode_spec_jm(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len);

} // namespace turboquant_xpu_sycl_jm
