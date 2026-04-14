#pragma once
#include <cstddef>
#include <cstdint>

namespace turboquant_xpu_sycl {

// Hello-world shims — Task 7
void hello_identity(const float* in, float* out, std::size_t n);
void hello_scale(const float* in, float* out, std::size_t n, float s);

// Real kernels — filled in Tasks 9+
void tq_decode_spec_scalar(
    const float* q_rot,            // [N_spec, B, Hq, D] fp32
    const uint8_t* k_idx_packed,   // [B, seqlen, Hk, D] uint8 (for this PoC: unpacked indices)
    const float* k_norm,           // [B, seqlen, Hk]
    const float* k_fp8_as_fp32,    // [B, seqlen, Hk, D] (k8v4 path; nullptr for k3v4_nc)
    const uint8_t* v_idx_packed,   // [B, seqlen, Hk, D] uint8 (unpacked for PoC)
    const float* v_scale,          // [B, seqlen, Hk]
    const float* v_zero,           // [B, seqlen, Hk]
    const float* centroids,        // [8] fp32 (k3v4_nc only)
    float* out,                    // [N_spec, B, Hq, D] fp32
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id                  // 0 = k8v4, 1 = k3v4_nc
);

void tq_decode_spec_dpas(
    const float* q_rot, const uint8_t* k_idx_packed, const float* k_norm,
    const float* k_fp8_as_fp32, const uint8_t* v_idx_packed,
    const float* v_scale, const float* v_zero, const float* centroids,
    float* out, int N_spec, int B, int Hq, int Hk, int D, int seqlen, int preset_id
);

} // namespace turboquant_xpu_sycl
