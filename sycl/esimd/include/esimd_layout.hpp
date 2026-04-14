#pragma once
#include <cstdint>

namespace turboquant_xpu_esimd::layout {

// SIMD / DPAS geometry for BMG-G31 (Arc Pro B70).
constexpr int SG_SIZE       = 16;   // Xe2 sub-group = SIMD16
constexpr int M_TILE        = 8;    // DPAS M dim = N_spec for spec decode
constexpr int N_TILE        = 16;   // DPAS N dim
constexpr int K_TILE        = 16;   // DPAS K dim
constexpr int BLK_KV        = 16;   // == N_TILE (one DPAS tile per KV step)
constexpr int D_DIM         = 128;  // PoC head dim
constexpr int K3_CENTROIDS  = 8;    // 3-bit Lloyd-Max table (7 active + 1 pad)

// Preset IDs — match turboquant_xpu_sycl_zc / turboquant_xpu_sycl convention.
enum Preset : int {
  PRESET_K8V4   = 0,
  PRESET_K3V4NC = 1,
};

} // namespace turboquant_xpu_esimd::layout
