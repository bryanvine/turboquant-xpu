#pragma once
#include <cstdint>

namespace turboquant_xpu_sycl_jm::config {

// SIMD / DPAS geometry for BMG-G31 (Arc Pro B70).
constexpr int SG_SIZE        = 16;
constexpr int M_TILE         = 8;
constexpr int N_TILE         = 16;
constexpr int K_TILE         = 16;
constexpr int BLK_KV         = 16;   // == N_TILE
constexpr int D_DIM          = 128;
constexpr int NUM_KV_SPLITS  = 8;
constexpr int N_D_SLICES     = D_DIM / N_TILE;  // 128 / 16 = 8

// Preset IDs — match turboquant_xpu_sycl_zc / turboquant_xpu_esimd convention.
enum Preset : int {
  PRESET_K8V4   = 0,
  PRESET_K3V4NC = 1,  // not implemented in phase (a); accepted only to keep
                     //   the signature stable across modules.
};

} // namespace turboquant_xpu_sycl_jm::config
