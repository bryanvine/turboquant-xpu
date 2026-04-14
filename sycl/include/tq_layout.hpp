#pragma once
#include <cstdint>

namespace turboquant_xpu_sycl::layout {

// PoC uses unpacked layouts — one uint8 per index. The production layout
// bit-packs indices per the turboquant_xpu.quantizer.config module, but this
// PoC defers bit-unpack to later; the correctness harness passes pre-unpacked
// uint8 tensors directly.

constexpr int SG_SIZE = 16;          // Required sub-group size on Xe2
constexpr int BLOCK_KV_DEFAULT = 64; // See SYCL_KERNEL_DESIGN.md — swept in Task 13
constexpr int K3_CENTROIDS = 8;      // 3-bit Lloyd-Max key table: 7 active + 1 pad

enum Preset : int {
  PRESET_K8V4   = 0,
  PRESET_K3V4NC = 1,
};

} // namespace turboquant_xpu_sycl::layout
