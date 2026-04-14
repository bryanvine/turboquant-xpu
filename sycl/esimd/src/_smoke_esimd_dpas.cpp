// SPDX-License-Identifier: Apache-2.0
//
// ESIMD DPAS smoke: C[8][16] = A[8][16] * B[16][16] (fp16 in, fp32 out).
// Proves xmx::dpas works on BMG-G31 via stock 2025.3 icpx (no nightly).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int M = 8, N = 16, K = 16;
  // Host-visible USM for easy verification.
  sycl::half* A      = sycl::malloc_shared<sycl::half>(M * K, q);
  sycl::half* B      = sycl::malloc_shared<sycl::half>(K * N, q);  // logical row-major
  sycl::half* B_vnni = sycl::malloc_shared<sycl::half>(K * N, q);  // VNNI-packed for DPAS
  float*      C      = sycl::malloc_shared<float>(M * N, q);
  for (int i = 0; i < M * K; ++i) A[i] = sycl::half(float(i % 5) - 2);   // small deterministic
  for (int i = 0; i < K * N; ++i) B[i] = sycl::half(float(i % 7) - 3);
  for (int i = 0; i < M * N; ++i) C[i] = 0.f;

  // VNNI pack B[K=16][N=16] (fp16) → B_vnni[K/2=8][N=16][2]:
  //   B_vnni[kp*N*2 + n*2 + i] = B[(2*kp+i)*N + n]
  // DPAS requires B in VNNI layout for fp16 × fp16 → fp32.
  for (int kp = 0; kp < K / 2; ++kp)
    for (int n = 0; n < N; ++n)
      for (int i = 0; i < 2; ++i)
        B_vnni[kp * N * 2 + n * 2 + i] = B[(2 * kp + i) * N + n];

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class esimd_dpas_smoke>(
      sycl::nd_range<1>{16, 16},   // 1 sub-group of 16 lanes
      [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
        // Load A into a simd<half, M*K>, row-major.
        esimd::simd<sycl::half, M * K> a_reg;
        a_reg.copy_from(A);

        // Load B (VNNI-packed) into a simd<half, K*N>.
        esimd::simd<sycl::half, K * N> b_reg;
        b_reg.copy_from(B_vnni);

        // Accumulator initialized to zero.
        esimd::simd<float, M * N> c_reg(0.f);

        // One DPAS call, systolic_depth=8, repeat_count=8 (matches M=8).
        c_reg = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(
            c_reg, b_reg, a_reg);

        c_reg.copy_to(C);
      });
  }).wait();

  // Reference CPU GEMM for check.
  float ref[M * N] = {0};
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k)
        ref[m * N + n] += float(A[m * K + k]) * float(B[k * N + n]);

  float max_err = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float e = std::abs(C[i] - ref[i]);
    if (e > max_err) max_err = e;
  }
  std::cout << "max_err = " << max_err << " (expected < 0.1)\n";
  std::cout << "C[0,0] = " << C[0] << ", ref[0,0] = " << ref[0] << "\n";

  sycl::free(A, q); sycl::free(B, q); sycl::free(B_vnni, q); sycl::free(C, q);
  return (max_err < 0.1f) ? 0 : 1;
}
