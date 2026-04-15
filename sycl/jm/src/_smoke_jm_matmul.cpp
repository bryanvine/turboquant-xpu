// SPDX-License-Identifier: Apache-2.0
//
// joint_matrix smoke: C[8][16] = A[8][16] * B[16][16] (fp16 in, fp32 out).
// Proves nightly clang++ 23 + joint_matrix works on BMG-G31 (Arc Pro B70).
//
// Build ad-hoc (no CMake):
//   sg render -c '
//     export PATH=/tmp/intel-llvm-nightly/bin:$PATH
//     export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
//     clang++ -fsycl -O2 sycl/jm/src/_smoke_jm_matmul.cpp -o _smoke_jm_matmul
//     ./_smoke_jm_matmul
//   '
//
// Kept as a reference under sycl/jm/src/ — not built by the main CMakeLists.
//
// API note (nightly 2026-04-13 header, matrix-unified.hpp):
//   - accumulator fragment MUST declare layout::dynamic (not omit layout).
//   - joint_matrix_mad arg order: (sg, D, A, B, C)  — D = A*B + C.
//   - joint_matrix_store takes runtime layout arg on the accumulator.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <cmath>

namespace jm = sycl::ext::oneapi::experimental::matrix;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int M = 8, N = 16, K = 16;
  sycl::half* A = sycl::malloc_shared<sycl::half>(M * K, q);
  sycl::half* B = sycl::malloc_shared<sycl::half>(K * N, q);
  float*      C = sycl::malloc_shared<float>(M * N, q);
  for (int i = 0; i < M * K; ++i) A[i] = sycl::half(float(i % 5) - 2);
  for (int i = 0; i < K * N; ++i) B[i] = sycl::half(float(i % 7) - 3);
  for (int i = 0; i < M * N; ++i) C[i] = 0.f;

  q.submit([&](sycl::handler& h) {
    h.parallel_for(
      sycl::nd_range<1>{16, 16},
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        // A and B use compile-time layout; accumulator uses layout::dynamic
        // per the nightly header's joint_matrix_mad / joint_matrix_store signatures.
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, M, K, jm::layout::row_major> ma;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, K, N, jm::layout::row_major> mb;
        jm::joint_matrix<sycl::sub_group, float,      jm::use::accumulator, M, N, jm::layout::dynamic> mc;
        jm::joint_matrix_fill(sg, mc, 0.f);
        jm::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                               sycl::access::decorated::no>(A), K);
        jm::joint_matrix_load(sg, mb, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                               sycl::access::decorated::no>(B), N);
        jm::joint_matrix_mad(sg, mc, ma, mb, mc);
        jm::joint_matrix_store(sg, mc,
                               sycl::address_space_cast<sycl::access::address_space::global_space,
                                                        sycl::access::decorated::no>(C),
                               N, jm::layout::row_major);
      });
  }).wait();

  // CPU reference
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

  sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
  return (max_err < 0.1f) ? 0 : 1;
}
