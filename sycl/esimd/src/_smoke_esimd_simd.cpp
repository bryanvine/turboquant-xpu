// SPDX-License-Identifier: Apache-2.0
//
// ESIMD smoke: 16-wide simd<float> add on the Arc Pro B70.
// Proves that ESIMD compiles + runs before we touch xmx::dpas in Task 2.
//
// Build ad-hoc (no CMake):
//   icpx -fsycl -fsycl-device-code-split=per_kernel \
//        sycl/esimd/src/_smoke_esimd_simd.cpp -o _smoke_esimd_simd
//   sg render -c './_smoke_esimd_simd'
//
// Kept as a reference under sycl/esimd/src/ — not built by the main CMakeLists.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>

namespace esimd = sycl::ext::intel::esimd;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int N = 16;
  float* in1 = sycl::malloc_shared<float>(N, q);
  float* in2 = sycl::malloc_shared<float>(N, q);
  float* out = sycl::malloc_shared<float>(N, q);
  for (int i = 0; i < N; ++i) { in1[i] = i;  in2[i] = i * 10.f; }

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class esimd_simd_smoke>(
      sycl::nd_range<1>{N, N},
      [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
        esimd::simd<float, N> a;
        a.copy_from(in1);
        esimd::simd<float, N> b;
        b.copy_from(in2);
        esimd::simd<float, N> c = a + b;
        c.copy_to(out);
      });
  }).wait();

  std::cout << "out[5] = " << out[5] << " (expected 55)\n";
  sycl::free(in1, q); sycl::free(in2, q); sycl::free(out, q);
  return 0;
}
