#include "tq_decode_spec.hpp"
#include "tq_layout.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstring>
#include <stdexcept>

namespace turboquant_xpu_sycl {

// Bring in only the names we need to avoid collision with tq_layout.hpp's
// turboquant_xpu_sycl::layout namespace.
namespace jm = sycl::ext::oneapi::experimental::matrix;

static sycl::queue& queue_dpas() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

// Smoke: C[8][16] = A[8][16] * B[16][16], fp16 in, fp32 out. Exercises the
// exact joint_matrix tile shape we plan to use for Q.Kᵀ with M = N_spec.
void joint_matrix_smoke(const sycl::half* A, const sycl::half* B, float* C) {
  auto& q = queue_dpas();
  auto* d_A = sycl::malloc_device<sycl::half>(8*16, q);
  auto* d_B = sycl::malloc_device<sycl::half>(16*16, q);
  auto* d_C = sycl::malloc_device<float>(8*16, q);
  q.memcpy(d_A, A, 8*16*sizeof(sycl::half)).wait();
  q.memcpy(d_B, B, 16*16*sizeof(sycl::half)).wait();

  q.submit([&](sycl::handler& h) {
    h.parallel_for(
      sycl::nd_range<2>({8, 16}, {8, 16}),
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> a;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> b;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c;
        jm::joint_matrix_fill(sg, c, 0.0f);
        jm::joint_matrix_load(sg, a,
          sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(d_A), 16);
        jm::joint_matrix_load(sg, b,
          sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(d_B), 16);
        jm::joint_matrix_mad(sg, c, a, b, c);
        jm::joint_matrix_store(sg, c,
          sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(d_C), 16, jm::layout::row_major);
      });
  }).wait();

  q.memcpy(C, d_C, 8*16*sizeof(float)).wait();
  sycl::free(d_A, q); sycl::free(d_B, q); sycl::free(d_C, q);
}

// Stub for the real kernel — body in Task 12.
void tq_decode_spec_dpas(const float*, const uint8_t*, const float*, const float*,
                         const uint8_t*, const float*, const float*, const float*,
                         float*, int, int, int, int, int, int, int) {
  throw std::runtime_error("tq_decode_spec_dpas: not implemented (Task 12)");
}

} // namespace turboquant_xpu_sycl
