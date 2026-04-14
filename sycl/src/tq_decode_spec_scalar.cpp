#include "tq_decode_spec.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>

namespace turboquant_xpu_sycl {

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void hello_identity(const float* in, float* out, std::size_t n) {
  auto& q = queue();
  float* d_in = sycl::malloc_device<float>(n, q);
  float* d_out = sycl::malloc_device<float>(n, q);
  q.memcpy(d_in, in, n * sizeof(float)).wait();
  q.parallel_for(n, [=](sycl::id<1> i) { d_out[i] = d_in[i]; }).wait();
  q.memcpy(out, d_out, n * sizeof(float)).wait();
  sycl::free(d_in, q); sycl::free(d_out, q);
}

void hello_scale(const float* in, float* out, std::size_t n, float s) {
  auto& q = queue();
  float* d_in = sycl::malloc_device<float>(n, q);
  float* d_out = sycl::malloc_device<float>(n, q);
  q.memcpy(d_in, in, n * sizeof(float)).wait();
  q.parallel_for(n, [=](sycl::id<1> i) { d_out[i] = d_in[i] * s; }).wait();
  q.memcpy(out, d_out, n * sizeof(float)).wait();
  sycl::free(d_in, q); sycl::free(d_out, q);
}

// Stubs — filled in Task 9 onward. Must compile but need not produce correct output yet.
void tq_decode_spec_scalar(const float*, const uint8_t*, const float*, const float*,
                           const uint8_t*, const float*, const float*, const float*,
                           float*, int, int, int, int, int, int, int) {
  throw std::runtime_error("tq_decode_spec_scalar: not implemented (Task 9)");
}

} // namespace turboquant_xpu_sycl
