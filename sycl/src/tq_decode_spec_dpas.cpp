#include "tq_decode_spec.hpp"
#include <stdexcept>

namespace turboquant_xpu_sycl {

void tq_decode_spec_dpas(const float*, const uint8_t*, const float*, const float*,
                         const uint8_t*, const float*, const float*, const float*,
                         float*, int, int, int, int, int, int, int) {
  throw std::runtime_error("tq_decode_spec_dpas: not implemented (Task 11)");
}

} // namespace turboquant_xpu_sycl
