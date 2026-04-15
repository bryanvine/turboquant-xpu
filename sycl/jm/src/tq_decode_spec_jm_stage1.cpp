#include "tq_decode_spec_jm.hpp"
#include <stdexcept>

namespace turboquant_xpu_sycl_jm {

void tq_decode_spec_jm_stage1(
    uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
    uintptr_t, uintptr_t,
    int, int, int, int, int, int,
    int, int, int) {
  throw std::runtime_error("tq_decode_spec_jm_stage1: not implemented yet (Task 5+)");
}

} // namespace turboquant_xpu_sycl_jm
