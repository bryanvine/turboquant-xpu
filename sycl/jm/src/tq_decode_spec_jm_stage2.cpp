#include "tq_decode_spec_jm.hpp"
#include <stdexcept>

namespace turboquant_xpu_sycl_jm {

void tq_decode_spec_jm_stage2(
    uintptr_t, uintptr_t, uintptr_t,
    int, int, int, int) {
  throw std::runtime_error("tq_decode_spec_jm_stage2: not implemented yet (Task 6+)");
}

void tq_decode_spec_jm(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  tq_decode_spec_jm_stage1(q_rot, k_fp8, v_idx, v_scale, v_zero,
                           partial_out, partial_lse,
                           N_spec, B, Hq, Hk, D, seqlen,
                           preset_id, causal, cached_len);
  tq_decode_spec_jm_stage2(partial_out, partial_lse, out,
                           N_spec, B, Hq, D);
}

} // namespace turboquant_xpu_sycl_jm
