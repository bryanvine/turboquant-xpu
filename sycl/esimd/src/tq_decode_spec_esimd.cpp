#include <cstdint>
#include <stdexcept>

namespace turboquant_xpu_esimd {

void tq_decode_spec_esimd(
    uintptr_t q_rot, uintptr_t k_idx, uintptr_t k_norm, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero, uintptr_t centroids,
    uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  (void)q_rot; (void)k_idx; (void)k_norm; (void)k_fp8;
  (void)v_idx; (void)v_scale; (void)v_zero; (void)centroids; (void)out;
  (void)N_spec; (void)B; (void)Hq; (void)Hk; (void)D; (void)seqlen;
  (void)preset_id; (void)causal; (void)cached_len;
  throw std::runtime_error("tq_decode_spec_esimd: not implemented yet (Task 5+)");
}

} // namespace turboquant_xpu_esimd
