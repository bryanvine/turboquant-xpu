#include "tq_decode_spec_jm.hpp"
#include "jm_layout.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace turboquant_xpu_sycl_jm {
using namespace config;

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void tq_decode_spec_jm_stage2(
    uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
    int N_spec, int B, int Hq, int D) {
  if (D != D_DIM)
    throw std::runtime_error("jm stage2 assumes D == 128");

  auto& q = queue();
  const auto* d_pout = reinterpret_cast<const float*>(partial_out);
  const auto* d_plse = reinterpret_cast<const float*>(partial_lse);
  auto* d_out        = reinterpret_cast<float*>(out);
  const int n_spec = N_spec;
  const int b_total = B;
  const int hq_total = Hq;

  // Grid: one work-item per output row [n, b, hq]. Each emits D floats.
  const sycl::range<1> global_range{std::size_t(N_spec) * B * Hq};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_jm_stage2_reduce>(
      global_range,
      [=](sycl::id<1> id) {
        const int n  = id[0] / (b_total * hq_total);
        const int rem = id[0] % (b_total * hq_total);
        const int b  = rem / hq_total;
        const int hq = rem % hq_total;

        // Find max lse across splits (skip -INF sentinels).
        float m = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < NUM_KV_SPLITS; ++s) {
          float lse_s = d_plse[(((s * n_spec + n) * b_total + b) * hq_total + hq)];
          if (lse_s > m) m = lse_s;
        }
        // If all splits are -INF (no valid tokens — shouldn't happen in phase a), write zero.
        if (m == -std::numeric_limits<float>::infinity()) {
          for (int d = 0; d < D_DIM; ++d)
            d_out[((n * b_total + b) * hq_total + hq) * D_DIM + d] = 0.f;
          return;
        }

        // denom = sum_s exp(lse_s - m)
        float denom = 0.f;
        for (int s = 0; s < NUM_KV_SPLITS; ++s) {
          float lse_s = d_plse[(((s * n_spec + n) * b_total + b) * hq_total + hq)];
          if (lse_s > -std::numeric_limits<float>::infinity())
            denom += sycl::exp(lse_s - m);
        }

        // out[d] = (sum_s exp(lse_s - m) * partial_out[s, n, b, hq, d]) / denom
        for (int d = 0; d < D_DIM; ++d) {
          float num = 0.f;
          for (int s = 0; s < NUM_KV_SPLITS; ++s) {
            float lse_s = d_plse[(((s * n_spec + n) * b_total + b) * hq_total + hq)];
            if (lse_s == -std::numeric_limits<float>::infinity()) continue;
            float w = sycl::exp(lse_s - m);
            num += w * d_pout[((((s * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM) + d];
          }
          d_out[((n * b_total + b) * hq_total + hq) * D_DIM + d] = num / denom;
        }
      });
  }).wait();
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
