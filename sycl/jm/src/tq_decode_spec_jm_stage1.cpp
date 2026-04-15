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

void tq_decode_spec_jm_stage1(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;
  if (N_spec > M_TILE)
    throw std::runtime_error("jm PoC assumes N_spec <= 8");
  if (D != D_DIM)
    throw std::runtime_error("jm PoC assumes D == 128");
  if (preset_id != PRESET_K8V4)
    throw std::runtime_error("jm phase (a) supports k8v4 only; k3v4_nc is phase (b)");
  if (seqlen % BLK_KV != 0)
    throw std::runtime_error("seqlen must be BLK_KV-aligned");

  auto& q = queue();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / std::sqrt(float(D));
  const int n_spec = N_spec;
  const int is_causal = causal;
  const int c_len = cached_len;
  const int b_total = B;
  const int hq_total = Hq;
  const int hk_total = Hk;
  const int seqlen_v = seqlen;

  const auto* d_q      = reinterpret_cast<const float*>(q_rot);
  const auto* d_kfp8   = reinterpret_cast<const float*>(k_fp8);
  const auto* d_vidx   = reinterpret_cast<const uint8_t*>(v_idx);
  const auto* d_vscale = reinterpret_cast<const float*>(v_scale);
  const auto* d_vzero  = reinterpret_cast<const float*>(v_zero);
  auto* d_pout         = reinterpret_cast<float*>(partial_out);
  auto* d_plse         = reinterpret_cast<float*>(partial_lse);

  // Task 5: no split-KV. One work-item per (b, hq). Writes into partial slot 0.
  // Fill other slots with sentinels so stage 2 can ignore them (lse = -INFINITY).
  q.memset(d_plse, 0, sizeof(float) * NUM_KV_SPLITS * n_spec * B * Hq).wait();
  // Then mark splits 1..N-1 as -INF so stage 2's log-sum-exp ignores them.
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>{std::size_t(NUM_KV_SPLITS - 1) * n_spec * B * Hq},
      [=](sycl::id<1> i) {
        d_plse[n_spec * B * Hq + i[0]] = -std::numeric_limits<float>::infinity();
      });
  }).wait();

  const sycl::range<1> global_range{std::size_t(B) * Hq};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_jm_stage1_scalar>(
      global_range,
      [=](sycl::id<1> id) {
        const int wg_id = id[0];
        const int b  = wg_id / hq_total;
        const int hq = wg_id % hq_total;
        const int h_k = hq / kv_group;

        float m_prev[M_TILE];
        float l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -std::numeric_limits<float>::infinity();
          l_prev[n] = 0.0f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.f;
        }

        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        for (int kv0 = 0; kv0 < seqlen_v; kv0 += BLK_KV) {
          float scores[M_TILE][BLK_KV];
          for (int n = 0; n < n_spec; ++n) {
            const float* q_ptr = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM);
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) {
                scores[n][t] = -std::numeric_limits<float>::infinity();
                continue;
              }
              float s = 0.f;
              const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              for (int d = 0; d < D_DIM; ++d) s += q_ptr[d] * kp[d];
              scores[n][t] = s * attn_scale;
            }
          }

          for (int n = 0; n < n_spec; ++n) {
            // row_has_any (causal mask may eliminate all positions)
            bool any_valid = false;
            float m_local = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              any_valid = true;
              float v = scores[n][t];
              if (v > m_local) m_local = v;
            }
            if (!any_valid) continue;

            float m_p = m_prev[n];
            float m_new = m_local > m_p ? m_local : m_p;
            float re = (m_p == -std::numeric_limits<float>::infinity())
                         ? 0.f : sycl::exp(m_p - m_new);
            for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
            l_prev[n] *= re;

            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              float p = sycl::exp(scores[n][t] - m_new);
              l_prev[n] += p;
              const uint8_t* vp = d_vidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              float vs = d_vscale[(b * seqlen_v + kv) * hk_total + h_k];
              float vz = d_vzero[(b * seqlen_v + kv) * hk_total + h_k];
              for (int d = 0; d < D_DIM; ++d)
                acc[n][d] += p * (float(vp[d]) * vs + vz);
            }
            m_prev[n] = m_new;
          }
        }

        // Write the normalized partial for split 0.
        // Stage 2 expects partial_out[s] = acc_s / l_s (normalized) so that
        // it can combine splits via weighted sum: sum_s exp(lse_s - m) * partial_out[s].
        for (int n = 0; n < n_spec; ++n) {
          // Flat index: partial_out[split=0, n, b, hq, d]
          //   offset = (((0 * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM + d
          //   = ((n * b_total + b) * hq_total + hq) * D_DIM + d
          float* o_ptr = d_pout + (((n * b_total + b) * hq_total + hq) * D_DIM);
          float inv_l = (l_prev[n] > 0.f) ? 1.0f / l_prev[n] : 0.0f;
          for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc[n][d] * inv_l;
          // lse = m + log(l). If row had zero valid tokens (impossible when not causal),
          // leave lse = 0 (memset). For split 0 with causal, row always has at least
          // cached_len+0+1 tokens so m_prev[n] is finite.
          float lse = (l_prev[n] > 0.f)
                        ? m_prev[n] + sycl::log(l_prev[n])
                        : -std::numeric_limits<float>::infinity();
          d_plse[((n * b_total + b) * hq_total + hq)] = lse;
        }
      });
  }).wait();
}

} // namespace turboquant_xpu_sycl_jm
