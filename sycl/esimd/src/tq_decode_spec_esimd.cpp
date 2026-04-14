#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "esimd_layout.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace esimd = sycl::ext::intel::esimd;

namespace turboquant_xpu_esimd {
using namespace layout;

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void tq_decode_spec_esimd(
    uintptr_t q_rot, uintptr_t k_idx, uintptr_t k_norm, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero, uintptr_t centroids,
    uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;
  if (N_spec > M_TILE)
    throw std::runtime_error("esimd PoC assumes N_spec <= 8");
  if (D != D_DIM)
    throw std::runtime_error("esimd PoC assumes D == 128");

  auto& q = queue();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / std::sqrt(float(D));
  const int n_spec = N_spec;
  const int pid = preset_id;
  const int is_causal = causal;
  const int c_len = cached_len;
  const int b_total = B;
  const int hq_total = Hq;
  const int hk_total = Hk;
  const int seqlen_v = seqlen;

  const auto* d_q      = reinterpret_cast<const float*>(q_rot);
  const auto* d_kidx   = reinterpret_cast<const uint8_t*>(k_idx);
  const auto* d_knorm  = reinterpret_cast<const float*>(k_norm);
  const auto* d_kfp8   = reinterpret_cast<const float*>(k_fp8);
  const auto* d_vidx   = reinterpret_cast<const uint8_t*>(v_idx);
  const auto* d_vscale = reinterpret_cast<const float*>(v_scale);
  const auto* d_vzero  = reinterpret_cast<const float*>(v_zero);
  const auto* d_cent   = reinterpret_cast<const float*>(centroids);
  auto* d_out          = reinterpret_cast<float*>(out);

  // Grid: one WG per (b, h_q), one sub-group of SG_SIZE lanes.
  const sycl::range<2> global_range{std::size_t(B) * Hq, SG_SIZE};
  const sycl::range<2> local_range{1, SG_SIZE};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_decode_spec_esimd_scalar>(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const int wg_id = it.get_global_id(0);
        const int lane  = it.get_local_id(1);
        const int b  = wg_id / hq_total;
        const int hq = wg_id % hq_total;
        const int h_k = hq / kv_group;

        // Only lane 0 runs the serial logic in this scalar fallback. The 15
        // idle lanes will be unlocked by DPAS in Task 6.
        if (lane != 0) return;

        // Online softmax state, per-query — plain registers.
        float m_prev[M_TILE];
        float l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -INFINITY;
          l_prev[n] = 0.0f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.f;
        }

        // Per-query effective seq_len (for causal mode).
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          {
            int eff = c_len + n + 1;
            if (eff > seqlen_v) eff = seqlen_v;
            eff_end_q[n] = is_causal ? eff : seqlen_v;
          }
        }

        for (int kv0 = 0; kv0 < seqlen_v; kv0 += BLK_KV) {
          float scores[M_TILE][BLK_KV];
          for (int n = 0; n < n_spec; ++n) {
            const float* q_ptr = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM);
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) {
                scores[n][t] = -INFINITY;
                continue;
              }
              float s = 0.f;
              if (pid == PRESET_K8V4) {
                const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
                for (int d = 0; d < D_DIM; ++d) s += q_ptr[d] * kp[d];
              } else {
                const uint8_t* kidx_p =
                    d_kidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
                float norm = d_knorm[(b * seqlen_v + kv) * hk_total + h_k];
                float term = 0.f;
                for (int d = 0; d < D_DIM; ++d)
                  term += q_ptr[d] * d_cent[kidx_p[d] & (K3_CENTROIDS - 1)];
                s = term * norm;
              }
              scores[n][t] = s * attn_scale;
            }
          }

          // Per-query online softmax + P·V update.
          for (int n = 0; n < n_spec; ++n) {
            float m_local = scores[n][0];
            for (int t = 1; t < BLK_KV; ++t)
              if (scores[n][t] > m_local) m_local = scores[n][t];
            float m_new = m_local > m_prev[n] ? m_local : m_prev[n];
            float re = sycl::ext::intel::esimd::exp(esimd::simd<float, 1>(m_prev[n] - m_new))[0];
            for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
            l_prev[n] *= re;

            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              float p = esimd::exp(esimd::simd<float, 1>(scores[n][t] - m_new))[0];
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

        for (int n = 0; n < n_spec; ++n) {
          float inv_l = 1.f / l_prev[n];
          float* o_ptr = d_out + (((n * b_total + b) * hq_total + hq) * D_DIM);
          for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc[n][d] * inv_l;
        }
      });
  }).wait();
}

} // namespace turboquant_xpu_esimd
