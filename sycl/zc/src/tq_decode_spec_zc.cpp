#include "tq_layout.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>

namespace turboquant_xpu_sycl_zc {

using namespace turboquant_xpu_sycl::layout;

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void tq_decode_spec_zc_scalar(
    uintptr_t q_rot, uintptr_t k_idx, uintptr_t k_norm, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero, uintptr_t centroids,
    uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen, int preset_id) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;
  auto& q = queue();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / std::sqrt(float(D));

  // All pointers are USM device pointers — no malloc/memcpy needed.
  const float*   d_q      = reinterpret_cast<const float*>(q_rot);
  const uint8_t* d_kidx   = reinterpret_cast<const uint8_t*>(k_idx);
  const float*   d_knorm  = reinterpret_cast<const float*>(k_norm);
  const float*   d_kfp8   = reinterpret_cast<const float*>(k_fp8);
  const uint8_t* d_vidx   = reinterpret_cast<const uint8_t*>(v_idx);
  const float*   d_vscale = reinterpret_cast<const float*>(v_scale);
  const float*   d_vzero  = reinterpret_cast<const float*>(v_zero);
  const float*   d_cent   = reinterpret_cast<const float*>(centroids);
  float*         d_out    = reinterpret_cast<float*>(out);

  // Grid: one work-item per (n_spec, b, h_q). Intentionally dumb — optimization
  // is the DPAS unit's job (Task 12). This path exists for correctness only.
  const sycl::range<3> g{(std::size_t)N_spec, (std::size_t)B, (std::size_t)Hq};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_decode_spec_zc_scalar_k>(g, [=](sycl::id<3> it) {
      int n = it[0], b = it[1], hq = it[2];
      int h_k = hq / kv_group;

      // q: [N_spec, B, Hq, D]
      const float* q_ptr = d_q + (((n*B + b)*Hq + hq)*D);

      // Online softmax state — fixed-size register arrays, D must be 128 for PoC.
      float m_prev = -INFINITY, l_prev = 0.0f;
      float acc[128];
      for (int d = 0; d < D; ++d) acc[d] = 0.0f;

      for (int kv0 = 0; kv0 < seqlen; kv0 += BLOCK_KV_DEFAULT) {
        float scores[BLOCK_KV_DEFAULT];
        for (int t = 0; t < BLOCK_KV_DEFAULT; ++t) {
          int kv = kv0 + t;
          if (kv >= seqlen) { scores[t] = -INFINITY; continue; }
          float s = 0.0f;
          if (preset_id == PRESET_K8V4) {
            const float* k_ptr = d_kfp8 + (((b*seqlen + kv)*Hk + h_k)*D);
            for (int d = 0; d < D; ++d) s += q_ptr[d] * k_ptr[d];
          } else {
            const uint8_t* kidx = d_kidx + (((b*seqlen + kv)*Hk + h_k)*D);
            float norm = d_knorm[(b*seqlen + kv)*Hk + h_k];
            float term = 0.0f;
            for (int d = 0; d < D; ++d) term += q_ptr[d] * d_cent[kidx[d] & (K3_CENTROIDS - 1)];
            s = term * norm;
          }
          scores[t] = s * attn_scale;
        }

        // Block-level max and renormalize
        float m_local = scores[0];
        for (int t = 1; t < BLOCK_KV_DEFAULT; ++t) m_local = sycl::fmax(m_local, scores[t]);
        float m_new = sycl::fmax(m_local, m_prev);
        float re = sycl::exp(m_prev - m_new);
        for (int d = 0; d < D; ++d) acc[d] *= re;
        l_prev *= re;

        for (int t = 0; t < BLOCK_KV_DEFAULT; ++t) {
          int kv = kv0 + t;
          if (kv >= seqlen) continue;
          float p = sycl::exp(scores[t] - m_new);
          l_prev += p;
          const uint8_t* vi = d_vidx + (((b*seqlen + kv)*Hk + h_k)*D);
          float vs = d_vscale[(b*seqlen + kv)*Hk + h_k];
          float vz = d_vzero[(b*seqlen + kv)*Hk + h_k];
          for (int d = 0; d < D; ++d) acc[d] += p * (float(vi[d]) * vs + vz);
        }
        m_prev = m_new;
      }

      float inv_l = 1.0f / l_prev;
      float* o_ptr = d_out + (((n*B + b)*Hq + hq)*D);
      for (int d = 0; d < D; ++d) o_ptr[d] = acc[d] * inv_l;
    });
  }).wait();

  // Output is already in device memory — no memcpy needed.
}

} // namespace turboquant_xpu_sycl_zc
