#include "tq_decode_spec.hpp"
#include "tq_layout.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <cmath>

namespace turboquant_xpu_sycl {

using namespace layout;

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void hello_identity(const float* in, float* out, std::size_t n) {
  if (n == 0) return;
  auto& q = queue();
  float* d_in = sycl::malloc_device<float>(n, q);
  float* d_out = sycl::malloc_device<float>(n, q);
  q.memcpy(d_in, in, n * sizeof(float)).wait();
  q.parallel_for(n, [=](sycl::id<1> i) { d_out[i] = d_in[i]; }).wait();
  q.memcpy(out, d_out, n * sizeof(float)).wait();
  sycl::free(d_in, q); sycl::free(d_out, q);
}

void hello_scale(const float* in, float* out, std::size_t n, float s) {
  if (n == 0) return;
  auto& q = queue();
  float* d_in = sycl::malloc_device<float>(n, q);
  float* d_out = sycl::malloc_device<float>(n, q);
  q.memcpy(d_in, in, n * sizeof(float)).wait();
  q.parallel_for(n, [=](sycl::id<1> i) { d_out[i] = d_in[i] * s; }).wait();
  q.memcpy(out, d_out, n * sizeof(float)).wait();
  sycl::free(d_in, q); sycl::free(d_out, q);
}

void tq_decode_spec_scalar(
    const float* q_rot_h, const uint8_t* k_idx_h, const float* k_norm_h,
    const float* k_fp8_h, const uint8_t* v_idx_h, const float* v_scale_h,
    const float* v_zero_h, const float* centroids_h, float* out_h,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen, int preset_id) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;
  auto& q = queue();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / std::sqrt(float(D));

  // Byte sizes for each buffer
  const std::size_t K_IDX_BYTES = std::size_t(B)*seqlen*Hk*D;
  const std::size_t V_IDX_BYTES = std::size_t(B)*seqlen*Hk*D;
  const std::size_t K_NORM_BYTES = std::size_t(B)*seqlen*Hk * sizeof(float);
  const std::size_t V_SC_BYTES   = std::size_t(B)*seqlen*Hk * sizeof(float);
  const std::size_t Q_BYTES      = std::size_t(N_spec)*B*Hq*D * sizeof(float);
  const std::size_t OUT_BYTES    = Q_BYTES;
  const std::size_t KFP8_BYTES   = (preset_id==PRESET_K8V4)
                                    ? std::size_t(B)*seqlen*Hk*D*sizeof(float) : 0;

  auto make_dev = [&](const void* src, std::size_t bytes) -> void* {
    if (bytes == 0) return nullptr;
    void* p = sycl::malloc_device(bytes, q);
    q.memcpy(p, src, bytes).wait();
    return p;
  };

  auto* d_q      = (float*)   make_dev(q_rot_h,     Q_BYTES);
  auto* d_kidx   = (uint8_t*) make_dev(k_idx_h,     K_IDX_BYTES);
  auto* d_knorm  = (float*)   make_dev(k_norm_h,    K_NORM_BYTES);
  auto* d_kfp8   = (float*)   make_dev(k_fp8_h,     KFP8_BYTES);
  auto* d_vidx   = (uint8_t*) make_dev(v_idx_h,     V_IDX_BYTES);
  auto* d_vscale = (float*)   make_dev(v_scale_h,   V_SC_BYTES);
  auto* d_vzero  = (float*)   make_dev(v_zero_h,    V_SC_BYTES);
  auto* d_cent   = (float*)   make_dev(centroids_h, 8*sizeof(float));
  auto* d_out    = sycl::malloc_device<float>(OUT_BYTES/sizeof(float), q);

  // Grid: one work-item per (n_spec, b, h_q). Intentionally dumb — optimization
  // is the DPAS unit's job (Task 12). This path exists for correctness only.
  const sycl::range<3> g{(std::size_t)N_spec, (std::size_t)B, (std::size_t)Hq};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_decode_spec_scalar_k>(g, [=](sycl::id<3> it) {
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
            for (int d = 0; d < D; ++d) term += q_ptr[d] * d_cent[kidx[d] & 7];
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

  q.memcpy(out_h, d_out, OUT_BYTES).wait();

  if (d_q)      sycl::free(d_q, q);
  if (d_kidx)   sycl::free(d_kidx, q);
  if (d_knorm)  sycl::free(d_knorm, q);
  if (d_kfp8)   sycl::free(d_kfp8, q);
  if (d_vidx)   sycl::free(d_vidx, q);
  if (d_vscale) sycl::free(d_vscale, q);
  if (d_vzero)  sycl::free(d_vzero, q);
  if (d_cent)   sycl::free(d_cent, q);
  sycl::free(d_out, q);
}

} // namespace turboquant_xpu_sycl
