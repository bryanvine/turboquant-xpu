#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include "esimd_layout.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

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
  if (seqlen % BLK_KV != 0)
    throw std::runtime_error("esimd PoC assumes seqlen % 16 == 0");

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

  // Grid: one WG per (b, h_q), one sub-group of SG_SIZE threads. Only thread 0
  // in each WG does real work in this hybrid DPAS+scalar version. Thread 0's
  // 16 SIMD lanes are used for the DPAS op itself (via simd<...> registers).
  const sycl::range<2> global_range{std::size_t(B) * Hq, SG_SIZE};
  const sycl::range<2> local_range{1, SG_SIZE};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_decode_spec_esimd_dpas_qk>(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const int wg_id = it.get_global_id(0);
        const int lane  = it.get_local_id(1);
        const int b  = wg_id / hq_total;
        const int hq = wg_id % hq_total;
        const int h_k = hq / kv_group;

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

        // Per-query effective seq_len.
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        for (int kv0 = 0; kv0 < seqlen_v; kv0 += BLK_KV) {
          // ---- Dequant K tile [BLK_KV][D] into a register simd<half, 2048> ----
          // Row-major: k_tile[t*D_DIM + d] = fp16 K[kv0+t, d].
          esimd::simd<sycl::half, BLK_KV * D_DIM> k_tile(sycl::half(0.f));
          for (int t = 0; t < BLK_KV; ++t) {
            int kv = kv0 + t;
            if (pid == PRESET_K8V4) {
              // fp32 → fp16 copy of one D-row.
              esimd::simd<float, D_DIM> k_f;
              k_f.copy_from(d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM));
              esimd::simd<sycl::half, D_DIM> k_h = k_f;
              k_tile.template select<D_DIM, 1>(t * D_DIM) = k_h;
            } else {
              const uint8_t* kp_idx = d_kidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              float norm = d_knorm[(b * seqlen_v + kv) * hk_total + h_k];
              esimd::simd<float, D_DIM> k_f;
              // Scalar gather from centroids (small D=128, per-lane table lookup).
              for (int d = 0; d < D_DIM; ++d) {
                k_f[d] = d_cent[kp_idx[d] & (K3_CENTROIDS - 1)] * norm;
              }
              esimd::simd<sycl::half, D_DIM> k_h = k_f;
              k_tile.template select<D_DIM, 1>(t * D_DIM) = k_h;
            }
          }

          // ---- DPAS Q·Kᵀ, tiled over D in slices of K_TILE=16 ----
          // scores tile: [M_TILE=8][N_TILE=16] fp32
          esimd::simd<float, M_TILE * N_TILE> c_scores(0.f);
          for (int ds = 0; ds < D_DIM; ds += K_TILE) {
            // A: Q slice [M_TILE][K_TILE] row-major, fp16. Zero-pad n >= n_spec.
            esimd::simd<sycl::half, M_TILE * K_TILE> a_reg(sycl::half(0.f));
            for (int n = 0; n < n_spec; ++n) {
              const float* q_ptr = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM) + ds;
              esimd::simd<float, K_TILE> q_slice;
              q_slice.copy_from(q_ptr);
              esimd::simd<sycl::half, K_TILE> q_h = q_slice;
              a_reg.template select<K_TILE, 1>(n * K_TILE) = q_h;
            }
            // B (VNNI-packed): K slice [K_TILE][N_TILE] col-major from k_tile, then
            //   B_vnni[kp*N*2 + nc*2 + i] = k_tile[nc*D + ds + 2*kp + i]
            esimd::simd<sycl::half, K_TILE * N_TILE> b_reg;
            for (int kp = 0; kp < K_TILE / 2; ++kp) {
              for (int nc = 0; nc < N_TILE; ++nc) {
                b_reg[kp * N_TILE * 2 + nc * 2 + 0] = k_tile[nc * D_DIM + ds + 2 * kp + 0];
                b_reg[kp * N_TILE * 2 + nc * 2 + 1] = k_tile[nc * D_DIM + ds + 2 * kp + 1];
              }
            }
            c_scores = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(
                c_scores, b_reg, a_reg);
          }

          // Apply attn_scale.
          c_scores = c_scores * attn_scale;

          // ---- Scalar online softmax + P·V (unchanged from Task 5) ----
          for (int n = 0; n < n_spec; ++n) {
            float m_local = c_scores[n * N_TILE];
            for (int t = 1; t < BLK_KV; ++t) {
              float v = c_scores[n * N_TILE + t];
              if (v > m_local) m_local = v;
            }
            // Causal mask on the local max: drop positions past eff_end.
            // (Already encoded below when we iterate; we just skip those t's.)
            // Re-compute m_local skipping masked-out positions for correctness.
            if (is_causal) {
              m_local = -INFINITY;
              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) continue;
                float v = c_scores[n * N_TILE + t];
                if (v > m_local) m_local = v;
              }
              // If all masked, m_local stays -INF — skip the block for this n.
              if (m_local == -INFINITY) continue;
            }
            float m_p = m_prev[n];
            float m_new = m_local > m_p ? m_local : m_p;
            float re = esimd::exp(esimd::simd<float, 1>(m_p - m_new))[0];
            for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
            l_prev[n] *= re;
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              float p = esimd::exp(esimd::simd<float, 1>(c_scores[n * N_TILE + t] - m_new))[0];
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
