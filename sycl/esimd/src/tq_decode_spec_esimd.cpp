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

static constexpr int N_D_SLICES = D_DIM / N_TILE;  // 128 / 16 = 8

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

  // Grid: one WG per (b, h_q). Thread 0 of each WG does the real work.
  const sycl::range<2> global_range{std::size_t(B) * Hq, SG_SIZE};
  const sycl::range<2> local_range{1, SG_SIZE};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_decode_spec_esimd_dpas_full>(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const int wg_id = it.get_global_id(0);
        const int lane  = it.get_local_id(1);
        const int b  = wg_id / hq_total;
        const int hq = wg_id % hq_total;
        const int h_k = hq / kv_group;

        if (lane != 0) return;

        // Online softmax state (per-query).
        float m_prev[M_TILE];
        float l_prev[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -INFINITY;
          l_prev[n] = 0.0f;
        }
        // 8 d-slice accumulators. Each: [M_TILE × N_TILE] fp32.
        esimd::simd<float, M_TILE * N_TILE> acc_d[N_D_SLICES];
        for (int i = 0; i < N_D_SLICES; ++i) acc_d[i] = 0.f;

        // Per-query eff_end_q.
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        for (int kv0 = 0; kv0 < seqlen_v; kv0 += BLK_KV) {
          // --- Dequant K tile [BLK_KV][D] into k_tile register ---
          esimd::simd<sycl::half, BLK_KV * D_DIM> k_tile(sycl::half(0.f));
          for (int t = 0; t < BLK_KV; ++t) {
            int kv = kv0 + t;
            if (pid == PRESET_K8V4) {
              esimd::simd<float, D_DIM> k_f;
              k_f.copy_from(d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM));
              esimd::simd<sycl::half, D_DIM> k_h = k_f;
              k_tile.template select<D_DIM, 1>(t * D_DIM) = k_h;
            } else {
              const uint8_t* kp_idx = d_kidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              float norm = d_knorm[(b * seqlen_v + kv) * hk_total + h_k];
              esimd::simd<uint8_t, D_DIM> idx_vec;
              idx_vec.copy_from(kp_idx);
              esimd::simd<uint32_t, D_DIM> offsets =
                  esimd::simd<uint32_t, D_DIM>(idx_vec & uint8_t(K3_CENTROIDS - 1))
                  * uint32_t(sizeof(float));
              esimd::simd<float, D_DIM> k_f =
                  esimd::gather<float, D_DIM>(d_cent, offsets) * norm;
              esimd::simd<sycl::half, D_DIM> k_h = k_f;
              k_tile.template select<D_DIM, 1>(t * D_DIM) = k_h;
            }
          }

          // --- Dequant V tile [BLK_KV][D] (fp16) into v_tile register ---
          // V is uint8; dequant is v_f = float(v_u8) * v_scale + v_zero. Vectorize.
          esimd::simd<sycl::half, BLK_KV * D_DIM> v_tile(sycl::half(0.f));
          for (int t = 0; t < BLK_KV; ++t) {
            int kv = kv0 + t;
            const uint8_t* vp = d_vidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
            float vs = d_vscale[(b * seqlen_v + kv) * hk_total + h_k];
            float vz = d_vzero[(b * seqlen_v + kv) * hk_total + h_k];
            esimd::simd<uint8_t, D_DIM> v_u8;
            v_u8.copy_from(vp);
            esimd::simd<float, D_DIM> v_f = esimd::simd<float, D_DIM>(v_u8) * vs + vz;
            esimd::simd<sycl::half, D_DIM> v_h = v_f;
            v_tile.template select<D_DIM, 1>(t * D_DIM) = v_h;
          }

          // --- DPAS Q·Kᵀ → c_scores [M_TILE × N_TILE] ---
          esimd::simd<float, M_TILE * N_TILE> c_scores(0.f);
          for (int ds = 0; ds < D_DIM; ds += K_TILE) {
            esimd::simd<sycl::half, M_TILE * K_TILE> a_reg(sycl::half(0.f));
            for (int n = 0; n < n_spec; ++n) {
              const float* q_ptr = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM) + ds;
              esimd::simd<float, K_TILE> q_slice;
              q_slice.copy_from(q_ptr);
              a_reg.template select<K_TILE, 1>(n * K_TILE) = esimd::simd<sycl::half, K_TILE>(q_slice);
            }
            // B_vnni [K/2][N_TILE][2] from k_tile: k_tile[t=nc][d = ds+2kp+i]
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
          c_scores = c_scores * attn_scale;

          // --- Softmax: mask, per-row max, re, p fp16, l-update, acc rescale ---
          float m_new_arr[M_TILE];
          float re_arr[M_TILE];
          bool row_has_any[M_TILE];
          // Matrix p[M_TILE * N_TILE] fp16 for DPAS.
          esimd::simd<sycl::half, M_TILE * N_TILE> p_reg(sycl::half(0.f));

          for (int n = 0; n < M_TILE; ++n) {
            row_has_any[n] = false;
            if (n >= n_spec) { m_new_arr[n] = -INFINITY; re_arr[n] = 0.f; continue; }
            float m_local = -INFINITY;
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              row_has_any[n] = true;
              float v = c_scores[n * N_TILE + t];
              if (v > m_local) m_local = v;
            }
            if (!row_has_any[n]) {
              // Nothing to merge this block for this query.
              m_new_arr[n] = m_prev[n];
              re_arr[n] = 1.f;
              continue;
            }
            float m_p = m_prev[n];
            float m_new = m_local > m_p ? m_local : m_p;
            // re = exp(m_p - m_new). If m_p is -INF, re = 0.
            float re;
            if (m_p == -INFINITY) re = 0.f;
            else re = esimd::exp(esimd::simd<float, 1>(m_p - m_new))[0];
            m_new_arr[n] = m_new;
            re_arr[n] = re;
            l_prev[n] *= re;
            float lsum = 0.f;
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              float pf = esimd::exp(esimd::simd<float, 1>(c_scores[n * N_TILE + t] - m_new))[0];
              lsum += pf;
              p_reg[n * N_TILE + t] = sycl::half(pf);
            }
            l_prev[n] += lsum;
            m_prev[n] = m_new;
          }

          // --- Rescale acc by re[n] (row-wise) ---
          for (int n = 0; n < n_spec; ++n) {
            if (!row_has_any[n]) continue;
            float re = re_arr[n];
            if (re == 1.f) continue;
            for (int ds_idx = 0; ds_idx < N_D_SLICES; ++ds_idx) {
              auto row_view = acc_d[ds_idx].template select<N_TILE, 1>(n * N_TILE);
              row_view = row_view * re;
            }
          }

          // --- DPAS P·V: for each d_slice, c_acc += p · V[:, ds:ds+N_TILE] ---
          for (int ds_idx = 0; ds_idx < N_D_SLICES; ++ds_idx) {
            int ds = ds_idx * N_TILE;
            // B_vnni from v_tile: B[k, nc, i] = v_tile[t = 2*kp+i][d = ds + nc]
            //   B_vnni[kp * N*2 + nc*2 + i] = v_tile[(2*kp+i) * D + ds + nc]
            esimd::simd<sycl::half, K_TILE * N_TILE> b_pv;
            for (int kp = 0; kp < K_TILE / 2; ++kp) {
              for (int nc = 0; nc < N_TILE; ++nc) {
                b_pv[kp * N_TILE * 2 + nc * 2 + 0] = v_tile[(2 * kp + 0) * D_DIM + ds + nc];
                b_pv[kp * N_TILE * 2 + nc * 2 + 1] = v_tile[(2 * kp + 1) * D_DIM + ds + nc];
              }
            }
            acc_d[ds_idx] = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(
                acc_d[ds_idx], b_pv, p_reg);
          }
        }

        // --- Emit output: acc_d[ds_idx] holds [M_TILE × N_TILE] for d in [ds..ds+N_TILE) ---
        for (int n = 0; n < n_spec; ++n) {
          float inv_l = 1.f / l_prev[n];
          float* o_ptr = d_out + (((n * b_total + b) * hq_total + hq) * D_DIM);
          for (int ds_idx = 0; ds_idx < N_D_SLICES; ++ds_idx) {
            int ds = ds_idx * N_TILE;
            for (int nc = 0; nc < N_TILE; ++nc) {
              o_ptr[ds + nc] = acc_d[ds_idx][n * N_TILE + nc] * inv_l;
            }
          }
        }
      });
  }).wait();
}

} // namespace turboquant_xpu_esimd
