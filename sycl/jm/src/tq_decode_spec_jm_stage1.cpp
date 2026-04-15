#include "tq_decode_spec_jm.hpp"
#include "jm_layout.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
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
    throw std::runtime_error("jm phase (a) supports k8v4 only");
  if (seqlen % BLK_KV != 0)
    throw std::runtime_error("seqlen must be BLK_KV-aligned");
  if (seqlen % NUM_KV_SPLITS != 0)
    throw std::runtime_error("seqlen must be NUM_KV_SPLITS-aligned");

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
  const int seqlen_per_split = seqlen / NUM_KV_SPLITS;

  const auto* d_q      = reinterpret_cast<const float*>(q_rot);
  const auto* d_kfp8   = reinterpret_cast<const float*>(k_fp8);
  const auto* d_vidx   = reinterpret_cast<const uint8_t*>(v_idx);
  const auto* d_vscale = reinterpret_cast<const float*>(v_scale);
  const auto* d_vzero  = reinterpret_cast<const float*>(v_zero);
  auto* d_pout         = reinterpret_cast<float*>(partial_out);
  auto* d_plse         = reinterpret_cast<float*>(partial_lse);

  // Grid: one sub-group (16 lanes) per (b, hq, split).
  const sycl::range<2> global_range{
      std::size_t(B) * Hq * NUM_KV_SPLITS,
      SG_SIZE
  };
  const sycl::range<2> local_range{1, SG_SIZE};

  q.submit([&](sycl::handler& h) {
    // SLM buffers shared across the sub-group.
    // joint_matrix_load/store requires local_space — private_space is not allowed.
    // Work-group is {1, 16} = exactly one sub-group, so SLM is per sub-group.
    sycl::local_accessor<sycl::half, 1> slm_q_buf(M_TILE * D_DIM, h);
    sycl::local_accessor<sycl::half, 1> slm_k_tile(BLK_KV * D_DIM, h);
    sycl::local_accessor<sycl::half, 1> slm_b_tile(K_TILE * N_TILE, h);
    sycl::local_accessor<float, 1>      slm_scores(M_TILE * N_TILE, h);

    h.parallel_for<class tq_jm_stage1_dpas_qk>(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        namespace jm = sycl::ext::oneapi::experimental::matrix;
        auto sg = it.get_sub_group();
        const int wg_id    = it.get_global_id(0);   // (b*Hq*NUM_SPLITS + ...)
        const int lane     = it.get_local_id(1);
        const int split_id = wg_id % NUM_KV_SPLITS;
        const int bh       = wg_id / NUM_KV_SPLITS;
        const int b        = bh / hq_total;
        const int hq       = bh % hq_total;
        const int h_k      = hq / kv_group;
        const int split_start = split_id * seqlen_per_split;
        const int split_end   = split_start + seqlen_per_split;

        // Per-query softmax state (scalar — each lane has its own copy but we
        // only use lane-0's values since these are logically "per work-item"
        // rather than per-lane. Future: broadcast via sub_group_broadcast).
        float m_prev[M_TILE], l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -std::numeric_limits<float>::infinity();
          l_prev[n] = 0.f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.f;
        }
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        // Get raw pointers to SLM buffers for joint_matrix_load/store.
        sycl::half* q_buf   = slm_q_buf.get_pointer();
        sycl::half* k_tile  = slm_k_tile.get_pointer();
        sycl::half* b_tile  = slm_b_tile.get_pointer();
        float*      scores_buf = slm_scores.get_pointer();

        // Load Q once per (b, hq, split). Only lane 0 does the scalar fill;
        // other lanes will load from this tile via joint_matrix_load.
        // (For phase a, keep the fill serial — Task b will parallelize.)
        if (lane == 0) {
          for (int n = 0; n < n_spec; ++n) {
            const float* qp = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM);
            for (int d = 0; d < D_DIM; ++d)
              q_buf[n * D_DIM + d] = sycl::half(qp[d]);
          }
          for (int n = n_spec; n < M_TILE; ++n) {
            for (int d = 0; d < D_DIM; ++d) q_buf[n * D_DIM + d] = sycl::half(0.f);
          }
        }
        sycl::group_barrier(sg);

        for (int kv0 = split_start; kv0 < split_end; kv0 += BLK_KV) {
          // K tile dequant (fp32 → fp16). Fully scalar fill by lane 0 for
          // phase (a) correctness; phase (b) parallelizes across the 16 lanes.
          if (lane == 0) {
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              for (int d = 0; d < D_DIM; ++d)
                k_tile[t * D_DIM + d] = sycl::half(kp[d]);
            }
          }
          sycl::group_barrier(sg);

          // DPAS fragments. joint_matrix is sub-group-collective — each lane
          // owns a portion of the fragment's elements.
          jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a,
                           M_TILE, K_TILE, jm::layout::row_major> ma;
          jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b,
                           K_TILE, N_TILE, jm::layout::row_major> mb;
          jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator,
                           M_TILE, N_TILE, jm::layout::dynamic> mc;   // <- layout::dynamic REQUIRED (smoke verified)
          jm::joint_matrix_fill(sg, mc, 0.f);

          for (int ds = 0; ds < D_DIM; ds += K_TILE) {
            // Load A = Q[0:M_TILE, ds:ds+K_TILE] from slm_q_buf (row-major stride D_DIM).
            auto a_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::no>(q_buf + ds);
            jm::joint_matrix_load(sg, ma, a_ptr, D_DIM);

            // B = K_tile[:, ds:ds+K_TILE]^T — for Q·K^T we need B[k][t] = K[t][ds+k].
            // Pre-transpose into slm_b_tile (lane 0), then load row-major.
            if (lane == 0) {
              for (int k = 0; k < K_TILE; ++k)
                for (int t = 0; t < N_TILE; ++t)
                  b_tile[k * N_TILE + t] = k_tile[t * D_DIM + ds + k];
            }
            sycl::group_barrier(sg);
            auto b_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::no>(b_tile);
            jm::joint_matrix_load(sg, mb, b_ptr, N_TILE);

            jm::joint_matrix_mad(sg, mc, ma, mb, mc);
          }

          // Store C (scores) back to SLM buffer via sub-group-collective store.
          auto c_ptr = sycl::address_space_cast<
              sycl::access::address_space::local_space,
              sycl::access::decorated::no>(scores_buf);
          jm::joint_matrix_store(sg, mc, c_ptr, N_TILE, jm::layout::row_major);
          sycl::group_barrier(sg);

          // Scale + mask + softmax + P·V — all scalar, lane 0 only.
          if (lane == 0) {
            // Apply attn_scale in-place.
            for (int i = 0; i < M_TILE * N_TILE; ++i) scores_buf[i] *= attn_scale;

            // Per-row: causal mask, online softmax, scalar P·V.
            for (int n = 0; n < n_spec; ++n) {
              bool any_valid = false;
              float m_local = -std::numeric_limits<float>::infinity();
              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) continue;
                any_valid = true;
                float v = scores_buf[n * N_TILE + t];
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
                float p = sycl::exp(scores_buf[n * N_TILE + t] - m_new);
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
          sycl::group_barrier(sg);
        }

        // Emit partials (lane 0 only). STORE NORMALIZED acc/l to match Task 5 invariant.
        if (lane == 0) {
          for (int n = 0; n < n_spec; ++n) {
            float* o_ptr = d_pout +
                ((((split_id * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM);
            float lse;
            if (l_prev[n] <= 0.f) {
              lse = -std::numeric_limits<float>::infinity();
              for (int d = 0; d < D_DIM; ++d) o_ptr[d] = 0.f;
            } else {
              float inv_l = 1.0f / l_prev[n];
              for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc[n][d] * inv_l;
              lse = m_prev[n] + sycl::log(l_prev[n]);
            }
            d_plse[((split_id * n_spec + n) * b_total + b) * hq_total + hq] = lse;
          }
        }
      });
  }).wait();
}

} // namespace turboquant_xpu_sycl_jm
