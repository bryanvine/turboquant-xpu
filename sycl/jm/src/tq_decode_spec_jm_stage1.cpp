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
    //
    // Task 7 buffers (Q·K DPAS):
    //   slm_q_buf   : half [M_TILE=8][D_DIM=128]   = 2 KB
    //   slm_k_tile  : half [BLK_KV=16][D_DIM=128]  = 4 KB
    //   slm_b_tile  : half [K_TILE=16][N_TILE=16]  = 512 B
    //   slm_scores  : float[M_TILE=8][N_TILE=16]   = 512 B
    // Task 8 new buffers (P·V DPAS):
    //   slm_v_tile  : half [BLK_KV=16][D_DIM=128]  = 4 KB
    //   slm_p_buf   : half [M_TILE=8][N_TILE=16]   = 256 B
    //   slm_b_pv    : half [K_TILE=16][N_TILE=16]  = 512 B
    //   slm_acc_in  : float[M_TILE=8][N_TILE=16]   = 512 B
    //   slm_acc_out : float[M_TILE=8][N_TILE=16]   = 512 B
    // Total: 2+4+0.5+0.5+4+0.25+0.5+0.5+0.5 ≈ 12.75 KB  (well within 64 KB BMG-G31 budget)
    sycl::local_accessor<sycl::half, 1> slm_q_buf(M_TILE * D_DIM, h);
    sycl::local_accessor<sycl::half, 1> slm_k_tile(BLK_KV * D_DIM, h);
    sycl::local_accessor<sycl::half, 1> slm_b_tile(K_TILE * N_TILE, h);
    sycl::local_accessor<float, 1>      slm_scores(M_TILE * N_TILE, h);
    sycl::local_accessor<sycl::half, 1> slm_v_tile(BLK_KV * D_DIM, h);
    sycl::local_accessor<sycl::half, 1> slm_p_buf(M_TILE * N_TILE, h);
    sycl::local_accessor<sycl::half, 1> slm_b_pv(K_TILE * N_TILE, h);
    sycl::local_accessor<float, 1>      slm_acc_in(M_TILE * N_TILE, h);
    sycl::local_accessor<float, 1>      slm_acc_out(M_TILE * N_TILE, h);

    h.parallel_for<class tq_jm_stage1_dpas_pv>(
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
        // acc_scalar replaces Task 7's acc[M_TILE][D_DIM] — persists across KV blocks.
        // Rescaled in-place by re_arr[n] before each DPAS P·V accumulation.
        float acc_scalar[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -std::numeric_limits<float>::infinity();
          l_prev[n] = 0.f;
          for (int d = 0; d < D_DIM; ++d) acc_scalar[n][d] = 0.f;
        }
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        // Get raw pointers to SLM buffers for joint_matrix_load/store.
        // Drive-by fix: use get_multi_ptr (replaces deprecated get_pointer()).
        sycl::half* q_buf      = slm_q_buf.get_multi_ptr<sycl::access::decorated::no>().get();
        sycl::half* k_tile     = slm_k_tile.get_multi_ptr<sycl::access::decorated::no>().get();
        sycl::half* b_tile     = slm_b_tile.get_multi_ptr<sycl::access::decorated::no>().get();
        float*      scores_buf = slm_scores.get_multi_ptr<sycl::access::decorated::no>().get();
        sycl::half* v_tile     = slm_v_tile.get_multi_ptr<sycl::access::decorated::no>().get();
        sycl::half* p_buf      = slm_p_buf.get_multi_ptr<sycl::access::decorated::no>().get();
        sycl::half* b_pv       = slm_b_pv.get_multi_ptr<sycl::access::decorated::no>().get();
        float*      acc_in     = slm_acc_in.get_multi_ptr<sycl::access::decorated::no>().get();
        float*      acc_out    = slm_acc_out.get_multi_ptr<sycl::access::decorated::no>().get();

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
          // K tile dequant (fp32 → fp16) + V tile dequant (uint8 → fp16).
          // Both filled by lane 0 in a single pass for phase (a) correctness.
          // Phase (b) parallelizes across the 16 lanes.
          if (lane == 0) {
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              for (int d = 0; d < D_DIM; ++d)
                k_tile[t * D_DIM + d] = sycl::half(kp[d]);
              const uint8_t* vp = d_vidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              float vs = d_vscale[(b * seqlen_v + kv) * hk_total + h_k];
              float vz = d_vzero[(b * seqlen_v + kv) * hk_total + h_k];
              for (int d = 0; d < D_DIM; ++d)
                v_tile[t * D_DIM + d] = sycl::half(float(vp[d]) * vs + vz);
            }
          }
          sycl::group_barrier(sg);

          // ---- Q·K^T DPAS ----
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

          // ---- Softmax: lane 0 computes re_arr + p_buf for DPAS P·V ----
          float re_arr[M_TILE];
          bool has_valid[M_TILE];
          if (lane == 0) {
            for (int i = 0; i < M_TILE * N_TILE; ++i) scores_buf[i] *= attn_scale;
            for (int n = 0; n < M_TILE; ++n) {
              re_arr[n] = 1.f;
              has_valid[n] = false;
              if (n >= n_spec) {
                for (int t = 0; t < N_TILE; ++t) p_buf[n * N_TILE + t] = sycl::half(0.f);
                continue;
              }
              bool any_valid = false;
              float m_local = -std::numeric_limits<float>::infinity();
              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) continue;
                any_valid = true;
                float v = scores_buf[n * N_TILE + t];
                if (v > m_local) m_local = v;
              }
              if (!any_valid) {
                for (int t = 0; t < N_TILE; ++t) p_buf[n * N_TILE + t] = sycl::half(0.f);
                continue;
              }
              has_valid[n] = true;
              float m_p = m_prev[n];
              float m_new = m_local > m_p ? m_local : m_p;
              float re = (m_p == -std::numeric_limits<float>::infinity())
                           ? 0.f : sycl::exp(m_p - m_new);
              re_arr[n] = re;
              l_prev[n] *= re;
              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) {
                  p_buf[n * N_TILE + t] = sycl::half(0.f);
                  continue;
                }
                float p = sycl::exp(scores_buf[n * N_TILE + t] - m_new);
                l_prev[n] += p;
                p_buf[n * N_TILE + t] = sycl::half(p);
              }
              m_prev[n] = m_new;
            }
            // Rescale acc_scalar by re_arr — must happen BEFORE the DPAS P·V
            // accumulation so the existing running sum is properly re-weighted.
            for (int n = 0; n < n_spec; ++n) {
              if (!has_valid[n]) continue;
              float re = re_arr[n];
              if (re != 1.f) {
                for (int d = 0; d < D_DIM; ++d) acc_scalar[n][d] *= re;
              }
            }
          }
          sycl::group_barrier(sg);

          // ---- P·V DPAS: 8 d_slice loop ----
          // P·V: out[n][ds+t] = sum_k P[n][k] * V[k][ds+t]
          // A = P  [M_TILE=8, K_TILE=16] — softmax probabilities (fp16)
          // B = V_slice [K_TILE=16, N_TILE=16] — direct slice, no transpose needed
          // C = acc_frag [M_TILE=8, N_TILE=16] — loaded from acc_scalar, stored back after MAD
          for (int ds_idx = 0; ds_idx < N_D_SLICES; ++ds_idx) {
            int ds = ds_idx * N_TILE;

            // Stage current acc_scalar slice (fp32) into SLM, then load as accumulator.
            if (lane == 0) {
              for (int n = 0; n < M_TILE; ++n)
                for (int t = 0; t < N_TILE; ++t)
                  acc_in[n * N_TILE + t] = acc_scalar[n][ds + t];
            }
            sycl::group_barrier(sg);

            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator,
                             M_TILE, N_TILE, jm::layout::dynamic> mc_out;
            auto acc_in_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::no>(acc_in);
            jm::joint_matrix_load(sg, mc_out, acc_in_ptr, N_TILE, jm::layout::row_major);

            // A = P [M_TILE=8, K_TILE=16] from p_buf (row-major, stride N_TILE=16=BLK_KV).
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a,
                             M_TILE, K_TILE, jm::layout::row_major> ma_pv;
            auto p_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::no>(p_buf);
            jm::joint_matrix_load(sg, ma_pv, p_ptr, N_TILE);

            // B = V_tile[:, ds:ds+N_TILE] — direct slice, row-major.
            // b_pv[k][t] = v_tile[k][ds+t]; k in [0,K_TILE), t in [0,N_TILE).
            // No transpose: P·V accumulates over kv tokens, not head-dim.
            if (lane == 0) {
              for (int k = 0; k < K_TILE; ++k)
                for (int t = 0; t < N_TILE; ++t)
                  b_pv[k * N_TILE + t] = v_tile[k * D_DIM + ds + t];
            }
            sycl::group_barrier(sg);
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b,
                             K_TILE, N_TILE, jm::layout::row_major> mb_pv;
            auto b_pv_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::no>(b_pv);
            jm::joint_matrix_load(sg, mb_pv, b_pv_ptr, N_TILE);

            jm::joint_matrix_mad(sg, mc_out, ma_pv, mb_pv, mc_out);

            // Store fragment back to SLM fp32, then scalar copy to acc_scalar.
            auto acc_out_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::no>(acc_out);
            jm::joint_matrix_store(sg, mc_out, acc_out_ptr, N_TILE, jm::layout::row_major);
            sycl::group_barrier(sg);
            if (lane == 0) {
              for (int n = 0; n < M_TILE; ++n)
                for (int t = 0; t < N_TILE; ++t)
                  acc_scalar[n][ds + t] = acc_out[n * N_TILE + t];
            }
            sycl::group_barrier(sg);
          }
        }

        // Emit partials (lane 0 only). STORE NORMALIZED acc_scalar/l to match Task 5 invariant.
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
              for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc_scalar[n][d] * inv_l;
              lse = m_prev[n] + sycl::log(l_prev[n]);
            }
            d_plse[((split_id * n_spec + n) * b_total + b) * hq_total + hq] = lse;
          }
        }
      });
  }).wait();
}

} // namespace turboquant_xpu_sycl_jm
