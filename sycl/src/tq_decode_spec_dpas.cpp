#include "tq_decode_spec.hpp"
#include "tq_layout.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstring>
#include <stdexcept>
#include <cmath>

namespace turboquant_xpu_sycl {

// Bring in only the names we need to avoid collision with tq_layout.hpp's
// turboquant_xpu_sycl::layout namespace.
namespace jm = sycl::ext::oneapi::experimental::matrix;

using namespace layout;

static sycl::queue& queue_dpas() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

// Smoke: C[8][16] = A[8][16] * B[16][16], fp16 in, fp32 out. Exercises the
// exact joint_matrix tile shape we plan to use for Q.Kᵀ with M = N_spec.
void joint_matrix_smoke(const sycl::half* A, const sycl::half* B, float* C) {
  auto& q = queue_dpas();
  auto* d_A = sycl::malloc_device<sycl::half>(8*16, q);
  auto* d_B = sycl::malloc_device<sycl::half>(16*16, q);
  auto* d_C = sycl::malloc_device<float>(8*16, q);
  q.memcpy(d_A, A, 8*16*sizeof(sycl::half)).wait();
  q.memcpy(d_B, B, 16*16*sizeof(sycl::half)).wait();

  q.submit([&](sycl::handler& h) {
    h.parallel_for(
      sycl::nd_range<2>({8, 16}, {8, 16}),
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> a;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> b;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c;
        jm::joint_matrix_fill(sg, c, 0.0f);
        jm::joint_matrix_load(sg, a,
          sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(d_A), 16);
        jm::joint_matrix_load(sg, b,
          sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(d_B), 16);
        jm::joint_matrix_mad(sg, c, a, b, c);
        jm::joint_matrix_store(sg, c,
          sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(d_C), 16, jm::layout::row_major);
      });
  }).wait();

  q.memcpy(C, d_C, 8*16*sizeof(float)).wait();
  sycl::free(d_A, q); sycl::free(d_B, q); sycl::free(d_C, q);
}

// ---------------------------------------------------------------------------
// DPAS decode kernel: hybrid design — DPAS for Q·Kᵀ, scalar for softmax+P·V.
//
// Work-group layout: nd_range<2>({B*Hq, SG_SIZE}, {1, SG_SIZE})
//   Each WG = one (b, hq) slot, one sub-group of 16 lanes.
//   All N_spec (≤8) queries for that (b, hq) are processed together using
//   an 8×16 A tile — rows beyond N_spec are zero-padded.
//
// Per-WG algorithm:
//   For each block of 16 KV tokens (kv0..kv0+15):
//     Dequant K[kv0:kv0+16, hk, :] → fp16 tile [16, D] in SLM
//     For d_slice in 0..D step 16:
//       Load A[N_spec, 16] from q[*, b, hq, d_slice:d_slice+16]  (fp16)
//       Load B_K[16, 16] from K_tile[kv0:kv0+16, d_slice:d_slice+16].T  (fp16)
//       C_scores += A * B_K   (accumulates [8, 16] fp32 scores)
//     Store C_scores to SLM → apply mask + scale → online softmax
//     P·V via scalar FMA ladder (matches scalar kernel exactly)
//   Normalize and write to d_out
// ---------------------------------------------------------------------------

void tq_decode_spec_dpas(
    const float* q_rot_h, const uint8_t* k_idx_h, const float* k_norm_h,
    const float* k_fp8_h, const uint8_t* v_idx_h, const float* v_scale_h,
    const float* v_zero_h, const float* centroids_h, float* out_h,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen, int preset_id) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;

  // PoC constraint: D must be 128, seqlen a multiple of 16.
  // N_spec must be <= 8 for the M=8 DPAS tile.
  if (N_spec > 8)
    throw std::runtime_error("tq_decode_spec_dpas: N_spec > 8 not supported");
  if (D != 128)
    throw std::runtime_error("tq_decode_spec_dpas: D must be 128 for PoC");
  if (seqlen % 16 != 0)
    throw std::runtime_error("tq_decode_spec_dpas: seqlen must be multiple of 16");

  auto& q = queue_dpas();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / sycl::sqrt(float(D));

  // Byte sizes — same pattern as scalar kernel
  const std::size_t K_IDX_BYTES  = std::size_t(B)*seqlen*Hk*D;
  const std::size_t V_IDX_BYTES  = std::size_t(B)*seqlen*Hk*D;
  const std::size_t K_NORM_BYTES = std::size_t(B)*seqlen*Hk * sizeof(float);
  const std::size_t V_SC_BYTES   = std::size_t(B)*seqlen*Hk * sizeof(float);
  const std::size_t Q_BYTES      = std::size_t(N_spec)*B*Hq*D * sizeof(float);
  const std::size_t OUT_BYTES    = Q_BYTES;
  const std::size_t KFP8_BYTES   = (preset_id == PRESET_K8V4)
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
  auto* d_cent   = (float*)   make_dev(centroids_h, K3_CENTROIDS*sizeof(float));
  auto* d_out    = sycl::malloc_device<float>(OUT_BYTES/sizeof(float), q);

  // WG layout: one WG per (b, hq), each WG has exactly one sub-group of 16.
  // Grid row = b*Hq + hq; WG local size = [1, SG_SIZE].
  const sycl::range<2> global_range{std::size_t(B) * Hq, SG_SIZE};
  const sycl::range<2> local_range{1, SG_SIZE};

  // DPAS tile geometry. BLK_KV is deliberately locked to N_TILE=16 because the
  // DPAS tile shape requires it — this is NOT the same knob as layout::BLOCK_KV_DEFAULT,
  // which tunes the scalar kernel. Task 13's BLOCK_KV sweep does not apply here.
  static constexpr int M_TILE  = 8;   // # speculative queries per DPAS tile
  static constexpr int K_TILE  = 16;  // K dimension of DPAS tile (= SG_SIZE)
  static constexpr int N_TILE  = 16;  // N dimension = KV tokens per DPAS block
  static constexpr int BLK_KV  = 16;  // KV tokens per outer block (== N_TILE)
  static constexpr int D_DIM   = 128; // fixed for PoC
  static_assert(K_TILE == turboquant_xpu_sycl::layout::SG_SIZE,
                "DPAS K tile must equal sub-group size on Xe2.");
  static_assert(BLK_KV == N_TILE, "Outer KV block must equal DPAS N tile.");

  // SLM sizes (per WG):
  //   scores_slm: [8, 16] fp32 — raw Q·Kᵀ scores from DPAS
  //   a_slm:      [8, 16] fp16 — Q fragment staged before joint_matrix_load
  //   k_fp16_slm: [16, D]  fp16 — dequanted K tile
  // The dedicated a_slm region replaces an earlier design that aliased it onto
  // scores_slm; keeping them distinct removes a latent footgun where an
  // early-exit could skip the separating barrier and corrupt scores.
  const int SCORES_SLM_FLOATS = M_TILE * N_TILE;             // 128
  const int A_SLM_HALVES      = M_TILE * K_TILE;             // 128
  const int K_SLM_HALVES      = BLK_KV * D_DIM;              // 2048

  const std::size_t SLM_BYTES =
      SCORES_SLM_FLOATS * sizeof(float) +
      A_SLM_HALVES      * sizeof(sycl::half) +
      K_SLM_HALVES      * sizeof(sycl::half);

  // Capture scalars for kernel lambda
  int n_spec = N_spec;
  int b_dim = B, hq_dim = Hq, hk_dim = Hk, d_dim = D, sl = seqlen;
  int kv_grp = kv_group;
  float asc = attn_scale;
  int pid = preset_id;

  q.submit([&](sycl::handler& h) {
    // Declare SLM accessor
    sycl::local_accessor<uint8_t, 1> slm(SLM_BYTES, h);

    h.parallel_for(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int wg_id = it.get_global_id(0);  // b*Hq + hq
        const int lane  = it.get_local_id(1);   // 0..15

        const int b  = wg_id / hq_dim;
        const int hq = wg_id % hq_dim;
        const int h_k = hq / kv_grp;

        // SLM pointers — dedicated, non-overlapping regions.
        float*       scores_slm = reinterpret_cast<float*>(&slm[0]);
        sycl::half*  a_slm      = reinterpret_cast<sycl::half*>(
                                     &slm[SCORES_SLM_FLOATS * sizeof(float)]);
        sycl::half*  k_fp16_slm = reinterpret_cast<sycl::half*>(
                                     &slm[SCORES_SLM_FLOATS * sizeof(float)
                                        + A_SLM_HALVES * sizeof(sycl::half)]);

        // Online softmax state: per-query (n) values
        // With M_TILE=8 rows, each lane holds ceil(8*16/16) = 8 accumulator
        // elements in the DPAS register file. For the scalar softmax+PV we
        // keep separate per-query accumulators.
        float m_prev[M_TILE];
        float l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -INFINITY;
          l_prev[n] = 0.0f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.0f;
        }

        // -------------------------------------------------------------------
        // Outer loop: KV blocks
        // -------------------------------------------------------------------
        for (int kv0 = 0; kv0 < sl; kv0 += BLK_KV) {

          // -----------------------------------------------------------------
          // Step 1: Dequant K[kv0:kv0+BLK_KV, h_k, :] → fp16 into SLM.
          // All 16 lanes collaborate: each lane fills 16 columns per row
          // across the 16-row tile (= 128 fp16 per lane total).
          // k_fp16_slm[t, d] with stride D_DIM.
          // -----------------------------------------------------------------
          for (int t = 0; t < BLK_KV; ++t) {
            int kv = kv0 + t;
            bool valid = (kv < sl);
            // Each lane handles 8 consecutive d values (8 * 16 lanes = 128)
            for (int col = lane; col < D_DIM; col += SG_SIZE) {
              sycl::half val;
              if (!valid) {
                val = sycl::half(0.0f);
              } else if (pid == PRESET_K8V4) {
                // k_fp8 stored as float32 — just cast to fp16
                const float* kp = d_kfp8 + ((b*sl + kv)*hk_dim + h_k)*D_DIM;
                val = sycl::half(kp[col]);
              } else {
                // k3v4_nc: centroid lookup + norm
                const uint8_t* ki = d_kidx + ((b*sl + kv)*hk_dim + h_k)*D_DIM;
                float norm = d_knorm[(b*sl + kv)*hk_dim + h_k];
                int idx = ki[col] & (K3_CENTROIDS - 1);
                val = sycl::half(d_cent[idx] * norm);
              }
              k_fp16_slm[t * D_DIM + col] = val;
            }
          }
          // Barrier: all lanes must finish filling k_fp16_slm before DPAS reads it
          sycl::group_barrier(it.get_group());

          // -----------------------------------------------------------------
          // Step 2: Q·Kᵀ via DPAS — accumulate over D in slices of K_TILE=16.
          //   A:  [M=8, K_TILE=16] fp16  — 8 queries × 16 dim-slice
          //   B:  [K_TILE=16, N_TILE=16] fp16 — transposed K tile (16 KV × 16 dim)
          //   C:  [M=8, N_TILE=16] fp32  — accumulator
          // Note: K^T means B is loaded from K[kv, d_slice] which is already
          //       column-major when we treat dim as the inner axis. We load
          //       B from k_fp16_slm transposed using col_major layout.
          // -----------------------------------------------------------------
          jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M_TILE, N_TILE>
              c_scores;
          jm::joint_matrix_fill(sg, c_scores, 0.0f);

          // Q·Kᵀ: for each d_slice, A = q[n, b, hq, d_slice:d_slice+16]
          //       B = K_tile[:, d_slice:d_slice+16]  but we need K^T so
          //       B dimension is [K_dim=16, N_KV=16] → load from column-major
          //       view of k_fp16_slm[kv, d_slice].
          //
          // k_fp16_slm layout: [BLK_KV=16, D=128] row-major
          // For B (fp16, K_TILE×N_TILE): we need K[d, kv] which is the
          // transpose of K[kv, d]. We achieve this by loading from SLM
          // with col_major layout: B_mat[k=d_idx, n=kv_idx].
          // col_major means element [k, n] is at offset n*K_TILE + k =
          // kv * K_TILE + d_in_slice. But k_fp16_slm[kv, d] = kv*128 + d.
          // For a col_major B where stride = BLK_KV (outer stride is KV dim):
          // element [k, n] at k + n*K_TILE. We need [d_in_slice, kv] →
          // address = d_in_slice + kv * K_TILE. That matches SLM layout
          // if we load with row stride = BLK_KV (stride between k-rows).
          //
          // Actually for joint_matrix B col_major: element at [K,N] is at
          // base + K + N * K_TILE. We want B[k=d_i, n=kv] = K[kv, d_slice+d_i].
          // In SLM: K[kv, d_slice+d_i] = kv*128 + d_slice + d_i.
          // For col_major with stride=BLK_KV: B[d_i, kv] stored as d_i + kv*16.
          // These don't match directly, so we use row_major for B but load K^T
          // by constructing a temporary in SLM.
          //
          // Simplest approach: build a [K_TILE=16, N_TILE=16] transposed scratch
          // in SLM before the DPAS. We use the scores_slm space (128 float =
          // 256 half bytes; K^T scratch also 256 half bytes). We'll reuse
          // k_fp16_slm for a transposed view by using col_major layout for B.
          //
          // joint_matrix B col_major with size [K=16, N=16]:
          // loads element [k][n] = ptr[k + n * 16]. We want B[k][n] = K[n][d_slice+k].
          // K[n][d_slice+k] in SLM = k_fp16_slm[n*128 + d_slice + k].
          // With col_major stride=128: ptr[k + n*16] but we need ptr at
          // base = k_fp16_slm + d_slice, and stride=128? Let's check:
          // col_major B[k][n] = base[k + n*stride_n]. If stride_n = 128:
          // base[k + n*128] = k_fp16_slm[d_slice + k + n*128]. That IS
          // K[n, d_slice+k] = k_fp16_slm[n*128 + d_slice + k]. YES — matches!
          //
          // So: load B col_major from (k_fp16_slm + d_slice) with stride=128.

          for (int ds = 0; ds < D_DIM; ds += K_TILE) {
            // Construct A in its dedicated SLM region: 8 queries × 16 dim elements.
            // q is laid out [N_spec, B, Hq, D] in global memory, so consecutive
            // n rows are D*Hq*B apart — not contiguous. All 16 lanes cooperatively
            // scatter-write 128 half-elements into a_slm.
            for (int elem = lane; elem < M_TILE * K_TILE; elem += SG_SIZE) {
              int n_row  = elem / K_TILE;
              int d_col  = elem % K_TILE;
              sycl::half v16;
              if (n_row < n_spec) {
                // q is [N_spec, B, Hq, D] float32
                int q_idx = n_row * (b_dim * hq_dim * D_DIM)
                          + b * (hq_dim * D_DIM)
                          + hq * D_DIM
                          + ds + d_col;
                v16 = sycl::half(d_q[q_idx]);
              } else {
                v16 = sycl::half(0.0f);  // zero-pad rows beyond N_spec
              }
              a_slm[n_row * K_TILE + d_col] = v16;
            }
            sycl::group_barrier(it.get_group());

            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, M_TILE, K_TILE, jm::layout::row_major>
                mat_a;
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, K_TILE, N_TILE, jm::layout::col_major>
                mat_b;

            // Load A from SLM (a_slm is float* reinterpreted as half* — aligned)
            jm::joint_matrix_load(sg, mat_a,
              sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(a_slm),
              K_TILE);

            // Load B = K^T: col_major from (k_fp16_slm + ds), stride=D_DIM
            // B[k=d_i, n=kv] = K[kv, ds+d_i] = k_fp16_slm[kv*128 + ds + d_i]
            //                = (k_fp16_slm + ds)[d_i + kv*128]
            // col_major B: element[k][n] at ptr[k + n*stride], stride=D_DIM
            jm::joint_matrix_load(sg, mat_b,
              sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(k_fp16_slm + ds),
              D_DIM);

            jm::joint_matrix_mad(sg, c_scores, mat_a, mat_b, c_scores);

            // Barrier before next iteration that overwrites a_slm
            sycl::group_barrier(it.get_group());
          }

          // Store scores [8, 16] to scores_slm (fp32, row_major)
          jm::joint_matrix_store(sg, c_scores,
            sycl::address_space_cast<sycl::access::address_space::local_space,
              sycl::access::decorated::no>(scores_slm),
            N_TILE, jm::layout::row_major);
          sycl::group_barrier(it.get_group());

          // -----------------------------------------------------------------
          // Step 3: Scalar online softmax + P·V (matches scalar kernel).
          // Lane 0 drives all scalar work; other lanes idle here.
          // This is the hybrid approach — correctness over performance.
          // -----------------------------------------------------------------
          if (lane == 0) {
            // Read scores from SLM: scores_slm[n, kv_local] for n=0..N_spec-1
            for (int n = 0; n < n_spec; ++n) {
              // Apply attn_scale and masking
              float scores_n[N_TILE];
              for (int t = 0; t < N_TILE; ++t) {
                int kv = kv0 + t;
                float s = scores_slm[n * N_TILE + t];
                if (kv >= sl) {
                  scores_n[t] = -INFINITY;
                } else {
                  scores_n[t] = s * asc;
                }
              }

              // Block-level max and online softmax renormalization
              float m_local = scores_n[0];
              for (int t = 1; t < N_TILE; ++t)
                m_local = sycl::fmax(m_local, scores_n[t]);
              float m_new = sycl::fmax(m_local, m_prev[n]);
              float re = sycl::exp(m_prev[n] - m_new);

              // Rescale accumulator
              for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
              l_prev[n] *= re;

              // P·V accumulation (scalar, matches scalar kernel)
              for (int t = 0; t < N_TILE; ++t) {
                int kv = kv0 + t;
                if (kv >= sl) continue;
                float p = sycl::exp(scores_n[t] - m_new);
                l_prev[n] += p;
                const uint8_t* vi = d_vidx + ((b*sl + kv)*hk_dim + h_k)*D_DIM;
                float vs = d_vscale[(b*sl + kv)*hk_dim + h_k];
                float vz = d_vzero [(b*sl + kv)*hk_dim + h_k];
                for (int d = 0; d < D_DIM; ++d)
                  acc[n][d] += p * (float(vi[d]) * vs + vz);
              }
              m_prev[n] = m_new;
            }
          }
          // Barrier before next kv0 block (next DPAS overwrites k_fp16_slm)
          sycl::group_barrier(it.get_group());
        } // end kv0 loop

        // -------------------------------------------------------------------
        // Step 4: Normalize and write output (lane 0 only)
        // d_out[n, b, hq, d] = acc[n][d] / l_prev[n]
        // -------------------------------------------------------------------
        if (lane == 0) {
          for (int n = 0; n < n_spec; ++n) {
            float inv_l = 1.0f / l_prev[n];
            float* o_ptr = d_out + ((n * b_dim + b) * hq_dim + hq) * D_DIM;
            for (int d = 0; d < D_DIM; ++d)
              o_ptr[d] = acc[n][d] * inv_l;
          }
        }
      });
  }).wait();

  q.memcpy(out_h, d_out, OUT_BYTES).wait();

  if (d_q)      sycl::free(d_q,      q);
  if (d_kidx)   sycl::free(d_kidx,   q);
  if (d_knorm)  sycl::free(d_knorm,  q);
  if (d_kfp8)   sycl::free(d_kfp8,   q);
  if (d_vidx)   sycl::free(d_vidx,   q);
  if (d_vscale) sycl::free(d_vscale, q);
  if (d_vzero)  sycl::free(d_vzero,  q);
  if (d_cent)   sycl::free(d_cent,   q);
  sycl::free(d_out, q);
}

} // namespace turboquant_xpu_sycl
