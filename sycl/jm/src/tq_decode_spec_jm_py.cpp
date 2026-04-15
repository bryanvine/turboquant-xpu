#include <pybind11/pybind11.h>
#include "tq_decode_spec_jm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(turboquant_xpu_sycl_jm, m) {
  m.doc() = "SYCL joint_matrix + split-KV TurboQuant decode-spec (phase a PoC, nightly-only)";

  m.def("tq_decode_spec_jm",
        [](uintptr_t q, uintptr_t kf, uintptr_t vi, uintptr_t vs, uintptr_t vz,
           uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
           int N_spec, int B, int Hq, int Hk, int D, int seqlen,
           int preset_id, int causal, int cached_len) {
          turboquant_xpu_sycl_jm::tq_decode_spec_jm(
              q, kf, vi, vs, vz, partial_out, partial_lse, out,
              N_spec, B, Hq, Hk, D, seqlen, preset_id, causal, cached_len);
        },
        "Full two-stage decode: stage1 DPAS+split-KV then stage2 reduce. "
        "All pointers are XPU USM (int-cast).");

  m.attr("PRESET_K8V4") = 0;
  m.attr("PRESET_K3V4NC") = 1;
  m.attr("NUM_KV_SPLITS") = 8;
}
