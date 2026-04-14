#include <pybind11/pybind11.h>
#include <cstdint>

namespace py = pybind11;

namespace turboquant_xpu_sycl_zc {
void tq_decode_spec_zc_scalar(
    uintptr_t, uintptr_t, uintptr_t, uintptr_t,
    uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
    int, int, int, int, int, int, int);
}

PYBIND11_MODULE(turboquant_xpu_sycl_zc, m) {
  m.doc() = "Zero-copy SYCL TurboQuant spec-decode (stock oneAPI 2025.3)";
  m.def("tq_decode_spec_scalar",
        [](uintptr_t q, uintptr_t ki, uintptr_t kn, uintptr_t kf,
           uintptr_t vi, uintptr_t vs, uintptr_t vz, uintptr_t ce, uintptr_t out,
           int N_spec, int B, int Hq, int Hk, int D, int seqlen, int preset_id) {
          turboquant_xpu_sycl_zc::tq_decode_spec_zc_scalar(
              q, ki, kn, kf, vi, vs, vz, ce, out,
              N_spec, B, Hq, Hk, D, seqlen, preset_id);
        },
        "Zero-copy scalar SYCL tq_decode_spec (all pointers are XPU USM)");
  m.attr("PRESET_K8V4") = 0;
  m.attr("PRESET_K3V4NC") = 1;
}
