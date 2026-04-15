#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstring>
#include "tq_decode_spec_jm.hpp"

namespace py = pybind11;

namespace {
// Module-private queue used by the USM helpers. Stage 1 / Stage 2 have their
// own static queue — they all resolve to the same default GPU queue in
// practice since sycl::queue is copy-assignable and the default-constructed
// selector picks the same device.
static sycl::queue& helper_queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}
} // namespace

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
        });

  // USM helpers (child-side allocators; avoid pulling in torch).
  m.def("alloc_device_f32", [](std::size_t n) -> uintptr_t {
    return reinterpret_cast<uintptr_t>(sycl::malloc_device<float>(n, helper_queue()));
  });
  m.def("alloc_device_u8", [](std::size_t n) -> uintptr_t {
    return reinterpret_cast<uintptr_t>(sycl::malloc_device<std::uint8_t>(n, helper_queue()));
  });
  m.def("free_device", [](uintptr_t p) {
    sycl::free(reinterpret_cast<void*>(p), helper_queue());
  });
  m.def("memcpy_to_device_f32",
        [](uintptr_t dst, py::array_t<float, py::array::c_style | py::array::forcecast> src) {
          helper_queue()
              .memcpy(reinterpret_cast<void*>(dst), src.data(), src.nbytes())
              .wait();
        });
  m.def("memcpy_to_device_u8",
        [](uintptr_t dst, py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> src) {
          helper_queue()
              .memcpy(reinterpret_cast<void*>(dst), src.data(), src.nbytes())
              .wait();
        });
  m.def("memcpy_from_device_f32",
        [](uintptr_t src, py::array_t<float, py::array::c_style> dst) {
          helper_queue()
              .memcpy(dst.mutable_data(), reinterpret_cast<void*>(src), dst.nbytes())
              .wait();
        });
  m.def("synchronize", []() { helper_queue().wait(); });

  m.attr("PRESET_K8V4") = 0;
  m.attr("PRESET_K3V4NC") = 1;
  m.attr("NUM_KV_SPLITS") = 8;
}
