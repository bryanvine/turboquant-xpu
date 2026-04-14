#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tq_decode_spec.hpp"

namespace py = pybind11;

py::array_t<float> hello_identity_py(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
  auto in = x.unchecked<1>();
  auto out = py::array_t<float>(in.shape(0));
  auto out_m = out.mutable_unchecked<1>();
  turboquant_xpu_sycl::hello_identity(in.data(0), out_m.mutable_data(0), in.shape(0));
  return out;
}

py::array_t<float> hello_scale_py(py::array_t<float, py::array::c_style | py::array::forcecast> x, float s) {
  auto in = x.unchecked<1>();
  auto out = py::array_t<float>(in.shape(0));
  auto out_m = out.mutable_unchecked<1>();
  turboquant_xpu_sycl::hello_scale(in.data(0), out_m.mutable_data(0), in.shape(0), s);
  return out;
}

py::array_t<float> tq_decode_spec_scalar_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> q,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> k_idx,
    py::array_t<float,   py::array::c_style | py::array::forcecast> k_norm,
    py::array_t<float,   py::array::c_style | py::array::forcecast> k_fp8,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> v_idx,
    py::array_t<float,   py::array::c_style | py::array::forcecast> v_scale,
    py::array_t<float,   py::array::c_style | py::array::forcecast> v_zero,
    py::array_t<float,   py::array::c_style | py::array::forcecast> centroids,
    int preset_id) {
  auto qb = q.request();
  int N_spec = (int)qb.shape[0], B = (int)qb.shape[1], Hq = (int)qb.shape[2], D = (int)qb.shape[3];
  // preset_id == 0 => k8v4 (k_fp8 populated, k_idx/k_norm are small stubs)
  // preset_id == 1 => k3v4_nc (k_idx + k_norm populated, k_fp8 is a small stub)
  int seqlen, Hk;
  if (preset_id == 0) {
    auto req = k_fp8.request();
    seqlen = (int)req.shape[1];
    Hk = (int)req.shape[2];
  } else {
    auto req = k_idx.request();
    seqlen = (int)req.shape[1];
    Hk = (int)req.shape[2];
  }
  auto out = py::array_t<float>({N_spec, B, Hq, D});
  turboquant_xpu_sycl::tq_decode_spec_scalar(
    (const float*)qb.ptr,
    (const uint8_t*)k_idx.request().ptr,
    (const float*)k_norm.request().ptr,
    (const float*)k_fp8.request().ptr,
    (const uint8_t*)v_idx.request().ptr,
    (const float*)v_scale.request().ptr,
    (const float*)v_zero.request().ptr,
    (const float*)centroids.request().ptr,
    (float*)out.request().ptr,
    N_spec, B, Hq, Hk, D, seqlen, preset_id);
  return out;
}

py::array_t<float> tq_decode_spec_dpas_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> q,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> k_idx,
    py::array_t<float,   py::array::c_style | py::array::forcecast> k_norm,
    py::array_t<float,   py::array::c_style | py::array::forcecast> k_fp8,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> v_idx,
    py::array_t<float,   py::array::c_style | py::array::forcecast> v_scale,
    py::array_t<float,   py::array::c_style | py::array::forcecast> v_zero,
    py::array_t<float,   py::array::c_style | py::array::forcecast> centroids,
    int preset_id) {
  auto qb = q.request();
  int N_spec = (int)qb.shape[0], B = (int)qb.shape[1], Hq = (int)qb.shape[2], D = (int)qb.shape[3];
  // preset_id == 0 => k8v4 (k_fp8 populated, k_idx/k_norm are small stubs)
  // preset_id == 1 => k3v4_nc (k_idx + k_norm populated, k_fp8 is a small stub)
  int seqlen, Hk;
  if (preset_id == 0) {
    auto req = k_fp8.request();
    seqlen = (int)req.shape[1];
    Hk = (int)req.shape[2];
  } else {
    auto req = k_idx.request();
    seqlen = (int)req.shape[1];
    Hk = (int)req.shape[2];
  }
  auto out = py::array_t<float>({N_spec, B, Hq, D});
  turboquant_xpu_sycl::tq_decode_spec_dpas(
    (const float*)qb.ptr,
    (const uint8_t*)k_idx.request().ptr,
    (const float*)k_norm.request().ptr,
    (const float*)k_fp8.request().ptr,
    (const uint8_t*)v_idx.request().ptr,
    (const float*)v_scale.request().ptr,
    (const float*)v_zero.request().ptr,
    (const float*)centroids.request().ptr,
    (float*)out.request().ptr,
    N_spec, B, Hq, Hk, D, seqlen, preset_id);
  return out;
}

PYBIND11_MODULE(turboquant_xpu_sycl, m) {
  m.doc() = "SYCL TurboQuant decode-spec PoC";
  m.def("hello_identity", &hello_identity_py, "Device-round-trip identity");
  m.def("hello_scale",    &hello_scale_py,    "Device-round-trip scalar multiply");
  m.def("tq_decode_spec_scalar", &tq_decode_spec_scalar_py, "Scalar SYCL tq_decode_spec");
  m.def("tq_decode_spec_dpas", &tq_decode_spec_dpas_py, "DPAS/joint_matrix SYCL tq_decode_spec");
  m.attr("PRESET_K8V4")   = 0;
  m.attr("PRESET_K3V4NC") = 1;
}
