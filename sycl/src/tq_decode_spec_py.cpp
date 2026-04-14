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

PYBIND11_MODULE(turboquant_xpu_sycl, m) {
  m.doc() = "SYCL TurboQuant decode-spec PoC";
  m.def("hello_identity", &hello_identity_py, "Device-round-trip identity");
  m.def("hello_scale",    &hello_scale_py,    "Device-round-trip scalar multiply");
  // tq_decode_spec_* bindings added in Task 9/11
}
