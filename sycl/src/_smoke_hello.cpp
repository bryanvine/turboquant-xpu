// SPDX-License-Identifier: Apache-2.0
//
// Minimal SYCL device-smoke program. Not built by CMake — compiled ad-hoc with
// `icpx -fsycl sycl/src/_smoke_hello.cpp -o smoke_hello`. Keep as a reference
// when diagnosing SYCL environment issues (queue selection, sub-group sizes,
// parallel_for correctness) without involving the full kernel/bindings stack.
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
  std::cout << "SG sizes: ";
  for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) std::cout << s << " ";
  std::cout << "\n";

  int N = 16;
  sycl::buffer<int, 1> buf(N);
  q.submit([&](sycl::handler& h) {
    sycl::accessor a{buf, h, sycl::write_only, sycl::no_init};
    h.parallel_for(N, [=](sycl::id<1> i) { a[i] = int(i) * 3; });
  }).wait();
  auto host = buf.get_host_access();
  std::cout << "host[5]=" << host[5] << " (expected 15)\n";
  return 0;
}
