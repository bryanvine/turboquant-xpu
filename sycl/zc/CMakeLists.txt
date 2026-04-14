cmake_minimum_required(VERSION 3.26)
project(turboquant_xpu_sycl_zc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CXX_COMPILER MATCHES "icpx$")
  message(FATAL_ERROR "Zero-copy build REQUIRES stock oneAPI 2025.3 icpx (to match torch-XPU's libsycl.so.8). Source /opt/intel/oneapi/setvars.sh and configure with -DCMAKE_CXX_COMPILER=icpx.")
endif()

find_package(pybind11 REQUIRED CONFIG)

# Detect the venv's SYCL include root (intel_sycl_rt 2025.1.x headers) so that
# icpx generates symbols compatible with torch-XPU's libsycl.so.8 (2025.1),
# not the newer 2025.3 libsycl that icpx itself ships with.
if(NOT DEFINED TORCH_VENV_SYCL_INCLUDE)
  set(TORCH_VENV_SYCL_INCLUDE "" CACHE STRING "Path to the venv include dir hosting SYCL 2025.1 headers")
endif()

# AOT target default for B70; empty string forces JIT.
if(NOT DEFINED VLLM_XPU_AOT_DEVICES)
  set(VLLM_XPU_AOT_DEVICES "intel_gpu_bmg_g31" CACHE STRING "")
endif()

set(SYCL_FLAGS -fsycl -O3 -fno-sycl-instrument-device-code)
if(NOT "${VLLM_XPU_AOT_DEVICES}" STREQUAL "")
  list(APPEND SYCL_FLAGS -fsycl-targets=${VLLM_XPU_AOT_DEVICES})
endif()

pybind11_add_module(turboquant_xpu_sycl_zc
  src/tq_decode_spec_zc.cpp
  src/tq_decode_spec_zc_py.cpp
)

target_include_directories(turboquant_xpu_sycl_zc PRIVATE ../include)

# Prepend venv SYCL headers so icpx uses 2025.1 ABI instead of its own 2025.3 ABI.
if(NOT "${TORCH_VENV_SYCL_INCLUDE}" STREQUAL "")
  target_compile_options(turboquant_xpu_sycl_zc PRIVATE
    -isystem ${TORCH_VENV_SYCL_INCLUDE}
  )
endif()

target_compile_options(turboquant_xpu_sycl_zc PRIVATE ${SYCL_FLAGS})
target_link_options(turboquant_xpu_sycl_zc PRIVATE ${SYCL_FLAGS})
