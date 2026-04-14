cmake_minimum_required(VERSION 3.26)
project(turboquant_xpu_esimd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CXX_COMPILER MATCHES "icpx$")
  message(FATAL_ERROR "ESIMD build requires stock oneAPI 2025.3 icpx (to match torch-XPU's libsycl.so.8). Configure with -DCMAKE_CXX_COMPILER=icpx.")
endif()

find_package(pybind11 REQUIRED CONFIG)

# Detect the venv's SYCL include root (intel_sycl_rt 2025.1.x headers) so that
# icpx generates symbols compatible with torch-XPU's libsycl.so.8 (2025.1),
# not the newer 2025.3 libsycl that icpx itself ships with. Mirrors sycl/zc/.
if(NOT DEFINED TORCH_VENV_SYCL_INCLUDE)
  set(TORCH_VENV_SYCL_INCLUDE "" CACHE STRING "Path to the venv include dir hosting SYCL 2025.1 headers")
endif()

# JIT only — stock 2025.3's AOT BMG-G31 target rejects ESIMD kernel combinations.
# The driver compiles at first dispatch; ~100-200 ms startup cost, no steady-state penalty.
if(NOT DEFINED VLLM_XPU_AOT_DEVICES)
  set(VLLM_XPU_AOT_DEVICES "" CACHE STRING "ESIMD PoC: JIT only, leave empty")
endif()

set(SYCL_FLAGS -fsycl -O3 -fno-sycl-instrument-device-code -fsycl-device-code-split=per_kernel)
if(NOT "${VLLM_XPU_AOT_DEVICES}" STREQUAL "")
  list(APPEND SYCL_FLAGS -fsycl-targets=${VLLM_XPU_AOT_DEVICES})
endif()

pybind11_add_module(turboquant_xpu_esimd
  src/tq_decode_spec_esimd.cpp
  src/tq_decode_spec_esimd_py.cpp
)

target_include_directories(turboquant_xpu_esimd PRIVATE include)

# Prepend venv SYCL headers so icpx uses 2025.1 ABI instead of its own 2025.3 ABI.
if(NOT "${TORCH_VENV_SYCL_INCLUDE}" STREQUAL "")
  target_compile_options(turboquant_xpu_esimd PRIVATE
    -isystem ${TORCH_VENV_SYCL_INCLUDE}
  )
endif()

target_compile_options(turboquant_xpu_esimd PRIVATE ${SYCL_FLAGS})
target_link_options(turboquant_xpu_esimd PRIVATE ${SYCL_FLAGS})
