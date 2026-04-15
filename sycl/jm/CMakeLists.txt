cmake_minimum_required(VERSION 3.26)
project(turboquant_xpu_sycl_jm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Phase (a) requires the intel/llvm nightly (clang++ 23) — stock oneAPI 2025.3
# lacks BMG-G31 in get_matrix_combinations(). Accept either compiler name so
# the CMakeLists is future-proof if a stock release ships the fix.
if(NOT CMAKE_CXX_COMPILER MATCHES "(clang\\+\\+|icpx)$")
  message(FATAL_ERROR
    "JM build requires intel/llvm clang++ (or a future stock icpx with "
    "BMG-G31 combinations). Configure with "
    "-DCMAKE_CXX_COMPILER=/tmp/intel-llvm-nightly/bin/clang++ after setting "
    "PATH + LD_LIBRARY_PATH per the plan's env section.")
endif()

find_package(pybind11 REQUIRED CONFIG)

# JIT only — nightly's AOT target list may not include intel_gpu_bmg_g31.
# Leaving VLLM_XPU_AOT_DEVICES empty forces JIT (~100-200ms first-dispatch).
if(NOT DEFINED VLLM_XPU_AOT_DEVICES)
  set(VLLM_XPU_AOT_DEVICES "" CACHE STRING "JM phase (a): JIT only, leave empty")
endif()

set(SYCL_FLAGS -fsycl -O3 -fno-sycl-instrument-device-code -fsycl-device-code-split=per_kernel)
if(NOT "${VLLM_XPU_AOT_DEVICES}" STREQUAL "")
  list(APPEND SYCL_FLAGS -fsycl-targets=${VLLM_XPU_AOT_DEVICES})
endif()

pybind11_add_module(turboquant_xpu_sycl_jm
  src/tq_decode_spec_jm_stage1.cpp
  src/tq_decode_spec_jm_stage2.cpp
  src/tq_decode_spec_jm_py.cpp
)

target_include_directories(turboquant_xpu_sycl_jm PRIVATE include)
target_compile_options(turboquant_xpu_sycl_jm PRIVATE ${SYCL_FLAGS})
target_link_options(turboquant_xpu_sycl_jm PRIVATE ${SYCL_FLAGS})
