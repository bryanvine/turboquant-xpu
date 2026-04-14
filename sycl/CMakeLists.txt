cmake_minimum_required(VERSION 3.26)
project(turboquant_xpu_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Require icpx
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx")
  message(FATAL_ERROR "Please configure with -DCMAKE_CXX_COMPILER=icpx after sourcing /opt/intel/oneapi/setvars.sh")
endif()

find_package(pybind11 REQUIRED CONFIG)

# Default AOT target: bmg family (covers B70 / BMG-G31). Override with
#   -DVLLM_XPU_AOT_DEVICES=intel_gpu_bmg_g21 on the cmake command line.
# Set to empty string to force JIT compilation (no `ocloc` needed).
if(NOT DEFINED VLLM_XPU_AOT_DEVICES)
  set(VLLM_XPU_AOT_DEVICES "intel_gpu_bmg_g31" CACHE STRING "AOT target list for SYCL build")
endif()

set(SYCL_FLAGS
  -fsycl
  -O3
  -fno-sycl-instrument-device-code
)
if(NOT "${VLLM_XPU_AOT_DEVICES}" STREQUAL "")
  list(APPEND SYCL_FLAGS -fsycl-targets=${VLLM_XPU_AOT_DEVICES})
endif()

# SPV extensions used by the DPAS + 2D block I/O path — safe to enable for scalar
# kernel too (compiler ignores if unused).
set(SPV_EXTS
  -Xspirv-translator
  -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_2d_block_io,+SPV_INTEL_split_barrier
)

pybind11_add_module(turboquant_xpu_sycl
  src/tq_decode_spec_scalar.cpp
  src/tq_decode_spec_dpas.cpp
  src/tq_decode_spec_py.cpp
)

target_include_directories(turboquant_xpu_sycl PRIVATE include src)
target_compile_options(turboquant_xpu_sycl PRIVATE ${SYCL_FLAGS} ${SPV_EXTS})
target_link_options(turboquant_xpu_sycl PRIVATE
  ${SYCL_FLAGS}
  -fsycl-max-parallel-link-jobs=8
  -flink-huge-device-code
)
