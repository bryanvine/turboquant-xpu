# SYCL PoC build runbook

## One-time setup (host)
1. `sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake pybind11-dev`
2. `source /opt/intel/oneapi/setvars.sh` (must be sourced in every shell that builds)
3. `sycl-ls` must list the B70 Level-Zero device

## Python env
`python3 -m venv .venv-sycl && source .venv-sycl/bin/activate && pip install -r sycl/requirements.txt`

## Build
`cd sycl && cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=icpx && cmake --build build`

## Run tests
`source .venv-sycl/bin/activate && python -m pytest tests/sycl/ -v`

## Run benchmark
`python scripts/bench_sycl_spec.py`
