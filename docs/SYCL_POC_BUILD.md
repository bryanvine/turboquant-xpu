# SYCL PoC build runbook

## One-time setup (host)
1. Add Intel apt repo (GPG key + `/etc/apt/sources.list.d/oneAPI.list`) and install:
   `sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake pybind11-dev`
2. Join the `render` group so Level-Zero can open `/dev/dri/renderD128`:
   `sudo usermod -a -G render $USER`, then log out/in (or use `sg render -c '<cmd>'` for a one-off shell)
3. `source /opt/intel/oneapi/setvars.sh` (must be sourced in every shell that builds)
4. `sycl-ls` must list the B70 Level-Zero device (appears as `Intel(R) Graphics [0xe223]`)

## Python env
`python3 -m venv .venv-sycl && source .venv-sycl/bin/activate && pip install -r sycl/requirements.txt`

## Build
`cd sycl && cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=icpx && cmake --build build`

## Run tests
`source .venv-sycl/bin/activate && python -m pytest tests/sycl/ -v`

## Run benchmark
`python scripts/bench_sycl_spec.py`
