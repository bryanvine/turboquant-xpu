# SYCL PoC build runbook

**Order of operations.** Only the `## One-time setup (host)` section below can be run after Task 1. The Python env, build, tests, and benchmark sections depend on artifacts created in later tasks (`sycl/requirements.txt` in Task 3, `sycl/CMakeLists.txt` in Task 6, `scripts/bench_sycl_spec.py` in Task 14). They are documented here up-front so the full pipeline is visible.

## One-time setup (host)
1. **Add the Intel oneAPI apt repo.** Keep the signing key in a dedicated keyring and pin the repo via `signed-by`:
   ```bash
   wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
     | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
     | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update
   ```
2. **Install the compiler, MKL headers, and CMake/Ninja/pybind11:**
   ```bash
   sudo apt install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel \
                      ninja-build cmake pybind11-dev
   ```
3. **Join the `render` group** so Level-Zero can open `/dev/dri/renderD128`:
   ```bash
   sudo usermod -a -G render $USER
   ```
   Log out and back in (or start a new SSH session) so the new group applies. For an ad-hoc shell in an already-running session: `sg render -c '<cmd>'`.
4. **Source the oneAPI env** in every shell that builds or runs SYCL:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
   Re-source after the render-group re-login — the new shell doesn't inherit it.
5. **Verify the compiler:**
   ```bash
   icpx --version   # expect: Intel(R) oneAPI DPC++/C++ Compiler 2025.x
   ```
6. **Verify the device:**
   ```bash
   sycl-ls
   ```
   Expected: at least one `[level_zero:gpu]` line for the B70, shown as `Intel(R) Graphics [0xe223]`.

**AOT vs. JIT compilation note.** The `-fsycl-targets=intel_gpu_bmg_g21` ahead-of-time flag requires `ocloc` (Intel Graphics Compiler offline tool), which is packaged separately from the base `intel-oneapi-compiler-dpcpp-cpp` install. If `ocloc` is absent, `icpx` will error with `ocloc tool could not be found`. In that case, drop the `-fsycl-targets` flag entirely and let the compiler emit a fat binary that JIT-compiles to the device at first run:
```bash
icpx -fsycl -O2 src/_smoke_hello.cpp -o smoke_hello   # JIT fallback, no ocloc needed
```
JIT compilation is transparent at runtime and produces correct results; startup latency for the first kernel launch is slightly higher. To restore AOT: `sudo apt install intel-ocloc` (ships in the Ubuntu `universe` archive, source package `intel-compute-runtime`) and re-compile with `-fsycl-targets=intel_gpu_bmg_g21`.

## Using the intel/llvm nightly (BMG-G31 joint_matrix)

oneAPI 2025.3's `libsycl.so` is missing the BMG-G31 arch enum in
`get_matrix_combinations()`, so `joint_matrix` calls throw
`no matrix hardware on the target device` at runtime on the Arc Pro B70.
Use Intel's `intel/llvm` nightly until the fix lands in a released oneAPI.

Download and extract (once):
```bash
mkdir -p /tmp/intel-llvm-nightly && cd /tmp/intel-llvm-nightly
curl -LO https://github.com/intel/llvm/releases/download/nightly-2026-04-13/sycl_linux.tar.gz
tar -xzf sycl_linux.tar.gz --strip-components=0
```

Build + run with the nightly:
```bash
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  # oneAPI 2025.3 lib dir is kept on LD_LIBRARY_PATH only for libhwloc.so.15
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  cd sycl
  rm -rf build
  cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++ \
        -Dpybind11_DIR=$(../.venv-sycl/bin/python -m pybind11 --cmakedir) \
        -DVLLM_XPU_AOT_DEVICES=""
  cmake --build build
  cd ..
  .venv-sycl/bin/python -m pytest tests/sycl/ -v
'
```

Do NOT source `/opt/intel/oneapi/setvars.sh` when using the nightly — it clobbers
PATH back to the stock 2025.3 toolchain. The two-entry `LD_LIBRARY_PATH` above
is sufficient to pick up libhwloc while keeping the nightly's libsycl first.

## Python env (after Task 3)
`python3 -m venv .venv-sycl && source .venv-sycl/bin/activate && pip install -r sycl/requirements.txt`

## Build (after Task 6)
`cd sycl && cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=icpx && cmake --build build`

## Run tests (after Task 9+)
`source .venv-sycl/bin/activate && python -m pytest tests/sycl/ -v`

## Run benchmark (after Task 14)
`python scripts/bench_sycl_spec.py`
