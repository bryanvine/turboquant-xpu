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

## Python env (after Task 3)
`python3 -m venv .venv-sycl && source .venv-sycl/bin/activate && pip install -r sycl/requirements.txt`

## Build (after Task 6)
`cd sycl && cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=icpx && cmake --build build`

## Run tests (after Task 9+)
`source .venv-sycl/bin/activate && python -m pytest tests/sycl/ -v`

## Run benchmark (after Task 14)
`python scripts/bench_sycl_spec.py`
