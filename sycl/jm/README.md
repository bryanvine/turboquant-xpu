# SYCL `joint_matrix` + split-KV TurboQuant decode (phase a)

Portable-API SYCL kernel for the TurboQuant speculative-decode verification path on
Arc Pro B70 (BMG-G31). Built with the intel/llvm nightly 2026-04-13 at
`/tmp/intel-llvm-nightly/` because stock oneAPI 2025.3's `libsycl.so.8` doesn't have
BMG-G31 in `get_matrix_combinations()`. The `.so` is ABI-incompatible with torch-XPU,
so tests and the benchmark run the module inside a child process via the nightly
`LD_LIBRARY_PATH`. See `../../docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md`
for the full design; see `../../docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md`
for the implementation plan.
