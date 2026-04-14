# TurboQuant ESIMD PoC

Intel-Explicit-SIMD TurboQuant decode kernel targeting Arc Pro B70 (BMG-G31).
Built with stock oneAPI 2025.3 icpx, links against torch-XPU's libsycl ABI,
co-loads with torch in one process. See `../../docs/ESIMD_DESIGN.md` for the
design rationale and `../../docs/superpowers/plans/2026-04-14-esimd-tq-decode-poc.md`
for the implementation plan.
