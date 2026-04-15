# SYCL JM phase (a) per-task PoC-causal timings

Captured during implementation at PoC shape (N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, causal, cached_len=8184, k8v4). These were ephemeral development measurements emitted by each implementer subagent's bench mode after its respective kernel change. Archived here so the phase (a) writeup's "what moved the needle" analysis has citable numbers.

Each measurement is one warmup=3, n_timed=10 run via `scripts/harness/bench_jm_child.py` at the listed commit.

| task | kernel state | commit | ms_per_iter | max_abs_err |
|---|---|---|---:|---:|
| 6 | scalar split-KV (no DPAS) | 7a0efac | 223.3 | 5.1e-08 |
| 7 | Q·Kᵀ DPAS + scalar P·V | 53fb11d | 244.3 | 1.18e-06 |
| 8 | Q·Kᵀ + P·V DPAS (full) | f9c4292 | 96.9  | 4.7e-06 |
| 9 | same as Task 8; warmup=5 n_timed=20 | d026ce6 | 96.921 | — |

## Observations

- Task 6 → Task 7 regression (223 → 244 ms): adding DPAS Q·Kᵀ replaced an 8192-iteration scalar Q·K inner loop, but the lane-0 serial SLM fill pattern (Q, K, per-d_slice b_tile pre-transpose) and the 16 SG barriers per KV block added enough overhead to slightly net-regress.
- Task 7 → Task 8 speedup (244 → 97 ms = 2.5×): adding DPAS P·V replaced a 1024-iteration scalar `acc[n][d] += p * v[d]` inner loop per (n, KV block). The DPAS savings dominate despite the added acc_scalar round-trip through SLM.
- Task 8 → Task 9 (96.9 → 96.9 ms, different warmup/timed): consistent measurement, confirming the 97 ms figure is stable at the 1-2% level.

## Source

These measurements are reported in the subagent execution logs of the superpowers:subagent-driven-development session that produced phase (a). The logs live in the session transcript, not in version control. The final Task 9 measurement (96.921 ms) is separately archived at `docs/tuning/sycl_jm_bench_2026-04-15.txt`.
