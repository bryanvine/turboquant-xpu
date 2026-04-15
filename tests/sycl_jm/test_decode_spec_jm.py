"""Correctness: SYCL JM kernel matches numpy reference.

Subprocess-bridged — child loads the nightly-built .so; parent only orchestrates.

Phase (a) matrix: 2 shapes × 2 modes × k8v4 only = 4 test cases.
"""
from __future__ import annotations

import pytest


def _req(shape: str, causal: int, seed: int = 42) -> dict:
    # For causal, use cached_len = seqlen - N_spec (the spec-verify pattern).
    # cached_len_adj = -N_spec is encoded here generically:
    #   cached_len = seqlen + cached_len_adj
    n_spec = {"small": 4, "poc": 8}[shape]
    return {
        "mode": "correctness",
        "shape": shape,
        "preset": "k8v4",
        "seed": seed,
        "causal": causal,
        "cached_len_adj": -n_spec if causal else 0,
    }


@pytest.mark.parametrize("shape", ["small", "poc"])
@pytest.mark.parametrize("causal", [0, 1])
def test_jm_matches_reference(shape, causal, run_child_fixture):
    result = run_child_fixture(_req(shape, causal))
    assert "pass" in result.parsed, (
        f"child returned malformed JSON.\n"
        f"rc={result.returncode}\nstdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    assert result.parsed["pass"], (
        f"correctness failed for shape={shape} causal={causal}: "
        f"{result.parsed}\nstderr={result.stderr}"
    )
