"""Shared helpers for SYCL JM subprocess-bridged pytest.

The JM .so cannot load in the same process as torch-XPU. Every test spawns a
child process with the nightly LD path, sends a JSON request, parses JSON from
stdout. `run_child()` is the common entry point.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(REPO, ".venv-jm", "bin", "python")
CHILD = os.path.join(REPO, "scripts", "harness", "bench_jm_child.py")

_NIGHTLY_PREFIX = (
    "export PATH=/tmp/intel-llvm-nightly/bin:$PATH; "
    "export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; "
)


@dataclass
class ChildResult:
    returncode: int
    stdout: str
    stderr: str
    parsed: dict


@pytest.fixture
def run_child_fixture():
    """Pytest fixture form — use in tests as `run_child_fixture(req)`."""
    return run_child


def run_child(req: dict, timeout: int = 180) -> ChildResult:
    cmd = _NIGHTLY_PREFIX + f"{PY} {CHILD}"
    proc = subprocess.run(
        ["sg", "render", "-c", cmd],
        input=json.dumps(req),
        capture_output=True, text=True, cwd=REPO, timeout=timeout,
    )
    # Child prints one JSON line on stdout; tolerate prefix noise.
    json_line = None
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
    parsed = json.loads(json_line) if json_line else {}
    return ChildResult(proc.returncode, proc.stdout, proc.stderr, parsed)
