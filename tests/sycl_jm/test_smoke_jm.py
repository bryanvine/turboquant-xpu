"""joint_matrix smoke test — subprocess-invoked.

Builds and runs _smoke_jm_matmul in the nightly env. If this passes, phase (a)
clears Task 2's exit gate.
"""
import os
import subprocess
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NIGHTLY_PREFIX = (
    "export PATH=/tmp/intel-llvm-nightly/bin:$PATH; "
    "export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; "
)


def test_joint_matrix_smoke():
    cmd = (
        NIGHTLY_PREFIX +
        "clang++ -fsycl -O2 sycl/jm/src/_smoke_jm_matmul.cpp -o /tmp/_smoke_jm_matmul && "
        "/tmp/_smoke_jm_matmul; "
        "rc=$?; rm -f /tmp/_smoke_jm_matmul; exit $rc"
    )
    result = subprocess.run(
        ["sg", "render", "-c", cmd],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"smoke failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "max_err = 0" in result.stdout or "max_err = " in result.stdout
    # Allow any value < 0.1; the binary itself exits 1 if max_err >= 0.1.
