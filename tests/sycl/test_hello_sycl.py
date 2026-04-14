"""Hello-world SYCL kernel smoke test.

This test exists to catch environmental problems (module not built,
device not visible, numpy<->SYCL tensor conversion broken) BEFORE any
TurboQuant math is in the picture.
"""
import numpy as np
import pytest


def test_hello_sycl_identity():
    import turboquant_xpu_sycl as tq_sycl
    x = np.arange(256, dtype=np.float32) + 0.5
    y = tq_sycl.hello_identity(x)
    np.testing.assert_array_equal(x, y)


def test_hello_sycl_scale_by_two():
    import turboquant_xpu_sycl as tq_sycl
    x = np.arange(256, dtype=np.float32)
    y = tq_sycl.hello_scale(x, 2.0)
    np.testing.assert_allclose(y, 2.0 * x, atol=1e-6)
