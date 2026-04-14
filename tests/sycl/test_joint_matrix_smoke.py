import numpy as np


def test_joint_matrix_smoke_against_numpy():
    import turboquant_xpu_sycl as tq_sycl
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 16)).astype(np.float16)
    B = rng.standard_normal((16, 16)).astype(np.float16)
    C = tq_sycl.joint_matrix_smoke(A.view(np.uint16), B.view(np.uint16))
    ref = (A.astype(np.float32) @ B.astype(np.float32))
    np.testing.assert_allclose(C, ref, atol=1e-2, rtol=1e-2)
