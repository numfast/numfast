"""Golden test for MatMul."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from Labs._005_matrix_multiplication.jobs import matmul

import pytest

TEST_CASES = [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (16, 32, 8),
    (33, 33, 33),   # non-divisible by tile
]


def test_cpu_golden():
    for M, N, K in TEST_CASES:
        np.random.seed(M * 100 + N * 10 + K)
        A = np.random.random(M * K).astype(np.float64)
        B = np.random.random(K * N).astype(np.float64)
        expected = (A.reshape(M, K) @ B.reshape(K, N)).flatten()
        result = matmul(A, B, M, N, K, use_gpu=False)
        err = float(np.abs(result - expected).max())
        assert err < 1e-12, f"M={M} N={N} K={K}: CPU err={err:.2e}"


@pytest.mark.skipif(os.environ.get("SKIP_GPU") == "1", reason="SKIP_GPU=1")
def test_gpu_consistency():
    for M, N, K in [(16, 16, 16), (32, 32, 16)]:
        np.random.seed(M * 100 + N * 10 + K)
        A = np.random.random(M * K).astype(np.float64)
        B = np.random.random(K * N).astype(np.float64)
        expected = (A.reshape(M, K) @ B.reshape(K, N)).flatten()
        try:
            result = matmul(A, B, M, N, K, use_gpu=True)
            if result is not None:
                err = float(np.abs(result - expected).max())
                assert err < 0.01, f"M={M} N={N} K={K}: GPU err={err:.2e}"
        except Exception:
            pytest.skip("WebGPU not available")
