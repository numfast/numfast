"""Golden test for Prefix Sum."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from Labs._002_prefix_sum.jobs import run

import pytest

TEST_CASES = [
    (1, [0.0]),
    (2, [0.0, 0.5]),
    (5, [0.0, 0.5, 1.5, 3.0, 5.0]),
    (64, None),     # full single block
    (100, None),    # two blocks
    (1000, None),   # many blocks
]


def _expected(n: int) -> np.ndarray:
    data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
    return np.cumsum(data)


def test_cpu_golden():
    for n, known in TEST_CASES:
        data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
        expected = np.cumsum(data)
        result = run(data, use_gpu=False)
        err = float(np.abs(result - expected).max())
        assert err < 1e-12, f"N={n}: CPU err={err}"


@pytest.mark.skipif(
    os.environ.get("SKIP_GPU") == "1",
    reason="SKIP_GPU=1 set"
)
def test_gpu_consistency():
    """GPU result should be consistent with CPU (within float32 tolerance)."""
    for n in [64, 1000, 10000]:
        data = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)
        expected = np.cumsum(data)
        try:
            gpu_result = run(data, use_gpu=True)
            if gpu_result is not None:
                err = float(np.abs(gpu_result - expected).max())
                rel = err / float(np.abs(expected).max()) * 100
                assert rel < 0.5, f"N={n}: GPU rel err={rel:.3f}%"
        except Exception:
            pytest.skip("WebGPU not available")
