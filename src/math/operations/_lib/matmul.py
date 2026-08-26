"""High-level matrix multiplication operation."""

import numpy as np
from Runtime import Runtime
from Compute import register_all


def matmul(A: np.ndarray, B: np.ndarray, M: int = None, N: int = None, K: int = None,
           use_gpu: bool = False) -> np.ndarray:
    """C = A × B.

    Args:
        A: 1D flatten array, M×K (or 2D)
        B: 1D flatten array, K×N (or 2D)
        M, N, K: dimensions (inferred if None and inputs are 2D)
        use_gpu: use WebGPU if True

    Returns:
        C: 1D flatten array, M×N
    """
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver

    # Handle 2D inputs
    if A.ndim == 2 and B.ndim == 2:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Shape mismatch: A({A.shape}) @ B({B.shape})"
        A_flat = A.flatten()
        B_flat = B.flatten()
    elif A.ndim == 1 and B.ndim == 1:
        assert M is not None and N is not None and K is not None
        A_flat = A
        B_flat = B
    else:
        raise ValueError(f"Input shapes: A{A.shape}, B{B.shape}")

    driver = WebGpuDriver() if use_gpu else None
    rt = Runtime(driver=driver)
    register_all(rt)

    jobs = [{
        "op": "MatMul",
        "inputs": ["A", "B"],
        "params": {"M": M, "N": N, "K": K},
        "out": "C",
    }]
    tasks = rt.compile(jobs)
    rt.execute(tasks, {"A": A_flat, "B": B_flat})
    result = rt.driver.resolve_output("C")

    if driver:
        driver.release()

    return result
