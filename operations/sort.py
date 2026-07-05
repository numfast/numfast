"""High-level sort operation."""
import math
import numpy as np
from numfast.Runtime import Runtime
from numfast.Compute import register_all


def sort(data: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Sort array using bitonic sort (power of 2 size).

    Creates Runtime ONCE and reuses it across all (stage, substage) iterations.
    Each iteration compiles a single job with current params and executes it.
    Output from previous iteration feeds as input to next.

    Args:
        data: 1D float64 array (power of 2)
        use_gpu: use WebGPU if True

    Returns:
        sorted array
    """
    from numfast.Runtime._lib.Drivers.WebGPU import WebGpuDriver

    n = len(data)
    log2n = int(math.log2(n))
    assert 2**log2n == n, f"Size {n} must be power of 2"

    # --- Runtime created ONCE ---
    driver = WebGpuDriver() if use_gpu else None
    rt = Runtime(driver=driver)
    register_all(rt)

    d = data.copy()
    for stage in range(1, log2n + 1):
        for substage in range(stage, 0, -1):
            jobs = [{
                "op": "Sort",
                "inputs": ["data"],
                "params": {"stage": stage, "substage": substage},
                "out": "data",
            }]
            rt.execute(rt.compile(jobs), {"data": d})
            d = rt.driver.resolve_output("data")

    if driver:
        driver.release()

    return d
