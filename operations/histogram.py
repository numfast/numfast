"""High-level histogram operation."""

import numpy as np
from numfast.Runtime import Runtime
from numfast.Compute import register_all


def histogram(data: np.ndarray, bins: int = 10,
              min_val: float = None, max_val: float = None,
              use_gpu: bool = False) -> np.ndarray:
    """Compute histogram of data.

    Args:
        data: 1D float64 array
        bins: number of bins
        min_val: minimum value (auto from data if None)
        max_val: maximum value (auto from data if None)
        use_gpu: use WebGPU if True

    Returns:
        histogram: 1D float64 array of bin counts
    """
    from numfast.Runtime._lib.Drivers.WebGPU import WebGpuDriver

    if min_val is None:
        min_val = float(data.min())
    if max_val is None:
        max_val = float(data.max())

    driver = WebGpuDriver() if use_gpu else None
    rt = Runtime(driver=driver)
    register_all(rt)

    jobs = [{
        "op": "Histogram",
        "inputs": ["data"],
        "params": {"min_val": min_val, "max_val": max_val, "num_bins": bins},
        "out": "bins",
    }]
    tasks = rt.compile(jobs)
    rt.execute(tasks, {"data": data})
    result = rt.driver.resolve_output("bins")

    if driver:
        driver.release()

    return result
