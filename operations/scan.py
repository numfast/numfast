"""High-level scan (prefix sum) operation."""

import numpy as np
from numfast.Runtime import Runtime
from numfast.Compute import register_all


def scan(data: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Inclusive prefix sum.

    Args:
        data: 1D float64 array
        use_gpu: use WebGPU if True

    Returns:
        prefix_sum: 1D float64 array, same length as data
    """
    from numfast.Runtime._lib.Drivers.WebGPU import WebGpuDriver

    driver = WebGpuDriver() if use_gpu else None
    rt = Runtime(driver=driver)
    register_all(rt)

    n = len(data)
    if n <= 64:
        jobs = [
            {"op": "ScanLocal", "inputs": ["data"], "params": {},
             "out": ["scan", "_bsum"]},
        ]
    else:
        jobs = [
            {"op": "ScanLocal", "inputs": ["data"], "params": {},
             "out": ["scan_local", "block_sum"]},
            {"op": "ScanTotals", "inputs": ["block_sum"], "params": {},
             "out": "block_prefix"},
            {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"], "params": {},
             "out": "scan"},
        ]

    tasks = rt.compile(jobs)
    rt.execute(tasks, {"data": data})
    result = rt.driver.resolve_output("scan")

    if driver:
        driver.release()

    return result
