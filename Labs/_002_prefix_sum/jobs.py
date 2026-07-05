"""Prefix Sum — запуск CPU/GPU."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Compute import register_all


def run(data: np.ndarray, use_gpu=False) -> np.ndarray:
    """Compute inclusive prefix sum.

    Args:
        data: 1D float64 array
        use_gpu: if True, use WebGPU driver

    Returns:
        prefix_sum: 1D float64 array
    """
    driver = WebGpuDriver() if use_gpu else None
    rt = Runtime(driver=driver)
    register_all(rt)

    n = len(data)
    if n <= 64:
        # Single block — ScanLocal produces 2 outputs, we keep both
        jobs = [
            {"op": "ScanLocal", "inputs": ["data"], "params": {},
             "out": ["scan", "_bsum"]},
        ]
    else:
        # Multi-block
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


if __name__ == "__main__":
    import time

    for N in [64, 1000, 10000, 100000]:
        data = np.array([float(i) * 0.5 for i in range(N)], dtype=np.float64)
        expected = np.cumsum(data)

        t0 = time.time()
        cpu_result = run(data, use_gpu=False)
        t1 = time.time()
        cpu_err = float(np.abs(cpu_result - expected).max())

        gpu_err = -1
        gpu_ms = -1
        try:
            t2 = time.time()
            gpu_result = run(data, use_gpu=True)
            t3 = time.time()
            gpu_err = float(np.abs(gpu_result - expected).max())
            gpu_ms = (t3 - t2) * 1000
        except Exception as e:
            pass

        print(f"N={N:>6}: CPU err={cpu_err:.2e} ({(t1-t0)*1000:.1f}ms)",
              end="")
        if gpu_err >= 0:
            rel = gpu_err / float(np.abs(expected).max()) * 100
            print(f" | GPU err={gpu_err:.2e} ({rel:.3f}%) {gpu_ms:.0f}ms",
                  end="")
        print()
