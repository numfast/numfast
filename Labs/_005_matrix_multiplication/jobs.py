"""Matrix Multiplication — запуск CPU/GPU."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Compute import register_all


def run(A: np.ndarray, B: np.ndarray, use_gpu=False) -> np.ndarray:
    """C = A × B.

    Args:
        A: 1D flatten, M×K
        B: 1D flatten, K×N
        use_gpu: if True, use WebGPU driver

    Returns:
        C: 1D flatten, M×N
    """
    M = int(np.sqrt(len(A))) if int(np.sqrt(len(A))) ** 2 == len(A) else 0
    # Infer dimensions from array sizes
    K = len(A)  # placeholder — actual M,N,K from params
    driver = WebGpuDriver() if use_gpu else None
    rt = Runtime(driver=driver)
    register_all(rt)
    # Return rt for external dimension setting
    return rt


def matmul(A, B, M, N, K, use_gpu=False):
    """Compute C = A × B with given dimensions."""
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
    rt.execute(tasks, {"A": A, "B": B})
    result = rt.driver.resolve_output("C")
    if driver:
        driver.release()
    return result


if __name__ == "__main__":
    import time

    sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256),
             (512, 512, 512), (1024, 1024, 1024)]

    for M, N, K in sizes:
        A = np.random.random(M * K).astype(np.float64)
        B = np.random.random(K * N).astype(np.float64)

        # CPU (numpy reference)
        expected = A.reshape(M, K) @ B.reshape(K, N)
        t0 = time.time()
        _ = A.reshape(M, K) @ B.reshape(K, N)
        t1 = time.time()
        cpu_ms = (t1 - t0) * 1000

        # CPU (NumFast)
        t2 = time.time()
        cpu_result = matmul(A, B, M, N, K, use_gpu=False)
        t3 = time.time()
        nf_cpu_ms = (t3 - t2) * 1000
        cpu_err = float(np.abs(cpu_result - expected.flatten()).max())

        # GPU
        gpu_ok = False
        gpu_err = -1
        gpu_ms = -1
        try:
            t4 = time.time()
            gpu_result = matmul(A, B, M, N, K, use_gpu=True)
            t5 = time.time()
            gpu_ms = (t5 - t4) * 1000
            if gpu_result is not None:
                gpu_err = float(np.abs(gpu_result - expected.flatten()).max())
                gpu_ok = True
        except Exception:
            pass

        print(f"M={M:>4} N={N:>4} K={K:>4}: "
              f"numpy={cpu_ms:.2f}ms "
              f"CPU={nf_cpu_ms:.2f}ms err={cpu_err:.2e}",
              end="")
        if gpu_ok:
            print(f" | GPU={gpu_ms:.2f}ms err={gpu_err:.2e}", end="")
        print()
