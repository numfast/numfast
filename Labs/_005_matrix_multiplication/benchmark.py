"""Micro Benchmark - Tiled Matrix Multiplication.

Measures CPU vs GPU performance with phase breakdown.

Usage:
    python -m Labs._005_matrix_multiplication.benchmark
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Runtime import Runtime
from Compute import register_all

SIZES = [64, 128, 256, 512]


def benchmark_cpu(M, N, K):
    """numpy matmul timing."""
    A = np.random.random(M * K).astype(np.float64)
    B = np.random.random(K * N).astype(np.float64)
    A_mat = A.reshape(M, K)
    B_mat = B.reshape(K, N)

    # Warmup
    _ = A_mat @ B_mat

    t0 = time.perf_counter()
    _ = A_mat @ B_mat
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms


def benchmark_gpu(M, N, K):
    """GPU matmul with phase breakdown."""
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver

    A = np.random.random(M * K).astype(np.float64)
    B = np.random.random(K * N).astype(np.float64)

    jobs = [{
        "op": "MatMul",
        "inputs": ["A", "B"],
        "params": {"M": M, "N": N, "K": K},
        "out": "C",
    }]

    # Compilation
    t0 = time.perf_counter()
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    tasks = rt.compile(jobs)
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000

    # Warmup
    rt.execute(tasks, {"A": A, "B": B})
    _ = rt.driver.resolve_output("C")

    # Timed execution
    t2 = time.perf_counter()
    rt.execute(tasks, {"A": A, "B": B})
    t3 = time.perf_counter()
    exec_ms = (t3 - t2) * 1000

    # Readback
    t4 = time.perf_counter()
    result = rt.driver.resolve_output("C")
    t5 = time.perf_counter()
    readback_ms = (t5 - t4) * 1000

    # Verify
    expected = (A.reshape(M, K) @ B.reshape(K, N)).flatten()
    err = float(np.abs(result - expected).max()) if result is not None else -1

    rt.driver.release()

    return compile_ms, exec_ms, readback_ms, err


# --- main ---

print()
print("=" * 90)
print("  Matrix Multiplication - Micro Benchmark")
print("  Tile: 16x16, Workgroup: 16x16x1")
print("=" * 90)
print(f"  {'Matrix':>14} {'numpy':>8} {'CPU':>8} {'GPU tot':>8} "
      f"{'Compile':>8} {'Exec':>8} {'Readback':>8} {'Error':>10}")
print(f"  {'-'*14:>14} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} "
      f"{'-'*8:>8} {'-':>8} {'-'*8:>8} {'-'*10:>10}")

for N in SIZES:
    M = N
    K = N

    numpy_ms = benchmark_cpu(M, N, K)

    # CPU (NumFast triple loop - slower reference)
    A = np.random.random(M * K).astype(np.float64)
    B = np.random.random(K * N).astype(np.float64)
    rt_cpu = Runtime()
    register_all(rt_cpu)
    jobs = [{"op": "MatMul", "inputs": ["A", "B"], "params": {"M": M, "N": N, "K": K}, "out": "C"}]
    t0 = time.perf_counter()
    rt_cpu.execute(rt_cpu.compile(jobs), {"A": A, "B": B})
    t1 = time.perf_counter()
    cpu_ms = (t1 - t0) * 1000

    # GPU
    compile_ms = exec_ms = readback_ms = err = None
    gpu_total = None
    try:
        compile_ms, exec_ms, readback_ms, err = benchmark_gpu(M, N, K)
        gpu_total = compile_ms + exec_ms + readback_ms
    except Exception:
        pass

    def _fmt(val):
        return f"{val:.2f}" if val is not None else "N/A"

    def _fmt_err(val):
        return f"{val:.2e}" if val is not None else "N/A"

    print(f"  {M:>4}x{N:>4}{K:>4}  {numpy_ms:>8.3f} {_fmt(cpu_ms):>8} "
          f"{_fmt(gpu_total):>8} {_fmt(compile_ms):>8} "
          f"{_fmt(exec_ms):>8} {_fmt(readback_ms):>8} {_fmt_err(err):>10}")

print("=" * 90)
print()
print("  Notes:")
print("  - numpy: np.dot (optimized BLAS, double precision)")
print("  - CPU: NumFast naive triple loop (double precision)")
print("  - GPU: NumFast tiled MatMul via WebGPU (float32)")
print("  - Compile includes WebGPU device init + shader compilation")
print("  - Error: max abs diff vs numpy reference")
print()
