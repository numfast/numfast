"""Micro Benchmark — Prefix Sum.

Measures CPU vs GPU performance with phase breakdown.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Runtime import Runtime
from Compute import register_all

SIZES = [64, 1000, 10000, 100000, 1_000_000]

# ---------- helpers ----------

def _make_data(n):
    return np.array([float(i) * 0.5 for i in range(n)], dtype=np.float64)


def _make_jobs(n):
    if n <= 64:
        return [{"op": "ScanLocal", "inputs": ["data"], "params": {},
                  "out": ["scan", "_bsum"]}]
    return [
        {"op": "ScanLocal", "inputs": ["data"], "params": {},
         "out": ["scan_local", "block_sum"]},
        {"op": "ScanTotals", "inputs": ["block_sum"], "params": {},
         "out": "block_prefix"},
        {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"], "params": {},
         "out": "scan"},
    ]


def benchmark_cpu(n):
    data = _make_data(n)
    t0 = time.perf_counter()
    _ = np.cumsum(data)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms


def benchmark_gpu(n):
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver

    data = _make_data(n)
    jobs = _make_jobs(n)

    # --- Compilation ---
    t0 = time.perf_counter()
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    tasks = rt.compile(jobs)
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000

    # --- Warmup ---
    rt.execute(tasks, {"data": data})
    _ = rt.driver.resolve_output("scan")

    # --- Timed execution ---
    t2 = time.perf_counter()
    rt.execute(tasks, {"data": data})
    t3 = time.perf_counter()
    exec_ms = (t3 - t2) * 1000

    # --- Readback ---
    t4 = time.perf_counter()
    result = rt.driver.resolve_output("scan")
    t5 = time.perf_counter()
    readback_ms = (t5 - t4) * 1000

    # Verify correctness
    expected = np.cumsum(data)
    err = float(np.abs(result - expected).max()) if result is not None else -1

    rt.driver.release()

    return compile_ms, exec_ms, readback_ms, err


# ---------- main ----------

print()
print("=" * 80)
print("  Prefix Sum — Micro Benchmark")
print("=" * 80)
print(f"  {'Size':>8} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>8} "
      f"{'Compile':>8} {'Exec':>8} {'Readback':>8} {'Error':>10}")
print(f"  {'-'*8:>8} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8} "
      f"{'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*10:>10}")

for n in SIZES:
    cpu_ms = benchmark_cpu(n)

    gpu_total = None
    compile_ms = exec_ms = readback_ms = err = None
    try:
        compile_ms, exec_ms, readback_ms, err = benchmark_gpu(n)
        gpu_total = compile_ms + exec_ms + readback_ms
    except Exception as e:
        pass

    speedup = cpu_ms / gpu_total if gpu_total and gpu_total > 0 else None

    err_str = f"{err:.2e}" if err is not None else "N/A"
    gpu_total_str = f"{gpu_total:.2f}" if gpu_total is not None else "N/A"
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
    compile_str = f"{compile_ms:.2f}" if compile_ms is not None else "N/A"
    exec_str = f"{exec_ms:.2f}" if exec_ms is not None else "N/A"
    readback_str = f"{readback_ms:.3f}" if readback_ms is not None else "N/A"

    print(f"  {n:>8} {cpu_ms:>10.3f} {gpu_total_str:>10} {speedup_str:>8} "
          f"{compile_str:>8} {exec_str:>8} {readback_str:>8} {err_str:>10}")

print("=" * 80)

# Summary
print()
print("  Notes:")
print("  - GPU includes first-run compilation (wgsl -> shader module)")
print("  - Warmup pass executed before timed pass")
print("  - CPU: numpy.cumsum (float64)")
print("  - GPU: 3-phase scan (ScanLocal + ScanTotals + ScanFinal)")
print("  - Error: max abs difference vs numpy.cumsum (float64 ref)")
print()
