"""Benchmark: Reduce (sum) GPU vs NumPy."""

import time

import numpy as np

import numfast  # noqa: F401  (adds src paths so Runtime/Compute are importable)

from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime import Runtime as RuntimeClass
from Compute import register_all

SIZES = [100_000, 1_000_000]


def _autotune(elapsed_run, lo=3, hi=200, budget=0.15):
    return max(lo, min(hi, int(budget / max(elapsed_run, 1e-6))))


def run() -> dict:
    rng = np.random.RandomState(42)

    driver = WebGpuDriver()
    rt = RuntimeClass(driver=driver)
    register_all(rt)

    job = {"op": "Reduce", "inputs": ["x"], "params": {}, "out": "y"}
    tasks = rt.compile([job])

    validation_failed = False
    results = {}
    total_elapsed = 0.0
    total_elements = 0

    print("=" * 60)
    print("REDUCE BENCHMARK (sum, GPU vs NumPy)")
    print("  NOTE: dispatch limit 65535 groups x 64 elems = ~4.19M elements per call")
    print("=" * 60)

    try:
        for n in SIZES:
            x = rng.randn(n) * 100.0 + 1000.0
            source = {"x": x}
            np_val = float(np.sum(x))

            # Warmup: 1 GPU + 1 numpy
            rt.execute(tasks, source)
            np.sum(x)

            # t1 probe
            t0 = time.perf_counter()
            rt.execute(tasks, source)
            t1_gpu = time.perf_counter() - t0

            t0 = time.perf_counter()
            np.sum(x)
            t1_np = time.perf_counter() - t0

            n_runs = _autotune(t1_gpu)

            # GPU timed loop
            t0 = time.perf_counter()
            for _ in range(n_runs):
                rt.execute(tasks, source)
            gpu_elapsed = time.perf_counter() - t0
            gpu_arr = np.asarray(rt.driver.resolve_output("y")); gpu_val = float(np.sum(gpu_arr))
            gpu_ms = gpu_elapsed / n_runs * 1000

            # NumPy timed loop
            t0 = time.perf_counter()
            for _ in range(n_runs):
                np.sum(x)
            np_elapsed = time.perf_counter() - t0
            numpy_ms = np_elapsed / n_runs * 1000

            total_elapsed += gpu_elapsed
            total_elements += n * n_runs

            speedup = numpy_ms / gpu_ms if gpu_ms > 0 else 0.0
            melem = (n * n_runs / gpu_elapsed) / 1e6

            # Validation
            rel_diff = abs(gpu_val - np_val) / max(1.0, abs(np_val))
            ok = rel_diff <= 1e-2
            if not ok:
                validation_failed = True
            print(f"  reduce n={n:<10,} gpu {gpu_ms:8.3f} ms  "
                  f"numpy {numpy_ms:8.3f} ms  speedup {speedup:5.1f}x  "
                  f"{melem:7.1f} M elem/s")
            if ok:
                print(f"    validation: OK (rel_diff={rel_diff:.3e})")
            else:
                print(f"    VALIDATION FAIL (rel_diff={rel_diff:.3e})")
            gpu_mem_mb = (n * 4 * 2) / 1e6  # input + output
            numpy_mem_mb = (n * 8) / 1e6  # float64 input
            print(f"    memory (n={n}): gpu {gpu_mem_mb:.1f} MB, numpy {numpy_mem_mb:.1f} MB")

            results[n] = {
                "gpu_ms": gpu_ms,
                "numpy_ms": numpy_ms,
                "speedup": speedup,
                "melem_per_sec": melem,
            }
    finally:
        driver.release()

    print()
    return {
        "benchmark": "reduce",
        "time_sec": total_elapsed,
        "bars_per_sec": total_elements / total_elapsed if total_elapsed > 0 else 0,
        "results": results,
        "validation_failed": validation_failed,
    }


if __name__ == "__main__":
    run()
