"""Benchmark: elementwise GPU ops (Runtime + WebGpuDriver) vs NumPy."""

import time

import numpy as np

import numfast  # noqa: F401  (adds src paths so Runtime/Compute are importable)

from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime import Runtime as RuntimeClass
from Compute import register_all

SIZES = [100_000, 1_000_000, 10_000_000]

# func code from Compute Map kernel
UNARY = {"sin": 0, "cos": 1, "exp": 2, "sqrt": 3, "log": 4, "abs": 5, "neg": 6, "square": 7}

# op code from Compute MapBinary kernel
BINARY = {"add": 0, "sub": 1, "mul": 2, "truediv": 3}

_UNARY_NP = {
    "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt,
    "log": np.log, "abs": np.abs, "neg": np.negative, "square": np.square,
}

_BINARY_NP = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "truediv": np.divide,
}


def _print_mem(n, is_binary, label):
    buffers = 2 if is_binary else 1
    gpu_mem_mb = (n * 4 * (buffers + 1)) / 1e6
    numpy_mem_mb = (n * 4 * (buffers + 1)) / 1e6
    print(f"  {label} memory (n={n}): gpu {gpu_mem_mb:.1f} MB, numpy {numpy_mem_mb:.1f} MB")
    return gpu_mem_mb, numpy_mem_mb


def _autotune(elapsed_run, lo=3, hi=200, budget=0.15):
    return max(lo, min(hi, int(budget / max(elapsed_run, 1e-6))))


def run() -> dict:
    rng = np.random.RandomState(42)

    driver = WebGpuDriver()
    rt = RuntimeClass(driver=driver)
    register_all(rt)

    validation_failed = False
    results = {}
    total_elapsed = 0.0
    total_elements = 0

    print("=" * 60)
    print("ELEMENTWISE BENCHMARK (GPU vs NumPy, float32)")
    print("  NOTE: tasks > 4,194,240 elements are auto-split into chunks by ExecutionScheduler (ADR-012)")
    print("=" * 60)

    try:
        for op_name, code in list(UNARY.items()) + list(BINARY.items()):
            is_binary = op_name in BINARY
            kind = "MapBinary" if is_binary else "Map"
            params = {"op": code} if is_binary else {"func": code}
            results[op_name] = {}

            job = {"op": kind, "inputs": ["a", "b"] if is_binary else ["x"],
                   "params": params, "out": "y"}
            tasks = rt.compile([job])

            print(f"\n  op: {op_name}  (job: {kind} code={code})")

            for n in SIZES:
                x = rng.randn(n) * 5.0 + 30.0
                ref_fn = _UNARY_NP[op_name] if not is_binary else _BINARY_NP[op_name]

                if is_binary:
                    y = rng.randn(n) * 5.0 + 30.0
                    source = {"a": x, "b": y}
                    ref_arr = ref_fn(x.astype(np.float32), y.astype(np.float32))
                else:
                    source = {"x": x}
                    ref_arr = ref_fn(x.astype(np.float32))

                # Warmup: 1 GPU + 1 numpy
                rt.execute(tasks, source)
                ref_fn(x.astype(np.float32)) if not is_binary else ref_fn(
                    x.astype(np.float32), y.astype(np.float32))

                # t1 probe
                t0 = time.perf_counter()
                rt.execute(tasks, source)
                t1_gpu = time.perf_counter() - t0

                t0 = time.perf_counter()
                if is_binary:
                    ref_fn(x.astype(np.float32), y.astype(np.float32))
                else:
                    ref_fn(x.astype(np.float32))
                t1_np = time.perf_counter() - t0

                n_runs = _autotune(t1_gpu)

                # GPU timed loop
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    rt.execute(tasks, source)
                gpu_elapsed = time.perf_counter() - t0
                gpu_arr = np.asarray(rt.driver.resolve_output("y"))
                gpu_ms = gpu_elapsed / n_runs * 1000

                # NumPy timed loop
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    if is_binary:
                        ref_fn(x.astype(np.float32), y.astype(np.float32))
                    else:
                        ref_fn(x.astype(np.float32))
                np_elapsed = time.perf_counter() - t0
                numpy_ms = np_elapsed / n_runs * 1000

                total_elapsed += gpu_elapsed
                total_elements += n * n_runs

                speedup = numpy_ms / gpu_ms if gpu_ms > 0 else 0.0
                melem = (n * n_runs / gpu_elapsed) / 1e6

                # Validation
                max_diff = float(np.max(np.abs(gpu_arr - ref_arr)))
                tol = 1e-3 * max(1.0, float(np.max(np.abs(ref_arr))))
                ok = max_diff <= tol
                if not ok:
                    validation_failed = True
                print(f"  {op_name:8s} n={n:<10,} gpu {gpu_ms:8.3f} ms  "
                      f"numpy {numpy_ms:8.3f} ms  speedup {speedup:5.1f}x  "
                      f"{melem:7.1f} M elem/s")
                if ok:
                    print(f"    validation: OK (max_diff={max_diff:.3e})")
                else:
                    print(f"    VALIDATION FAIL (max_diff={max_diff:.3e} > tol={tol:.3e})")
                _print_mem(n, is_binary, op_name)

                results[op_name][n] = {
                    "gpu_ms": gpu_ms,
                    "numpy_ms": numpy_ms,
                    "speedup": speedup,
                    "melem_per_sec": melem,
                }
    finally:
        driver.release()

    print()
    return {
        "benchmark": "elementwise",
        "time_sec": total_elapsed,
        "bars_per_sec": total_elements / total_elapsed if total_elapsed > 0 else 0,
        "results": results,
        "validation_failed": validation_failed,
    }


if __name__ == "__main__":
    run()
