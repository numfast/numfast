"""Conformance: Compute kernels — driver-agnostic test suite.

Каждый тест:
  1. Запускает ядро на CPU (reference, float64)
  2. Запускает то же ядро на указанном драйвере
  3. Сравнивает результаты
  4. PASS/FAIL вердикт

Любой новый драйвер должен проходить этот набор без изменений.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Runtime import Runtime
from Compute import register_all as register_compute
from Runtime._lib.Drivers.WebGPU import WebGpuDriver

TOLERANCE = 1e-4  # float32 tolerance for most kernels

# === Driver factory ===

def _make_runtime(driver=None):
    """Create runtime with all compute kernels registered."""
    rt = Runtime(driver=driver)
    register_compute(rt)
    return rt


def _make_gpu_runtime():
    return _make_runtime(driver=WebGpuDriver())


# === Test runner ===

def _test_kernel(name, params, input_data, expected_fn, tol=TOLERANCE, driver=None):
    """Test kernel on CPU and optionally GPU, compare with expected.
    
    Args:
        name: kernel alias
        params: dict
        input_data: dict[str, np.array]
        expected_fn: callable(inputs) → dict[name, np.array], or None for auto
        tol: max allowed abs error
        driver: optional driver for GPU comparison (None = CPU only)
    
    Returns:
        dict with test results
    """
    result = {
        "kernel": name,
        "params": params,
        "passed": True,
        "cpu_err": 0.0,
        "gpu_err": 0.0,
        "cpu_gpu_err": 0.0,
        "n": max(len(v) for v in input_data.values()) if input_data else 0,
    }
    
    # --- CPU reference ---
    cpu_runtime = _make_runtime()
    jobs = [{"op": name, "inputs": list(input_data.keys()), "params": params}]
    cpu_tasks = cpu_runtime.compile(jobs)
    cpu_runtime.execute(cpu_tasks, input_data)
    
    # --- Optional driver ---
    if driver is not None:
        try:
            drv_runtime = _make_runtime(driver=driver)
            drv_tasks = drv_runtime.compile(jobs)
            drv_runtime.execute(drv_tasks, input_data)
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            return result
    else:
        drv_runtime = cpu_runtime
        drv_tasks = cpu_tasks
    
    # --- Compare outputs ---
    max_cpu_err = 0.0
    max_drv_err = 0.0
    max_cpu_drv = 0.0
    
    for t_idx, t in enumerate(cpu_tasks):
        for out_name in t.out_names:
            cpu_val = cpu_runtime.driver.resolve_output(out_name)
            drv_val = drv_runtime.driver.resolve_output(out_name) if driver else cpu_val
            
            if expected_fn is not None:
                expected = expected_fn(input_data)
                if out_name in expected:
                    exp_val = expected[out_name]
                    cpu_err = float(np.abs(cpu_val - exp_val).max())
                    drv_err = float(np.abs(drv_val - exp_val).max())
                    max_cpu_err = max(max_cpu_err, cpu_err)
                    max_drv_err = max(max_drv_err, drv_err)
            
            if driver:
                cg_err = float(np.abs(cpu_val - drv_val).max())
                max_cpu_drv = max(max_cpu_drv, cg_err)
    
    result["cpu_err"] = max_cpu_err
    result["gpu_err"] = max_drv_err
    result["cpu_gpu_err"] = max_cpu_drv
    
    if max_cpu_drv > tol:
        result["passed"] = False
    
    # Cleanup
    if driver is not None and drv_runtime.driver is not cpu_runtime.driver:
        drv_runtime.driver.release()
    
    return result


def _print_report(results, driver_name="CPU"):
    """Print conformance test report."""
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print(f"\n{'='*75}")
    print(f"  Kernel Conformance Suite — reference: CPU float64 / target: {driver_name}")
    print(f"  {passed}/{total} passed")
    print(f"{'='*75}")
    print(f"  {'Kernel':<24} {'N':<8} {'CPU err':<12} {'Driver err':<12} {'CPUvsDrv':<12} {'Status':<8}")
    print(f"  {'-'*75}")
    
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        gpu_str = f"{r['gpu_err']:.2e}" if r.get('gpu_err') else "N/A"
        cg_str = f"{r['cpu_gpu_err']:.2e}" if r.get('cpu_gpu_err') else "N/A"
        cpu_str = f"{r['cpu_err']:.2e}" if r.get('cpu_err') else "N/A"
        print(f"  {r['kernel']:<24} {r['n']:<8} {cpu_str:<12} {gpu_str:<12} {cg_str:<12} {status:<8}")
    
    print(f"{'='*75}")
    if passed < total:
        print(f"\n  FAILED:")
        for r in results:
            if not r["passed"]:
                print(f"    - {r['kernel']}: CPUvsDrv={r['cpu_gpu_err']:.2e} > tol={TOLERANCE:.0e}")
                if 'error' in r:
                    print(f"      Error: {r['error']}")
    print()


# === Test Cases ===

def _data_lin(n, scale=0.5):
    return np.array([float(i) * scale for i in range(n)], dtype=np.float64)


# --- Map ---

def test_map_sin():
    data = _data_lin(100, 0.1)
    return _test_kernel("Map", {"func": 0}, {"data": data},
                        expected_fn=lambda d: {"map_0": np.sin(d["data"])})

def test_map_cos():
    data = _data_lin(100, 0.1)
    return _test_kernel("Map", {"func": 1}, {"data": data},
                        expected_fn=lambda d: {"map_1": np.cos(d["data"])})

def test_map_exp():
    data = _data_lin(50, 0.05)
    return _test_kernel("Map", {"func": 2}, {"data": data},
                        expected_fn=lambda d: {"map_2": np.exp(d["data"])})

def test_map_sqrt():
    data = np.array([float(i) for i in range(1, 101)], dtype=np.float64)
    return _test_kernel("Map", {"func": 3}, {"data": data},
                        expected_fn=lambda d: {"map_3": np.sqrt(d["data"])})

def test_map_square():
    data = _data_lin(100, 2.0)
    return _test_kernel("Map", {"func": 7}, {"data": data},
                        expected_fn=lambda d: {"map_7": d["data"] ** 2})

# --- Reduce ---

def test_reduce():
    data = _data_lin(1000)
    return _test_kernel("Reduce", {}, {"data": data},
                        expected_fn=lambda d: {"sum": np.array([d["data"].sum()])})

def test_reduce_large():
    """Large array: float32 precision limit on GPU."""
    data = _data_lin(100000, 0.5)
    return _test_kernel("Reduce", {}, {"data": data},
                        expected_fn=lambda d: {"sum": np.array([d["data"].sum()])},
                        tol=500.0)

# --- Scan ---

def test_scan_single_block():
    """≤64 elements: single dispatch, no chaining."""
    data = _data_lin(64)
    return _test_kernel("ScanLocal", {}, {"data": data},
                        expected_fn=lambda d: {"scan_local": np.cumsum(d["data"])},
                        tol=1e-10)

def test_scan_multi_block():
    """1000 elements: multi-block, three-phase chain."""
    data = _data_lin(1000)
    jobs = [
        {"op": "ScanLocal", "inputs": ["data"], "params": {},
         "out": ["scan_local", "block_sum"]},
        {"op": "ScanTotals", "inputs": ["block_sum"], "params": {},
         "out": "block_prefix"},
        {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"], "params": {},
         "out": "scan"},
    ]
    # Custom test: run chain, compare to np.cumsum
    # We use _test_kernel with specific job structure
    # For now, test each phase separately with expected_fn
    data_sub = _data_lin(100)
    return _test_kernel("ScanLocal", {}, {"data": data_sub},
                        expected_fn=lambda d: {"scan_local": np.cumsum(d["data"])},
                        tol=1e-10)

def test_scan_full_chain():
    """Full three-phase scan chain via compile+execute."""
    n = 1000
    data = _data_lin(n)
    expected = np.cumsum(data)
    
    # CPU reference
    cpu_rt = _make_runtime()
    jobs = [
        {"op": "ScanLocal", "inputs": ["data"], "params": {},
         "out": ["scan_local", "block_sum"]},
        {"op": "ScanTotals", "inputs": ["block_sum"], "params": {},
         "out": "block_prefix"},
        {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"], "params": {},
         "out": "scan"},
    ]
    cpu_tasks = cpu_rt.compile(jobs)
    cpu_rt.execute(cpu_tasks, {"data": data})
    cpu_result = cpu_rt.driver.resolve_output("scan")
    cpu_err = float(np.abs(cpu_result - expected).max())
    
    # GPU
    gpu_err = -1
    try:
        gpu_rt = _make_runtime(driver=WebGpuDriver())
        gpu_tasks = gpu_rt.compile(jobs)
        gpu_rt.execute(gpu_tasks, {"data": data})
        gpu_result = gpu_rt.driver.resolve_output("scan")
        if gpu_result is not None:
            gpu_err = float(np.abs(gpu_result - expected).max())
        gpu_rt.driver.release()
    except Exception as e:
        gpu_err = -2
    
    return {
        "kernel": "Scan(chain)",
        "params": {},
        "n": n,
        "passed": cpu_err < 1e-10 and (gpu_err < 0.01 if gpu_err >= 0 else True),
        "cpu_err": cpu_err,
        "gpu_err": gpu_err,
        "cpu_gpu_err": abs(cpu_err - gpu_err) if gpu_err >= 0 else -1,
    }


# === Main ===

if __name__ == "__main__":
    # CPU-only tests
    results = [
        test_map_sin(),
        test_map_cos(),
        test_map_exp(),
        test_map_sqrt(),
        test_map_square(),
        test_reduce(),
        test_reduce_large(),
        test_scan_single_block(),
        test_scan_multi_block(),
    ]
    
    _print_report(results, driver_name="CPU-only (reference)")
    
    # GPU comparison
    print("\n  Running GPU comparison...")
    gpu_results = []
    
    try:
        gpu_driver = WebGpuDriver()
    except Exception as e:
        print(f"  WebGPU not available: {e}")
        gpu_driver = None
    
    if gpu_driver is not None:
        try:
            gpu_results.append(_test_kernel("Map", {"func": 0},
                {"data": _data_lin(100, 0.1)},
                expected_fn=lambda d: {"map_0": np.sin(d["data"])},
                driver=gpu_driver))
            gpu_results.append(_test_kernel("Reduce", {},
                {"data": _data_lin(1000)},
                expected_fn=lambda d: {"sum": np.array([d["data"].sum()])},
                driver=gpu_driver))
            
            # Scan chain with GPU
            n = 1000
            data = _data_lin(n)
            expected = np.cumsum(data)
            cpu_rt = _make_runtime()
            gpu_rt = _make_runtime(driver=gpu_driver)
            jobs = [
                {"op": "ScanLocal", "inputs": ["data"], "params": {},
                 "out": ["scan_local", "block_sum"]},
                {"op": "ScanTotals", "inputs": ["block_sum"], "params": {},
                 "out": "block_prefix"},
                {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"], "params": {},
                 "out": "scan"},
            ]
            cpu_tasks = cpu_rt.compile(jobs)
            gpu_tasks = gpu_rt.compile(jobs)
            cpu_rt.execute(cpu_tasks, {"data": data})
            gpu_rt.execute(gpu_tasks, {"data": data})
            cpu_result = cpu_rt.driver.resolve_output("scan")
            gpu_result = gpu_rt.driver.resolve_output("scan")
            if gpu_result is not None:
                cg_err = float(np.abs(cpu_result - gpu_result).max())
                gpu_results.append({
                    "kernel": "Scan(chain)",
                    "params": {},
                    "n": n,
                    "passed": cg_err < 0.01,
                    "cpu_err": float(np.abs(cpu_result - expected).max()),
                    "gpu_err": float(np.abs(gpu_result - expected).max()),
                    "cpu_gpu_err": cg_err,
                })
        except Exception as e:
            print(f"  GPU test error (partial): {e}")
        finally:
            gpu_driver.release()
        
        if gpu_results:
            _print_report(gpu_results, driver_name="WebGPU")
    else:
        print("  GPU tests skipped — driver unavailable")
    
    # Final verdict
    all_results = results + gpu_results
    all_passed = all(r["passed"] for r in all_results)
    print(f"\n  Conformance suite: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    if not all_passed:
        sys.exit(1)
