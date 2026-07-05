"""Test Histogram kernel — CPU and GPU."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from Runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Compute import register_all

np.random.seed(42)

print("=== Histogram Tests ===\n")

# Test 1: Simple uniform data
print("Test 1: Uniform data, 10 bins [0, 10)")
data1 = np.array([float(i % 10) for i in range(1000)], dtype=np.float64)
expected1 = np.array([100.0] * 10)  # 100 elements per bin

rt = Runtime()
register_all(rt)
jobs1 = [{"op": "Histogram", "inputs": ["data1"], "params": {"min_val": 0.0, "max_val": 10.0, "num_bins": 10},
          "out": "bins"}]
tasks1 = rt.compile(jobs1)
# Rename input to match job
rt.execute(tasks1, {"data1": data1})
cpu_result = rt.driver.resolve_output("bins")
err1 = float(np.abs(cpu_result - expected1).max())
print(f"  CPU result: {cpu_result}")
print(f"  Expected:   {expected1}")
print(f"  Error: {err1:.2e} {'PASS' if err1 < 1e-10 else 'FAIL'}")

# Test 2: Gaussian-like data
print("\nTest 2: Random data in [0, 1), 5 bins")
data2 = np.random.random(500).astype(np.float64)
rt2 = Runtime()
register_all(rt2)
jobs2 = [{"op": "Histogram", "inputs": ["data2"], "params": {"min_val": 0.0, "max_val": 1.0, "num_bins": 5},
          "out": "bins"}]
tasks2 = rt2.compile(jobs2)
rt2.execute(tasks2, {"data2": data2})
cpu_result2 = rt2.driver.resolve_output("bins")
# Verify sum of bins = total elements
print(f"  CPU result: {cpu_result2}")
print(f"  Sum: {cpu_result2.sum():.0f} (expected 500) {'PASS' if abs(cpu_result2.sum() - 500) < 1 else 'FAIL'}")

# Test 3: Edge case — all same value
print("\nTest 3: All same value (5.0), range [0, 10), 10 bins")
data3 = np.array([5.0] * 200, dtype=np.float64)
rt3 = Runtime()
register_all(rt3)
jobs3 = [{"op": "Histogram", "inputs": ["data3"], "params": {"min_val": 0.0, "max_val": 10.0, "num_bins": 10},
          "out": "bins"}]
tasks3 = rt3.compile(jobs3)
rt3.execute(tasks3, {"data3": data3})
cpu_result3 = rt3.driver.resolve_output("bins")
# 5.0 is 0.5 normalized, bin = 0.5 * 9 = 4.5 → bin 4 or 5
print(f"  CPU result: {cpu_result3}")
expected_bin = 4  # (5.0 - 0.0) / 10.0 = 0.5, * 9 = 4.5 → int = 4
print(f"  Expected all in bin {expected_bin}: {cpu_result3[expected_bin]:.0f} of 200")
print(f"  {'PASS' if cpu_result3[expected_bin] == 200 else 'FAIL'}")

# Test 4: GPU comparison (if available)
print("\nTest 4: GPU comparison")
try:
    gpu_rt = Runtime(driver=WebGpuDriver())
    register_all(gpu_rt)
    gpu_tasks = gpu_rt.compile(jobs1)
    gpu_rt.execute(gpu_tasks, {"data1": data1})
    gpu_result = gpu_rt.driver.resolve_output("bins")
    if gpu_result is not None:
        gpu_err = float(np.abs(gpu_result - expected1).max())
        cpu_gpu_err = float(np.abs(gpu_result - cpu_result).max())
        print(f"  GPU result: {gpu_result}")
        print(f"  GPU vs expected err: {gpu_err:.2e}")
        print(f"  CPU vs GPU diff: {cpu_gpu_err:.2e}")
        print(f"  {'PASS' if gpu_err < 0.5 else 'FAIL'}")
    gpu_rt.driver.release()
except Exception as e:
    print(f"  GPU error: {e}")

os.remove(__file__)
print("\nDone.")
