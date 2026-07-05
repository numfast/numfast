"""Test Histogram kernel fixes."""
import numpy as np
from Runtime import Runtime
from Compute import register_all

# Test 1: Uniform data
data = np.array([float(i % 10) for i in range(1000)], dtype=np.float64)
rt = Runtime()
register_all(rt)
rt.execute(rt.compile([{'op': 'Histogram', 'inputs': ['data'], 'params': {'min_val': 0.0, 'max_val': 10.0, 'num_bins': 10}, 'out': 'bins'}]), {'data': data})
r = rt.driver.resolve_output('bins')
print('Uniform 10 bins:', r)
err = float(np.abs(r - np.array([100.0]*10)).max())
status = 'PASS' if err < 1 else 'FAIL'
print(f'Error: {err:.2e} {status}')

# Test 2: All same value
data2 = np.array([5.0]*200, dtype=np.float64)
rt2 = Runtime()
register_all(rt2)
rt2.execute(rt2.compile([{'op': 'Histogram', 'inputs': ['data2'], 'params': {'min_val': 0.0, 'max_val': 10.0, 'num_bins': 10}, 'out': 'bins'}]), {'data2': data2})
r2 = rt2.driver.resolve_output('bins')
print('All 5.0:', r2)
print('All in bin 5:', r2[5] == 200)

# Test 3: GPU
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
try:
    rt3 = Runtime(driver=WebGpuDriver())
    register_all(rt3)
    rt3.execute(rt3.compile([{'op': 'Histogram', 'inputs': ['data3'], 'params': {'min_val': 0.0, 'max_val': 10.0, 'num_bins': 10}, 'out': 'bins'}]), {'data3': data})
    r3 = rt3.driver.resolve_output('bins')
    if r3 is not None:
        cg = float(np.abs(r3 - r).max())
        status2 = 'PASS' if cg < 1 else 'FAIL'
        print(f'GPU result: {r3}')
        print(f'CPU-GPU diff: {cg:.2e} {status2}')
    rt3.driver.release()
except Exception as e:
    print(f'GPU error: {e}')
