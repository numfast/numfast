# NumFast Core Index

## Purpose
GPU-accelerated numerical computing runtime. Python + wgpu-py + WGSL.

## Key Files

| File | Purpose |
|------|---------|
| `driver.py` | CpuDriver --- read/write/allocate/dispatch/barrier. No operation knowledge. |
| `executor.py` | Planner (merge, operation-agnostic) + CpuKernel (15 ops) + CoreAPI |
| `gpu_driver.py` | MockGpuDriver + WebGpuDriver (wgpu StorageBuffer) |
| `nseries.py` | NumericSeries --- logical abstraction read(i)/write(i). NOT container. |
| `profiler.py` | RuntimeProfiler + GraphVisualizer + OptimizationReport |
| `_evil_ops.py` | Stress test operations (zero, negative, random, throws, manifest) |
| `gpu/shaders.py` | WGSL shader generators: sma_wgsl, copy_wgsl, pointwise_wgsl, roc_wgsl, sliding_wgsl |
| `gpu/device.py` | wgpu device singleton (get_device) |

## Tests (`tests/`)

| File | Tests |
|------|-------|
| `test_conformance.py` | 30 parametrized tests across 15 ops |
| `test_gpu_conformance.py` | Same 30 tests on GPU (MockGpuDriver + WebGpuDriver) |
| `test_gpu_dispatch.py` | GPU dispatch: SMA, ROC, MIN, MAX, Pointwise |
| `test_webgpu_driver.py` | WebGpuDriver read/write/allocate/dispatch |
| `test_stress_runtime.py` | 1000 ops stress, memory pressure |
| `test_break_architecture.py` | Evil ops + bad DAGs. 144 tests total. |
| `test_operations_pack.py` | 8 new ops (Phase 17): WMA, TRANGE, RSI, STOCH, MEDIAN, OBV, VWAP, CCI |
| `test_pandas_compatibility.py` | 49 tests: 99.6% signal match vs pandas |
| `test_wgsl_pipeline.py` | WGSL compile + dispatch pipeline |

## Architecture Invariants

1. **Runtime frozen** (ADR-010): NumericSeries, Planner, Driver ABI, Kernel ABI
2. **No materialization**: only series.read(i), never to_numpy()
3. **Runtime not storage**: no format checks in Runtime
4. **Planner agnostic**: knows only OpID, Inputs, Params
5. **New ops = 0 Core changes**: add _run_* + _ops dict + CoreAPI method
6. **int32 logical**: prices * SCALE, accumulators int64
7. **min(read()) == 1**: no zero values in computational series

## GPU Status

- WGSL: SMA, ROC, MIN, MAX, Pointwise, Copy
- CPU fallback: CCI, MEDIAN, STOCH, STDDEV, RSI, EMA, WMA, TRANGE, ATR, OBV, VWAP
- Priority: CCI -> MEDIAN -> STOCH -> STDDEV -> RSI

## Related Specs

`_specs/numfast/compute/` --- full specification set.