"""Test ExecutionProfiler — профилирование на уровне ExecutionPacket."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "core")))

import numpy as np
from Runtime._lib.runtime import Runtime
from Runtime._lib.profiler import ExecutionProfiler
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Runtime._lib.mod_iface import (
    InputSlot, OutputSlot, ExecutionPlan
)


def _describe_rolling_sum(params):
    period = int(params.get("period", 3))
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="roll_{period}")],
        workspace=[],
        uniforms={"period": period},
    )


def _cpu_rolling_sum(ctx):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    period = int(ctx.uniforms.get("period", 3))
    n = src.length()
    for i in range(n):
        s = 0.0
        lo = max(0, i - period + 1)
        for j in range(lo, i + 1):
            s += src.read(j)
        dst.write(i, s)


def _describe_state_kernel(params):
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="state")],
        workspace=[],
        uniforms={},
    )


def _cpu_state_kernel(ctx):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    n = src.length()
    prev = 0.0
    for i in range(n):
        val = src.read(i)
        dst.write(i, val - prev)
        prev = val


def _describe_map(params):
    scale = float(params.get("scale", 2.0))
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="map_{scale}")],
        workspace=[],
        uniforms={"scale": scale},
    )


def _cpu_map(ctx):
    src = ctx.inputs[0].view
    dst = ctx.outputs[0].view
    scale = ctx.uniforms.get("scale", 2.0)
    n = src.length()
    for i in range(n):
        dst.write(i, src.read(i) * scale)


def test_exec_profiler():
    runtime = Runtime()
    driver = CpuDriver()
    profiler = ExecutionProfiler()
    driver.profiler = profiler
    runtime.driver = driver

    runtime.register_kernel("RollingSum",
        describe=_describe_rolling_sum, cpu=_cpu_rolling_sum)
    runtime.register_kernel("StateKernel",
        describe=_describe_state_kernel, cpu=_cpu_state_kernel)
    runtime.register_kernel("Map",
        describe=_describe_map, cpu=_cpu_map)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    tasks = runtime.compile([
        {"op": "RollingSum", "params": {"period": 3}, "inputs": ["Close"]},
        {"op": "StateKernel", "params": {}, "inputs": ["Close"]},
        {"op": "Map", "params": {"scale": 2.0}, "inputs": ["Close"]},
    ])

    runtime.execute(tasks, {"Close": data})

    # Profiler should have collected 3 records
    assert profiler.total_dispatches == 3, f"Expected 3, got {profiler.total_dispatches}"
    assert profiler.total_time_ms > 0, "Total time should be > 0"
    assert profiler.total_bytes_moved > 0, "Total bytes moved should be > 0"

    # print_report should produce valid output
    profiler.print_report()

    # summary should have 3 kernels
    summary = profiler.summary()
    assert len(summary) == 3, f"Expected 3 kernels in summary, got {len(summary)}"
    for name in ("RollingSum", "StateKernel", "Map"):
        assert name in summary, f"Missing kernel {name} in summary"
        assert summary[name]["count"] == 1, f"Expected 1 dispatch for {name}"

    print("  OK ExecutionProfiler test passed")


if __name__ == "__main__":
    test_exec_profiler()
    print("\n=== Execution Profiler test PASSED ===")
