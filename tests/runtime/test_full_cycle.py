"""Integration test: full cycle Jobs → Compiler → Runtime → Driver → kernels.

Phase 4: No classes. Only kernel_table with {describe, cpu} functions.
ABI-hardened: ExecutionPacket, BlockView.read/write/length, uniform validation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from Runtime import Runtime
from Trading import register_all as register_trading_kernels


def make_runtime():
    """Create Runtime with all Trading kernels."""
    runtime = Runtime()
    register_trading_kernels(runtime)
    return runtime


def test_single_job():
    """Test 1: Single EMA job via Runtime.execute()."""
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    jobs = [
        {"op": "EMA", "inputs": ["Close"], "params": {"period": 3}, "resource": "@tmp"},
    ]
    tasks = runtime.compile(jobs)

    assert len(tasks) == 1
    assert tasks[0].op == "EMA"
    assert tasks[0].out_names == ["ema_3"]
    assert tasks[0].params == {"period": 3}
    assert tasks[0].num_outputs == 1
    assert tasks[0].workspace == []

    runtime.execute(tasks, {"Close": data})

    result = runtime.driver.memory.resolve_output("ema_3")
    assert result is not None, "ema_3 should exist"
    assert abs(result[-1] - 4.0625) < 0.01
    print(f"  OK test_single_job: EMA(3) last={result[-1]:.4f}")


def test_cartesian_product():
    """Test 2: Multiple parameter values -> cartesian product."""
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    jobs = [
        {"op": "EMA", "inputs": ["Close"], "params": {"period": [3, 5]}, "resource": "@tmp"},
    ]
    tasks = runtime.compile(jobs)

    assert len(tasks) == 2
    assert tasks[0].out_names == ["ema_3"]
    assert tasks[1].out_names == ["ema_5"]

    runtime.execute(tasks, {"Close": data})

    assert runtime.driver.memory.resolve_output("ema_3") is not None
    assert runtime.driver.memory.resolve_output("ema_5") is not None
    print(f"  OK test_cartesian_product: {len(tasks)} tasks")


def test_task_chaining():
    """Test 3: SMA -> RSI chain."""
    runtime = make_runtime()
    data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)

    jobs = [
        {"op": "SMA", "inputs": ["Close"], "params": {"period": 3}, "resource": "@tmp"},
        {"op": "RSI", "inputs": ["sma_3"], "params": {"period": 14}, "resource": "Indicators"},
    ]
    tasks = runtime.compile(jobs)

    assert len(tasks) == 2
    assert tasks[1].inputs[0].type == "task"
    assert tasks[1].inputs[0].task_id == 0

    runtime.execute(tasks, {"Close": data})

    assert runtime.driver.memory.resolve_output("sma_3") is not None
    assert runtime.driver.memory.resolve_output("rsi_14") is not None
    print(f"  OK test_task_chaining: SMA->RSI chain OK")


def test_triple_chain():
    """Test 4: EMA -> SMA -> RSI three-stage chain."""
    runtime = make_runtime()
    data = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)

    jobs = [
        {"op": "EMA", "inputs": ["Close"], "params": {"period": 5}, "resource": "@tmp"},
        {"op": "SMA", "inputs": ["ema_5"], "params": {"period": 3}, "resource": "@tmp"},
        {"op": "RSI", "inputs": ["sma_3"], "params": {"period": 14}, "resource": "Signals"},
    ]
    tasks = runtime.compile(jobs)

    assert len(tasks) == 3
    assert tasks[1].inputs[0].type == "task"
    assert tasks[2].inputs[0].type == "task"

    runtime.execute(tasks, {"Close": data})

    for name in ["ema_5", "sma_3", "rsi_14"]:
        val = runtime.driver.memory.resolve_output(name)
        assert val is not None, f"{name} should exist"
    print(f"  OK test_triple_chain: EMA->SMA->RSI OK")


def test_unknown_kernel():
    """Test 5: Unknown kernel raises error."""
    runtime = make_runtime()
    try:
        runtime.compile([{"op": "UNKNOWN", "inputs": ["Close"], "params": {}}])
        assert False, "Should have raised KeyError"
    except KeyError:
        print(f"  OK test_unknown_kernel: correctly raised KeyError")


def test_runtime_execute():
    """Test 6: Runtime.execute() convenience method."""
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    tasks = runtime.compile([
        {"op": "EMA", "inputs": ["Close"], "params": {"period": 2}},
    ])
    runtime.execute(tasks, {"Close": data})

    assert runtime.driver is not None
    result = runtime.driver.memory.resolve_output("ema_2")
    assert result is not None
    print(f"  OK test_runtime_execute: EMA(2) last={result[-1]:.4f}")


if __name__ == "__main__":
    test_single_job()
    test_cartesian_product()
    test_task_chaining()
    test_triple_chain()
    test_unknown_kernel()
    test_runtime_execute()
    print("\n=== All Phase 4 integration tests PASSED ===")
