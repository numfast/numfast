"""Test Runtime Profiler."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from Runtime import Runtime
from Trading import register_all as register_trading_kernels
from Runtime._lib.profiler import RuntimeProfiler, phase, get_profiler


def make_runtime():
    runtime = Runtime()
    register_trading_kernels(runtime)
    return runtime


def test_profiler_basic():
    profiler = RuntimeProfiler()
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    with profiler.phase("compile"):
        tasks = runtime.compile([
            {"op": "EMA", "inputs": ["Close"], "params": {"period": 3}},
        ])

    with profiler.phase("execute"):
        runtime.execute(tasks, {"Close": data})

    report = profiler.to_dict()
    assert "compile" in report
    assert "execute" in report
    assert report["compile"]["calls"] == 1
    assert report["execute"]["calls"] == 1
    assert report["compile"]["elapsed"] > 0
    assert report["execute"]["elapsed"] > 0

    profiler.print_report()
    print("  OK Profiler basic timing OK")


def test_profiler_multiple_phases():
    profiler = RuntimeProfiler()
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    for period in [3, 5, 10, 20]:
        with profiler.phase("compile"):
            tasks = runtime.compile([
                {"op": "EMA", "inputs": ["Close"], "params": {"period": period}},
            ])
        with profiler.phase("execute"):
            runtime.execute(tasks, {"Close": data})

    report = profiler.to_dict()
    assert report["compile"]["calls"] == 4
    assert report["execute"]["calls"] == 4

    profiler.print_report(sort_by="calls")
    print(f"  OK Multiple phases: compile={report['compile']['calls']}x {report['compile']['elapsed']*1000:.3f}ms total")


def test_profiler_multi_kernel():
    profiler = RuntimeProfiler()
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    with profiler.phase("compile"):
        tasks = runtime.compile([
            {"op": "EMA",  "inputs": ["Close"], "params": {"period": 3}, "resource": "@tmp"},
            {"op": "RSI",  "inputs": ["Close"], "params": {"period": 14}},
        ])

    with profiler.phase("scheduler"):
        from Runtime._lib.scheduler import Scheduler
        scheduler = Scheduler()
        waves = scheduler.schedule(tasks)

    with profiler.phase("execute"):
        runtime.execute(tasks, {"Close": data})

    profiler.print_report()
    print(f"  OK Multi-kernel profiling: {len(waves)} waves")
    print(f"  OK Report printed OK")


if __name__ == "__main__":
    test_profiler_basic()
    test_profiler_multiple_phases()
    test_profiler_multi_kernel()
    print("\n=== All profiler tests PASSED ===")
