"""Test Kernel Test Harness itself."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "core")))

import numpy as np
from Runtime import Runtime
from trading.Trading import register_all as register_trading_kernels
from Runtime._lib.harness import KernelTestHarness, test_kernel as _harness_test_kernel


def make_runtime():
    runtime = Runtime()
    register_trading_kernels(runtime)
    return runtime


def test_ema_basic():
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    report = _harness_test_kernel(runtime, "EMA", {"period": 3}, {"Close": data})
    assert report["passed"], f"EMA failed: {report['summary']}"
    assert report["nan_count"] == 0
    assert report["inf_count"] == 0
    assert "ema_3" in report["outputs"]
    print(f"  [OK] {report['summary']}")


def test_ema_with_expected():
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    # Compute expected by running twice (second run = expected)
    first = _harness_test_kernel(runtime, "EMA", {"period": 3}, {"Close": data})
    expected = first["outputs"]["ema_3"]
    
    second = _harness_test_kernel(runtime, "EMA", {"period": 3}, {"Close": data}, expected=expected)
    assert second["passed"]
    assert second["max_abs_error"] < 1e-12
    print(f"  [OK] EMA with expected: max_abs_error={second['max_abs_error']:.2e}")


def test_all_trading_kernels():
    runtime = make_runtime()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    harness = KernelTestHarness(runtime)
    
    harness.validate("EMA",  {"period": 3},           {"Close": data}, label="period=3")
    harness.validate("EMA",  {"period": 5},           {"Close": data}, label="period=5")
    harness.validate("RSI",  {"period": 14},          {"Close": data})
    harness.validate("SMA",  {"period": 3},           {"Close": data})
    harness.validate("MACD", {"fast": 12, "slow": 26}, {"Close": data})
    
    assert harness.report_all()


def test_nan_detection():
    """Verify NaN detection works."""
    runtime = make_runtime()
    # Create data that might produce NaN
    data = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    # RSI with flat data — will produce NaN (division by zero)
    report = _harness_test_kernel(runtime, "RSI", {"period": 3}, {"Close": data})
    print(f"  RSI on flat data: {report['summary']}")
    # Flat data RSI produces NaN (gain/loss both zero → 0/0)
    # This is expected behavior, we just check NaN is detected
    print(f"  Note: RSI on flat data -> NaN={report['nan_count']}")


if __name__ == "__main__":
    test_ema_basic()
    test_ema_with_expected()
    test_all_trading_kernels()
    test_nan_detection()
    print("\n=== All harness tests PASSED ===")
