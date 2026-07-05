"""Generate golden reference data for all kernels.

Usage:
    python -m numfast.tests.golden.generate
    
This creates .npy files in data/ with reference outputs from CPU execution.
"""

import sys, os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from numfast.Runtime import Runtime
from numfast import register_trading_kernels

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "data")


def ensure_dir():
    os.makedirs(GOLDEN_DIR, exist_ok=True)


def make_runtime():
    runtime = Runtime()
    register_trading_kernels(runtime)
    return runtime


def _generate_prices(n=100):
    """Generate realistic OHLC price data."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    close = np.maximum(close, 10.0)
    high = close * (1.0 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(low, close * 0.95)
    volume = np.maximum(rng.integers(1000, 10000, n), 100)
    return high, low, close, volume


CASES = [
    # Phase 3 — базовые
    ("EMA",  {"period": 3}, {"Close": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)}),
    ("EMA",  {"period": 5}, {"Close": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)}),
    ("SMA",  {"period": 3}, {"Close": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)}),
    ("SMA",  {"period": 5}, {"Close": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)}),
    ("RSI",  {"period": 14}, {"Close": np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)}),
    ("MACD", {"fast": 12, "slow": 26}, {"Close": np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype=np.float64)}),
    # Phase 5 — новые
    ("ATR",  {"period": 14}, {"High": ..., "Low": ..., "Close": ...}),
    ("ROC",  {"period": 10}, {"Close": ...}),
    ("Momentum", {"period": 10}, {"Close": ...}),
    ("VWAP", {}, {"High": ..., "Low": ..., "Close": ..., "Volume": ...}),
    ("BollingerBands", {"period": 20}, {"Close": ...}),
    ("Stochastic", {"period_k": 14, "period_d": 3}, {"High": ..., "Low": ..., "Close": ...}),
    ("WilliamsR", {"period": 14}, {"High": ..., "Low": ..., "Close": ...}),
    ("OBV", {}, {"Close": ..., "Volume": ...}),
    ("CCI", {"period": 20}, {"High": ..., "Low": ..., "Close": ...}),
    ("KeltnerChannels", {"period_ema": 20, "period_atr": 10}, {"High": ..., "Low": ..., "Close": ...}),
    ("ADX", {"period": 14}, {"High": ..., "Low": ..., "Close": ...}),
    ("SuperTrend", {"period": 10, "mult": 3.0}, {"High": ..., "Low": ..., "Close": ...}),
]


def make_data_for_case(alias, input_keys):
    """Generate realistic OHLCV data for the given input keys."""
    high, low, close, volume = _generate_prices(80)
    mapping = {
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }
    return {k: mapping[k] for k in input_keys}


def generate_all():
    """Generate golden data for all test cases."""
    ensure_dir()
    runtime = make_runtime()
    manifest = []

    for entry in CASES:
        alias = entry[0]
        params = entry[1]
        input_data = entry[2]

        # Fill Ellipsis placeholders with generated data
        if any(v is Ellipsis for v in input_data.values()):
            input_data = make_data_for_case(alias, list(input_data.keys()))

        print(f"  Generating golden for {alias}({params})...", end=" ")

        params_str = "_".join(f"{k}{v}" for k, v in params.items())
        key = f"{alias}_{params_str}"

        jobs = [{"op": alias, "inputs": list(input_data.keys()), "params": params}]
        tasks = runtime.compile(jobs)
        runtime.execute(tasks, input_data)

        saved_outputs = {}
        for t in tasks:
            for name in t.out_names:
                arr = runtime.driver.memory.resolve_output(name)
                if arr is not None:
                    filepath = os.path.join(GOLDEN_DIR, f"{key}_{name}.npy")
                    np.save(filepath, arr)
                    saved_outputs[name] = os.path.basename(filepath)
                    print(f"{name} -> {os.path.basename(filepath)}", end=" ")

        manifest.append({
            "key": key,
            "alias": alias,
            "params": params,
            "inputs": {k: v.tolist() for k, v in input_data.items()},
            "outputs": saved_outputs,
        })
        print()

    manifest_path = os.path.join(GOLDEN_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest saved: {manifest_path}")
    print(f"  Total cases: {len(manifest)}")


if __name__ == "__main__":
    print("Generating golden reference data...")
    generate_all()
    print("\nDone.")
