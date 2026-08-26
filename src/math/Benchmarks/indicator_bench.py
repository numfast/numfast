"""Benchmark: individual indicator computation throughput."""

import time
import numpy as np
from ..AST._lib import parse, evaluate
from ._lib.data_gen import generate_ohlc
from ._lib.benchmark_base import BenchmarkTimer, format_bars_per_sec, format_profile

def run() -> dict:
    n_bars = 100000
    data = generate_ohlc(n_bars, base_price=50000.0, volatility=0.02)

    indicators = {
        "SMA(close, 14)": {"close": data["close"]},
        "SMA(close, 50)": {"close": data["close"]},
        "EMA(close, 20)": {"close": data["close"]},
        "ATR(high, low, close, 14)": {"high": data["high"], "low": data["low"], "close": data["close"]},
        "high - low": {"high": data["high"], "low": data["low"]},
        "close - open": {"close": data["close"], "open": data["open"]},
    }

    print("=" * 60)
    print("INDICATOR BENCHMARK")
    print("=" * 60)
    print(f"  Bars: {n_bars:,}")
    print()

    results = {}
    for expr, scope in indicators.items():
        ast = parse(expr)

        # Warmup
        evaluate(ast, scope=scope)

        # Benchmark
        n_runs = max(1, 50000 // n_bars)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            evaluate(ast, scope=scope)
        elapsed = time.perf_counter() - t0

        total_bars = n_bars * n_runs
        rate = format_bars_per_sec(total_bars, elapsed)
        ms_per_eval = (elapsed / n_runs) * 1000

        print(f"  {expr:40s}  {ms_per_eval:8.3f} ms  {rate:>15s}")
        results[expr] = {
            "ms_per_eval": ms_per_eval,
            "bars_per_sec": total_bars / elapsed if elapsed > 0 else 0,
            "time_sec": elapsed,
        }

    print()
    return {
        "benchmark": "indicators",
        "n_bars": n_bars,
        "results": results,
    }


if __name__ == "__main__":
    run()
