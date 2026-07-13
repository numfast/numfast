"""Benchmark: QuoteTable operations."""

import time
import sys
import numpy as np

# QuoteTable is in develop/backtest/Backtest/_lib
sys.path.insert(0, r"C:\App\numfast\develop\backtest\Backtest\_lib")
from quote_table import QuoteTable
from ._lib.data_gen import generate_ohlc
from ._lib.benchmark_base import BenchmarkTimer, format_bars_per_sec

def run() -> dict:
    bar_sizes = [1000, 10000, 100000]

    print("=" * 60)
    print("QUOTETABLE BENCHMARK")
    print("=" * 60)

    results = {}
    for n_bars in bar_sizes:
        print(f"\n  --- Bars: {n_bars:,} ---")
        data = generate_ohlc(n_bars, base_price=50000.0, volatility=0.02)
        timer = BenchmarkTimer()

        # We need int32 arrays for QuoteTable
        # Compute low_offset, multiplier etc
        low_min = int(np.floor(data["low"].min()))
        low_offset_val = low_min
        multiplier = 10000  # 4 decimal places
        power = 0

        low_stored = ((data["low"] - low_offset_val) * multiplier).astype(np.int64)
        close_stored = ((data["close"] - low_offset_val) * multiplier).astype(np.int64)
        high_stored = ((data["high"] - low_offset_val) * multiplier).astype(np.int64)
        open_stored = ((data["open"] - low_offset_val) * multiplier).astype(np.int64)

        d_open = (open_stored - low_stored).astype(np.int64)
        d_high = (high_stored - low_stored).astype(np.int64)
        d_close = (close_stored - low_stored).astype(np.int64)

        # Construction
        t0 = time.perf_counter()
        qt = QuoteTable(
            n_rows=n_bars,
            low_offset=low_offset_val,
            multiplier=multiplier,
            power=power,
            has_open=True,
            vol_mode='none',
            low=low_stored,
            d_open=d_open,
            d_high=d_high,
            d_close=d_close,
        )
        t_constr = time.perf_counter() - t0
        timer.lap("construct")

        # Access close prices
        t0 = time.perf_counter()
        _ = qt.close_prices
        t_close = time.perf_counter() - t0
        timer.lap("close_prices")

        # Access OHLC
        t0 = time.perf_counter()
        _ = qt.ohlc
        t_ohlc = time.perf_counter() - t0
        timer.lap("ohlc_all")

        # Memory
        mem = qt.memory_bytes()
        timer.lap("memory")

        print(f"    Construct:    {t_constr*1000:8.2f} ms")
        print(f"    Close access: {t_close*1000:8.2f} ms")
        print(f"    OHLC access:  {t_ohlc*1000:8.2f} ms")
        print(f"    Memory:       {mem/1024/1024:.2f} MB")

        results[str(n_bars)] = {
            "n_bars": n_bars,
            "construct_ms": t_constr * 1000,
            "close_access_ms": t_close * 1000,
            "ohlc_access_ms": t_ohlc * 1000,
            "memory_bytes": mem,
            "memory_mb": mem / 1024 / 1024,
        }

    print()
    return {
        "benchmark": "quote_table",
        "results_by_size": results,
    }


if __name__ == "__main__":
    run()
