"""Benchmark: ISA execution (Python evaluator) throughput."""

import time
import numpy as np
from ..AST._lib import parse, evaluate
from ._lib.data_gen import generate_ohlc, generate_expression_set
from ._lib.benchmark_base import BenchmarkTimer, format_bars_per_sec

def run() -> dict:
    bar_sizes = [1000, 10000, 100000]
    expression = "SMA(close, 14) + EMA(close, 20)"

    print("=" * 60)
    print("ISA EXECUTION BENCHMARK")
    print("=" * 60)
    print(f"  Expression: {expression}")

    results = {}
    ast = parse(expression)

    for n_bars in bar_sizes:
        data = generate_ohlc(n_bars, base_price=50000.0, volatility=0.02)
        scope = {"close": data["close"]}

        # Warmup
        evaluate(ast, scope=scope)

        # Benchmark
        n_runs = max(1, 100000 // n_bars)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            evaluate(ast, scope=scope)
        elapsed = time.perf_counter() - t0

        total_bars = n_bars * n_runs
        rate = format_bars_per_sec(total_bars, elapsed)
        time_per_eval = (elapsed / n_runs) * 1000

        print(f"\n  Bars: {n_bars:>8,}  Runs: {n_runs:>4}  "
              f"Time: {elapsed*1000:>8.2f} ms  "
              f"Per eval: {time_per_eval:>7.3f} ms  "
              f"Throughput: {rate}")

        results[str(n_bars)] = {
            "n_bars": n_bars,
            "n_runs": n_runs,
            "total_bars": total_bars,
            "time_sec": elapsed,
            "bars_per_sec": total_bars / elapsed if elapsed > 0 else 0,
            "ms_per_eval": time_per_eval,
        }

    print()
    return {
        "benchmark": "isa_exec",
        "expression": expression,
        "results_by_size": results,
    }


if __name__ == "__main__":
    run()
