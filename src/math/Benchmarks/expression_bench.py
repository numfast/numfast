"""Benchmark: full expression pipeline (parse + compile + evaluate) throughput."""

import time
import numpy as np
from ..AST._lib import parse, compile_to_isa, evaluate
from ._lib.data_gen import generate_ohlc, generate_expression_set
from ._lib.benchmark_base import BenchmarkTimer, format_bars_per_sec, format_exprs_per_sec, format_profile

def run() -> dict:
    n_bars = 10000
    n_exprs = 100

    data = generate_ohlc(n_bars, base_price=50000.0, volatility=0.02)
    exprs = generate_expression_set(seed=42, count=n_exprs)
    scope = {"close": data["close"], "high": data["high"],
             "low": data["low"], "open": data["open"]}

    print("=" * 60)
    print("EXPRESSION BENCHMARK (full pipeline)")
    print("=" * 60)
    print(f"  Expressions: {n_exprs}")
    print(f"  Bars:        {n_bars:,}")

    timer = BenchmarkTimer()

    # Phase 1: Parse all
    asts = []
    for expr in exprs:
        asts.append(parse(expr))
    timer.lap("parse")

    # Phase 2: Compile all
    isas = []
    for ast in asts:
        isas.append(compile_to_isa(ast))
    timer.lap("compile")

    # Phase 3: Evaluate all
    for ast in asts:
        evaluate(ast, scope=scope)
    timer.lap("evaluate")

    profile = timer.results()
    total_time = sum(profile.values())

    print(f"\n  Phase breakdown:")
    for label, sec in profile.items():
        pct = (sec / total_time * 100) if total_time > 0 else 0
        hrs = format_exprs_per_sec(n_exprs, sec)
        print(f"    {label:12s}: {sec*1000:8.2f} ms  ({pct:5.1f}%)  {hrs:>12s}")

    pipeline_expr_sec = n_exprs / total_time if total_time > 0 else 0
    bars_sec = n_bars * n_exprs / total_time if total_time > 0 else 0
    print(f"\n  Pipeline throughput: {format_exprs_per_sec(n_exprs, total_time)}")
    print(f"  Bars processed:      {format_bars_per_sec(n_bars * n_exprs, total_time)}")

    print()
    return {
        "benchmark": "expression_pipeline",
        "n_expressions": n_exprs,
        "n_bars": n_bars,
        "total_time_sec": total_time,
        "parse_time_sec": profile.get("parse", 0),
        "compile_time_sec": profile.get("compile", 0),
        "evaluate_time_sec": profile.get("evaluate", 0),
        "expr_per_sec": pipeline_expr_sec,
        "bars_per_sec": bars_sec,
    }


if __name__ == "__main__":
    run()
