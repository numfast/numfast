"""Benchmark: AST parsing throughput."""

import time
import numpy as np
from ..AST._lib import parse
from ._lib.data_gen import generate_expression_set
from ._lib.benchmark_base import BenchmarkTimer, format_exprs_per_sec

def run() -> dict:
    n_exprs = 1000
    exprs = generate_expression_set(seed=42, count=n_exprs)

    # Warmup
    for e in exprs[:10]:
        parse(e)

    # Benchmark
    timer = BenchmarkTimer()
    timer.lap("warmup_done")

    parsed = 0
    errors = 0
    for expr in exprs:
        try:
            parse(expr)
            parsed += 1
        except Exception:
            errors += 1

    timer.lap("parse_all")

    elapsed = timer.results()["parse_all"]
    rate = format_exprs_per_sec(n_exprs, elapsed)

    print("=" * 60)
    print("AST PARSE BENCHMARK")
    print("=" * 60)
    print(f"  Expressions:       {n_exprs}")
    print(f"  Parsed:            {parsed}")
    print(f"  Errors:            {errors}")
    print(f"  Total time:        {elapsed*1000:.2f} ms")
    print(f"  Throughput:        {rate}")

    results = {
        "benchmark": "ast_parse",
        "n_expressions": n_exprs,
        "parsed": parsed,
        "errors": errors,
        "time_sec": elapsed,
        "expr_per_sec": n_exprs / elapsed if elapsed > 0 else 0,
    }
    return results


if __name__ == "__main__":
    run()
