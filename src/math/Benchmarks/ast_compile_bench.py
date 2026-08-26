"""Benchmark: AST -> ISA compilation throughput."""

import time
import numpy as np
from ..AST._lib import parse, compile_to_isa
from ._lib.data_gen import generate_expression_set
from ._lib.benchmark_base import BenchmarkTimer, format_exprs_per_sec

def run() -> dict:
    n_exprs = 500
    exprs = generate_expression_set(seed=42, count=n_exprs)

    # Pre-parse all
    asts = []
    for expr in exprs:
        asts.append(parse(expr))

    # Warmup
    for ast in asts[:5]:
        compile_to_isa(ast)

    # Benchmark
    compiled = 0
    errors = 0
    total_instrs = 0

    t0 = time.perf_counter()
    for ast in asts:
        try:
            isa = compile_to_isa(ast)
            compiled += 1
            total_instrs += len(isa)
        except Exception:
            errors += 1
    elapsed = time.perf_counter() - t0

    rate = format_exprs_per_sec(n_exprs, elapsed)
    avg_instrs = total_instrs / compiled if compiled > 0 else 0

    print("=" * 60)
    print("AST COMPILE BENCHMARK")
    print("=" * 60)
    print(f"  Expressions:        {n_exprs}")
    print(f"  Compiled:           {compiled}")
    print(f"  Errors:             {errors}")
    print(f"  Total instructions: {total_instrs}")
    print(f"  Avg instr/expr:     {avg_instrs:.1f}")
    print(f"  Total time:         {elapsed*1000:.2f} ms")
    print(f"  Throughput:         {rate}")
    print(f"  Instr/sec:          {total_instrs/elapsed:.0f}")

    return {
        "benchmark": "ast_compile",
        "n_expressions": n_exprs,
        "compiled": compiled,
        "errors": errors,
        "total_instructions": total_instrs,
        "avg_instructions_per_expr": avg_instrs,
        "time_sec": elapsed,
        "expr_per_sec": n_exprs / elapsed if elapsed > 0 else 0,
        "instr_per_sec": total_instrs / elapsed if elapsed > 0 else 0,
    }


if __name__ == "__main__":
    run()
