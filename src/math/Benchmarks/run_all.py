"""Run all benchmarks within numfast package."""

import time
import numpy as np

from ._lib.benchmark_base import ResultTable
from . import ast_parse_bench
from . import ast_compile_bench
from . import isa_exec_bench
from . import indicator_bench
from . import expression_bench
from . import bench_elementwise
from . import bench_reduce


def main():
    print("=" * 60)
    print("NumFast BENCHMARK SUITE")
    print("=" * 60)
    print()

    all_results = {}
    total_t0 = time.perf_counter()

    benchmarks = [
        ("ast_parse", ast_parse_bench),
        ("ast_compile", ast_compile_bench),
        ("isa_exec", isa_exec_bench),
        ("indicator", indicator_bench),
        ("expression", expression_bench),
        ("elementwise", bench_elementwise),
        ("reduce", bench_reduce),
    ]

    for name, mod in benchmarks:
        print(f"\n  Running {name}...")
        try:
            result = mod.run()
            all_results[name] = result
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            all_results[name] = {"benchmark": name, "error": str(e)}

    total_time = time.perf_counter() - total_t0

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    table = ResultTable(["Benchmark", "Status", "Time", "Throughput"])
    for name, result in all_results.items():
        status = "OK" if "error" not in result else "FAIL"
        t = f"{result.get('time_sec', 0)*1000:.1f} ms" if 'time_sec' in result else "-"
        tp = ""
        if 'expr_per_sec' in result:
            v = result['expr_per_sec']
            tp = f"{v/1000:.1f} K expr/s" if v > 1000 else f"{v:.1f} expr/s"
        elif 'bars_per_sec' in result:
            v = result['bars_per_sec']
            tp = f"{v/1_000_000:.2f} M bars/s" if v > 1_000_000 else f"{v/1000:.1f} K bars/s"
        table.add_row([name, status, t, tp])

    print(table.render())
    print(f"\n  Total suite time: {total_time:.2f} s")
    print()
    print("MISSION COMPLETE")

    return all_results


if __name__ == "__main__":
    main()
