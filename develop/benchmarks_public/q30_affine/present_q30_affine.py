# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Q30 affine presentation (packaging only, stdlib, no remeasure)."""
import json
from pathlib import Path

SUMMARY = Path(__file__).with_name("summary.json")


def main():
    s = json.loads(SUMMARY.read_text())
    print("=" * 64)
    print("Q30 affine: 181 -> 181 nodes EXACT (current CSE+DCE); 181->2 / 43.02x UNCONFIRMED prototype-only")
    print("=" * 64)
    print(f"workload : {s['workload']}")
    print(f"env      : {s['env']}")
    cw = s["method"]["cold_warm"]
    print(f"method   : warm={cw['warm']} | A warm {cw['A_warm_med_total_ms']}ms | "
          f"AUTO warm {cw['AUTO_warm_med_total_ms']}ms | duck {cw['duck_best_of_3_ms']}ms")
    print(f"correctness: graph {s['correctness']['graph']}")
    print(f"             all diffs {s['correctness']['all_pairs']}")
    th = s["timing"]["throughput_rows_s"]
    print(f"timing   : {th['A']} rows/s (A) vs {th['AUTO']} rows/s (AUTO)")
    print("comparison:")
    for k, v in s["comparison"].items():
        print(f"  {k:14s} {v}")
    print(f"conclusion: {s['conclusion']}")
    print(f"NOT claimed: {'; '.join(s['not_claimed'])}")


if __name__ == "__main__":
    main()
