# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""H2O presentation (packaging only, stdlib, no remeasure)."""
import json
from pathlib import Path

SUMMARY = Path(__file__).with_name("summary.json")


def main():
    s = json.loads(SUMMARY.read_text())
    print("=" * 64)
    print("H2O GROUPBY Q1-Q5 + Join @ 10M  (GOLD, CPU-only)")
    print("=" * 64)
    print(f"workload : {s['workload']}")
    print(f"env      : {s['env']}")
    m = s["method"]
    print(f"method   : seed={m['seed']} | {m['cold_warm']}")
    print("correctness:")
    for k, v in s["correctness"].items():
        print(f"  {k:10s} {v}")
    print("timing (ms):")
    print(f"  Q1 HLL {s['timing']['Q1']['HLL_ms']} vs A {s['timing']['Q1']['A_ms']}")
    print(f"  Q2 HLL {s['timing']['Q2']['HLL_ms']} vs A {s['timing']['Q2']['A_ms']}")
    print(f"  Q3-Q5/Join-HLL: {s['timing']['Q3_Q4_Q5_JoIN_HLL']['status']}")
    c = s["comparison"]
    print("comparison (baseline vs optimized):")
    print(f"  Q1 {c['Q1_HLL_vs_A_ms']} — {c['Q1_note']}")
    print(f"  Q2 {c['Q2_HLL_vs_A_ms']} — {c['Q2_note']}")
    print(f"conclusion: {s['conclusion']}")
    print(f"measured: {s['measured_vs_proof']}")
    print(f"NOT claimed: {'; '.join(s['not_claimed'])}")


if __name__ == "__main__":
    main()
