# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""HLL Stage-5 presentation (packaging only, stdlib, no remeasure)."""
import json
from pathlib import Path

SUMMARY = Path(__file__).with_name("summary.json")


def main():
    s = json.loads(SUMMARY.read_text())
    print("=" * 64)
    print("HLL Stage-5 FULL: 7 waves  (zero-copy proof, CPU-only)")
    print("=" * 64)
    print(f"workload : {s['workload']}")
    print(f"env      : {s['env']}")
    print(f"method   : seed={s['method']['seed']} small_n={s['method']['small_n']}")
    print("correctness:")
    for k, v in s["correctness"].items():
        print(f"  {k:18s} {v}")
    print(f"timing   : {s['timing']['note']}")
    print(f"conclusion: {s['conclusion']}")
    print(f"NOT claimed: {'; '.join(s['not_claimed'])}")


if __name__ == "__main__":
    main()
