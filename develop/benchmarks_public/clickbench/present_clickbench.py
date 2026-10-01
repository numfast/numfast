# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""ClickBench presentation (packaging only, stdlib, no remeasure)."""
import json
from pathlib import Path

SUMMARY = Path(__file__).with_name("summary.json")


def main():
    s = json.loads(SUMMARY.read_text())
    print("=" * 64)
    print("ClickBench Q1-Q43 matrix @ ~1M  (EXACT/TOL, ingest split)")
    print("=" * 64)
    print(f"workload : {s['workload']}")
    print(f"env      : {s['env']}")
    m = s["method"]
    print(f"method   : seed={m['seed']} | {m['cold_warm']}")
    ig = m["ingest_split_ms"]
    print(f"ingest   : parquet {ig['parquet_read']}ms + to_numpy {ig['to_numpy']}ms"
          f" + dict_url {ig['dict_url']}ms + date {ig['date']}ms (rows {ig['rows']})")
    print("correctness:")
    for k, v in s["correctness"].items():
        print(f"  {k:22s} {v}")
    print("comparison (measured AB-probes, all EXACT):")
    for k, v in s["comparison"].items():
        print(f"  {k:16s} {v}")
    print(f"conclusion: {s['conclusion']}")
    print(f"NOT claimed: {'; '.join(s['not_claimed'])}")


if __name__ == "__main__":
    main()
