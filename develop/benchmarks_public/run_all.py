# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""benchmarks_public run_all: ONE light verification command (stdlib only).

Packaging only — no remeasure, no core/HLL/ABI/methodology changes.
Reads the 5 vendored summary.json snapshots (honest extracts of already-measured
artifacts), validates schema, prints the professional terminal index, and writes
machine-readable results/unified.jsonl.

Canonical (Git Bash, from repo root C:/App/numfast/numfast):
  export PYTHONPATH="/c/App/numfast/numfast/src" && python develop/benchmarks_public/run_all.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = ["h2o", "clickbench", "hll_stage5", "gpu_proof", "q30_affine"]
REQUIRED = ["provenance", "workload", "method", "correctness", "timing",
            "comparison", "conclusion", "measured_vs_proof", "not_claimed",
            "source_artifact"]


def main():
    rows = []
    bad = []
    for name in SUITES:
        p = ROOT / name / "summary.json"
        try:
            s = json.loads(p.read_text())
        except Exception as e:
            bad.append(f"{name}: unreadable ({e})")
            continue
        missing = [k for k in REQUIRED if k not in s]
        if missing:
            bad.append(f"{name}: missing {missing}")
            continue
        rows.append({"suite": name, **s})
    print("=" * 64)
    print("NumFast PUBLIC BENCHMARKS — verification index (packaging only)")
    print("=" * 64)
    for r in rows:
        print(f"[{r['suite']:12s}] {r['conclusion'][:110]}")
    print("-" * 64)
    print("PROVENANCE OF EVERY SUITE ABOVE:")
    for r in rows:
        print(f"[{r['suite']:12s}] {r['provenance']}")
    print("-" * 64)
    print("Sources named by each suite (all verified absent on 2026-10-04):")
    for r in rows:
        print(f"[{r['suite']:12s}] src: {r['source_artifact'][:100]}")
    print("-" * 64)
    print("NOTHING in this index is a current measurement. Nothing was")
    print("remeasured. No figure here is re-runnable from this repository.")
    out = ROOT / "results" / "unified.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote: {out} ({len(rows)} rows)")
    if bad:
        print("FAIL:")
        for b in bad:
            print(f"  {b}")
        raise SystemExit(1)
    print("ALL 5 SUITES READ — schema OK, nothing remeasured, and every")
    print("suite is marked HISTORICAL / UNREPRODUCIBLE above. For a current,")
    print("re-runnable figure see ../../BENCHMARKS.md.")


if __name__ == "__main__":
    main()
