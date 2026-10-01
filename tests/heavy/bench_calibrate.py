# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration bench: nf.calibrate() -> TOML profile + dataset (SPEC-DELTA-03C).

Measures the allowed matrix ONLY (compare/mask/filter/gather/reduce,
seed 42 synthetic, cold vs warm, host + resident chains) through PUBLIC
aliases. No historical numbers: every cost number is measured this run.
Writes calibration.toml + calibration_dataset.json at the fork root, then
prints predicted-vs-measured (validation Ns excluded from fits). If the
measured composition diverges, the gap is REPORTED, never fitted away.
1B never runs here (max 1M rows).

Usage (Git Bash, sequential, timeout):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python.exe \
    tests/heavy/bench_calibrate.py [--quick] [--force]
"""
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builder import MAIN  # noqa: E402


def main():
    t0 = time.perf_counter()
    quick = "--quick" in sys.argv
    force = True  # bench always remeasures; nf.calibrate() w/o force reuses
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    info = a["calibrate"](quick=quick, force=force)
    print(f"calibrate status={info['status']} path={info['path']}")
    print(f"profile={info['profile']}")
    for line in info.get("stages", []):
        print("  stage " + line)
    print("predicted-vs-measured (validation Ns, fit-excluded):")
    bad = 0
    for k, v in sorted(info.get("validation", {}).items()):
        if "rel_err" in v:
            print(f"  {k}: pred={v['predicted_ms']:.3f}ms "
                  f"meas={v['measured_ms']:.3f}ms rel_err={v['rel_err']:+.2%} "
                  f"abs_err={v['abs_err_ms']:+.3f}ms")
            if abs(v["rel_err"]) > 0.5 and abs(v["abs_err_ms"]) > 1.0:
                bad += 1
        else:
            print(f"  {k}: measured_warm={v['measured_warm_ms']:.3f}ms "
                  f"({v['note']})")
    el = time.perf_counter() - t0
    print(f"elapsed={el:.1f}s quick={quick}")
    if info["status"] == "measured" and bad:
        print(f"NOTE: {bad} validation rows diverge >50%: reported, not tuned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
