# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage-6 GPU proof presentation (packaging only, stdlib, no remeasure)."""
import json
from pathlib import Path

SUMMARY = Path(__file__).with_name("summary.json")


def main():
    s = json.loads(SUMMARY.read_text())
    print("=" * 64)
    print("Stage-6 GPU proof: same packet, CPU wins honest (N=1M)")
    print("=" * 64)
    print(f"workload : {s['workload']}")
    print(f"env      : {s['env']}")
    cw = s["method"]["cold_warm"]
    print(f"method   : seed={s['method']['seed']} reps={cw['reps']} warm={cw['warm']}")
    print("correctness:")
    for k, v in s["correctness"].items():
        print(f"  {k:10s} {v}")
    t = s["timing"]
    print(f"timing   : cpu warm {t['cpu_end_to_end_ms']['warm_med']}ms | "
          f"gpu H2D {t['gpu_end_to_end_ms']['h2d_med']} + exec {t['gpu_end_to_end_ms']['exec_med']} "
          f"+ D2H {t['gpu_end_to_end_ms']['d2h_med']} + fin {t['gpu_end_to_end_ms']['finalize_med']} "
          f"= total {t['gpu_end_to_end_ms']['total_med']}ms (cold {t['gpu_end_to_end_ms']['cold_total']}ms)")
    print(f"memory   : {t['memory_residency']['note']} "
          f"(H2D {t['memory_residency']['h2d_bytes']}B, D2H {t['memory_residency']['d2h_bytes']}B)")
    print(f"comparison: {s['comparison']['baseline_vs_optimized']} [{s['comparison']['honest']}]")
    print(f"conclusion: {s['conclusion']}")
    print(f"NOT claimed: {'; '.join(s['not_claimed'])}")


if __name__ == "__main__":
    main()
