# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Report + calibrated_v1 TOML writer. Pure formatting, no prod writes."""
import json
from datetime import datetime, timezone
from pathlib import Path


def write_calibrated_toml(path, demo, command):
    b, t = demo["baseline"], demo["best"]
    now = datetime.now(timezone.utc).isoformat()
    txt = f"""# Auto-tuner SEARCH V1 -- research only (no prod Planner/Driver change).
# Generated: {now} | command: {command}
# Calibration data: tune split seed42 N={demo['fp_tune']['n']}, eval split seed43 (held-out).
model_version = "calibrated_v1"
profile_kind = "workload_tuning"
source = "measured:seed42:tune-eval-split"

[workload]
query = "{demo['fp_tune']['query']}"
n = {demo['fp_tune']['n']}
tune_seed = {demo['fp_tune']['seed']}
eval_seed = {demo['fp_eval']['seed']}
tune_selectivity = {demo['fp_tune']['selectivity']}
eval_selectivity = {demo['fp_eval']['selectivity']}
tune_nunique = {demo['fp_tune']['nunique_filtered']}
tune_span = {demo['fp_tune']['span_filtered']}
is_sorted = {str(demo['fp_tune']['is_sorted']).lower()}

[baseline]
id = "{b['id']}"
exec = "{b['exec']}"
threads = {b['threads']}
groupby = "{b['groupby']}"
path = "{b['path']}"
ms = {b['ms']}

[chosen]
id = "{t['id']}"
exec = "{t['exec']}"
threads = {t['threads']}
groupby = "{t['groupby']}"
path = "{t['path']}"
ms = {t['ms']}
reason = "{demo['reason']}"

[guard]
scope_A = "customer/workload tuning (tune split only)"
scope_B = "benchmark research (eval split held-out)"
best_eval_id = "{demo['best_eval_id']}"
overfit_flag = {str(demo['overfit_flag']).lower()}
rule = "winner picked on tune split; reported on eval split; NOT benchmark-specific; NOT OPT4/5/6; no hot-path training"
"""
    Path(path).write_text(txt, encoding="utf-8")
    return path


def write_report(path, demo, command):
    lines = []
    lines.append("# Auto-tuner SEARCH V1 -- reproducible report (research only)")
    lines.append(f"command: `{command}`")
    lines.append(f"tune fingerprint: {json.dumps(demo['fp_tune'])}")
    lines.append(f"eval fingerprint: {json.dumps(demo['fp_eval'])}")
    b, t = demo["baseline"], demo["best"]
    lines.append(f"baseline {b['id']} ({b['exec']}/{b['groupby']}/{b['path']}): {b['ms']}ms")
    lines.append(f"best {t['id']} ({t['exec']}/{t['groupby']}/{t['path']}): {t['ms']}ms | reason: {demo['reason']}")
    lines.append(f"baseline vs tuned: {b['ms']}->{t['ms']}ms (measured, tune split)")
    lines.append(f"eval winner: {demo['best_eval_id']} | overfit_flag={demo['overfit_flag']}")
    lines.append("scope A = workload tuning (tune split); scope B = benchmark research (eval split).")
    lines.append("candidates (tune split, ms | correct | mem | rows_read | mats | native):")
    for r in demo["tune"]["rows"]:
        lines.append(f"  {r['id']}: {r['exec']}t{r['threads']} {r['groupby']}/{r['path']} = {r['ms']}ms ok={r['correct']} mem={r['memory_bytes']} rows={r['rows_read']} mats={r['materializations']} nat={r['native_calls']}")
    lines.append("proof: mechanism works = search enumerates 17, oracle gates correctness, min-ms wins, eval split confirms (or flags overfit). Proof stops here.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
