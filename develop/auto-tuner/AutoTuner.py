# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""AutoTuner Extension entry: imports from _lib only + setup(kernel)."""
from _lib.candidates import list_candidates
from _lib.fingerprint import fingerprint
from _lib.report import write_calibrated_toml, write_report
from _lib.search import run_demo, search


def setup(kernel):
    kernel.metadata.setdefault("AutoTuner", {})
    kernel.metadata["AutoTuner"]["version"] = "0.1.0"
    kernel.metadata["AutoTuner"]["types"] = [
        "search", "fingerprint", "list_candidates", "run_demo",
    ]


PUBLIC = {
    "search": search,
    "fingerprint": fingerprint,
    "list_candidates": list_candidates,
    "run_demo": run_demo,
}


if __name__ == "__main__":
    import json
    from pathlib import Path
    here = Path(__file__).resolve().parent
    demo = run_demo()
    cmd = 'export PYTHONPATH="develop/auto-tuner" && python develop/auto-tuner/AutoTuner.py'
    toml_p = here / "calibrated_v1.toml"
    rep_p = here / "REPORT.md"
    json_p = here / "search_results.json"
    write_calibrated_toml(str(toml_p), demo, cmd)
    write_report(str(rep_p), demo, cmd)
    json_p.write_text(json.dumps(demo, indent=2, default=str), encoding="utf-8")
    b, t = demo["baseline"], demo["best"]
    print(f"ordinary query: {demo['fp_tune']['query']} N={demo['fp_tune']['n']} sel~{demo['fp_tune']['selectivity']}")
    print(f"searched {len(demo['tune']['rows'])} variants (tune seed42 / eval seed43)")
    print(f"best {t['id']} vs {b['id']}: {b['ms']}->{t['ms']}ms | {demo['reason']}")
    print(f"eval winner={demo['best_eval_id']} overfit={demo['overfit_flag']}")
    print(f"wrote {toml_p.name}, {rep_p.name}, {json_p.name}")
