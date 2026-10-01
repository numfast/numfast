# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Guardian: calibrated routing has no hidden thresholds (SPEC-DELTA-03C/04R).

Fails on: hardcoded physics literals in Planner/Runtime, N->backend
branches, per-op routing, selectivity-gated strategy/backend, missing
coverage gate. The filter compact-method crossover (host vs gpu_blocks at
n=2048, DELTA-8) is a measured METHOD choice inside the GPU path, not
backend routing, and is out of scope here.
"""

import re
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2]
RLIB = FORK / "src" / "Runtime"


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](str(FORK))

# Physics-looking literals: never a routing input (all costs are fitted).
BANNED = [
    (r"18\s*GB|18\.?\d*\s*\*\s*1024|bandwidth_gb_s\s*=", "bandwidth literal"),
    (r"dispatch_overhead_us|dispatch.*12\.0|12e-6", "dispatch literal"),
    (r"compile_miss_ms|compile_hit_ms", "compile literal"),
    (r"\b0\.55\b", "fusion factor literal"),
    (r"5e-0?5|0\.00005", "conversion-cost literal"),
    (r"if\s+.*n\s*>\s*\d+.*(gpu|backend)", "N->backend branch"),
    (r">\s*(5000|10000|100000)\s*.*gpu|gpu.*(5000|10000|100000)\s*(rows|>)",
     "N-threshold near gpu"),
]


@pytest.mark.fast
def test_no_hardcoded_physics_in_routing():
    hits = []
    for py in list((RLIB / "Planner" / "_lib").glob("*.py")) + \
            list((RLIB / "Runtime" / "_lib").glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for pat, label in BANNED:
            for m in re.finditer(pat, text, re.IGNORECASE):
                # STUB placeholders carry an explicit unmeasured warning.
                lo, hi = max(0, m.start() - 200), m.end() + 200
                ctx = text[lo:hi]
                if "STUB" in ctx and "stub costs" in ctx:
                    continue
                hits.append(f"{py.name}: {label}: ...{m.group(0)}...")
    assert not hits, "hardcoded routing numbers:\n" + "\n".join(hits)


@pytest.mark.fast
def test_select_backend_called_once_per_graph(kernel):
    a = kernel.alias
    calls = []
    real = a["select_backend"]
    a["select_backend"] = lambda *args, **kw: (calls.append(1), real(*args, **kw))[1]
    try:
        jobs = [a["ir_series"]("s", [1, 2, 3, 4]),
                a["ir_map"]("m", "s", "mul", 2),
                a["ir_reduce"]("r", "m", "sum")]
        a["evaluate"](a["compile"](jobs), "auto", 4)
    finally:
        a["select_backend"] = real
    assert len(calls) == 1, f"select_backend per graph != 1: {len(calls)}"


@pytest.mark.fast
def test_selectivity_sizes_only_downstream(kernel):
    a = kernel.alias
    import numpy as np
    rng = np.random.default_rng(42)
    V = rng.integers(0, 100, 2000).astype(np.int32)
    jobs = [a["ir_series"]("v", V),
            a["ir_compare"]("m", "v", 50, ">"),
            a["ir_filter"]("f", "v", "m"),
            a["ir_reduce"]("r", "f", "sum")]
    g = a["compile"](jobs)
    lo = a["select_backend"](g, 2000, hints={"selectivity": 0.1})
    hi = a["select_backend"](g, 2000, hints={"selectivity": 0.9})
    assert lo["backend"] == hi["backend"], (lo, hi)
    assert lo["gpu_eligible"] == hi["gpu_eligible"]
    # Only downstream n_op / tail numbers move, never eligibility.
    el = {b["node"]: b["n_op"] for b in lo["estimate"]["breakdown"]} \
        if lo["estimate"] else {}
    eh = {b["node"]: b["n_op"] for b in hi["estimate"]["breakdown"]} \
        if hi["estimate"] else {}
    if el:
        assert el != eh, "selectivity must resize downstream n_op"
    assert a["plan_filter"](10**6, selectivity=0.1)["strategy"] == \
        a["plan_filter"](10**6, selectivity=0.9)["strategy"]


@pytest.mark.fast
def test_gates_before_cost_and_coverage_observable(kernel):
    a = kernel.alias
    # Ineligible op: gates fire, no cost comparison.
    jobs = [a["ir_series"]("s", [1, 2, 3]),
            a["ir_encode_pattern"]("e", ["a1", "a2", "a3"], "a")]
    sel = a["select_backend"](a["compile"](jobs), 3)
    assert sel["backend"] == "cpu" and sel["gpu_eligible"] is False
    assert sel["gpu_blockers"], sel
    # Unmeasured-but-eligible op (groupby_multi): coverage gate, still explicit-gpu-able.
    jobs = [a["ir_series"]("v", [1, 2, 2, 3]),
            a["ir_series"]("k", [0, 0, 1, 1]),
            a["ir_groupby_multi"]("g", "v", "k", ("sum",))]
    sel = a["select_backend"](a["compile"](jobs), 4)
    assert sel["gpu_eligible"] is True
    assert sel["coverage"]["ok"] is False, sel
    assert sel["backend"] == "cpu" and "calibrated" in sel["reason"]
    # f64-unscaled gate.
    jobs = [a["ir_series"]("v", [1.0, 2.0], "float64"),
            a["ir_reduce"]("r", "v", "sum")]
    sel = a["select_backend"](a["compile"](jobs), 2)
    assert sel["backend"] == "cpu" and sel["gpu_eligible"] is False
    # Memory gate is structural: absurd n without allocating.
    jobs = [a["ir_series"]("v", [1, 2]),
            a["ir_groupby"]("g", "v", "v", "sum")]
    sel = a["select_backend"](a["compile"](jobs), 10**12)
    assert sel["backend"] == "cpu" and sel["gpu_eligible"] is False
    assert any("memory" in b for b in sel["gpu_blockers"]), sel
