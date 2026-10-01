# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Auto Planner v1 (P7): 11 planner tests. No timings, no benchmark numbers.

All profiles here are in-memory synthetic dicts (source='test-synthetic')
passed explicitly — never written to calibration.toml, never claimed as
measurements. They exercise routing LOGIC only (structure, gates, slots).
"""

import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK / "src" / "Runtime" / "Planner"))

from _lib.calibrate import propagate_residence as _propagate_residence  # noqa: E402
from _lib.calibrate import transfer_slots as _transfer_slots  # noqa: E402


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](str(FORK))


def _compare_graph(a, n=1000):
    import numpy as np

    v = np.arange(n, dtype=np.int32)
    return a["compile"]([a["ir_series"]("x", v),
                         a["ir_compare"]("m", "x", 50, ">")])


def _synth_profile(cpu_a=0.001, gpu_a=0.0005, cold=False):
    cost = {"compare_cpu_a": cpu_a, "compare_cpu_b": 0.1,
            "compare_gpu_a": gpu_a, "compare_gpu_b": 0.2,
            "compare_gpu_compile_ms": 5.0,
            "compare_cpu_compile_ms": 0.0,
            "groupby_cardinalities": [100, 10000, 100000]}
    for m in (100, 10000, 100000):
        cost[f"groupby_M{m}_cpu_a"] = 0.002
        cost[f"groupby_M{m}_cpu_b"] = 0.3
        cost[f"groupby_M{m}_gpu_a"] = 0.001
        cost[f"groupby_M{m}_gpu_b"] = 0.4
        cost[f"groupby_M{m}_gpu_compile_ms"] = 6.0
        cost[f"groupby_M{m}_cpu_compile_ms"] = 0.0
    if cold:
        cost["device_init_one_time_ms"] = 7.0
    return {"model_version": "calibrated_v1",
            "generated": "2026-09-10T00:00:00+00:00",
            "source": "test-synthetic", "quick": True,
            "hardware": {"gpu_device": "unknown"},
            "cost": cost, "measurements": {}}


# 1. P1: additive context, old callers unaffected.
@pytest.mark.fast
def test_context_additive_fields(kernel):
    a = kernel.alias
    g = _compare_graph(a)
    sel = a["select_backend"](g, 1000, _synth_profile())
    ctx = sel["context"]
    for k in ("op_multiset", "n", "n_ops", "dtypes", "cardinalities",
              "nearest", "selectivity", "residence_from", "device_state",
              "profile"):
        assert k in ctx, k
    assert ctx["n"] == 1000 and ctx["n_ops"] == 1
    assert ("compare", 1) in ctx["op_multiset"]
    # Old caller: no hints/profile args at all.
    res = a["evaluate"](g, "auto", 1000)
    assert res["execution_info"]["actual"] in ("cpu", "gpu")


# 2. P2: Cost v1 arithmetic; CPU always a candidate.
@pytest.mark.fast
def test_cost_v1_formula_cpu_candidate(kernel):
    a = kernel.alias
    g = _compare_graph(a)
    prof = _synth_profile(cpu_a=0.001, gpu_a=10.0)  # cpu cheaper by structure
    sel = a["select_backend"](g, 1000, prof)
    est = sel["estimate"]
    assert est["cpu_ms"] == pytest.approx(0.001 * 1000 + 0.1)
    assert sel["backend"] == "cpu"
    assert "measured" in sel["reason"]
    prof2 = _synth_profile(cpu_a=10.0, gpu_a=0.0005)  # gpu cheaper
    sel2 = a["select_backend"](g, 1000, prof2)
    assert sel2["backend"] == "gpu"
    assert sel2["estimate"]["gpu_host_ms"] < sel2["estimate"]["cpu_ms"]


# 3. P3: cold slots structured; unmeasured device-init forces safe CPU.
@pytest.mark.fast
def test_cold_slots_unmeasured_safe_cpu(kernel):
    a = kernel.alias
    g = _compare_graph(a)
    sel = a["select_backend"](g, 1000, _synth_profile(),  # no device-init slot
                              hints={"device_state": "cold"})
    assert sel["backend"] == "cpu"
    assert "device_init_one_time" in sel["estimate"]["unknown_slots"]
    assert sel["estimate"]["slots"]["cold"]["device_init_one_time"]["ms"] is None
    assert sel["estimate"]["slots"]["cold"]["compile_miss"]["ms"] is not None
    # Measured cold still structural (never silent zero).
    sel2 = a["select_backend"](g, 1000, _synth_profile(cold=True),
                               hints={"device_state": "cold"})
    assert sel2["estimate"]["slots"]["cold"]["device_init_one_time"] == \
        {"status": "measured", "ms": 7.0, "note": ""}


# 4. P3/P4: transfer slot matrix (same-side zero needs no measurement).
@pytest.mark.fast
def test_transfer_slots_matrix():
    z = _transfer_slots("host", "host", 1000, None)
    assert z["h2d"] == {"status": "zero-by-residence", "ms": 0.0,
                        "note": "host->host: no bytes cross"}
    assert z["d2h"]["ms"] == 0.0
    r = _transfer_slots("resident", "resident", 1000, None)
    assert r["h2d"]["ms"] == 0.0 and r["d2h"]["ms"] == 0.0
    u = _transfer_slots("host", "resident", 1000, None)  # no evidence
    assert u["h2d"] == {"status": "unmeasured", "ms": None,
                        "note": "host->resident: no measured transfer "
                                "matrix; safe CPU choice downstream"}
    ab = _transfer_slots("host", "resident", 1000, None, absorbed=True)
    assert ab["h2d"] == {"status": "absorbed-in-fit", "ms": 0.0,
                         "note": "host->resident: warm fits absorb "
                                 "steady-state transfer (DELTA-10)"}
    d = _transfer_slots("resident", "host", 1000, None)
    assert d["d2h"]["status"] == "unmeasured" and d["h2d"]["ms"] == 0.0


# 5. P4: residence — at most one crossing, ping-pong impossible.
@pytest.mark.fast
def test_residence_no_pingpong(kernel):
    a = kernel.alias
    g = _compare_graph(a, 64)
    for frm, be, crosses in (("host", "cpu", []), ("host", "gpu", ["h2d"]),
                             ("resident", "cpu", ["d2h"]),
                             ("resident", "gpu", [])):
        exp = _propagate_residence(g, frm, be)
        assert exp["crossings"] == crosses and exp["ping_pong"] is False
        res = a["evaluate"](g, be if be != "gpu" else "auto", 64,
                            hints={"residence_from": frm})
        got = res["execution_info"]["residence"]
        assert got["ping_pong"] is False and len(got["crossings"]) <= 1


# 6. P5: override — cpu always CPU; gpu ineligible raises; auto delegates.
@pytest.mark.fast
def test_override_semantics(kernel):
    a = kernel.alias
    g = _compare_graph(a, 64)
    assert a["evaluate"](g, "cpu", 64)["execution_info"]["actual"] == "cpu"
    bad = a["compile"]([a["ir_series"]("s", [1, 2, 3]),
                        a["ir_encode_pattern"]("e", ["a1", "a2", "a3"], "a")])
    with pytest.raises(RuntimeError, match="backend='gpu' ineligible"):
        a["evaluate"](bad, "gpu", 3)
    # Ineligible op on auto/cpu never silently routes to GPU.
    assert a["evaluate"](bad, "auto", 3)["execution_info"]["actual"] == "cpu"
    auto = a["evaluate"](g, "auto", 64)
    assert auto["execution_info"]["requested"] == "auto"
    assert auto["execution_info"]["actual"] in ("cpu", "gpu")


# 7. P6: execution_info = old fields + requested/actual/reason/cost/dispatch.
@pytest.mark.fast
def test_execution_info_p6_fields(kernel):
    a = kernel.alias
    info = a["evaluate"](_compare_graph(a, 32), "auto", 32)["execution_info"]
    for k in ("requested", "actual", "reason", "estimated_cost", "dispatches",
              "h2d", "d2h", "placement", "residence", "cost_estimate",
              "profile", "coverage", "estimate", "edges"):
        assert k in info, k
    assert info["residence"]["from"] in ("host", "resident")
    assert info["placement"] in ("host", "resident", "gpu_chunked")


# 8. P9: without calibration GPU never wins on its own (any N).
@pytest.mark.fast
def test_no_profile_never_gpu(kernel, tmp_path, monkeypatch):
    monkeypatch.setenv("NUMFAST_CALIBRATION_DIR", str(tmp_path))
    a = kernel.alias
    for n in (1000, 100000, 10000000):
        sel = a["select_backend"](_compare_graph(a, 8), n)
        assert sel["backend"] == "cpu", n
        assert sel["cost_estimate"] == {"cpu": None, "gpu": None}
        assert "calibration" in sel["reason"] or "CPU" in sel["reason"]


# 9. P9: GroupBy cardinality conservative (unknown/out-of-span -> CPU).
@pytest.mark.fast
def test_groupby_cardinality_conservative(kernel):
    a = kernel.alias
    import numpy as np

    rng = np.random.default_rng(42)
    v = rng.integers(0, 100, 200).astype(np.int32)
    k = rng.integers(0, 50, 200).astype(np.int32)
    prof = _synth_profile()

    def _gb(extra=None):
        p = {"op": "sum"}
        if extra:
            p.update(extra)
        return a["compile"]([{"op": "series", "inputs": [], "params": {
            "values": v, "dtype": "int32"}, "out": "vv"},
            {"op": "series", "inputs": [], "params": {
                "values": k, "dtype": "int32"}, "out": "kk"},
            {"op": "groupby", "inputs": ["vv", "kk"], "params": p,
             "out": "g"}])

    sel = a["select_backend"](_gb(), 200, prof)  # no cardinality anywhere
    assert sel["coverage"]["ok"] is False and sel["backend"] == "cpu"
    sel = a["select_backend"](_gb(), 200, prof,
                              hints={"cardinality": 10_000_000})  # out of span
    assert sel["coverage"]["ok"] is False and sel["backend"] == "cpu"
    sel = a["select_backend"](_gb({"cardinality": 50000}), 200, prof)  # red flag
    assert sel["coverage"]["ok"] is True  # nearest-measured, in span
    notes = " ".join(b.get("note", "") for b in sel["estimate"]["breakdown"])
    assert "nearest-measured" in notes  # conservative upper bound, not a fit


# 10. WASM never an auto candidate.
@pytest.mark.fast
def test_no_wasm_in_auto_candidates(kernel):
    a = kernel.alias
    assert "wasm" not in [o.lower() for o in a["gpu_capability"]()["ops"]]
    assert "wasm" not in [o.lower() for o in a["cpu_capability"]()["ops"]]
    for n in (100, 100000):
        sel = a["select_backend"](_compare_graph(a, 8), n, _synth_profile())
        assert sel["backend"] in ("cpu", "gpu")


# 11. No N->backend thresholds; selectivity never routes.
@pytest.mark.fast
def test_no_n_threshold_selectivity_never_routes(kernel):
    a = kernel.alias
    prof = _synth_profile(cpu_a=0.001, gpu_a=10.0)
    backs = {a["select_backend"](_compare_graph(a, 8), n, prof)["backend"]
             for n in (1000, 10000, 100000, 1000000, 10000000)}
    assert backs == {"cpu"}  # cost structure decides, not N
    g = _compare_graph(a)
    lo = a["select_backend"](g, 1000, prof, hints={"selectivity": 0.1})
    hi = a["select_backend"](g, 1000, prof, hints={"selectivity": 0.9})
    assert (lo["backend"], lo["gpu_eligible"]) == \
        (hi["backend"], hi["gpu_eligible"])
