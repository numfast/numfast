# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden e2e: Schema -> IR -> Planner -> Runtime -> CPU.

Expected values hand-derived (NumPy semantics per spec 07/08);
old NumFast used only as behavior reference, no code copied.
Stage breakdown (ms) printed per run. Seed: no RNG in skeleton path.
"""

import time
from pathlib import Path

import pytest

from harness import assert_float_close, assert_int_exact, load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _stages(alias, jobs, n, backend="auto"):
    t = {}
    s = time.perf_counter()
    schema = alias["column_schema"]("v", jobs[0]["params"].get("dtype", "int32"))
    t["schema"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    graph = alias["compile"](jobs)
    t["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    graph = alias["optimize"](graph)
    t["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    sel = alias["select_backend"](graph, n)
    t["planner"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    res = alias["evaluate"](graph, backend, n)
    t["execute"] = (time.perf_counter() - s) * 1000
    print("\nstages ms: " + " ".join(f"{k}={v:.3f}" for k, v in t.items()))
    return schema, graph, sel, res


@pytest.mark.fast
def test_schema_scaled_roundtrip_exact(kernel):
    a = kernel.alias
    phys = a["schema_transform"]([1.0, 2.0, 3.0, 4.0], 0.5, 0)
    assert [int(v) for v in phys] == [2, 4, 6, 8]
    back = a["schema_untransform"](phys, 0.5, 0)
    for got, want in zip(back, [1.0, 2.0, 3.0, 4.0]):
        assert_float_close(got, want, PROFILE, "f64", label="untransform")
    col = a["column_schema"]("price", "float32", 0.5, 0)
    assert (col["logical"], col["physical"], col["scale"], col["offset"], col["auto"]) == (
        "float32", "int32", 0.5, 0, True,
    )


@pytest.mark.fast
def test_e2e_int_map_reduce_exact(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4]), a["ir_map"]("m", "s", "mul", 2), a["ir_reduce"]("r", "m", "sum")]
    _, graph, sel, res = _stages(a, jobs, 4)
    assert graph["outputs"] == ["r"]
    assert sel["backend"] == "cpu"
    assert_int_exact(res["result"], 20, label="sum([1,2,3,4]*2)")
    assert res["execution_info"]["actual"] == "cpu"


@pytest.mark.fast
def test_e2e_float_map_mean_tolerance(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("s", [1.5, 2.5, 3.0], "float32"),
        a["ir_map"]("m", "s", "mul", 2.0),
        a["ir_reduce"]("r", "m", "mean"),
    ]
    _, _, _, res = _stages(a, jobs, 3)
    assert_float_close(res["result"], 14.0 / 3.0, PROFILE, "f32", label="mean([1.5,2.5,3]*2)")


@pytest.mark.fast
def test_planner_selects_cpu_with_why(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2]), a["ir_reduce"]("r", "s", "sum")]
    graph = a["compile"](jobs)
    sel = a["select_backend"](graph, 2)
    assert sel["backend"] == "cpu"
    assert sel["reason"]
    # The branch keys on the ROUTING verdict, not on whether a profile FILE
    # exists. Those two are deliberately different since the profile-validity
    # change: `calibrate_info()['profile']` describes the file, which is a
    # complete measured profile on every machine, while
    # `calibrate_info()['routing']` is whether it may drive routing HERE. On a
    # foreign machine the old condition took the else-branch and asserted
    # matched is True against a stub selection -- the file exists, the routing
    # decision is stub, and the test was reading the wrong one.
    info = a["calibrate_info"]()
    if info["routing"]["routing"]:
        assert sel["profile"]["matched"] is True, (info["routing"], sel["profile"])
        assert sel["profile"]["version"] == "calibrated_v1"
        assert sel["cost_estimate"]["cpu"] is not None, sel["cost_estimate"]
    else:
        # Nothing may drive routing here, and the selection says so rather than
        # reporting a calibrated profile it did not use.
        assert sel["profile"]["matched"] is False, (info["routing"], sel["profile"])
        assert sel["profile"]["version"] == "stub", sel["profile"]
        assert sel["cost_estimate"] == {"cpu": None, "gpu": None}
        assert "NUMFAST_CALIBRATION_DIR" in sel["profile"]["warning"], \
            sel["profile"]
    assert "cost_estimate" in sel
    rep = a["explain"](graph, {"actual": "cpu", "requested": "auto",
                               "reason": sel["reason"], "cost_estimate": sel["cost_estimate"],
                               "profile": sel["profile"]})
    assert "reduce" in rep and "cpu" in rep
