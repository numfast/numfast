# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 2 gate: generic TEXT->dictionary encode via resident_prepare.

unit (explicit/auto/pattern-precedence/numeric-untouched/NULL/determinism)
+ differential (seed 42, resident vs direct encode) + boundary restore +
ClickBench hits_1m queries (exact vs pandas). No per-column branches.
"""

import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
HITS = Path("C:/App/competitions/ClickBench/data/hits_1m.parquet")
SEED = 42


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


# ---------- unit: generic routing ----------

@pytest.mark.fast
def test_explicit_dtype_dictionary(kernel):
    a = kernel.alias
    res = a["resident_prepare"]({"t": {"values": ["b", "a", None], "dtype": "dictionary"}})
    r = res["t"]
    assert r["kind"] == "dictionary" and r["physical"] == "int32"
    assert r["codes"].dtype == np.int32 and list(r["codes"]) == [1, 0, 0]
    assert r["dictionary"] == ["a", "b"]
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, True, False]
    assert r["n"] == 3


@pytest.mark.fast
def test_auto_detect_numpy_and_list(kernel):
    a = kernel.alias
    u = np.array(["x", "y", "x"], dtype="U")
    r1 = a["resident_prepare"]({"t": {"values": u}})["t"]
    r2 = a["resident_prepare"]({"t": {"values": ["x", "y", "x"]}})["t"]
    r3 = a["resident_prepare"]({"t": {"values": np.array(
        ["x", "y", "x"], dtype=object)}})["t"]
    for r in (r1, r2, r3):
        assert r["kind"] == "dictionary"
        assert list(r["codes"]) == [0, 1, 0] and r["dictionary"] == ["x", "y"]
        assert r["validity"] is None  # all valid -> None (DELTA-3)


@pytest.mark.fast
def test_pattern_precedence_and_numeric_untouched(kernel):
    a = kernel.alias
    res = a["resident_prepare"]({
        "p": {"values": ["id1", "id2"], "prefix": "id"},
        "v": {"values": [3, 1], "dtype": "int32"},
    })
    assert res["p"]["kind"] == "pattern"
    assert res["v"]["kind"] == "series"
    assert list(res["v"]["codes"]) == [3, 1]


@pytest.mark.fast
def test_null_explicit_and_explicit_validity(kernel):
    a = kernel.alias
    r = a["resident_prepare"]({"t": {"values": ["a", None, "b", None]}})["t"]
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, False, True, False]
    back = a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"])
    assert back == ["a", None, "b", None]
    r2 = a["resident_prepare"]({"t": {"values": ["a", "b"], "validity": [1, 0]}})["t"]
    assert list(np.asarray(r2["validity"], dtype=bool)) == [True, False]


@pytest.mark.fast
def test_deterministic_across_runs(kernel):
    a = kernel.alias
    vals = ["k3", "k1", None, "k2", "k1"]
    r1 = a["resident_prepare"]({"t": {"values": vals}})["t"]
    r2 = a["resident_prepare"]({"t": {"values": list(vals)}})["t"]
    assert r1["dictionary"] == r2["dictionary"] == ["k1", "k2", "k3"]
    assert list(r1["codes"]) == list(r2["codes"])


# ---------- differential (seed 42) ----------

@pytest.mark.fast
def test_differential_resident_vs_direct(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    pool = [f"w-{i:03d}" for i in range(200)]
    vals = [None if rng.random() < 0.1 else pool[int(rng.integers(0, len(pool)))]
            for _ in range(30_000)]
    s = time.perf_counter()
    res = a["resident_prepare"]({"t": {"values": vals}})["t"]
    t_res = (time.perf_counter() - s) * 1000
    ref = a["dictionary_encode"](vals)
    assert list(res["codes"]) == list(ref["codes"])
    assert res["dictionary"] == ref["values"]
    back = a["dictionary_decode"](res["codes"], res["dictionary"], res["validity"])
    assert back == vals
    print(f"\nstages ms: resident-TEXT-30K={t_res:.2f} D={len(ref['values'])}")


# ---------- ClickBench hits_1m ----------

def _cb_col(col):
    import pandas as pd

    df = pd.read_parquet(HITS, columns=[col])
    return [None if (v is None or (isinstance(v, float) and v != v)) else str(v)
            for v in df[col].tolist()]


@pytest.mark.fast
def test_clickbench_url_via_resident_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    t0 = time.perf_counter()
    url = _cb_col("URL")
    t_load = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    res = a["resident_prepare"]({"url": {"values": url}})["url"]
    t_enc = (time.perf_counter() - t0) * 1000
    assert res["kind"] == "dictionary" and res["codes"].dtype == np.int32
    target = res["dictionary"][len(res["dictionary"]) // 2]
    want = sum(1 for v in url if v == target)
    t0 = time.perf_counter()
    code = res["dictionary"].index(target)
    bufs = a["cpu_execute"]([a["ir_series"]("codes", res["codes"], dtype="int32"),
                             a["ir_compare"]("m", "codes", int(code), op="==")])
    n = int(np.count_nonzero(bufs["m"].to_array()))
    t_cmp = (time.perf_counter() - t0) * 1000
    assert n == want
    print(f"\nstages ms: load={t_load:.1f} resident-encode={t_enc:.1f} "
          f"compare={t_cmp:.2f} N={len(url)} D={len(res['dictionary'])} hits={n}")


@pytest.mark.fast
def test_clickbench_model_groupby_via_resident_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    col = _cb_col("MobilePhoneModel")
    res = a["resident_prepare"]({"m": {"values": col}})["m"]
    assert res["kind"] == "dictionary"
    import pandas as pd

    want = pd.Series([v for v in col if v is not None]).value_counts().to_dict()
    nodes = [a["ir_series"]("codes", res["codes"], dtype="int32",
                            validity=None if res["validity"] is None
                            else np.asarray(res["validity"], dtype=np.int8).tolist()),
             a["ir_groupby"]("g", "codes", "codes", "count")]
    bufs = a["cpu_execute"](nodes)
    got = {res["dictionary"][int(k)]: int(v) for k, v in bufs["g"].items()}
    assert got == want, f"groupby mismatch: {len(got)} vs {len(want)} keys"
    print(f"\nMobilePhoneModel N={len(col)} D={len(res['dictionary'])} groups={len(got)}")
