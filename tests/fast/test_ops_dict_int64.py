# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage int64 gate: Dictionary<int64> on the proven TEXT ABI (second real type).

Same resident envelope (codes int32[N] + validity sidecar + int64[D]
sorted-unique via np.unique, D<2^31); TEXT ABI untouched. Numeric domain:
lookup / equality / IN via searchsorted (D-scale), range via searchsorted,
min/max O(1) from LUT borders. SUM/AVG/arithmetic ONLY via LUT->gather->
numeric kernel (ir_gather + ir_groupby/ir_reduce); SUM(codes) is forbidden
(codes are ranks, not values). GPU int64 gather NOT implemented (CPU only).
No generic Dictionary<T>, no packed-bit stage (int32 codes suffice).

unit + differential (seed 42) + parity vs pandas on ClickBench hits_1m
(IsRefresh/CounterID/OS/RegionID/UserID/WatchID): memory, encode/decode,
==/IN/range, GROUP BY/JOIN over codes, SUM via gather (int64-wrap parity).
"""

import time
from pathlib import Path

import numpy as np
import pytest

from harness import load_profile  # noqa: F401  (profile-parity with suite)

APP_DIR = str(Path(__file__).resolve().parents[2])
HITS = Path("C:/App/competitions/ClickBench/data/hits_1m.parquet")
SEED = 42
INT_COLS = ["IsRefresh", "CounterID", "OS", "RegionID", "UserID", "WatchID"]


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


# ---------- unit: encode ----------

@pytest.mark.fast
def test_int64_registered_with_range(kernel):
    a = kernel.alias
    assert "dict_range_codes" in a
    assert kernel.metadata["Dictionary"]["version"] == "0.1.0"


@pytest.mark.fast
def test_encode_small_exact(kernel):
    a = kernel.alias
    r = a["dictionary_encode"]([5, 3, 5, None, 3, -7])
    assert r["dtype"] == "int64"
    assert r["codes"].dtype == np.int32 and list(r["codes"]) == [2, 1, 2, 0, 1, 0]
    assert list(np.asarray(r["dictionary"])) == [-7, 3, 5]
    assert r["dictionary"].dtype == np.int64
    assert r["values"] == [-7, 3, 5]
    assert list(np.asarray(r["validity"], dtype=bool)) == [True] * 3 + [False] + [True] * 2
    assert r["metadata"] == {"encoding": "dictionary-sorted-v1", "d": 3, "n": 6,
                             "sorted": True, "nulls": 1}


@pytest.mark.fast
def test_encode_ndarray_and_all_null_and_empty(kernel):
    a = kernel.alias
    r = a["dictionary_encode"](np.array([7, 7, 9], dtype=np.int64))
    assert r["dtype"] == "int64" and list(r["codes"]) == [0, 0, 1]
    assert r["validity"] is None
    r2 = a["dictionary_encode"]([None, None])
    assert r2["dtype"] == "text"  # empty stays TEXT (legacy, untouched)
    r3 = a["dictionary_encode"](np.zeros(0, dtype=np.int64))
    assert r3["dtype"] == "int64" and r3["codes"].size == 0
    assert r3["dictionary"].dtype == np.int64


@pytest.mark.fast
def test_encode_explicit_validity_and_rejections(kernel):
    a = kernel.alias
    r = a["dictionary_encode"]([4, 5, 6], validity=[1, 0, 1])
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, False, True]
    assert a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"]) == [4, None, 6]
    with pytest.raises(ValueError, match="validity size"):
        a["dictionary_encode"]([1], validity=[1, 0])
    for bad in ([1.5, 2], ["a", 1], [True, False], [1, "x"]):
        with pytest.raises(ValueError, match="string column"):
            a["dictionary_encode"](bad)
    with pytest.raises(ValueError, match="out of range"):
        a["dictionary_decode"](np.array([7], dtype=np.int32),
                               np.array([1, 2], dtype=np.int64))


@pytest.mark.fast
def test_decode_and_metadata_carriers(kernel):
    a = kernel.alias
    r = a["dictionary_encode"]([10, -1, 10])
    assert a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"]) == [10, -1, 10]
    assert a["dictionary_decode"](r["codes"], r["values"], r["validity"]) == [10, -1, 10]
    assert a["dictionary_decode"](r["codes"], {"dtype": "int64",
                                               "dictionary": r["dictionary"]},
                                  r["validity"]) == [10, -1, 10]
    assert a["dictionary_metadata"](r["dictionary"]) == {
        "encoding": "dictionary-sorted-v1", "d": 2, "sorted": True, "unique": True}
    assert a["dictionary_metadata"]([3, 1])["sorted"] is False
    # TEXT ABI untouched spot-check
    t = a["dictionary_encode"](["b", "a"])
    assert t["dtype"] == "text" and hasattr(t["dictionary"], "utf8_data")


@pytest.mark.fast
def test_int64_decode_d0_returns_nulls_not_out_of_range(kernel):
    """M4b regression: D=0 -> every row NULL, not "code 0 out of range D=0".

    An empty int64 LUT cannot have a value at code 0, so under DELTA-3 every
    code is the NULL placeholder. True on both the validity-supplied and the
    validity-omitted path.
    """
    a = kernel.alias
    lut0 = {"dtype": "int64", "dictionary": np.zeros(0, dtype=np.int64)}
    codes = np.array([0, 0], dtype=np.int32)
    assert a["dictionary_decode"](codes, lut0) == [None, None]
    assert a["dictionary_decode"](codes, lut0, [False, False]) == [None, None]
    assert a["dictionary_decode"](codes, np.zeros(0, dtype=np.int64)) == [None, None]
    with pytest.raises(ValueError, match="out of range"):
        a["dictionary_decode"](np.array([-1], dtype=np.int32), lut0)
    with pytest.raises(ValueError, match="validity size"):
        a["dictionary_decode"](codes, lut0, [False])


# ---------- unit: numeric domain ----------

@pytest.mark.fast
def test_lookup_equal_in_range_minmax(kernel):
    a = kernel.alias
    lut = np.array([-7, 3, 5, 100], dtype=np.int64)
    assert a["dict_lookup"](lut, 3) == 1
    assert a["dict_lookup"]([5, 6], 6) == 1  # [int] list carrier
    assert a["dict_lookup"](lut, 99) is None
    assert list(a["dict_equal_codes"](lut, 5)) == [2]
    assert list(a["dict_equal_codes"](lut, 99)) == []
    assert list(a["dict_range_codes"](lut, 0, 5)) == [1, 2]
    assert list(a["dict_range_codes"](lut, -100, 100)) == [0, 1, 2, 3]
    assert list(a["dict_range_codes"](lut, 10, 5)) == []  # lo > hi -> empty
    assert a["dict_min_max"](lut) == (-7, 100)
    assert a["dict_min_max"](np.zeros(0, dtype=np.int64)) == (None, None)
    with pytest.raises(ValueError, match="must be int"):
        a["dict_lookup"](lut, "3")
    with pytest.raises(ValueError, match="must be int"):
        a["dict_lookup"](lut, True)
    with pytest.raises(ValueError, match="bounds must be int"):
        a["dict_range_codes"](lut, 0.5, 5)
    with pytest.raises(ValueError, match="needs an int64 dictionary"):
        a["dict_range_codes"](["a", "b"], 0, 5)
    # IN = equal-codes composed through the numeric member mask (row-side)
    codes = np.array([0, 1, 3, 1, 2], dtype=np.int32)
    allowed = np.concatenate([a["dict_equal_codes"](lut, 3),
                              a["dict_equal_codes"](lut, 100)]).astype(np.int32)
    m = a["codes_member_mask"](codes, allowed, None)
    assert list(np.asarray(m, dtype=bool)) == [False, True, True, True, False]


@pytest.mark.fast
def test_text_only_ops_reject_int64(kernel):
    a = kernel.alias
    lut = np.array([1, 2], dtype=np.int64)
    for op, args in [("dict_contains", ("x",)), ("dict_startswith", ("x",)),
                     ("dict_not_empty_codes", ()), ("dict_len_lut", ()),
                     ("dict_ordering", ())]:
        with pytest.raises(ValueError, match="TEXT-only"):
            a[op](lut, *args)


@pytest.mark.fast
def test_sum_codes_forbidden_sum_lut_gather(kernel):
    """SUM(codes) is meaningless (ranks) and FORBIDDEN; SUM = gather LUT first."""
    a = kernel.alias
    r = a["dictionary_encode"]([30, 10, 30, 20])
    codes, lut = r["codes"], r["dictionary"]
    assert int(codes.astype(np.int64).sum()) != 90  # ranks sum != value sum
    bufs = a["cpu_execute"]([a["ir_series"]("lut", lut, dtype="int64"),
                             a["ir_series"]("c", codes, dtype="int32"),
                             a["ir_gather"]("rows", "lut", "c"),
                             a["ir_series"]("k0", np.zeros(4, dtype=np.int32)),
                             a["ir_groupby"]("g", "rows", "k0", "sum")])
    assert int(bufs["g"][0]) == 90
    # int64 code order == value order: sort over codes sorts values
    bufs2 = a["cpu_execute"]([a["ir_series"]("c2", codes, dtype="int32"),
                              a["ir_sort"]("p", "c2")])
    assert [int(lut[int(codes[i])]) for i in np.asarray(bufs2["p"]).tolist()] == [10, 20, 30, 30]


# ---------- differential (seed 42) ----------

@pytest.mark.fast
def test_differential_vs_numpy_reference(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    pool = rng.integers(-10**12, 10**12, size=500, dtype=np.int64)
    vals = [None if rng.random() < 0.08 else int(pool[int(rng.integers(0, len(pool)))])
            for _ in range(50_000)]
    s = time.perf_counter()
    r = a["dictionary_encode"](vals)
    t_enc = (time.perf_counter() - s) * 1000
    ref_lut = np.unique(np.array([v for v in vals if v is not None], dtype=np.int64))
    assert np.array_equal(np.asarray(r["dictionary"]), ref_lut)
    idx = {int(v): i for i, v in enumerate(ref_lut.tolist())}
    assert [int(c) for c, v in zip(r["codes"].tolist(), vals) if v is not None] == [
        idx[v] for v in vals if v is not None]
    back = a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"])
    assert back == vals
    # range/equality/minmax vs row scan
    lo, hi, key = int(ref_lut[len(ref_lut) // 4]), int(ref_lut[3 * len(ref_lut) // 4]), \
        int(ref_lut[len(ref_lut) // 2])
    assert (int(r["dictionary"][a["dict_lookup"](r["dictionary"], key)]) == key)
    n_eq = int(np.count_nonzero(a["codes_member_mask"](
        r["codes"], a["dict_equal_codes"](r["dictionary"], key), r["validity"])))
    assert n_eq == sum(1 for v in vals if v == key)
    n_rg = int(np.count_nonzero(a["codes_member_mask"](
        r["codes"], a["dict_range_codes"](r["dictionary"], lo, hi), r["validity"])))
    assert n_rg == sum(1 for v in vals if v is not None and lo <= v <= hi)
    assert a["dict_min_max"](r["dictionary"]) == (int(ref_lut[0]), int(ref_lut[-1]))
    print(f"\nstages ms: diff-int64-50K={t_enc:.1f} D={len(ref_lut)} eq={n_eq} range={n_rg}")


# ---------- resident hint ----------

@pytest.mark.fast
def test_resident_hint_dictionary_vs_series(kernel):
    a = kernel.alias
    v = np.array([5, 3, 5], dtype=np.int64)
    res = a["resident_prepare"]({"v": {"values": v, "encoding": "dictionary"}})["v"]
    assert res["kind"] == "dictionary" and res["dtype"] == "int64"
    assert res["physical"] == "int32" and res["codes"].dtype == np.int32
    assert list(np.asarray(res["dictionary"])) == [3, 5] and res["n"] == 3
    ser = a["resident_prepare"]({"v": {"values": v}})["v"]
    assert ser["kind"] == "series"  # no hint -> legacy series, no churn
    back = a["dictionary_decode"](res["codes"], res["dictionary"], res["validity"])
    assert back == [5, 3, 5]


# ---------- ClickBench hits_1m ----------

def _cb_all():
    import pandas as pd

    t0 = time.perf_counter()
    df = pd.read_parquet(HITS, columns=INT_COLS)
    return df, (time.perf_counter() - t0) * 1000


@pytest.mark.fast
def test_clickbench_memory_encode_decode(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    df, t_load = _cb_all()
    stages = [f"load={t_load:.0f}"]
    for col in INT_COLS:
        v = df[col].to_numpy()
        t0 = time.perf_counter()
        r = a["dictionary_encode"](v)
        t_enc = (time.perf_counter() - t0) * 1000
        raw_mb = v.nbytes / 1e6
        dict_mb = (r["codes"].nbytes + np.asarray(r["dictionary"]).nbytes) / 1e6
        t0 = time.perf_counter()
        back = a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"])
        t_dec = (time.perf_counter() - t0) * 1000
        assert back == v.tolist()
        assert r["metadata"]["d"] == int(df[col].nunique())
        stages.append(f"{col}:D={r['metadata']['d']} raw={raw_mb:.1f}MB "
                      f"dict={dict_mb:.2f}MB enc={t_enc:.0f}ms dec={t_dec:.0f}ms")
    print("\nstages ms: " + " | ".join(stages))


@pytest.mark.fast
def test_clickbench_predicates_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    df, _ = _cb_all()
    import pandas as pd

    stages = []
    for col in INT_COLS:
        v = df[col].to_numpy()
        r = a["dictionary_encode"](v)
        codes, lut = r["codes"], np.asarray(r["dictionary"])
        key = int(lut[len(lut) // 2])
        t0 = time.perf_counter()
        m_eq = a["codes_member_mask"](codes, a["dict_equal_codes"](lut, key), r["validity"])
        n_eq = int(np.count_nonzero(m_eq))
        keys = [int(lut[0]), int(lut[len(lut) // 3]), int(lut[-1]) if len(lut) > 2 else int(lut[0])]
        allowed = np.concatenate([a["dict_equal_codes"](lut, k) for k in dict.fromkeys(keys)]
                                 ).astype(np.int32)
        n_in = int(np.count_nonzero(a["codes_member_mask"](codes, allowed, r["validity"])))
        lo, hi = int(lut[len(lut) // 4]), int(lut[3 * len(lut) // 4])
        n_rg = int(np.count_nonzero(a["codes_member_mask"](
            codes, a["dict_range_codes"](lut, lo, hi), r["validity"])))
        t_q = (time.perf_counter() - t0) * 1000
        s = pd.Series(v)
        assert n_eq == int((s == key).sum())
        assert n_in == int(s.isin(list(dict.fromkeys(keys))).sum())
        assert n_rg == int(((s >= lo) & (s <= hi)).sum())
        assert a["dict_min_max"](lut) == (int(s.min()), int(s.max()))
        stages.append(f"{col}:eq={n_eq} in={n_in} range={n_rg} q={t_q:.0f}ms")
    print("\nstages ms: " + " | ".join(stages))


@pytest.mark.fast
def test_clickbench_groupby_sum_via_gather_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    import pandas as pd

    df, _ = _cb_all()
    stages = []
    for col in INT_COLS:
        v = df[col].to_numpy()
        r = a["dictionary_encode"](v)
        codes, lut = r["codes"], np.asarray(r["dictionary"])
        t0 = time.perf_counter()
        bufs = a["cpu_execute"]([a["ir_series"]("c", codes, dtype="int32"),
                                 a["ir_groupby"]("g", "c", "c", "count")])
        got_cnt = {int(lut[int(k)]): int(cnt) for k, cnt in bufs["g"].items()}
        bufs2 = a["cpu_execute"]([a["ir_series"]("lut", lut, dtype="int64"),
                                  a["ir_series"]("c2", codes, dtype="int32"),
                                  a["ir_gather"]("rows", "lut", "c2"),
                                  a["ir_series"]("k0", np.zeros(codes.size, dtype=np.int32)),
                                  a["ir_groupby"]("s", "rows", "k0", "sum")])
        got_sum = int(bufs2["s"][0])
        t_q = (time.perf_counter() - t0) * 1000
        s = pd.Series(v)
        assert got_cnt == s.value_counts().to_dict()
        assert got_sum == int(s.sum())  # int64-wrap parity with pandas
        bufs3 = a["cpu_execute"]([a["ir_series"]("rows2", lut[codes], dtype="int64"),
                                  a["ir_series"]("c3", codes, dtype="int32"),
                                  a["ir_groupby"]("gs", "rows2", "c3", "sum")])
        want_grp = s.groupby(s).sum().to_dict()
        assert {int(lut[int(k)]): int(x) for k, x in bufs3["gs"].items()} == want_grp
        stages.append(f"{col}:groups={len(got_cnt)} sum={got_sum} q={t_q:.0f}ms")
    print("\nstages ms: " + " | ".join(stages))


@pytest.mark.fast
def test_clickbench_join_over_codes_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    import pandas as pd

    df, _ = _cb_all()
    v = df["CounterID"].to_numpy()
    r = a["dictionary_encode"](v)
    codes, lut = r["codes"], np.asarray(r["dictionary"])
    d = len(lut)
    t0 = time.perf_counter()
    build, _build_ms = a["join_build"](np.arange(d, dtype=np.int32), lut.astype(np.int32))
    n_probe = 50_000
    probe_keys, probe_v1 = codes[:n_probe].copy(), np.arange(n_probe, dtype=np.int32) + 1
    (keys, got_v1, got_v2), t_join = a["join_inner"](probe_keys, probe_v1, build)
    t_q = (time.perf_counter() - t0) * 1000
    assert np.array_equal(np.asarray(got_v2), lut[np.asarray(probe_keys)])
    chk1, chk2 = a["join_chk"](np.asarray(got_v1), np.asarray(got_v2))
    assert chk1 + chk2 == int(np.asarray(got_v1).astype(np.int64).sum()
                              + np.asarray(got_v2).astype(np.int64).sum())
    mapping = pd.DataFrame({"k": lut.tolist(), "v": lut.tolist()})
    want = pd.DataFrame({"k": [int(v[i]) for i in range(n_probe)]}).merge(
        mapping, on="k", how="inner")["v"].tolist()
    assert np.asarray(got_v2).tolist() == want
    print(f"\nstages ms: join-codes-50K={t_q:.0f} D={d} rows={len(np.asarray(keys))}")


@pytest.mark.fast
def test_clickbench_avg_via_sum_count(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    df, _ = _cb_all()
    stages = []
    # UserID/WatchID excluded: true sums exceed int64 (wrap parity already
    # covers SUM; AVG over a wrapped sum is meaningless -- same as pandas).
    for col in ["IsRefresh", "CounterID", "OS", "RegionID"]:
        v = df[col].to_numpy()
        r = a["dictionary_encode"](v)
        codes, lut = r["codes"], np.asarray(r["dictionary"])
        bufs = a["cpu_execute"]([a["ir_series"]("lut", lut, dtype="int64"),
                                 a["ir_series"]("c", codes, dtype="int32"),
                                 a["ir_gather"]("rows", "lut", "c"),
                                 a["ir_series"]("k0", np.zeros(codes.size, dtype=np.int32)),
                                 a["ir_groupby"]("s", "rows", "k0", "sum"),
                                 a["ir_reduce"]("n", "c", op="count")])
        avg = float(int(bufs["s"][0])) / int(bufs["n"])
        want_avg = float(sum(int(x) for x in v)) / v.size  # exact-int reference
        assert abs(avg - want_avg) <= 1e-12 * max(1.0, abs(want_avg))  # float64 display
        stages.append(f"{col}:avg={avg}")
    print("\nAVG via LUT->gather->groupby-sum/count: " + " | ".join(stages))
