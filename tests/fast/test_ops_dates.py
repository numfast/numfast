# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 4 gate: DATE as int32 epoch seconds (date-epoch-s-v1, single variant).

unit (mixed inputs/midnight/naive-UTC/bounds/tz-reject/NULL/determinism) +
differential (seed 42 vs timestamp()) + boundary decode + ClickBench
hits_1m EventTime/EventDate queries (exact vs pandas). No calendar.
"""

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
HITS = Path("C:/App/competitions/ClickBench/data/hits_1m.parquet")
SEED = 42
EPOCH = dt.datetime(1970, 1, 1)


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


# ---------- unit ----------

@pytest.mark.fast
def test_date_registered_and_encoding_tag(kernel):
    a = kernel.alias
    assert "date_encode" in a and "date_decode" in a
    r = a["date_encode"]([dt.date(2013, 7, 15)])
    assert r["codes"].dtype == np.int32
    assert r["metadata"]["encoding"] == "date-epoch-s-v1"


@pytest.mark.fast
def test_mixed_inputs_midnight_naive_utc(kernel):
    a = kernel.alias
    d = dt.date(2013, 7, 15)
    r = a["date_encode"]([d, dt.datetime(2013, 7, 15), np.datetime64("2013-07-15"), None])
    want = int((dt.datetime(2013, 7, 15) - EPOCH).total_seconds())
    assert list(r["codes"]) == [want, want, want, 0]
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, True, True, False]
    assert r["metadata"] == {"encoding": "date-epoch-s-v1", "n": 4, "nulls": 1}
    # midnight normalization: time-of-day kept for datetimes, dropped for dates
    r2 = a["date_encode"]([dt.datetime(2013, 7, 15, 20, 0, 0)])
    assert list(r2["codes"]) == [want + 72000]
    assert r2["validity"] is None


@pytest.mark.fast
def test_bounds_tz_and_type_rejects(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="exceed int32"):
        a["date_encode"]([dt.datetime(2100, 1, 1)])
    with pytest.raises(ValueError, match="tz-aware"):
        a["date_encode"]([dt.datetime(2013, 7, 15, tzinfo=dt.timezone.utc)])
    with pytest.raises(ValueError, match="DATE/DATETIME"):
        a["date_encode"]([1373832000])
    with pytest.raises(ValueError, match="DATE/DATETIME"):
        a["date_encode"](["2013-07-15"])
    with pytest.raises(ValueError, match="validity size"):
        a["date_encode"]([dt.date(2013, 7, 15)], validity=[1, 0])
    r = a["date_encode"]([])
    assert r["codes"].size == 0 and r["metadata"]["nulls"] == 0


@pytest.mark.fast
def test_decode_roundtrip_and_determinism(kernel):
    a = kernel.alias
    vals = [dt.datetime(2013, 7, 14, 20), None, dt.date(2013, 7, 15)]
    r = a["date_encode"](vals)
    back = a["date_decode"](r["codes"], r["validity"])
    assert back == [dt.datetime(2013, 7, 14, 20), None, dt.datetime(2013, 7, 15)]
    r2 = a["date_encode"](list(vals))
    assert list(r2["codes"]) == list(r["codes"])


@pytest.mark.fast
def test_resident_date_kind(kernel):
    a = kernel.alias
    res = a["resident_prepare"]({"d": {"values": [dt.date(2013, 7, 15), None],
                                       "dtype": "date"}})
    r = res["d"]
    assert r["kind"] == "date" and r["physical"] == "int32"
    assert r["encoding"] == "date-epoch-s-v1" and r["n"] == 2
    res2 = a["resident_prepare"]({"d": {"values": np.array(["2013-07-15"],
                                                           dtype="datetime64[D]")}})
    assert res2["d"]["kind"] == "date"


# ---------- differential (seed 42) ----------

@pytest.mark.fast
def test_differential_vs_timestamp(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    base = dt.datetime(2013, 1, 1)
    vals = [None if rng.random() < 0.05
            else base + dt.timedelta(seconds=int(rng.integers(0, 60 * 86400)))
            for _ in range(20_000)]
    s = time.perf_counter()
    r = a["date_encode"](vals)
    t_enc = (time.perf_counter() - s) * 1000
    ref = [0 if v is None else int((v - EPOCH).total_seconds()) for v in vals]
    assert list(r["codes"]) == ref
    back = a["date_decode"](r["codes"], r["validity"])
    assert back == [None if v is None else v for v in vals]
    print(f"\nstages ms: date-encode-20K={t_enc:.2f}")


# ---------- ClickBench hits_1m ----------

@pytest.mark.fast
def test_clickbench_eventtime_range_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    import pandas as pd

    a = kernel.alias
    df = pd.read_parquet(HITS, columns=["EventTime"])
    raw = df["EventTime"].tolist()
    t0 = time.perf_counter()
    res = a["resident_prepare"]({"t": {"values": list(raw)}})["t"]
    t_enc = (time.perf_counter() - t0) * 1000
    assert res["kind"] == "date" and res["validity"] is None
    lo = int((dt.datetime(2013, 7, 14, 22) - EPOCH).total_seconds())
    hi = int((dt.datetime(2013, 7, 15, 2) - EPOCH).total_seconds())
    want = int(((df["EventTime"] >= pd.Timestamp("2013-07-14 22:00")) &
                (df["EventTime"] < pd.Timestamp("2013-07-15 02:00"))).sum())
    t0 = time.perf_counter()
    bufs = a["cpu_execute"]([a["ir_series"]("t", res["codes"], dtype="int32"),
                             a["ir_compare"]("ge", "t", lo, op=">="),
                             a["ir_compare"]("lt", "t", hi, op="<"),
                             a["ir_mask"]("m", "ge", "lt", op="and"),
                             a["ir_filter"]("f", "t", "m")])
    n = int(bufs["f"].size)
    t_q = (time.perf_counter() - t0) * 1000
    assert n == want, f"EventTime range: {n} != pandas {want}"
    print(f"\nstages ms: encode={t_enc:.1f} range-compare+filter={t_q:.2f} "
          f"N={len(raw)} hits={n}")


@pytest.mark.fast
def test_clickbench_eventdate_equality_and_decode_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    import pandas as pd

    a = kernel.alias
    df = pd.read_parquet(HITS, columns=["EventDate"])
    col = df["EventDate"].tolist()
    res = a["resident_prepare"]({"d": {"values": list(col)}})["d"]
    assert res["kind"] == "date"
    day = int((dt.datetime(2013, 7, 15) - EPOCH).total_seconds())
    want = int((pd.to_datetime(df["EventDate"]) == pd.Timestamp("2013-07-15")).sum())
    bufs = a["cpu_execute"]([a["ir_series"]("d", res["codes"], dtype="int32"),
                             a["ir_compare"]("m", "d", day, op="=="),
                             a["ir_filter"]("f", "d", "m")])
    assert int(bufs["f"].size) == want
    back = a["date_decode"](res["codes"][:1000],
                            None if res["validity"] is None
                            else np.asarray(res["validity"])[:1000])
    ref = [dt.datetime.combine(v, dt.time()) if not pd.isna(v) else None
           for v in col[:1000]]
    assert back == ref, "boundary decode mismatch"
    print(f"\nEventDate == 2013-07-15: hits={want} decode-head-1000 exact")
