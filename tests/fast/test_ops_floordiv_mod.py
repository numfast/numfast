# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 5 gate: floor_div + mod (universal map fns, chunkable=true).

unit (floor semantics incl. negatives/floats/zero-div numpy parity/validity)
+ differential (seed 42, 100K) + capability (map chunkable) + ClickBench
hits_1m EventTime day/hour groupby (exact vs pandas). GPU code untouched;
WGSL-portability by capability read (SPEC 06 Map/MapBinary GPU ✅).
"""

import time
import warnings
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


def _run(a, jobs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero-div numpy parity: values compared
        return a["cpu_execute"](jobs)


# ---------- unit ----------

@pytest.mark.fast
def test_map_chunkable_and_fns_known(kernel):
    a = kernel.alias
    cap = a["cpu_capability"]()
    assert cap["chunkable_hints"]["map"] is True
    assert "map" in cap["ops"]
    with pytest.raises(ValueError, match="floor_div/mod"):
        a["ir_map"]("o", "x", fn="floordiv", value=2)


@pytest.mark.fast
def test_floor_semantics_negatives(kernel):
    a = kernel.alias
    x = np.array([-7, -6, 7, 6, 0], dtype=np.int32)
    d = np.array([3, 3, -3, -3, 5], dtype=np.int32)
    bufs = _run(a, [a["ir_series"]("x", x, dtype="int32"),
                    a["ir_series"]("d", d, dtype="int32"),
                    a["ir_map"]("q", "x", fn="floor_div", value="d"),
                    a["ir_map"]("r", "x", fn="mod", value="d")])
    assert bufs["q"].tolist() == [-3, -2, -3, -2, 0]  # floor, not trunc
    assert bufs["r"].tolist() == [2, 0, -2, 0, 0]  # sign of divisor
    assert bufs["q"].dtype == np.int32 and bufs["r"].dtype == np.int32
    # scalar form
    bufs2 = _run(a, [a["ir_series"]("x", x, dtype="int32"),
                     a["ir_map"]("q", "x", fn="floor_div", value=3),
                     a["ir_map"]("r", "x", fn="mod", value=3)])
    assert bufs2["q"].tolist() == [-3, -2, 2, 2, 0]
    assert bufs2["r"].tolist() == [2, 0, 1, 0, 0]


@pytest.mark.fast
def test_float_and_zero_div_numpy_parity(kernel):
    a = kernel.alias
    x = np.array([7.5, -7.5, 1.0, 0.0], dtype=np.float32)
    bufs = _run(a, [a["ir_series"]("x", x, dtype="float32"),
                    a["ir_map"]("q", "x", fn="floor_div", value=2.0),
                    a["ir_map"]("r", "x", fn="mod", value=2.0)])
    assert bufs["q"].tolist() == [3.0, -4.0, 0.0, 0.0]
    assert bufs["r"].tolist() == [1.5, 0.5, 1.0, 0.0]
    xi = np.array([1, -1, 0], dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_q = np.floor_divide(xi, np.int32(0))
        ref_r = np.remainder(xi, np.int32(0))
    bufs2 = _run(a, [a["ir_series"]("x", xi, dtype="int32"),
                     a["ir_map"]("q", "x", fn="floor_div", value=0),
                     a["ir_map"]("r", "x", fn="mod", value=0)])
    assert bufs2["q"].tolist() == ref_q.tolist()  # numpy parity, whatever it is
    assert bufs2["r"].tolist() == ref_r.tolist()


@pytest.mark.fast
def test_validity_propagates(kernel):
    a = kernel.alias
    bufs = _run(a, [a["ir_series"]("x", np.array([7, 8], dtype=np.int32),
                                   validity=[1, 0]),
                    a["ir_map"]("q", "x", fn="floor_div", value=3),
                    a["ir_map"]("r", "x", fn="mod", value=3)])
    assert bufs["q"].tolist() == [2, 2] and bufs["r"].tolist() == [1, 2]
    assert list(np.asarray(bufs["q#validity"], dtype=bool)) == [True, False]
    assert list(np.asarray(bufs["r#validity"], dtype=bool)) == [True, False]


# ---------- differential (seed 42, 100K) ----------

@pytest.mark.fast
def test_differential_100k_vs_numpy(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    x = rng.integers(-10_000, 10_000, size=100_000).astype(np.int32)
    d = rng.integers(-500, 500, size=100_000).astype(np.int32)
    d[d == 0] = 7
    s = time.perf_counter()
    bufs = _run(a, [a["ir_series"]("x", x, dtype="int32"),
                    a["ir_series"]("d", d, dtype="int32"),
                    a["ir_map"]("q", "x", fn="floor_div", value="d"),
                    a["ir_map"]("r", "x", fn="mod", value="d")])
    t_map = (time.perf_counter() - s) * 1000
    assert bufs["q"].tolist() == np.floor_divide(x, d).tolist()
    assert bufs["r"].tolist() == np.remainder(x, d).tolist()
    # identity: x == q*d + r (exact for all rows)
    assert (((bufs["q"].astype(np.int64) * d.astype(np.int64) + bufs["r"]) == x).all())
    print(f"\nstages ms: floordiv+mod-100K={t_map:.2f}")


# ---------- ClickBench hits_1m: EventTime -> day/hour ----------

@pytest.mark.fast
def test_clickbench_day_groupby_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    import pandas as pd

    a = kernel.alias
    df = pd.read_parquet(HITS, columns=["EventTime"])
    res = a["resident_prepare"]({"t": {"values": df["EventTime"].tolist()}})["t"]
    assert res["kind"] == "date"
    t0 = time.perf_counter()
    bufs = a["cpu_execute"]([a["ir_series"]("t", res["codes"], dtype="int32"),
                             a["ir_map"]("day", "t", fn="floor_div", value=86400),
                             a["ir_groupby"]("g", "t", "day", "count")])
    t_q = (time.perf_counter() - t0) * 1000
    want = (df["EventTime"].dt.floor("D").value_counts())
    got = {int(k) * 86400: int(v) for k, v in bufs["g"].items()}
    want = {int(v.timestamp()): int(c) for v, c in want.items()}
    assert got == want, f"day groupby: {got} != {want}"
    print(f"\nstages ms: floordiv-day+groupby={t_q:.2f} groups={got}")


@pytest.mark.fast
def test_clickbench_hour_groupby_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    import pandas as pd

    a = kernel.alias
    df = pd.read_parquet(HITS, columns=["EventTime"])
    res = a["resident_prepare"]({"t": {"values": df["EventTime"].tolist()}})["t"]
    t0 = time.perf_counter()
    bufs = a["cpu_execute"]([a["ir_series"]("t", res["codes"], dtype="int32"),
                             a["ir_map"]("tod", "t", fn="mod", value=86400),
                             a["ir_map"]("hr", "tod", fn="floor_div", value=3600),
                             a["ir_groupby"]("g", "t", "hr", "count")])
    t_q = (time.perf_counter() - t0) * 1000
    want = df["EventTime"].dt.hour.value_counts().to_dict()
    got = {int(k): int(v) for k, v in bufs["g"].items()}
    assert got == {int(k): int(v) for k, v in want.items()}, "hour groupby mismatch"
    print(f"\nstages ms: mod+floordiv-hour+groupby={t_q:.2f} groups={len(got)}")
