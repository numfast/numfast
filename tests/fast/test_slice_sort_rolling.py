# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Vertical slice PoC: sort_by(timestamp) -> rolling_sum -> rolling_mean.

Scope: ONLY existing IR (ir_sort/ir_gather/ir_rolling_sum/ir_map). No new
nodes, no Window framework, no Planner/GPU/WASM changes, no chunking
redesign (single-chunk CPU, honest PoC, not a contract).

Helpers below are test-local composition (not shipped public API).
Oracle: independent NumPy (no numfast imports inside oracle fns).
Seed 42 everywhere RNG is used. Stage breakdown printed per run.
"""

import math
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _bufs(a, jobs):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


# -- independent oracle (NumPy only, no numfast) ---------------------------

def oracle_stable_perm(keys, validity=None, descending=False):
    """Stable perm, invalid rows last in input order (spec: sort contract)."""
    keys = np.asarray(keys)
    n = keys.size
    if validity is not None:
        m = np.asarray(validity, dtype=bool)
        vpos = np.flatnonzero(m)
        ipos = np.flatnonzero(~m)
    else:
        vpos = np.arange(n)
        ipos = np.zeros(0, dtype=np.int64)
    kv = keys[vpos] if n else keys[:0]
    asc = np.argsort(kv, kind="stable")
    if descending and asc.size:
        sk = kv[asc]
        ch = np.empty(sk.shape, dtype=bool)
        ch[0] = True
        if sk.size > 1:
            ch[1:] = sk[1:] != sk[:-1]
        grp = np.cumsum(ch)
        rev = grp.max() - grp
        asc = asc[np.argsort(rev, kind="stable")]
    perm = np.empty(n, dtype=np.int64)
    perm[: vpos.size] = vpos[asc]
    perm[vpos.size :] = ipos
    return perm.astype(np.int32)


def oracle_rolling_sum(vals, window, min_periods=None):
    """Pandas-like: NaN in window -> NaN; count<min_periods -> NaN."""
    vals = np.asarray(vals, dtype=np.float64)
    mp = window if min_periods is None else min_periods
    n = vals.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        cnt = i - lo + 1
        if cnt >= mp and not np.isnan(vals[lo : i + 1]).any():
            out[i] = float(np.nansum(vals[lo : i + 1]))
    return out


def _isnan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _assert_rolling_close(got, ref, label):
    assert len(got) == len(ref), f"{label}: len {len(got)} != {len(ref)}"
    for i, (g, r) in enumerate(zip(got, ref)):
        if _isnan(r):
            assert _isnan(g), f"{label}[{i}]: expected NaN, got {g!r}"
        else:
            assert abs(float(g) - float(r)) < 1e-5, \
                f"{label}[{i}]: {g!r} != {r!r}"


# -- slice composition (existing IR only) -----------------------------------

def slice_sort_gather(a, cols, key, dtypes=None):
    """Table.sort_by(key): one ir_sort perm, ir_gather per column."""
    dtypes = dtypes or {}
    jobs = [a["ir_series"](f"s_{k}", np.ascontiguousarray(v),
                           dtypes.get(k, "int32")) for k, v in cols.items()]
    jobs.append(a["ir_sort"]("p", f"s_{key}"))
    for k in cols:
        jobs.append(a["ir_gather"](f"g_{k}", f"s_{k}", "p"))
    bufs = _bufs(a, jobs)
    return np.asarray(bufs["p"]).astype(np.int64), {k: np.asarray(bufs[f"g_{k}"]) for k in cols}


def slice_rolling_sum(a, vals, window, min_periods=None, dtype="float32"):
    jobs = [a["ir_series"]("s", np.ascontiguousarray(vals), dtype),
            a["ir_rolling_sum"]("r", "s", window,
                                **({} if min_periods is None else {"min_periods": min_periods}))]
    return np.asarray(_bufs(a, jobs)["r"])


def slice_rolling_mean(a, vals, window, dtype="float32"):
    """rolling_mean = RollingSum + MapBinary(div) (spec 02, no new node)."""
    jobs = [a["ir_series"]("s", np.ascontiguousarray(vals), dtype),
            a["ir_rolling_sum"]("rs", "s", window),
            a["ir_map"]("rm", "rs", "div", float(window))]
    return np.asarray(_bufs(a, jobs)["rm"])


# -- tests -------------------------------------------------------------------

@pytest.mark.fast
def test_sort_preserves_columns_and_alignment(kernel):
    a = kernel.alias
    ts = [30, 10, 20, 10]
    px = [300.0, 100.0, 200.0, 150.0]
    perm, out = slice_sort_gather(a, {"ts": ts, "px": px}, "ts")
    assert [int(i) for i in perm] == [1, 3, 2, 0]  # stable dup order
    assert list(out["ts"]) == [10, 10, 20, 30]
    assert list(out["px"]) == [100.0, 150.0, 200.0, 300.0]  # aligned


@pytest.mark.fast
def test_sort_matches_oracle_unsorted_and_sorted(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    for trial in range(3):
        n = int(rng.integers(5, 30))
        ts = rng.integers(0, 10, size=n).tolist()
        px = rng.normal(0, 1, size=n).tolist()
        perm, out = slice_sort_gather(a, {"ts": ts, "px": px}, "ts",
                                        {"px": "float32"})
        ref = oracle_stable_perm(np.asarray(ts))
        assert [int(i) for i in perm] == [int(i) for i in ref]
        assert list(out["ts"]) == [ts[int(i)] for i in ref]
        assert list(out["px"]) == [px[int(i)] for i in ref]


@pytest.mark.fast
def test_sort_duplicates_stable(kernel):
    a = kernel.alias
    ts = [5, 5, 5, 1, 5]
    tag = [0, 1, 2, 3, 4]
    perm, out = slice_sort_gather(a, {"ts": ts, "tag": tag}, "ts")
    assert [int(i) for i in perm] == [3, 0, 1, 2, 4]
    assert list(out["tag"]) == [3, 0, 1, 2, 4]


@pytest.mark.fast
def test_rolling_sum_parity_oracle(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    vals = rng.normal(0, 5, size=25).astype(np.float32)
    for window, mp in [(3, None), (5, 3), (4, 1), (1, None), (25, None), (40, None)]:
        got = slice_rolling_sum(a, vals, window, mp)
        ref = oracle_rolling_sum(vals.astype(np.float64), window, mp)
        _assert_rolling_close(list(got), list(ref), f"w={window} mp={mp}")


@pytest.mark.fast
def test_rolling_mean_is_sum_plus_div(kernel):
    a = kernel.alias
    vals = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    got = slice_rolling_mean(a, vals, 3)
    ref = oracle_rolling_sum(vals.astype(np.float64), 3) / 3.0
    _assert_rolling_close(list(got), list(ref), "mean=sum/div")


@pytest.mark.fast
def test_rolling_edges_window1_windowN(kernel):
    a = kernel.alias
    vals = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    assert list(slice_rolling_sum(a, vals, 1)) == [7.0, 8.0, 9.0]
    got = slice_rolling_sum(a, vals, 9)  # window > N -> all NaN
    assert all(_isnan(v) for v in got)
    got = slice_rolling_sum(a, vals, 3)
    assert _isnan(got[0]) and _isnan(got[1]) and abs(float(got[2]) - 24.0) < 1e-5


@pytest.mark.fast
def test_rolling_nan_in_window_is_nan(kernel):
    a = kernel.alias
    vals = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    got = slice_rolling_sum(a, vals, 2)
    assert _isnan(got[0]) and _isnan(got[1]) and _isnan(got[2])


@pytest.mark.fast
def test_rolling_nan_tail_clean_window(kernel):
    """Fixed: no cumsum poisoning — clean windows after a NaN sum normally
    (oracle parity: out[3]=7.0 for clean window [3,4], window=2)."""
    a = kernel.alias
    vals = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    got = slice_rolling_sum(a, vals, 2)
    ref = oracle_rolling_sum(vals.astype(np.float64), 2)
    _assert_rolling_close(list(got), list(ref), "post-NaN tail")
    assert abs(float(got[3]) - 7.0) < 1e-5


@pytest.mark.fast
def test_rolling_validity_gap_is_nan(kernel):
    """Fixed: rolling_sum honours the validity sidecar — invalid rows act
    as NaN (logical oracle: [1.0, nan, 3.0], window=2)."""
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1.0, 2.0, 3.0], "float32", validity=[1, 0, 1]),
            a["ir_rolling_sum"]("r", "s", 2)]
    got = list(_bufs(a, jobs)["r"])
    logical = oracle_rolling_sum(np.array([1.0, np.nan, 3.0]), 2)
    _assert_rolling_close(got, list(logical), "validity gap")
    assert _isnan(got[1])


@pytest.mark.fast
def test_full_slice_fuzz_parity(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    max_diff = 0.0
    for trial in range(10):
        n = int(rng.integers(5, 60))
        ts = rng.integers(0, 15, size=n)
        vals = rng.normal(100, 10, size=n).astype(np.float32)
        window = int(rng.integers(1, 12))
        perm, out = slice_sort_gather(a, {"ts": ts.tolist(), "px": vals.tolist()}, "ts")
        ref_perm = oracle_stable_perm(ts)
        assert [int(i) for i in perm] == [int(i) for i in ref_perm]
        svals = vals[np.asarray(ref_perm)]
        got = slice_rolling_sum(a, svals, window)
        ref = oracle_rolling_sum(svals.astype(np.float64), window)
        for g, r in zip(got, ref):
            if not _isnan(r):
                max_diff = max(max_diff, abs(float(g) - float(r)))
        _assert_rolling_close(list(got), list(ref), f"fuzz t={trial} w={window}")
    print(f"\nfuzz max_diff={max_diff:.3e} (10 trials, seed 42)")
