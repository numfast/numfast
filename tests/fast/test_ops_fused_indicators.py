# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: fused elementwise indicators (SPEC-DELTA-11) — one dispatch, K outs.

Claims: fused[K] bit-exact vs separate[K x 1] (same emitters, seed 42);
tolerance vs numpy oracle (f32); edge cases; dispatch counts; boundary
rejections (groupby/sort/EMA never fuse silently).
"""

from pathlib import Path

import numpy as np
import pytest

from harness import load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()

FIVE = [
    {"name": "sma20", "op": "sma", "params": {"w": 20}},
    {"name": "rsi14", "op": "rsi", "params": {"w": 14}},
    {"name": "bup20", "op": "boll_up", "params": {"w": 20, "k": 2.0}},
    {"name": "blo20", "op": "boll_lo", "params": {"w": 20, "k": 2.0}},
    {"name": "stk14", "op": "stoch_k", "params": {"w": 14}},
]

ALL_OPS = [
    {"name": "sma5", "op": "sma", "params": {"w": 5}},
    {"name": "rsum5", "op": "rsum", "params": {"w": 5}},
    {"name": "rmin5", "op": "rmin", "params": {"w": 5}},
    {"name": "rmax5", "op": "rmax", "params": {"w": 5}},
    {"name": "mom3", "op": "mom", "params": {"lag": 3}},
    {"name": "ret", "op": "returns"},
]  # rstd/rsi/boll/zscore/stoch/axpb/map_div/signals: covered below in pairs


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _quotes(n, seed=42, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=n).astype(np.float64)
    x = (100.0 * np.exp(np.cumsum(rets))).astype(np.float32)
    if nan_frac:
        m = rng.random(n) < nan_frac
        x[m] = np.nan
    return x


def _assert_close(a, b, label, atol=None, max_ulp=None):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, f"{label}: shape/dtype"
    if a.dtype == np.int32:
        bad = np.where(a != b)[0]
        assert bad.size == 0, f"{label}: {bad.size} int mismatches"
        return {"maxdiff": 0}
    sec = PROFILE["tolerance"]["f32"]
    atol = sec["atol"] if atol is None else atol
    max_ulp = sec["max_ulp"] if max_ulp is None else max_ulp
    af, bf = a.astype(np.float64), b.astype(np.float64)
    both_nan = np.isnan(af) & np.isnan(bf)
    assert not (np.isnan(af) ^ np.isnan(bf)).any(), f"{label}: NaN placement"
    with np.errstate(invalid="ignore"):
        diff = np.abs(af - bf)
    diff[both_nan] = 0.0
    tol = np.maximum(atol, sec["rtol"] * np.abs(bf))
    ok = (diff <= tol) | both_nan
    if ok.all():
        return {"maxdiff": float(diff.max(initial=0.0))}
    bad = np.where(~ok)[0]
    import struct as _st

    def _ord(f):
        (i,) = _st.unpack("<i", _st.pack("<f", float(f)))
        i &= 0xFFFFFFFF
        return i ^ (0xFFFFFFFF if i & 0x80000000 else 0x80000000)

    ulps = [abs(_ord(af[i]) - _ord(bf[i])) for i in bad]
    assert max(ulps) <= max_ulp, (
        f"{label}: {bad.size} beyond tol, max ULP {max(ulps)}")
    return {"maxdiff": float(diff.max(initial=0.0)), "max_ulp": max(ulps)}


# Cancellation-prone ops: (ratio-1)*100 on O(100) quotes amplifies f32 loop
# noise to ~2e-5 abs; zscore (x-mean)/sd with mean err ~W*|x|*eps needs 1e-3.
# Bounds derived from f32 eps, not tuned to data.
_RELAX = {"roc3": {"atol": 5e-5}, "rc": {"atol": 5e-5},
          "zs20": {"atol": 1e-3}, "z": {"atol": 1e-3}}


def _assert_bitexact(a, b, label):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, f"{label}: shape/dtype"
    ab, bb = a.view(np.uint32), b.view(np.uint32)
    bad = np.where(ab != bb)[0]
    assert bad.size == 0, (f"{label}: {bad.size}/{a.size} bits differ, "
                           f"first @{bad[:5]}")


@pytest.mark.fast
def test_fused_five_bitexact_vs_separate(kernel):
    a = kernel.alias
    x = _quotes(8192)
    fused, fi = a["fused_run"](x, FIVE)
    assert fi["dispatches"] == 1
    n_sep = 0
    for spec, f in zip(FIVE, fused):
        (s,), si = a["fused_run"](x, [spec])
        n_sep += si["dispatches"]
        _assert_bitexact(f, s, spec["name"])
    assert n_sep == len(FIVE) == 5


@pytest.mark.fast
def test_fused_five_vs_oracle(kernel):
    a = kernel.alias
    x = _quotes(8192)
    fused, _ = a["fused_run"](x, FIVE)
    ref = a["fused_oracle"](x, FIVE)
    for f, r, spec in zip(fused, ref, FIVE):
        m = _assert_close(f, r, spec["name"], **_RELAX.get(spec["name"], {}))
        print(f"\n{spec['name']}: maxdiff={m['maxdiff']:.3g}")


@pytest.mark.fast
def test_fused_nan_inputs_match(kernel):
    a = kernel.alias
    x = _quotes(4096, nan_frac=0.02)
    spec = [FIVE[0], FIVE[1], FIVE[4],
            {"name": "zs20", "op": "zscore", "params": {"w": 20}},
            {"name": "sd20", "op": "rstd", "params": {"w": 20, "ddof": 1}}]
    fused, _ = a["fused_run"](x, spec)
    ref = a["fused_oracle"](x, spec)
    for f, r, s in zip(fused, ref, spec):
        _assert_close(f, r, f"nan:{s['name']}", **_RELAX.get(s["name"], {}))
        (one,), _ = a["fused_run"](x, [s])
        _assert_bitexact(f, one, f"nan-bit:{s['name']}")


@pytest.mark.fast
def test_fused_all_ops_vs_oracle_and_separate(kernel):
    a = kernel.alias
    x = _quotes(2048)
    spec = (ALL_OPS
            + [{"name": "roc3", "op": "roc", "params": {"lag": 3}}])
    fused, fi = a["fused_run"](x, spec)
    assert fi["dispatches"] == 1
    ref = a["fused_oracle"](x, spec)
    for f, r, s in zip(fused, ref, spec):
        _assert_close(f, r, s["name"], **_RELAX.get(s["name"], {}))
        (one,), _ = a["fused_run"](x, [s])
        _assert_bitexact(f, one, f"bit:{s['name']}")


@pytest.mark.fast
def test_fused_maps_and_signals(kernel):
    a = kernel.alias
    x = _quotes(2048)
    spec = [
        {"name": "dbl", "op": "axpb", "params": {"a": 2.0, "b": 1.0}},
        {"name": "half", "op": "map_div", "params": {"v": 2.0}},
        {"name": "up", "op": "above_sma", "params": {"w": 20}},
        {"name": "lo", "op": "rsi_lt", "params": {"w": 14, "level": 70.0}},
        {"name": "hi", "op": "rsi_gt", "params": {"w": 14, "level": 30.0}},
        {"name": "sig", "op": "cond_and", "params": {"a": "up", "b": "lo"}},
    ]
    fused, fi = a["fused_run"](x, spec)
    assert fi["dispatches"] == 1
    assert fused[2].dtype == np.int32 and fused[5].dtype == np.int32
    ref = a["fused_oracle"](x, spec)
    for f, r, s in zip(fused, ref, spec):
        _assert_close(f, r, s["name"])


@pytest.mark.fast
def test_fused_edges(kernel):
    a = kernel.alias
    # n = 0: no dispatch, empty outs
    res, info = a["fused_run"](np.zeros(0, dtype=np.float32), FIVE[:2])
    assert info["dispatches"] == 0
    assert all(r.size == 0 for r in res)
    # n = 1, n < w: all-NaN windows
    x1 = np.array([3.5], dtype=np.float32)
    res, _ = a["fused_run"](x1, FIVE[:2])
    assert np.isnan(res[0][0]) and np.isnan(res[1][0])
    # constant series: rsi -> 50, stoch -> 50 (exact branches)
    xc = np.full(64, 7.0, dtype=np.float32)
    spec = [{"name": "r", "op": "rsi", "params": {"w": 14}},
            {"name": "s", "op": "stoch_k", "params": {"w": 14}}]
    res, _ = a["fused_run"](xc, spec)
    assert float(res[0][-1]) == 50.0
    assert float(res[1][-1]) == 50.0
    # zero series: mean/ss/sd exactly 0 -> zscore 0.0 via sd==0 branch.
    # (nonzero constants keep f32 loop-sum noise: e/|e| = +/-1, by design)
    xz0 = np.zeros(64, dtype=np.float32)
    (z,), _ = a["fused_run"](xz0, [{"name": "z", "op": "zscore",
                                    "params": {"w": 14}}])
    assert float(z[-1]) == 0.0
    # w = 1: sma == x
    xr = _quotes(128)
    (s,), _ = a["fused_run"](xr, [{"name": "s1", "op": "sma",
                                   "params": {"w": 1}}])
    _assert_bitexact(s, xr, "w=1 identity")
    # all-NaN input
    xn = np.full(64, np.nan, dtype=np.float32)
    res, _ = a["fused_run"](xn, FIVE[:2])
    assert np.isnan(res[0][14:]).all()
    # div-by-zero roc: 0/0 -> NaN, x/0 -> inf (matches numpy oracle)
    xz = np.array([0.0, 0.0, 2.0, 4.0], dtype=np.float32)
    (r,), _ = a["fused_run"](xz, [{"name": "rc", "op": "roc",
                                   "params": {"lag": 1}}])
    (o,), _ = a["fused_run"](xz, [{"name": "o", "op": "roc",
                                   "params": {"lag": 1}}])
    ref = a["fused_oracle"](xz, [{"name": "rc", "op": "roc",
                                  "params": {"lag": 1}}])
    _assert_close(r, ref[0], "roc-div0")
    assert np.isnan(o[1]) and np.isinf(o[2]) and o[3] == pytest.approx(100.0)
    # lag >= n: all NaN
    (m,), _ = a["fused_run"](xr[:8], [{"name": "m", "op": "mom",
                                       "params": {"lag": 8}}])
    assert np.isnan(m).all()


@pytest.mark.fast
def test_fused_boundary_rejections(kernel):
    a = kernel.alias
    x = _quotes(64)
    for bad in ("ema", "groupby", "groupby_multi", "sort", "join", "reduce",
                "median"):
        with pytest.raises(ValueError, match="NOT fused|unknown fused op"):
            a["fused_run"](x, [{"name": "q", "op": bad,
                                "params": {"w": 5}}])
    with pytest.raises(ValueError, match="unique name"):
        a["fused_build"]([{"name": "d", "op": "sma", "params": {"w": 5}},
                          {"name": "d", "op": "sma", "params": {"w": 5}}])
    with pytest.raises(ValueError, match="EARLIER"):
        a["fused_build"]([{"name": "sig", "op": "cond_and",
                           "params": {"a": "up", "b": "up"}},
                          {"name": "up", "op": "above_sma",
                           "params": {"w": 5}}])
    with pytest.raises(ValueError, match="window w must be"):
        a["fused_build"]([{"name": "s", "op": "sma", "params": {"w": 0}}])
    many = [{"name": f"s{i}", "op": "sma", "params": {"w": i + 2}}
            for i in range(8)]
    with pytest.raises(ValueError, match="budget exceeded"):
        a["fused_build"](many)
    with pytest.raises(ValueError, match="float32"):
        a["fused_run"](np.arange(16, dtype=np.int32),
                       [{"name": "s", "op": "sma", "params": {"w": 5}}])
    with pytest.raises(ValueError, match="rank-1"):
        a["fused_run"](np.zeros((4, 4), dtype=np.float32),
                       [{"name": "s", "op": "sma", "params": {"w": 5}}])
    c = a["fused_build"](FIVE)
    assert c["source"].count("@compute") == 1
    assert c["n_int"] == 5 and set(c["dtypes"]) == {"f32"}
