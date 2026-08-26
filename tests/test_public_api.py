# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Conformance tests for the public API surface (``import numfast as nf``).

Spec: PUBLIC_API_AUDIT_SPEC section 8 (+ section 3 examples E1-E8,
adapted to the real API where it differs).
"""

import inspect
import operator

import numpy as np
import pytest

import numfast as nf


# ── import & version ─────────────────────────────────────────────────


def test_import_clean():
    """Import works with no sys.path manipulation inside the test."""
    assert nf.__name__ == "numfast"
    exported = dir(nf)
    for name in ("zeros", "ones", "series", "random", "math",
                 "topk", "groupby", "device_info", "profile"):
        assert name in exported, name


def test_version():
    assert nf.__version__ == "1.0.0a1"


# ── creation surface (spec 2.1) ──────────────────────────────────────

_CREATION_PARAMS = {
    "zeros": ("shape", "dtype"),
    "ones": ("shape", "dtype"),
    "full": ("shape", "value", "dtype"),
    "arange": ("start", "stop", "step", "dtype"),
    "linspace": ("start", "stop", "num", "dtype"),
    "index": ("n", "dtype"),
    "tile": ("pattern", "n", "dtype"),
    "repeat": ("pattern", "repeats", "dtype"),
}


def test_creation_surface():
    for name, params in _CREATION_PARAMS.items():
        fn = getattr(nf, name)
        assert callable(fn), name
        sig = inspect.signature(fn)
        for p in params:
            assert p in sig.parameters, f"{name}() missing param {p!r}"
    assert len(nf.zeros(4)) == 4
    assert len(nf.ones(3)) == 3
    assert len(nf.full(3, 7.0)) == 3
    assert len(nf.arange(0, 5)) == 5
    assert len(nf.linspace(0.0, 1.0, 5)) == 5
    assert len(nf.index(6)) == 6  # native int32 path
    assert list(nf.tile([1.0, 2.0], 5).to_numpy()) == [1.0, 2.0, 1.0, 2.0, 1.0]
    assert len(nf.repeat([1.0, 2.0], 3)) == 6


# ── random (spec 2.2) ────────────────────────────────────────────────


def test_random_surface():
    ns = nf.random
    for name in ("uniform", "normal", "integers"):
        fn = getattr(ns, name)
        assert callable(fn), name
        # seed is keyword-only: missing -> TypeError
        with pytest.raises(TypeError):
            fn(shape=4)
    assert len(ns.uniform(low=0.0, high=1.0, shape=4, seed=42)) == 4
    assert len(ns.normal(loc=0.0, scale=1.0, shape=4, seed=42)) == 4
    assert len(ns.integers(low=0, high=10, shape=4, seed=42)) == 4


# ── series operators (spec 2.3) ──────────────────────────────────────


def test_series_operators():
    s = nf.series([1.0, 2.0, 3.0])
    t = nf.series([10.0, 20.0, 30.0])
    assert (s + t).data() == [11.0, 22.0, 33.0]
    assert (t - s).data() == [9.0, 18.0, 27.0]
    assert (s * t).data() == [10.0, 40.0, 90.0]
    assert (t / s).data()[0] == pytest.approx(10.0)
    assert (t % s).data()[0] == pytest.approx(0.0)
    assert (s ** 2).data() == [1.0, 4.0, 9.0]
    cmp_ops = (operator.lt, operator.le, operator.gt,
               operator.ge, operator.eq, operator.ne)
    for op in cmp_ops:
        mask = op(s, 2.0)
        assert len(mask.data()) == 3, op


def test_rtruediv():
    s = nf.series([1.0, 2.0, 4.0])
    r = 8 / s
    assert r.data() == [8.0, 4.0, 2.0]


def test_filter_method():
    s = nf.series([1.0, 2.0, 3.0])
    picked = (s > 1).filter(s)
    assert picked.data() == [2.0, 3.0]


def test_topk_groupby_exist():
    assert callable(nf.topk)
    assert callable(nf.groupby)


# ── stats & ops (spec 2.5) ───────────────────────────────────────────


def test_stats_ops():
    data = [1.0, 2.0, 3.0]
    assert nf.total(data) == pytest.approx(6.0)
    assert nf.mean(data) == pytest.approx(2.0)
    assert nf.minimum(data) == pytest.approx(1.0)
    assert nf.maximum(data) == pytest.approx(3.0)
    assert nf.count(data) == 3
    assert nf.var(data) >= 0.0
    assert nf.std(data) >= 0.0
    for name in ("scan", "sort", "histogram", "matmul", "fft"):
        assert callable(getattr(nf, name)), name


# ── math namespace + top-level aliases (spec 2.3) ────────────────────

_MATH_NAMES = ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "square")


def test_math_namespace():
    for name in _MATH_NAMES:
        mfn = getattr(nf.math, name)
        tfn = getattr(nf, name)
        assert callable(mfn), f"math.{name}"
        assert callable(tfn), name
    x = nf.series([0.0])
    assert nf.math.sin(x).compute().data()[0] == pytest.approx(0.0)


# ── diagnostics (spec 2.6) ───────────────────────────────────────────


def test_diagnostics():
    d = nf.device_info()
    assert isinstance(d, dict) and len(d) > 0
    nf.set_backend("gpu")  # must not raise
    nf.set_backend("cpu")  # restore deterministic default
    with nf.profile():
        pass


# ── encapsulation (spec 4: internal machinery is private) ────────────


def test_no_internal_leak():
    leaked = [name for name in (
        "compile", "execute", "register_kernel",
        "PackingPlan", "BlockView", "ExecutionPacket",
    ) if hasattr(nf, name)]
    assert leaked == []


# ── spec section 3 examples E1-E8 (adapted to real API) ──────────────


def test_e1_arange_expression_compute():
    x_arr = nf.arange(0, 10_000, dtype="float32")
    x = nf.series(x_arr.to_numpy().tolist())  # CreatedArray -> Series bridge
    y = (x * 2 + 1).compute()
    d = y.data()  # materialization boundary
    assert len(d) == 10_000
    assert list(d[:3]) == [1.0, 3.0, 5.0]


def test_e2_random_stats():
    u = nf.random.uniform(shape=50_000, seed=42)
    arr = u.to_numpy()
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
    assert nf.mean(arr.tolist()) == pytest.approx(0.5, abs=0.01)
    assert nf.std(arr.tolist()) == pytest.approx(0.2887, abs=0.02)


def test_e3_compare_filter_wave():
    i_arr = nf.index(10_000)
    x = nf.series(i_arr.to_numpy().tolist())
    wave = nf.math.sin(x * 0.01).compute()
    peaks = (wave > 0.5).filter(wave)
    picked = peaks.data()
    assert len(picked) > 0
    assert all(v > 0.5 for v in picked)


def test_e4_topk_normal():
    big = nf.random.normal(shape=100_000, seed=42)
    top10 = nf.topk(big.to_numpy(), 10)
    vals = list(top10)
    assert len(vals) == 10
    mx = float(big.to_numpy().max())
    assert vals[0] == pytest.approx(mx, rel=1e-6)
    assert all(vals[i] >= vals[i + 1] for i in range(9))


def test_e5_groupby_mean():
    keys = nf.random.integers(low=0, high=5, shape=1_000, seed=42).to_numpy()
    vals = nf.random.normal(loc=0.0, scale=1.0, shape=1_000, seed=43).to_numpy()
    res = nf.groupby(keys, vals)  # real API: groupby(keys, vals) -> dict
    means = res["mean"]
    n_groups = len(set(keys.tolist()))
    assert len(means) == n_groups
    ref = {}
    for k, v in zip(keys.tolist(), vals.tolist()):
        ref.setdefault(int(k), []).append(v)
    for gk, row in zip(res["keys"].tolist(), means.tolist()):
        assert row == pytest.approx(float(np.mean(ref[int(gk)])), rel=1e-4)


def test_e6_creation_limit_documented():
    # v1 limit N <= 4_194_240; chunking (S209) raises ValueError for now
    with pytest.raises(ValueError):
        nf.zeros(5_000_000)


def test_e7_pipeline_composition():
    close = nf.random.normal(loc=0.0, scale=1.0, shape=10_000, seed=42).to_numpy()

    def close_sma(arr, p):
        cs = np.cumsum(arr, dtype=np.float64)
        sma = np.full(arr.shape, np.nan, dtype=np.float32)
        sma[p - 1:] = ((cs[p - 1:] - np.concatenate(([0.0], cs[:len(arr) - p]))) / p)
        return sma.astype(np.float32)

    results = {p: nf.topk(close_sma(close, p), 100) for p in (10, 20, 50, 100)}
    assert all(len(v) == 100 for v in results.values())


def test_e8_diagnostics_profile_pipeline():
    info = nf.device_info()
    assert isinstance(info, dict)
    ones = nf.ones(1_000).to_numpy().tolist()
    with nf.profile():
        z = (nf.series(ones) * 3).compute()
        assert len(z.data()) == 1_000
