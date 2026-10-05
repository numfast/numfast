# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: ir_cumsum N->N inclusive CPU (v1, cumsum only).

Contract (explicit, v1 -- see ir_cumsum docstring): inclusive prefix
out[i] = sum(x[0..i]), dtype preserved (int32/float32/float64);
invalid rows contribute 0 and stay invalid (per-row validity carry,
downstream resumes); NaN is a value with IEEE forward propagation
(never a validity signal); int32 wraps mod 2**32 (never trap);
empty -> empty same dtype. Oracle: independent NumPy (no numfast
imports inside oracle fns). Seed 42 where RNG is used. Float
tolerance read from specs-rebuilt/conformance-profile.toml (no
hardcoded thresholds).
"""

import math
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE_PATH = Path(APP_DIR) / "specs-rebuilt" / "conformance-profile.toml"


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


@pytest.fixture(scope="module")
def profile():
    with open(PROFILE_PATH, "rb") as f:
        return tomllib.load(f)


def _tol(profile, dtype):
    key = "f32" if np.dtype(dtype) == np.dtype(np.float32) else "f64"
    sec = profile["tolerance"][key]
    return sec["atol"], sec["rtol"]


def _bufs(a, jobs):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _isnan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def oracle_cumsum(vals, validity=None):
    """Independent oracle: (out_vals, out_valid_or_None).

    Invalid rows contribute 0; output validity is a per-row copy.
    int32 wraps mod 2**32 (two's complement); floats accumulate in
    their own dtype; NaN propagates forward via IEEE.
    """
    vals = np.asarray(vals)
    n = vals.size
    dt = vals.dtype
    if validity is None:
        fill, ov = vals, None
    else:
        m = np.asarray(validity, dtype=bool)
        fill = np.where(m, vals, dt.type(0))
        ov = m.copy()
    if dt == np.dtype(np.int32):
        acc = np.cumsum(fill.astype(np.int64), dtype=np.int64)
        w = acc % np.int64(2 ** 32)
        out = np.where(w >= np.int64(2 ** 31),
                       w - np.int64(2 ** 32), w).astype(np.int32)
    else:
        out = np.cumsum(fill.astype(dt), dtype=dt)
    return out, ov


def _assert_valid(got_bufs, tag, ref_valid):
    if ref_valid is None:
        assert tag + "#validity" not in got_bufs, f"{tag}: unexpected sidecar"
        return
    gotv = np.asarray(got_bufs[tag + "#validity"])
    assert [bool(v) for v in gotv] == [bool(v) for v in ref_valid], \
        f"{tag}: validity {list(gotv)!r} != {list(ref_valid)!r}"


def _assert_float_close(got, ref, atol, rtol, label):
    assert len(got) == len(ref), f"{label}: len {len(got)} != {len(ref)}"
    for i, (g, r) in enumerate(zip(got, ref)):
        if _isnan(r):
            assert _isnan(g), f"{label}[{i}]: expected NaN, got {g!r}"
        else:
            assert not _isnan(g), f"{label}[{i}]: expected {r!r}, got NaN"
            assert abs(float(g) - float(r)) <= max(atol, rtol * abs(float(r))), \
                f"{label}[{i}]: {g!r} != {r!r}"


@pytest.mark.fast
def test_cumsum_1_int32_exact(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4], "int32"),
            a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [1, 3, 6, 10]
    assert bufs["h"].dtype == np.dtype(np.int32)
    assert "h#validity" not in bufs  # all-valid -> no sidecar


@pytest.mark.fast
def test_cumsum_float_dtypes_preserved(kernel, profile):
    a = kernel.alias
    atol32, rtol32 = _tol(profile, np.float32)
    atol64, rtol64 = _tol(profile, np.float64)
    cases = [
        ("float32", np.array([1.5, 2.5, -1.0], dtype=np.float32),
         (atol32, rtol32)),
        ("float64", np.array([1.5, 2.5, -1.0], dtype=np.float64),
         (atol64, rtol64)),
    ]
    for dtype, vals, (atol, rtol) in cases:
        jobs = [a["ir_series"]("s", vals, dtype), a["ir_cumsum"]("h", "s")]
        bufs = _bufs(a, jobs)
        assert bufs["h"].dtype == np.asarray(vals).dtype, dtype
        ref, refv = oracle_cumsum(vals)
        _assert_float_close(list(np.asarray(bufs["h"]).tolist()),
                            list(ref.tolist()), atol, rtol, dtype)
        _assert_valid(bufs, "h", refv)


@pytest.mark.fast
def test_cumsum_empty(kernel):
    a = kernel.alias
    for dtype in ("int32", "float32", "float64"):
        jobs = [a["ir_series"]("s", np.zeros(0, dtype=dtype), dtype),
                a["ir_cumsum"]("h", "s")]
        bufs = _bufs(a, jobs)
        assert np.asarray(bufs["h"]).size == 0
        assert bufs["h"].dtype == np.dtype(dtype)
        assert "h#validity" not in bufs
    jobs = [a["ir_series"]("s", np.zeros(0, dtype=np.int32), "int32",
                           validity=[]),
            a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    assert np.asarray(bufs["h"]).size == 0
    assert np.asarray(bufs["h#validity"]).size == 0


@pytest.mark.fast
def test_cumsum_single(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [7], "int32"), a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [7]


@pytest.mark.fast
def test_cumsum_validity_gaps(kernel):
    a = kernel.alias
    # Invalid rows contribute 0, stay invalid, downstream resumes.
    jobs = [a["ir_series"]("s", [10, 20, 30, 40], "int32",
                           validity=[1, 0, 1, 1]),
            a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    ref, refv = oracle_cumsum([10, 20, 30, 40], [1, 0, 1, 1])
    assert list(np.asarray(bufs["h"]).tolist()) == list(ref.tolist())
    _assert_valid(bufs, "h", refv)
    assert list(ref.tolist()) == [10, 10, 40, 80]


@pytest.mark.fast
def test_cumsum_all_invalid(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("s", [5, 6, 7], "int32",
                           validity=[0, 0, 0]),
            a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [0, 0, 0]
    _assert_valid(bufs, "h", np.zeros(3, dtype=bool))


@pytest.mark.fast
def test_cumsum_nan_vs_validity(kernel):
    a = kernel.alias
    # NaN is a VALUE: poisons its own and all downstream prefix sums.
    jobs = [a["ir_series"]("s", np.array([1.0, np.nan, 3.0], dtype=np.float32),
                           "float32"),
            a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    got = list(np.asarray(bufs["h"]).tolist())
    assert abs(float(got[0]) - 1.0) < 1e-6
    assert _isnan(got[1]) and _isnan(got[2])
    assert "h#validity" not in bufs  # NaN never creates validity
    # Invalid row (sidecar) vs NaN row: invalid contributes 0 and
    # downstream resumes; NaN propagates. Never conflated.
    jobs = [a["ir_series"]("s", np.array([1.0, 2.0, 3.0], dtype=np.float32),
                           "float32", validity=[1, 0, 1]),
            a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    got = np.asarray(bufs["h"])
    assert [bool(v) for v in bufs["h#validity"]] == [True, False, True]
    assert float(got[0]) == 1.0 and float(got[1]) == 1.0 \
        and float(got[2]) == 4.0


@pytest.mark.fast
def test_cumsum_overflow_wrap(kernel):
    a = kernel.alias
    # int32 wraps mod 2**32 (two's complement), never saturates/traps.
    x = np.full(5, 2147483647, dtype=np.int32)
    jobs = [a["ir_series"]("s", np.ascontiguousarray(x), "int32"),
            a["ir_cumsum"]("h", "s")]
    bufs = _bufs(a, jobs)
    ref, _ = oracle_cumsum(x)
    got = np.asarray(bufs["h"])
    assert bufs["h"].dtype == np.dtype(np.int32)
    assert (got == ref).all()
    # Sequential-prefix property under wrap: out[i]-out[i-1] == x[i].
    d = (got[1:].astype(np.int64) - got[:-1].astype(np.int64)) % 2 ** 32
    assert (d == (2147483647 % 2 ** 32)).all()
    assert got[0] == 2147483647


@pytest.mark.fast
def test_cumsum_rejects(kernel):
    a = kernel.alias
    # Packed bool has no cumsum semantics (explicit error, never cast).
    jobs = [a["ir_series"]("s", [True, False, True], "bool"),
            a["ir_cumsum"]("h", "s")]
    with pytest.raises(ValueError, match="cumsum"):
        _bufs(a, jobs)
    import numfast as nf

    b = nf.from_numpy(np.array([True, False], dtype=bool))
    with pytest.raises(ValueError, match="cumsum"):
        b.cumsum()
    with pytest.raises(ValueError, match="cumsum"):
        nf.cumsum(nf.from_numpy(np.array([1, 2, 3], dtype=np.int32))
                  .to_numpy())


@pytest.mark.fast
def test_series_and_nf_cumsum_api(kernel):
    import numfast as nf

    s = nf.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    g = s.cumsum()
    assert g.dtype == "int32"
    assert list(np.asarray(g.to_numpy()).tolist()) == [1, 3, 6, 10]
    assert g.validity is None  # all-valid -> no sidecar
    g2 = nf.cumsum(s)
    assert list(np.asarray(g2.to_numpy()).tolist()) == [1, 3, 6, 10]
    sv = nf.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                       validity=np.array([True, False, True]))
    gv = sv.cumsum()
    assert [bool(v) for v in gv.validity] == [True, False, True]
    # float invalid rows restored as NaN on readback (DELTA-3).
    assert list(np.asarray(gv.to_numpy(), dtype=np.float64)[[0, 2]]) == \
        pytest.approx([1.0, 4.0], abs=1e-5)
    assert _isnan(gv.to_numpy()[1])


@pytest.mark.fast
def test_cumsum_fuzz_N_validity(kernel, profile):
    a = kernel.alias
    atol32, rtol32 = _tol(profile, np.float32)
    rng = np.random.default_rng(42)
    max_n = 0
    for trial in range(40):
        n = int(rng.integers(0, 60))
        max_n = max(max_n, n)
        kind = trial % 3
        if kind == 0:
            vals = rng.integers(-5000, 5000, size=n).astype(np.int32)
            dtype = "int32"
        elif kind == 1:
            vals = rng.normal(0, 100, size=n).astype(np.float32)
            dtype = "float32"
        else:
            vals = rng.normal(0, 100, size=n).astype(np.float64)
            dtype = "float64"
        use_valid = bool(rng.integers(0, 2))
        valid = None
        if use_valid and n:
            valid = [bool(x) for x in rng.integers(0, 2, size=n)]
        kw = {"validity": [int(v) for v in valid]} if valid is not None else {}
        jobs = [a["ir_series"]("s", np.ascontiguousarray(vals), dtype, **kw),
                a["ir_cumsum"]("h", "s")]
        bufs = _bufs(a, jobs)
        ref, refv = oracle_cumsum(vals, None if valid is None else valid)
        got = np.asarray(bufs["h"])
        assert got.dtype == np.asarray(ref).dtype, f"t={trial} dtype"
        if dtype == "int32":
            assert list(got.tolist()) == list(ref.tolist()), \
                f"t={trial} n={n}"
        else:
            atol, rtol = (atol32, rtol32) if dtype == "float32" \
                else _tol(profile, np.float64)
            _assert_float_close(list(got.tolist()), list(ref.tolist()),
                                atol, rtol, f"t={trial} n={n}")
        _assert_valid(bufs, "h", refv)
    print(f"\nfuzz trials=40 max_n={max_n} seed=42")


@pytest.mark.fast
def test_regression_shift_rolling_sort(kernel):
    """Regression: shift + rolling_sum + sort paths untouched by cumsum."""
    a = kernel.alias
    jobs = [a["ir_series"]("s", [1, 2, 3, 4], "int32"),
            a["ir_shift"]("h", "s", 1)]
    bufs = _bufs(a, jobs)
    assert list(np.asarray(bufs["h"]).tolist()) == [0, 1, 2, 3]
    assert [bool(v) for v in bufs["h#validity"]] == [False, True, True, True]
    jobs = [a["ir_series"]("s", [1, 2, 3, 4, 5], "float32"),
            a["ir_rolling_sum"]("r", "s", 3)]
    got = list(_bufs(a, jobs)["r"])
    assert _isnan(got[0]) and _isnan(got[1])
    assert [float(v) for v in got[2:]] == pytest.approx([6.0, 9.0, 12.0],
                                                        abs=1e-5)
    jobs = [a["ir_series"]("s", [30, 10, 20, 10], "int32"),
            a["ir_sort"]("p", "s"),
            a["ir_gather"]("g", "s", "p")]
    bufs = _bufs(a, jobs)
    assert [int(i) for i in np.asarray(bufs["p"]).tolist()] == [1, 3, 2, 0]
    assert list(np.asarray(bufs["g"]).tolist()) == [10, 10, 20, 30]
