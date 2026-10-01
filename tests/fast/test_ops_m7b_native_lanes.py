# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""M7b: the three CPU-driver native lanes wired to the already-bound DLL.

`cumsum`, `unique_inverse` and `map` `div` now have a call site into
`native_cpu` (`nf_cumsum_*`, `nf_unique_inverse_*`, `nf_map_*`). The values
are bit-identical to the NumPy lanes they replaced, so a value-only test
cannot tell the two apart -- and these tests are also the *fail-before*
signal for the wiring itself.

Each test therefore counts the lane calls by swapping the module-global lane
helper in `_lib/cpu` for a counting wrapper (reached through
`alias["cpu_execute"].__globals__["_cpu_execute_impl"].__globals__`, so no
import machinery and no private path). Pre-M7b those helpers do not exist, so
the swap raises AttributeError -> FAIL. Post-M7b the counter must be exactly
what the documented gate predicts:

    cumsum          no gate     (native wins at every n)
    unique_inverse  n >= 10 000
    map div         n >= 100 000, binary only, same input dtype
    everything else (add/sub/mul/mod, every scalar map, map pow) never lanes.

Oracle: independent NumPy, verbatim from the pre-M7b driver branch. Seed 42.
"""

import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])

CUMSUM_FLOAT_MIN = 8192
UNIQUE_MIN = 100_000
MAP_DIV_MIN = 100_000


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _cpu_globals(a):
    """Module globals of `_lib/cpu`, reached without importing it."""
    return a["cpu_execute"].__globals__["_cpu_execute_impl"].__globals__


class _Counter:
    """Swaps a `_lib/cpu` lane helper for a counting wrapper."""

    def __init__(self, a, name):
        self.g = _cpu_globals(a)
        self.name = name
        self.calls = 0
        self.native = 0          # lane actually executed the DLL
        self.fell_back = 0       # lane returned None -> NumPy reference

    def __enter__(self):
        if self.name not in self.g:
            raise AttributeError(
                "_lib/cpu has no %r -- the M7b native lane is not wired"
                % self.name)
        self.orig = self.g[self.name]
        outer = self

        def counting(*args, **kw):
            r = outer.orig(*args, **kw)
            outer.calls += 1
            if r is None:
                outer.fell_back += 1
            else:
                outer.native += 1
            return r

        self.g[self.name] = counting
        return self

    def __exit__(self, *exc):
        self.g[self.name] = self.orig
        return False


def _bufs(a, jobs):
    s = time.perf_counter()
    graph = a["optimize"](a["compile"](jobs))
    t_compile = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: compile+optimize={t_compile:.3f} execute={t_exec:.3f}")
    return bufs


def _srs(a, name, arr, dtype, validity=None):
    if validity is None:
        return a["ir_series"](name, arr, dtype)
    return a["ir_series"](name, arr, dtype,
                          validity=[int(t) for t in np.asarray(validity)])


def _exact(got, ref, label):
    got, ref = np.asarray(got), np.asarray(ref)
    assert got.dtype == ref.dtype, "%s: dtype %s != %s" % (label, got.dtype,
                                                           ref.dtype)
    gn = np.isnan(got.astype(np.float64))
    rn = np.isnan(ref.astype(np.float64))
    assert np.array_equal(gn, rn), "%s: NaN mask differs at %s" % (
        label, np.flatnonzero(gn != rn))
    assert np.array_equal(got[~rn], ref[~rn]), "%s: values differ" % label


def _oracle_cumsum(vals, validity=None):
    """Pre-M7b `_cumsum_ref` body, verbatim."""
    a = np.ascontiguousarray(np.asarray(vals))
    is_int = np.issubdtype(a.dtype, np.integer)

    def wrap(x):
        acc = np.cumsum(x.astype(np.int64, copy=False), dtype=np.int64)
        w = acc % np.int64(2 ** 32)
        return np.ascontiguousarray(
            np.where(w >= np.int64(2 ** 31), w - np.int64(2 ** 32),
                     w).astype(np.int32))

    if validity is None:
        return wrap(a) if is_int else np.ascontiguousarray(
            np.cumsum(a, dtype=a.dtype))
    filled = np.where(np.asarray(validity, dtype=bool), a, a.dtype.type(0))
    return wrap(filled) if is_int else np.ascontiguousarray(
        np.cumsum(filled, dtype=filled.dtype))


def _oracle_div(x, y):
    r = x.astype(np.float64) / y
    return np.rint(r).astype(x.dtype) if np.issubdtype(
        x.dtype, np.integer) else r


# ---------------------------------------------------------------- cumsum
@pytest.mark.fast
def test_m7b_cumsum_native_lane_runs_and_is_bit_identical(kernel):
    """The cumsum lane is entered and the DLL kernel executes (not the
    NumPy fallback); the values stay bit-identical to the 5-pass chain."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    with _Counter(a, "_native_cumsum_lane") as c:
        for dtype in ("int32", "float32", "float64"):
            for n in ((1, 17, 1000, 100000) if dtype == "int32"
                      else (CUMSUM_FLOAT_MIN, 4 * CUMSUM_FLOAT_MIN, 100000)):
                if dtype == "int32":
                    vals = rng.integers(-1000, 1000, size=n).astype(dtype)
                else:
                    vals = (rng.standard_normal(n) * 100.0).astype(dtype)
                vals = np.ascontiguousarray(vals)
                got = np.asarray(_bufs(a, [_srs(a, "s", vals, dtype),
                                           a["ir_cumsum"]("r", "s")])["r"])
                _exact(got, _oracle_cumsum(vals), "cumsum %s n=%d" % (dtype, n))
        # validity sidecar path (0-fill then scan) also lanes
        vals = np.ascontiguousarray(rng.integers(-50, 50, size=4096).astype(
            np.int32))
        vl = (np.arange(4096) % 3 != 1).astype(np.int8)
        got = np.asarray(_bufs(a, [_srs(a, "s", vals, "int32", vl),
                                   a["ir_cumsum"]("r", "s")])["r"])
        _exact(got, _oracle_cumsum(vals, vl), "cumsum int32 validity")
    assert c.calls == 11, c.calls
    # the DLL executed every time -- a None return would mean NumPy owned it
    assert c.native == 11 and c.fell_back == 0, (c.native, c.fell_back)


@pytest.mark.fast
def test_m7b_cumsum_lane_int32_ungated_float_gated_at_8192(kernel):
    """int32 lanes at every n (its 5-pass NumPy chain repays the wrapper tax
    immediately); float32/float64 lanes only from 8192 up (their NumPy chain
    is one np.cumsum and does not repay it below that -- measured 0.60-0.87x,
    penalty <= 9 us)."""
    a = kernel.alias
    with _Counter(a, "_native_cumsum_lane") as c:
        for n in (1, 2, 7, 100, 1000):
            vals = np.arange(1, n + 1, dtype=np.int32)
            _bufs(a, [_srs(a, "s", vals, "int32"), a["ir_cumsum"]("r", "s")])
        assert c.native == 5, c.native
        for n in (1, 7, 100, 1000, CUMSUM_FLOAT_MIN - 1):
            vals = np.arange(1, n + 1, dtype=np.float64)
            _bufs(a, [_srs(a, "s", vals, "float64"), a["ir_cumsum"]("r", "s")])
        assert c.native == 5, "a float lane fired below %d" % CUMSUM_FLOAT_MIN
        for n in (CUMSUM_FLOAT_MIN, 4 * CUMSUM_FLOAT_MIN):
            vals = np.arange(1, n + 1, dtype=np.float64)
            _bufs(a, [_srs(a, "s", vals, "float64"), a["ir_cumsum"]("r", "s")])
        assert c.native == 7, c.native


@pytest.mark.fast
def test_m7b_cumsum_int32_wraps_mod_2_32(kernel):
    """int32 wrapping is the one place the two lanes could disagree
    (int64 accumulator + %2**32 vs Rust wrapping_add). It must not."""
    a = kernel.alias
    for vals in (np.full(8, 2_000_000_000, dtype=np.int32),
                 np.full(8, -2_000_000_000, dtype=np.int32),
                 np.array([2_147_483_647, 1, 1, 1, -2_147_483_648] * 3,
                          dtype=np.int32)):
        got = np.asarray(_bufs(a, [_srs(a, "s", vals, "int32"),
                                   a["ir_cumsum"]("r", "s")])["r"])
        _exact(got, _oracle_cumsum(vals), "cumsum wrap %s" % list(vals[:2]))


# -------------------------------------------------------- unique_inverse
@pytest.mark.fast
def test_m7b_unique_inverse_lane_gated_at_1e5(kernel):
    """The fused radix lane is entered from n >= 100 000 and NOT below it
    (np.unique is measurably faster on short / low-cardinality keys), and
    both sides return exactly np.unique(return_inverse=True)."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    for n, want_calls in ((UNIQUE_MIN - 1, 0), (UNIQUE_MIN, 1),
                          (4 * UNIQUE_MIN, 1)):
        keys = (rng.integers(0, max(1, n // 100), size=n)).astype(np.int32)
        with _Counter(a, "_native_unique_lane") as c:
            bufs = _bufs(a, [_srs(a, "k", keys, "int32"),
                             a["ir_unique_inverse"]("u", "k")])
        ref_u, ref_i = np.unique(keys, return_inverse=True)
        _exact(bufs["u"], ref_u, "uniq n=%d" % n)
        _exact(bufs["u#inv"], np.asarray(ref_i, dtype=np.int32),
               "inv n=%d" % n)
        assert int(bufs["u#ng"]) == int(ref_u.size)
        assert np.array_equal(np.asarray(bufs["u"])[bufs["u#inv"]], keys), \
            "uniq[inv] == keys invariant"
        # Below the gate the helper IS called but returns None immediately
        # (NumPy owns the row), so `native` -- not `calls` -- is the gate.
        assert c.calls == 1, "n=%d: %d helper calls" % (n, c.calls)
        assert c.native == want_calls, "n=%d: %d native" % (n, c.native)
        assert c.fell_back == 1 - want_calls, "n=%d: %d fell back" % (
            n, c.fell_back)


@pytest.mark.fast
@pytest.mark.parametrize("card", [2, 8, 1000, UNIQUE_MIN])
def test_m7b_unique_inverse_lane_wins_at_every_cardinality(kernel, card):
    """Above the gate the lane must win at EVERY cardinality: the 4-pass LSD
    radix is bandwidth-bound, so low cardinality is where it could regress
    (it did at n < 1e5)."""
    a = kernel.alias
    n = UNIQUE_MIN
    keys = (np.arange(n, dtype=np.int64) * 2654435761 % card).astype(np.int32)
    with _Counter(a, "_native_unique_lane") as c:
        bufs = _bufs(a, [_srs(a, "k", keys, "int32"),
                         a["ir_unique_inverse"]("u", "k")])
    ref_u, ref_i = np.unique(keys, return_inverse=True)
    _exact(bufs["u"], ref_u, "uniq card=%d" % card)
    _exact(bufs["u#inv"], np.asarray(ref_i, dtype=np.int32),
           "inv card=%d" % card)
    assert c.native == 1, (card, c.native)


@pytest.mark.fast
def test_m7b_unique_inverse_lane_int64_and_all_duplicate(kernel):
    """int64 keys and the fully-duplicated / all-unique extremes all take the
    lane and stay exact."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    for keys in (rng.integers(-2 ** 40, 2 ** 40, size=UNIQUE_MIN).astype(
                     np.int64),
                 np.zeros(UNIQUE_MIN, dtype=np.int32),
                 rng.permutation(UNIQUE_MIN).astype(np.int32)):
        with _Counter(a, "_native_unique_lane") as c:
            bufs = _bufs(a, [_srs(a, "k", keys, keys.dtype.name),
                             a["ir_unique_inverse"]("u", "k")])
        ref_u, ref_i = np.unique(keys, return_inverse=True)
        _exact(bufs["u"], ref_u, "uniq %s" % keys.dtype)
        _exact(bufs["u#inv"], np.asarray(ref_i, dtype=np.int32),
               "inv %s" % keys.dtype)
        assert c.native == 1, (keys.dtype, c.native)


@pytest.mark.fast
def test_m7b_unique_inverse_validity_path_never_lanes(kernel):
    """The validity sidecar path compacts rows first, so its key count is not
    the array length the gate keys on. It must stay on NumPy verbatim."""
    a = kernel.alias
    n = 4 * UNIQUE_MIN
    keys = np.arange(n, dtype=np.int32) % 977
    vl = (np.arange(n) % 2 == 0).astype(np.int8)
    with _Counter(a, "_native_unique_lane") as c:
        bufs = _bufs(a, [_srs(a, "k", keys, "int32", vl),
                         a["ir_unique_inverse"]("u", "k")])
    m = vl.astype(bool)
    ref_u, ref_iv = np.unique(keys[m], return_inverse=True)
    ref_i = np.full(keys.shape, -1, dtype=np.int32)
    ref_i[m] = ref_iv.astype(np.int32)
    _exact(bufs["u"], ref_u, "uniq valid-gap")
    _exact(bufs["u#inv"], ref_i, "inv valid-gap")
    assert c.calls == 0, c.calls


# --------------------------------------------------------------- map div
@pytest.mark.fast
def test_m7b_map_div_lane_gated_at_1e5_and_binary_only(kernel):
    """Binary div lanes from n >= 100 000, not below it; the value is
    bit-identical to astype(f64)/divide/rint/astype on both sides."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    for dtype in ("int32", "float32", "float64"):
        for n, want in ((MAP_DIV_MIN - 1, 0), (MAP_DIV_MIN, 1),
                        (3 * MAP_DIV_MIN, 1)):
            if dtype == "int32":
                x = rng.integers(-1000, 1000, size=n).astype(dtype)
                y = rng.integers(1, 1000, size=n).astype(dtype)
            else:
                x = (rng.standard_normal(n) * 100.0).astype(dtype)
                y = (np.abs(rng.standard_normal(n)) * 100.0 + 1.0).astype(dtype)
            with _Counter(a, "_native_map_div_lane") as c:
                got = np.asarray(_bufs(a, [_srs(a, "x", x, dtype),
                                           _srs(a, "y", y, dtype),
                                           a["ir_map"]("r", "x", "div", "y")
                                           ])["r"])
            _exact(got, _oracle_div(x, y), "div %s n=%d" % (dtype, n))
            assert c.calls == want, "%s n=%d: %d calls" % (dtype, n, c.calls)
            assert c.native == want, "%s n=%d: %d native" % (dtype, n, c.native)


@pytest.mark.fast
def test_m7b_map_scalar_div_never_lanes(kernel):
    """The scalar map lanes measured 0.12-1.06x and stay on NumPy at every n."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 3 * MAP_DIV_MIN
    x = rng.standard_normal(n).astype(np.float32)
    with _Counter(a, "_native_map_div_lane") as c:
        got = np.asarray(_bufs(a, [_srs(a, "x", x, "float32"),
                                   a["ir_map"]("r", "x", "div", 3.0)])["r"])
    _exact(got, x.astype(np.float64) / 3.0, "scalar div f32")
    assert c.calls == 0, c.calls


@pytest.mark.fast
@pytest.mark.parametrize("fn", ["add", "sub", "mul", "floor_div", "mod"])
def test_m7b_map_rejected_ops_never_lane(kernel, fn):
    """add/sub/mul/mod measured 0.94-1.34x (noise) and were REJECTED; they
    must never enter a native lane at any n, gated or not."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 3 * MAP_DIV_MIN
    x = rng.integers(1, 1000, size=n).astype(np.int32)
    y = rng.integers(1, 1000, size=n).astype(np.int32)
    with _Counter(a, "_native_map_div_lane") as c:
        _bufs(a, [_srs(a, "x", x, "int32"), _srs(a, "y", y, "int32"),
                  a["ir_map"]("r", "x", fn, "y")])
    assert c.calls == 0, "%s entered the native lane %d times" % (fn, c.calls)


@pytest.mark.fast
def test_m7b_map_div_zero_and_i32_min_by_minus_one(kernel):
    """The two integer-division corners where float64 and the fused lane
    could round differently: b == 0, and INT32_MIN / -1 (both end in an
    out-of-range int32 cast)."""
    a = kernel.alias
    n = MAP_DIV_MIN
    zero = np.zeros(n, dtype=np.int32)
    one = np.ones(n, dtype=np.int32)
    cases = [
        (one.copy(), zero.copy()),                      # 1/0 -> inf
        (zero.copy(), one.copy()),                      # 0/1 -> 0
        (np.full(n, -2_147_483_648, dtype=np.int32),
         np.full(n, -1, dtype=np.int32)),               # INT32_MIN / -1
        (np.full(n, 2_147_483_647, dtype=np.int32),
         np.full(n, 3, dtype=np.int32)),                # rint overflows int32
    ]
    for x, y in cases:
        got = np.asarray(_bufs(a, [_srs(a, "x", x, "int32"),
                                   _srs(a, "y", y, "int32"),
                                   a["ir_map"]("r", "x", "div", "y")])["r"])
        _exact(got, _oracle_div(x, y), "div %s/%s" % (x[0], y[0]))


@pytest.mark.fast
def test_m7b_map_pow_binary_still_rejected(kernel):
    """map pow with an array exponent is rejected by the driver BEFORE the
    M7b gate; M7b did not change that."""
    a = kernel.alias
    x = np.arange(1, 8, dtype=np.int32)
    y = np.full(7, 2, dtype=np.int32)
    graph = a["optimize"](a["compile"]([_srs(a, "x", x, "int32"),
                                        _srs(a, "y", y, "int32"),
                                        a["ir_map"]("r", "x", "pow", "y")]))
    with pytest.raises(Exception, match="array exponent"):
        a["cpu_execute"](graph["nodes"])
