# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""M5b regression: a bad dense key must NEVER kill the CPython process.

`numfast-native/Cargo.toml` sets `panic = "abort"`, so a Rust panic through
the FFI boundary is `0xC0000409`: no traceback, no `finally`, no cleanup --
the whole interpreter dies. `nf_group_variant_i64` with `variant=0` reaches
`dense_scatter_i64` (`groupby/scaled.rs:28-48`), which indexes
`sums[widen_usize(keys[i])]` with NO `key_in_range`, and `widen_usize`
sign-extends, so a NEGATIVE key becomes `usize::MAX`.

The binding must therefore request `V_CHECKED` (variant=1 ->
`checked_scatter_i64`, `scaled.rs:53-76`, which returns BAD_RANGE = -2).

These tests are IN-PROCESS on purpose: with the fix in place the bad call
returns a Python exception, so a plain `pytest.raises` is enough and no
subprocess is needed. Pre-fix the process dies, so the test file cannot even
report -- which is exactly what `probes/m5b_make_pre_fix.py` demonstrates
mechanically (it reverses the fix, runs this file, and restores it).

Seed: fixed ids / small explicit shapes, no RNG dependency for the contract.
"""

import re
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
NATIVE_CPU = Path(APP_DIR) / "src" / "Drivers" / "CPU" / "_lib" / "native_cpu.py"
RC_BAD_RANGE = -2
RC_OVERFLOW = -4
RC_OK = 0


@pytest.fixture(scope="module")
def nc():
    from Drivers.CPU._lib import native_cpu as mod

    if not mod.variant_available():
        pytest.skip("native variant symbols absent (old DLL)")
    return mod


# --------------------------------------------------------------- the fix
@pytest.mark.fast
def test_binding_requests_checked_variant():
    """The i64 binding must pass V_CHECKED (1), never V_DENSE (0).

    Source-level assertion on purpose: it is the one check that fails
    cleanly (an assert) instead of killing the interpreter, so it is the
    fail-before canary for the whole file.
    """
    src = NATIVE_CPU.read_text(encoding="utf-8")
    i32 = [m for m in re.finditer(r"nf_group_variant_i32\(", src)]
    i64 = [m for m in re.finditer(r"nf_group_variant_i64\(", src)]
    assert len(i64) == 1, "expected exactly one i64 variant call site"
    body = src[i64[0].end():i64[0].end() + 200]
    variant = int(re.search(r"\.size,\s*(\d+)\s*,", body).group(1))
    assert variant == 1, (
        "nf_group_variant_i64 must be called with variant=1 (V_CHECKED); "
        "variant=0 reaches dense_scatter_i64, which has no key_in_range and "
        "aborts CPython (0xC0000409) on any out-of-range dense key")
    # i32 is safe already: V_DENSE for the i32 family IS checked_scatter_i32.
    body32 = src[i32[0].end():i32[0].end() + 200]
    assert int(re.search(r"\.size,\s*(\d+)\s*,", body32).group(1)) == 0


@pytest.mark.fast
def test_bad_range_maps_to_runtime_error(nc):
    """rc == -2 (BAD_RANGE) must surface as RuntimeError, not an abort."""
    keys = np.array([0, 1, 9, 7], dtype=np.int32)  # 9 >= g=8
    ticks = np.array([10, 20, 30, 40], dtype=np.int64)
    with pytest.raises(RuntimeError, match="dense key outside"):
        nc.sum_count_i64(keys, ticks, 8)


# ------------------------------------------------- the crash, in-process
@pytest.mark.fast
@pytest.mark.parametrize("bad", [-1, 8, 2 ** 31 - 1, -(2 ** 31)])
def test_out_of_range_key_does_not_kill_process(nc, bad):
    """Every out-of-range key shape: exception, process still alive.

    Pre-fix each of these is a 0xC0000409 process kill; the `alive` probe
    below is unreachable in that case because the interpreter is gone.
    """
    keys = np.array([0, 1, bad, 7], dtype=np.int32)
    ticks = np.array([10, 20, 30, 40], dtype=np.int64)
    with pytest.raises(RuntimeError, match="dense key outside"):
        nc.sum_count_i64(keys, ticks, 8)
    # alive: a subsequent good call must still be correct
    sums, counts = nc.sum_count_i64(np.array([0, 3, 3, 7], dtype=np.int32),
                                    ticks, 8)
    assert sums.tolist() == [10, 0, 0, 50, 0, 0, 0, 40]
    assert counts.tolist() == [1, 0, 0, 2, 0, 0, 0, 1]


@pytest.mark.fast
def test_production_wrapper_degrades_instead_of_dying(nc):
    """mixed_sum_count must return None (caller runs the f64 fallback)."""
    keys = np.array([0, 1, -1, 7], dtype=np.int32)
    col = np.array([10, 20, 30, 40], dtype=np.int64)
    assert nc.mixed_sum_count(keys, [col], 8) is None
    # and the same wrapper is still correct for in-range input
    ok = nc.mixed_sum_count(np.array([0, 1, 2, 7], dtype=np.int32), [col], 8)
    assert ok is not None
    sums, counts = ok
    assert counts.tolist() == [1, 1, 1, 0, 0, 0, 0, 1]
    assert int(sums[0][7]) == 40


# ------------------------------------------------------------ exact parity
_GRID_D = (1, 2, 8, 1000, 65536)
_GRID_N = (1, 2, 7, 1000, 100_000)
_PATTERNS = ("zeros", "ramp", "uniform", "reversed", "max_key", "single_hot")


def _keys_for(pattern, n, d, rng):
    if pattern == "zeros":
        return np.zeros(n, dtype=np.int32)
    if pattern == "ramp":
        return (np.arange(n, dtype=np.int64) % d).astype(np.int32)
    if pattern == "uniform":
        return rng.integers(0, d, size=n, dtype=np.int32)
    if pattern == "reversed":
        return ((n - 1 - np.arange(n, dtype=np.int64)) % d).astype(np.int32)
    if pattern == "max_key":  # key == D-1, the upper in-range edge
        return np.full(n, d - 1, dtype=np.int32)
    if pattern == "single_hot":  # one row at D-1
        k = np.zeros(n, dtype=np.int32)
        k[n - 1] = d - 1
        return k
    raise ValueError(pattern)


@pytest.mark.fast
def test_checked_variant_bit_identical_to_unchecked(nc):
    """V_CHECKED must be BIT-IDENTICAL to V_DENSE on every in-range input.

    5 D x 5 n x 6 key patterns = 150 cells for the i64 family (the lane that
    actually changed), asserted inside one test so the parity contract reads
    as one fact. Integer adds commute, so the range check cannot change a
    sum; this is the measurement that proves it on the shipped DLL.

    i32 is DELIBERATELY not in this comparison: the i32 family has no
    V_CHECKED lane at all -- `dispatch_i32` (variants.rs:88-100) routes
    variant=0 to `checked_scatter_i32` and returns BAD_VARIANT (-3) for
    variant=1, by design ("int32 lanes exist to prove the overflow contract,
    not speed"). Its range safety is asserted by
    `test_i32_lane_already_range_checked` instead.
    """
    lib = nc._req_variant()
    rng = np.random.default_rng(42)
    cells = 0
    for d in _GRID_D:
        for n in _GRID_N:
            for pattern in _PATTERNS:
                keys = _keys_for(pattern, n, d, rng)
                data = rng.integers(-(2 ** 20), 2 ** 20,
                                    size=n).astype(np.int64)
                kp, dp = keys.ctypes.data, data.ctypes.data
                out = []
                for v in (0, 1):  # V_DENSE (unchecked) vs V_CHECKED
                    s = np.zeros(d, dtype=np.int64)
                    c = np.zeros(d, dtype=np.int64)
                    rc = lib.nf_group_variant_i64(kp, dp, n, v,
                                                  s.ctypes.data,
                                                  c.ctypes.data, d)
                    assert rc == RC_OK, (
                        "in-range input rejected: %r rc=%d"
                        % ((d, n, pattern), rc))
                    out.append((rc, s.tobytes(), c.tobytes()))
                assert out[0] == out[1], (
                    "V_CHECKED diverged from V_DENSE on %r: rc %r vs %r"
                    % ((d, n, pattern), out[0][0], out[1][0]))
                cells += 1
    assert cells == 150


@pytest.mark.fast
def test_i32_lane_already_range_checked(nc):
    """The i32 family's V_DENSE is already range-checked; keep it that way.

    `dispatch_i32` (variants.rs:96-98) sends variant=0 to
    `checked_scatter_i32`, so an out-of-range i32 key already returns -2
    rather than aborting -- unlike the i64 family, which is why only the
    i64 binding needed the V_CHECKED flip.
    """
    lib = nc._req_variant()
    keys = np.array([0, 1, 9, 7], dtype=np.int32)  # 9 >= g=8
    ticks = np.array([10, 20, 30, 40], dtype=np.int32)
    sums = np.zeros(8, dtype=np.int32)
    counts = np.zeros(8, dtype=np.int64)
    rc = lib.nf_group_variant_i32(keys.ctypes.data, ticks.ctypes.data,
                                  keys.size, 0, sums.ctypes.data,
                                  counts.ctypes.data, 8)
    assert rc == RC_BAD_RANGE
    # the i32 family has no V_CHECKED lane at all (by design, not a bug)
    rc2 = lib.nf_group_variant_i32(keys.ctypes.data, ticks.ctypes.data,
                                   keys.size, 1, sums.ctypes.data,
                                   counts.ctypes.data, 8)
    assert rc2 == -3, "variant=1 must stay BAD_VARIANT for the i32 family"


@pytest.mark.fast
@pytest.mark.parametrize("d,n", [(1, 1), (1, 100_000), (8, 1), (2, 7)])
def test_edge_shapes_exact(nc, d, n):
    """D=1, n=1 and friends: exact dense result, no silent zero-fill."""
    rng = np.random.default_rng(42)
    for pattern in _PATTERNS:
        keys = _keys_for(pattern, n, d, rng)
        ticks = rng.integers(-(2 ** 20), 2 ** 20, size=n).astype(np.int64)
        sums, counts = nc.sum_count_i64(keys, ticks, d)
        ref = np.zeros(d, dtype=np.int64)
        cnt = np.zeros(d, dtype=np.int64)
        for k, t in zip(keys.tolist(), ticks.tolist()):
            ref[k] += t
            cnt[k] += 1
        assert sums.tolist() == ref.tolist(), (d, n, pattern)
        assert counts.tolist() == cnt.tolist(), (d, n, pattern)


@pytest.mark.fast
def test_max_key_is_in_range_and_exact(nc):
    """key == D-1 is IN range: it must be accepted, not rejected."""
    d, n = 8, 1000
    keys = np.full(n, d - 1, dtype=np.int32)
    ticks = np.arange(1, n + 1, dtype=np.int64)
    sums, counts = nc.sum_count_i64(keys, ticks, d)
    assert counts.tolist() == [0] * 7 + [n]
    assert sums.tolist() == [0] * 7 + [int(ticks.sum())]


@pytest.mark.fast
def test_overflow_still_reported_distinctly(nc):
    """-4 OVERFLOW must stay OverflowError, not collapse into BAD_RANGE."""
    keys = np.array([0, 0], dtype=np.int32)
    ticks = np.array([2 ** 62, 2 ** 62], dtype=np.int64)
    with pytest.raises(OverflowError, match="accumulator overflow"):
        nc.sum_count_i64(keys, ticks, 2)
    # and the process is still alive and correct afterwards
    sums, counts = nc.sum_count_i64(np.array([0, 1], dtype=np.int32),
                                    np.array([5, 7], dtype=np.int64), 2)
    assert sums.tolist() == [5, 7] and counts.tolist() == [1, 1]


@pytest.mark.fast
def test_raw_checked_variant_returns_minus_two(nc):
    """The raw symbol returns -2, proving the guard is Rust-side not Python-side."""
    lib = nc._req_variant()
    keys = np.array([0, 1, -1, 7], dtype=np.int32)
    ticks = np.array([10, 20, 30, 40], dtype=np.int64)
    sums = np.zeros(8, dtype=np.int64)
    counts = np.zeros(8, dtype=np.int64)
    rc = lib.nf_group_variant_i64(keys.ctypes.data, ticks.ctypes.data,
                                  keys.size, 1, sums.ctypes.data,
                                  counts.ctypes.data, 8)
    assert rc == RC_BAD_RANGE
