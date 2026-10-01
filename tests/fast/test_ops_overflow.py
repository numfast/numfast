# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Regression: Invariant #1 — int64->int32 narrowing refuses overflow loudly.

BIGINT (ClickBench UserID ~9e18) has no Core dtype (GPU-portable); any
implicit narrowing to int32 raises OverflowError (what+range+value+fix),
never a silent wrap (the old garbage class, e.g. -6384394 for AVG(UserID)).
Q4/Q16/Q17/Q20/Q32/Q33-style queries over BIGINT are therefore explicit
unsupported, not wrong numbers. Seed: no RNG (fixed ids).
"""

import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
I32MIN, I32MAX = -(2 ** 31), 2 ** 31 - 1
# Real ClickBench BIGINT magnitudes (Q20 literal + int64-scale neighbours).
USERIDS = [435090932899640449, 1234567890123456789, 9223372036854775807]


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


@pytest.mark.fast
def test_userid_series_overflow_loud(kernel):
    a = kernel.alias
    with pytest.raises(OverflowError, match="narrowing overflow") as ei:
        _bufs(a, [a["ir_series"]("u", USERIDS)])
    msg = str(ei.value)
    assert "int32 range" in msg and "Fix:" in msg
    assert "observed range" in msg


@pytest.mark.fast
def test_silent_wrap_would_be_garbage(kernel):
    a = kernel.alias
    wrapped = np.asarray(USERIDS).astype(np.int32)
    assert (wrapped != np.asarray(USERIDS)).all(), "wrap must corrupt (else no bug)"
    with pytest.raises(OverflowError):
        _bufs(a, [a["ir_series"]("u", USERIDS)])


@pytest.mark.fast
def test_int32_edges_exact(kernel):
    a = kernel.alias
    vals = [I32MIN, -1, 0, 1, I32MAX]
    out = _bufs(a, [a["ir_series"]("e", vals)])["e"]
    assert [int(v) for v in out] == vals


@pytest.mark.fast
def test_int32_bounds_overflow(kernel):
    a = kernel.alias
    for bad in (I32MAX + 1, I32MIN - 1):
        with pytest.raises(OverflowError, match="narrowing overflow"):
            _bufs(a, [a["ir_series"]("b", [0, bad])])


@pytest.mark.fast
def test_schema_transform_overflow_loud(kernel):
    a = kernel.alias
    s = time.perf_counter()
    assert [int(v) for v in a["schema_transform"]([1.0, 2.0], 0.5, 0)] == [2, 4]
    t_ok = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    with pytest.raises(OverflowError, match="narrowing overflow"):
        a["schema_transform"]([4e9], 0.5, 0)
    t_err = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: schema_ok={t_ok:.3f} schema_overflow={t_err:.3f}")


@pytest.mark.fast
def test_q4_avg_userid_unsupported_not_garbage(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("u", USERIDS), a["ir_reduce"]("r", "u", "mean")]
    with pytest.raises(OverflowError, match="narrowing overflow"):
        _bufs(a, jobs)


@pytest.mark.fast
def test_q16_groupby_userid_unsupported_not_garbage(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [1, 2, 3]), a["ir_series"]("k", USERIDS),
            a["ir_groupby"]("g", "v", "k", "count")]
    with pytest.raises(OverflowError, match="narrowing overflow"):
        _bufs(a, jobs)


@pytest.mark.fast
def test_persist_bigint_narrowing_overflow(kernel, tmp_path):
    a = kernel.alias
    s = time.perf_counter()
    with pytest.raises(OverflowError, match="narrowing overflow"):
        a["persist_table"]({"u": {"values": USERIDS, "dtype": "int32"}},
                           str(tmp_path / "big.npz"))
    t_err = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: persist_overflow={t_err:.3f}")
