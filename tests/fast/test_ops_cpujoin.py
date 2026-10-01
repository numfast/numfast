# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: CpuJoin production vs scratch/hash_join_oracle.py (P1) + pandas.

int exact (chk sum(v1)+sum(v2)); J3 NULL contract: miss -> valid 0
(payload 0 + invalid, never NaN/sentinel).
"""

import importlib.util as _ilu
from pathlib import Path as _Path
from types import SimpleNamespace as _NS

import numpy as _np
import pytest

_FORK = _Path(__file__).resolve().parents[2]
APP_DIR = str(_FORK)


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


@pytest.fixture(scope="module")
def J(kernel):
    a = kernel.alias
    return _NS(
        build_timed=a["join_build"],
        join_inner=a["join_inner"],
        join_left=a["join_left"],
        make_pair=a["join_pair"],
        chk_inner=lambda o1, o2: a["join_chk"](o1, o2),
        chk_left=lambda o1, o2, valid: a["join_chk"](o1, o2, valid),
    )


def _oracle():
    spec = _ilu.spec_from_file_location(
        "nfcpujoin_oracle", str(_FORK / "scratch" / "hash_join_oracle.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORACLE = _oracle()


@pytest.mark.fast
def test_cpujoin_inner_oracle_exact(J):
    lk, lv1 = [1, 2, 3], [10, 20, 30]
    rk, rv2 = [2, 3, 4], [200, 300, 400]
    b, _ = J.build_timed(rk, rv2)
    (ok, o1, o2), _ = J.join_inner(lk, lv1, b, threads=2)
    exp = ORACLE.join_inner(lk, lv1, rk, rv2)
    assert [(int(a), int(c), int(d)) for a, c, d in
            zip(ok.tolist(), o1.tolist(), o2.tolist())] == exp
    assert J.chk_inner(o1, o2) == ORACLE.check_exact_sum(exp)


@pytest.mark.fast
def test_cpujoin_left_null_contract(J):
    lk, lv1 = [1, 2, 3], [10, 20, 30]
    rk, rv2 = [2, 3, 4], [200, 300, 400]
    b, _ = J.build_timed(rk, rv2)
    (ok, o1, o2, valid), _ = J.join_left(lk, lv1, b, threads=2)
    exp = ORACLE.join_left_outer(lk, lv1, rk, rv2)
    assert ok.tolist() == [r[0] for r in exp]
    assert o1.tolist() == [r[1] for r in exp]
    assert valid.tolist() == [r[2] is not None for r in exp]
    # miss payload is 0 + invalid (NULL), never NaN/sentinel
    assert int(o2[0]) == 0 and not bool(valid[0])
    assert o2[1:].tolist() == [200, 300]
    assert J.chk_left(o1, o2, valid) == (60, 500)


@pytest.mark.fast
def test_cpujoin_fuzz_vs_oracle_and_pandas(J):
    pd = pytest.importorskip("pandas")
    rng = _np.random.default_rng(42)
    for t in range(30):
        n = int(rng.integers(5, 200))
        s = int(rng.integers(1, max(2, n)))
        xk, xv, rk, rv = J.make_pair(seed=42 + t, n=n, s=s)
        b, _ = J.build_timed(rk, rv)
        (ok, o1, o2), _ = J.join_inner(xk, xv, b, threads=4)
        rows = ORACLE.join_inner([int(x) for x in xk], [int(x) for x in xv],
                                 [int(x) for x in rk], [int(x) for x in rv])
        es1, es2 = ORACLE.check_exact_sum(rows)
        assert len(ok) == len(rows)
        assert J.chk_inner(o1, o2) == (es1, es2)
        m = pd.DataFrame({"k": _np.asarray(xk), "v1": _np.asarray(xv)}).merge(
            pd.DataFrame({"k": _np.asarray(rk), "v2": _np.asarray(rv)}),
            on="k", how="inner")
        assert len(ok) == len(m)
        assert J.chk_inner(o1, o2) == (int(m["v1"].sum()), int(m["v2"].sum()))
        (lk2, lv12, lo2, valid), _ = J.join_left(xk, xv, b, threads=4)
        assert len(lk2) == n
        assert int(valid.sum()) == len(rows)
        # left preserves ALL v1 (misses included); v2 only over hits
        assert J.chk_left(lv12, lo2, valid) == (
            int(_np.asarray(xv).astype(_np.int64).sum()), es2)
