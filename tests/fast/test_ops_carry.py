# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Carry result-path: ColumnCarry primary, dict explicit compat (parity + stages).

Default result stays dict (frozen shape); result='carry' returns ColumnCarry
(ukeys/counts/sums, mean derived). Compat materialization is exact in all
shapes, serial or threaded.
"""

import time
from pathlib import Path

import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _bufs(a, jobs):
    graph = a["optimize"](a["compile"](jobs))
    return a["cpu_execute"](graph["nodes"])


def _carry_of(bufs, out):
    c = bufs[out + "#carry"]
    for attr in ("ukeys", "counts", "sums", "means", "to_dict_flat",
                 "to_dict_single", "to_dict_multi", "ngroups"):
        assert hasattr(c, attr), attr
    return c


# --- composite (tuple) key lane ------------------------------------------
# The tuple lane is pack_keys(mode="hash", N cols) -> groupby /
# groupby_multi. `result` is lane-independent by construction
# (nodes.py ir_groupby/ir_groupby_multi accept the same {dict, carry}
# set for both), so "carry on tuple keys" is the same contract as
# "carry on packed keys": ColumnCarry in `#carry` (and in `out` when
# result='carry'), dict only as explicit materialization.
#
# The tests below were written as xfail(strict=True) while the tuple
# branch bypassed _carry_result; all of them XPASSed when it was wired,
# and strict=True is what forced the marker off again.


def _tuple_keyed(a, key_cols, values, validities=None, op="sum", result=None,
                 multi_ops=None):
    """Series + pack_keys(mode='hash') + groupby/groupby_multi on tuples."""
    jobs = []
    for i, c in enumerate(key_cols):
        kw = {}
        if validities is not None:
            kw["validity"] = validities[i]
        jobs.append(a["ir_series"](f"k{i}", c, **kw))
    jobs.append(a["ir_series"]("v", values))
    jobs.append(a["ir_pack_keys"]("pk", *[f"k{i}"
                                         for i in range(len(key_cols))],
                                  mode="hash"))
    res = {} if result is None else {"result": result}
    if multi_ops is None:
        jobs.append(a["ir_groupby"]("g", "v", "pk", op, **res))
    else:
        jobs.append(a["ir_groupby_multi"]("g", "v", "pk", multi_ops, **res))
    return jobs


def _row_valid(validities, i):
    """Row survives the validity AND; a None column validity = all valid."""
    return all(v is None or v[i] for v in validities)


def _ref_groups(key_cols, values, validities=None):
    """Reference {tuple: sum} over valid rows, lex key order (groupindex
    contract: lex-sorted unique, validity AND over key columns)."""
    out = {}
    for i in range(len(values)):
        if validities is not None and not _row_valid(validities, i):
            continue
        k = tuple(int(c[i]) for c in key_cols)
        out[k] = out.get(k, 0) + values[i]
    return out


def _ref_counts(key_cols, values, validities=None):
    out = {}
    for i in range(len(values)):
        if validities is not None and not _row_valid(validities, i):
            continue
        k = tuple(int(c[i]) for c in key_cols)
        out[k] = out.get(k, 0) + 1
    return out


@pytest.mark.fast
def test_tuple_lane_documents_observed_default_dict_shape(kernel):
    """DOCUMENTING test. Pins the tuple lane's default output shape, and
    records what this lane did BEFORE result='carry' reached it.

    Observed v0.2.1 (now changed, deliberately, by V1): result='carry'
    was IGNORED here -- it returned the same dict as the default lane and
    `#carry` held that dict instead of a ColumnCarry. That was the
    un-wired branch, and the target-contract tests below are the change
    that closed it. This test keeps the durable half, so any FUTURE
    change to the default lane is still a reviewed one.

    The durable part: keys are tuples, values are the exact per-group
    sums, group order is lex. That is the 0 %-regression contract.
    """
    a = kernel.alias
    kc = [[0, 0, 1, 1, 2, 2, 0, 1], [5, 6, 5, 6, 5, 6, 5, 6],
          [1, 1, 2, 2, 1, 1, 2, 2], [9, 9, 9, 9, 8, 8, 8, 8]]
    v = [10, 20, 30, 40, 50, 60, 70, 80]
    bd = _bufs(a, _tuple_keyed(a, kc, v))
    bc = _bufs(a, _tuple_keyed(a, kc, v, result="carry"))
    d, c = bd["g"], bc["g"]

    # durable: default lane shape (this is the 0 %-regression contract)
    assert type(d) is dict
    ref = _ref_groups(kc, v)
    assert d == ref
    assert list(d) == sorted(ref)          # lex-sorted group order

    # result='carry' now bites on this lane: carry in `out` and in
    # `#carry`, default lane untouched (its #carry is the same carry).
    _carry_of(bd, "g")
    assert bc["g"] is bc["g#carry"]
    assert d == c.to_dict_flat("sum")      # same values, same group set


@pytest.mark.fast
def test_carry_flat_parity_dict(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [10, 20, 30, 40]),
            a["ir_series"]("k", [0, 1, 0, 1])]
    d = _bufs(a, jobs + [a["ir_groupby"]("g", "v", "k", "sum")])["g"]
    b = _bufs(a, jobs + [a["ir_groupby"]("g", "v", "k", "sum", result="carry")])
    c = _carry_of(b, "g")
    assert b["g"] is c
    assert list(c.ukeys) == [0, 1] and list(c.counts) == [2, 2]
    assert c.to_dict_flat("sum") == d
    assert c.to_dict_flat("count") == {0: 2, 1: 2}
    assert c.to_dict_flat("mean") == {0: 20.0, 1: 30.0}


@pytest.mark.fast
def test_carry_mean_derived_no_rescan(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("v", [1, 2, 3, 4], "float32"),
            a["ir_series"]("k", [0, 1, 0, 1]),
            a["ir_groupby_multi"]("g", "v", "k", ("sum", "count", "mean"),
                                  result="carry")]
    b = _bufs(a, jobs)
    c = _carry_of(b, "g")
    m1, m2 = c.means("v"), c.means("v")
    assert m1 is m2  # derived once, cached
    assert [float(x) for x in m1] == [2.0, 3.0]
    d = _bufs(a, [a["ir_series"]("v", [1, 2, 3, 4], "float32"),
                  a["ir_series"]("k", [0, 1, 0, 1]),
                  a["ir_groupby_multi"]("g", "v", "k",
                                        ("sum", "count", "mean"))])["g"]
    assert c.to_dict_single("v", ("sum", "count", "mean")) == d


@pytest.mark.fast
def test_carry_multi_threads_parity(kernel):
    a = kernel.alias
    n = 20_000
    import numpy as np  # noqa: PLC0415 -- deterministic data only

    rng = np.random.default_rng(42)
    v1 = rng.integers(0, 100, size=n).astype(np.int32)
    v3 = (rng.standard_normal(n)).astype(np.float64)
    kk = rng.integers(0, 5_000, size=n).astype(np.int32)
    mk = lambda res: [a["ir_series"]("v1", v1), a["ir_series"]("v3", v3, "float64"),  # noqa: E731
                      a["ir_series"]("k", kk),
                      a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                            {"v1": ("sum",), "v3": ("mean",)},
                                            result=res)]
    c = _carry_of(_bufs(a, mk("carry")), "g")
    d = _bufs(a, [a["ir_series"]("v1", v1), a["ir_series"]("v3", v3, "float64"),
                  a["ir_series"]("k", kk),
                  a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                        {"v1": ("sum",), "v3": ("mean",)})])["g"]
    ops = {"v1": ("sum",), "v3": ("mean",)}
    assert c.to_dict_multi(["v1", "v3"], ops) == d
    assert c.to_dict_multi(["v1", "v3"], ops, threads=4) == d
    s = time.perf_counter()
    c.to_dict_multi(["v1", "v3"], ops, threads=1)
    t1 = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    c.to_dict_multi(["v1", "v3"], ops, threads=4)
    t4 = (time.perf_counter() - s) * 1000
    print(f"\ncompat threads=1 {t1:.1f}ms vs threads=4 {t4:.1f}ms")


# --- target contract: result='carry' on the composite (tuple) lane --------
# All xfail(strict=True): each must XPASS once the tuple branch routes
# through _carry_result, and strict=True then forces the marker off.


@pytest.mark.fast
def test_tuple_lane_carry_contract_dict_multi(kernel):
    """Target: result='carry' on tuple keys returns a ColumnCarry whose
    ngroups is the number of distinct tuples, and whose explicit dict
    materialization is exactly the default lane's dict -- same values,
    same group set, same lex group order, same Python scalar types."""
    a = kernel.alias
    kc = [[0, 0, 1, 1, 2, 2, 0, 1], [5, 6, 5, 6, 5, 6, 5, 6],
          [1, 1, 2, 2, 1, 1, 2, 2], [9, 9, 9, 9, 8, 8, 8, 8]]
    v = [10, 20, 30, 40, 50, 60, 70, 80]
    d = _bufs(a, _tuple_keyed(a, kc, v, op="sum"))["g"]
    b = _bufs(a, _tuple_keyed(a, kc, v, op="sum", result="carry"))
    c = _carry_of(b, "g")
    assert b["g"] is c                       # result='carry' -> the carry
    ref = _ref_groups(kc, v)
    assert c.ngroups == len(ref)            # groups, NOT groups * ncols
    assert c.to_dict_flat("sum") == d
    assert c.to_dict_flat("count") == _ref_counts(kc, v)
    assert list(c.to_dict_flat("sum")) == list(d)   # same lex group order
    for got, want in zip(c.to_dict_flat("sum").values(), d.values()):
        assert type(got) is type(want)


@pytest.mark.fast
def test_tuple_lane_carry_d1_nc1_ngroups(kernel):
    """Edge D=1, nc=1: the degenerate case where ngroups is CORRECT by
    accident (ukeys.size == 1*1 == groups). Pins it anyway -- an
    ngroups-only check cannot see the nc>1 bug, so this case is what
    proves the check itself is not the thing being fixed."""
    a = kernel.alias
    kc = [[0, 1, 0, 1, 1, 0]]
    v = [1, 2, 3, 4, 5, 6]
    d = _bufs(a, _tuple_keyed(a, kc, v))["g"]
    b = _bufs(a, _tuple_keyed(a, kc, v, result="carry"))
    c = _carry_of(b, "g")
    assert c.ngroups == 2
    assert c.to_dict_flat("sum") == d == {(0,): 1 + 3 + 6, (1,): 2 + 4 + 5}


@pytest.mark.fast
def test_tuple_lane_carry_d1_nc4_ngroups(kernel):
    """Edge D=1, nc=4: the divergent case. One distinct tuple over four
    key columns, so a ngroups that reads ukeys.size reports 1*nc = 4
    instead of 1. The nc=1 case above cannot see that; this one can."""
    a = kernel.alias
    kc = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    v = [1, 2, 3, 4]
    d = _bufs(a, _tuple_keyed(a, kc, v))["g"]
    b = _bufs(a, _tuple_keyed(a, kc, v, result="carry"))
    c = _carry_of(b, "g")
    assert c.ngroups == 1
    assert c.ukeys.shape == (1, 4)      # 2-D view is (groups, cols)
    assert c.to_dict_flat("sum") == d == {(0, 1, 2, 3): 10}
    assert c.to_dict_flat("count") == {(0, 1, 2, 3): 4}


@pytest.mark.fast
def test_tuple_lane_carry_d3_nc4_ngroups(kernel):
    """Same divergence with more than one group: 3 groups over 4 key
    columns, so ukeys.size = 12 and the truth is 3 (ratio exactly nc)."""
    a = kernel.alias
    kc = [[0, 0, 1, 1, 2], [1, 1, 2, 2, 3], [5, 5, 5, 5, 5], [9, 9, 8, 8, 9]]
    v = [10, 20, 30, 40, 50]
    d = _bufs(a, _tuple_keyed(a, kc, v))["g"]
    b = _bufs(a, _tuple_keyed(a, kc, v, result="carry"))
    c = _carry_of(b, "g")
    assert d == {(0, 1, 5, 9): 30, (1, 2, 5, 8): 70, (2, 3, 5, 9): 50}
    assert c.ngroups == 3
    assert c.ukeys.shape == (3, 4)
    assert c.to_dict_flat("sum") == d


@pytest.mark.fast
def test_tuple_lane_carry_null_in_key_column_row_set(kernel):
    """Edge: NULL in a key column drops the whole row (3VL), and both
    lanes must drop the SAME rows -- the validity mask is applied before
    the result lane, so carry must not see the NULL rows at all."""
    a = kernel.alias
    kc = [[0, 0, 1, 1, 2], [5, 6, 5, 6, 5], [1, 1, 1, 1, 1], [9, 9, 9, 9, 9]]
    v = [10, 20, 30, 40, 50]
    # rows 1 and 3 have a NULL in key column 1 -> excluded from both lanes
    val = [1, None, 1, None, 1]
    valid = [None, val, None, None]
    d = _bufs(a, _tuple_keyed(a, kc, v, validities=valid))
    ref = _ref_groups(kc, v, validities=valid)
    assert ref == {(0, 5, 1, 9): 10, (1, 5, 1, 9): 30, (2, 5, 1, 9): 50}
    assert d["g"] == ref
    b = _bufs(a, _tuple_keyed(a, kc, v, validities=valid, result="carry"))
    c = _carry_of(b, "g")
    assert c.ngroups == len(ref)
    assert c.to_dict_flat("sum") == ref
    assert c.to_dict_flat("count") == {k: 1 for k in ref}
    assert sum(int(x) for x in c.counts) == 3   # NULL rows never reach carry


@pytest.mark.fast
def test_tuple_lane_carry_group_order_matches_dict_lane(kernel):
    """Edge: group ORDER. read.py consumes the groupby dict through
    record.items(), so the order is an observable (OFFSET/LIMIT, top-k).
    The contract is PARITY with the default lane: carry materialization
    must reproduce that lane's group order exactly, for every key width.
    (The order itself is groupindex's business; the carry must not
    reorder it.)"""
    a = kernel.alias
    for kc in ([[3, 1, 3, 1, 2], [7, 7, 0, 0, 5], [1, 1, 2, 2, 3]],
               [[3, 1, 3, 1, 2], [7, 7, 0, 0, 5]],
               [[3, 1, 3, 1, 2], [7, 7, 0, 0, 5], [1, 1, 2, 2, 3],
                [2, 2, 9, 9, 2]]):
        v = list(range(1, 6))
        d = _bufs(a, _tuple_keyed(a, kc, v))["g"]
        b = _bufs(a, _tuple_keyed(a, kc, v, result="carry"))
        c = _carry_of(b, "g")
        assert list(c.to_dict_flat("sum")) == list(d)
        assert list(c.to_dict_flat("count")) == list(d)
        assert len(set(list(d))) == c.ngroups      # distinct, no reordering


@pytest.mark.fast
def test_tuple_lane_carry_groupby_multi_contract(kernel):
    """groupby_multi on tuple keys: same contract, nested dict shape."""
    a = kernel.alias
    kc = [[0, 0, 1, 1, 1], [1, 2, 1, 2, 1]]
    v = [10, 20, 30, 40, 50]
    d = _bufs(a, _tuple_keyed(a, kc, v,
                              multi_ops=("sum", "count", "mean")))["g"]
    b = _bufs(a, _tuple_keyed(a, kc, v, multi_ops=("sum", "count", "mean"),
                              result="carry"))
    c = _carry_of(b, "g")
    assert d == {(0, 1): {"sum": 10, "count": 1, "mean": 10.0},
                 (0, 2): {"sum": 20, "count": 1, "mean": 20.0},
                 (1, 1): {"sum": 30 + 50, "count": 2, "mean": 40.0},
                 (1, 2): {"sum": 40, "count": 1, "mean": 40.0}}
    assert c.ngroups == 4
    assert c.to_dict_single("v", ("sum", "count", "mean")) == d
    assert list(c.to_dict_single("v", ("sum",))) == list(d)

