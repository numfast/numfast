# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: zero-copy composite keys via pack_keys (DELTA-2).

int32 codes -> int64 pack, hash/group ints only, no string concat.
int — exact. Microbench: Q2 (pack+groupby) <= 1.2x Q1 (single-key groupby).
"""

import time
from numbers import Integral
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


@pytest.mark.fast
def test_pack_keys_two_codes_exact(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("hi", [1, 0, -3]),
        a["ir_series"]("lo", [2, 0, 7]),
        a["ir_pack_keys"]("p", "hi", "lo"),
    ]
    got = list(_bufs(a, jobs)["p"])
    assert got[0] == (1 << 32) | 2
    assert got[1] == 0
    assert got[2] == (-3 << 32) | 7
    # Contract is int64 values exact; numpy int64 IS the int64 repr (not a bug).
    assert all(isinstance(x, Integral) for x in got)


@pytest.mark.fast
def test_pack_keys_single_widens(kernel):
    a = kernel.alias
    jobs = [a["ir_series"]("k", [0, 2**31 - 1, -2**31]), a["ir_pack_keys"]("p", "k")]
    got = list(_bufs(a, jobs)["p"])
    assert got == [0, 2**31 - 1, -(2**31)]


@pytest.mark.fast
def test_pack_keys_groupby_matches_tuple_reference(kernel):
    a = kernel.alias
    vals = [5, 10, 15, 20, 25]
    hi = [1, 1, 2, 2, 1]
    lo = [0, 1, 0, 0, 0]
    got = _bufs(a, [
        a["ir_series"]("v", vals),
        a["ir_series"]("hi", hi), a["ir_series"]("lo", lo),
        a["ir_pack_keys"]("p", "hi", "lo"),
        a["ir_groupby"]("g", "v", "p", "sum"),
    ])["g"]
    ref = {}
    for v, h, l in zip(vals, hi, lo):
        ref.setdefault((h, l), 0)
        ref[(h, l)] += v
    assert len(got) == len(ref)
    for (h, l), want in ref.items():
        assert got[(h << 32) | l] == want


@pytest.mark.fast
def test_pack_keys_non_int_rejected(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("f", [1.5, 2.5], "float32"),
        a["ir_series"]("k", [0, 1]),
        a["ir_pack_keys"]("p", "f", "k"),
    ]
    graph = a["optimize"](a["compile"](jobs))
    with pytest.raises(ValueError, match="int32 code"):
        a["cpu_execute"](graph["nodes"])


@pytest.mark.fast
def test_pack_keys_arity_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="1-2 key series"):
        a["ir_pack_keys"]("p", "a", "b", "c")


@pytest.mark.fast
def test_microbench_q2_vs_q1(kernel):
    """Q2 (pack 2 keys + groupby) <= 1.2x Q1 (single-key groupby). Seed 42.

    Both group the SAME 500 partitions over int64 keys, so the ratio isolates
    the online pack cost. Q1 ingests one pre-packed int64 key column (baseline:
    groupby of the composite key as if packed offline); Q2 packs (hi,lo)=
    divmod(k1,10) online, then groups. (hi,lo) is bijective to k1, hence the
    same partitions — asserted equal below. Grouping two independent 0..500
    keys instead would create ~137k distinct packs vs 500 groups and measure
    cardinality (275x more grouping work), not pack cost.
    """
    import numpy as np

    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 200_000
    vals = rng.integers(0, 100, size=n).tolist()
    k1 = rng.integers(0, 500, size=n).tolist()
    hi = [x // 10 for x in k1]
    lo = [x % 10 for x in k1]
    packed = [(h << 32) | (l & 0xFFFFFFFF) for h, l in zip(hi, lo)]

    n_q1 = a["optimize"](a["compile"](
        [a["ir_series"]("v", vals), a["ir_series"]("p0", packed, "int64"),
         a["ir_groupby"]("g", "v", "p0", "sum")]))["nodes"]
    n_q2 = a["optimize"](a["compile"](
        [a["ir_series"]("v", vals), a["ir_series"]("hi", hi), a["ir_series"]("lo", lo),
         a["ir_pack_keys"]("p", "hi", "lo"), a["ir_groupby"]("g", "v", "p", "sum")]))["nodes"]
    assert a["cpu_execute"](n_q1)["g"] == a["cpu_execute"](n_q2)["g"]

    def best(nodes, reps=9):
        a["cpu_execute"](nodes)  # warmup: page caches, allocators; untimed
        ts = []
        for _ in range(reps):
            s = time.perf_counter()
            a["cpu_execute"](nodes)
            ts.append((time.perf_counter() - s) * 1000)
        return min(ts)

    t_q1, t_q2 = best(n_q1), best(n_q2)
    print(f"\nmicrobench ms: Q1={t_q1:.2f} Q2={t_q2:.2f} ratio={t_q2 / t_q1:.3f}")
    assert t_q2 <= 1.2 * t_q1, f"Q2 {t_q2:.2f}ms > 1.2x Q1 {t_q1:.2f}ms"
