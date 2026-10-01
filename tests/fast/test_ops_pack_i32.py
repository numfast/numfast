# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: Planner-gated int32-direct radix pack (production Q2 path).

mode='radix' on 2 int32 code columns routes via plan_pack (generic
observables kmin/kmax/M2, no dataset branches): bound fits int32 -> one
int32 alloc, no casts, bit-identical to int64 radix; otherwise legacy int64
radix verbatim with #pack sidecar {strategy, reason}. mode='pack'
(bitpack) untouched. Overflow raises explicit error, never wraps.
"""

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
def test_plan_pack_gate_boundaries(kernel):
    a = kernel.alias
    ok = a["plan_pack"](0, 99, 0, 100)
    assert ok["strategy"] == "int32_direct" and "2**31" in ok["reason"]
    big = a["plan_pack"](0, 2**31 - 1, 0, 1)
    assert big["strategy"] == "radix" and "2**31" in big["reason"]
    neg = a["plan_pack"](-1, 99, 0, 100)
    assert neg["strategy"] == "radix" and "kmin" in neg["reason"]
    edge = a["plan_pack"](0, 213, 0, 10000000, 10000001)
    assert edge["strategy"] == "int32_direct"  # 213*10000001+1e7 < 2**31
    over = a["plan_pack"](0, 214, 0, 10000000, 10000001)
    assert over["strategy"] == "radix"  # one step over the bound
    with pytest.raises(ValueError, match="M2>=1"):
        a["plan_pack"](0, 1, 0, 1, 0)


@pytest.mark.fast
def test_radix_i32_bit_identical_and_grouping_equiv(kernel):
    import numpy as np

    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 50_000
    k1 = rng.integers(0, 100, size=n, dtype=np.int32)
    k2 = rng.integers(0, 101, size=n, dtype=np.int32)
    vals = rng.integers(0, 100, size=n).tolist()
    b = _bufs(a, [a["ir_series"]("hi", k1), a["ir_series"]("lo", k2),
                  a["ir_pack_keys"]("p", "hi", "lo", mode="radix")])
    assert b["p"].dtype == np.int32
    assert b["p#pack"]["strategy"] == "int32_direct"
    m2 = int(k2.max()) + 1
    ref = (k1.astype(np.int64) * np.int64(m2) + k2.astype(np.int64))
    assert bool((b["p"].astype(np.int64) == ref).all()), "bit-identical"
    # grouping equivalence vs bitpack (same partition sizes, own labels)
    bit = (k1.astype(np.int64) << np.int64(32)) | (
        k2.astype(np.int64) & np.int64(0xFFFFFFFF))
    co = np.sort(np.unique(ref, return_counts=True)[1])
    cb = np.sort(np.unique(bit, return_counts=True)[1])
    assert bool((co == cb).all())
    # end-to-end dict exact vs int64 reference
    got = _bufs(a, [a["ir_series"]("v", vals),
                    a["ir_series"]("hi", k1), a["ir_series"]("lo", k2),
                    a["ir_pack_keys"]("p", "hi", "lo", mode="radix"),
                    a["ir_groupby"]("g", "v", "p", "sum")])["g"]
    want = {}
    for kk, vv in zip(ref.tolist(), vals):
        want[kk] = want.get(kk, 0) + vv
    assert got == want


@pytest.mark.fast
def test_radix_fallbacks_verbatim(kernel):
    import numpy as np

    a = kernel.alias
    # negative codes -> legacy int64 radix, values follow c0*M2+c1
    b = _bufs(a, [a["ir_series"]("hi", [1, -3]),
                  a["ir_series"]("lo", [2, 7]),
                  a["ir_pack_keys"]("p", "hi", "lo", mode="radix")])
    assert b["p"].dtype == np.int64 and list(b["p"]) == [10, -17]
    assert b["p#pack"]["strategy"] == "radix"
    # int64 inputs -> legacy path (no silent narrow), values exact
    k1 = np.array([1, 2], dtype=np.int64)
    b = _bufs(a, [a["ir_series"]("hi", k1, "int64"),
                  a["ir_series"]("lo", [0, 1], "int64"),
                  a["ir_pack_keys"]("p", "hi", "lo", mode="radix")])
    assert b["p"].dtype == np.int64 and list(b["p"]) == [2, 5]
    # default pack bitpack untouched, no sidecar
    b = _bufs(a, [a["ir_series"]("hi", [1]), a["ir_series"]("lo", [2]),
                  a["ir_pack_keys"]("p", "hi", "lo")])
    assert list(b["p"]) == [(1 << 32) | 2] and "p#pack" not in b


@pytest.mark.fast
def test_i32_boundary_exact_no_wrap(kernel):
    """Bound edge through the public path: max fitting composite stays
    int32_direct with exact values (never wraps); +1 step falls back to
    legacy int64 radix, also exact."""
    import numpy as np

    a = kernel.alias
    m2, k2max = 101, 100
    k1max = (2**31 - 1 - k2max) // m2
    assert k1max * m2 + k2max <= 2**31 - 1 < (k1max + 1) * m2 + k2max
    b = _bufs(a, [a["ir_series"]("hi", np.array([0, k1max], dtype=np.int32)),
                  a["ir_series"]("lo", np.array([0, k2max], dtype=np.int32)),
                  a["ir_pack_keys"]("p", "hi", "lo", mode="radix")])
    assert b["p#pack"]["strategy"] == "int32_direct"
    assert list(b["p"]) == [0, k1max * m2 + k2max]
    b = _bufs(a, [a["ir_series"]("hi", np.array([0, k1max + 1], dtype=np.int32)),
                  a["ir_series"]("lo", np.array([0, k2max], dtype=np.int32)),
                  a["ir_pack_keys"]("p", "hi", "lo", mode="radix")])
    assert b["p#pack"]["strategy"] == "radix"
    assert list(b["p"]) == [0, (k1max + 1) * m2 + k2max]
