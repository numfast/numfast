# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: bulk sort (stable permutation) + slice (LIMIT/OFFSET) (spec 02).

Universal CPU primitives, no SQL: ORDER BY/LIMIT lower to sort -> slice ->
gather. sort returns the stable permutation (rank order); per-row rank is its
inverse (second sort of the permutation -- composition). Dictionary TEXT sorts
by precomputed numeric rank codes, never strings. Invalid key rows sort last.
Seed 42 wherever RNG is used; stage breakdown printed per run.
"""

import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
I32MIN, I32MAX = -(2 ** 31), 2 ** 31 - 1


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
def test_sort_ascending_stable(kernel):
    a = kernel.alias
    vals = [3, 1, 2, 1, 3, 0]
    perm = _bufs(a, [a["ir_series"]("v", vals), a["ir_sort"]("p", "v")])["p"]
    assert perm.dtype == np.dtype(np.int32)
    assert [int(i) for i in perm] == [5, 1, 3, 2, 0, 4]
    assert [vals[int(i)] for i in perm] == [0, 1, 1, 2, 3, 3]


@pytest.mark.fast
def test_sort_descending_stable_int32min_safe(kernel):
    a = kernel.alias
    vals = [I32MIN, 5, I32MIN, 0, 5, I32MAX]
    perm = _bufs(a, [a["ir_series"]("v", vals),
                     a["ir_sort"]("p", "v", descending=True)])["p"]
    assert [vals[int(i)] for i in perm] == [I32MAX, 5, 5, 0, I32MIN, I32MIN]
    assert [int(i) for i in perm] == [5, 1, 4, 3, 0, 2]


@pytest.mark.fast
def test_sort_composite_lexicographic_mixed(kernel):
    a = kernel.alias
    k1 = [1, 0, 1, 0, 1, 0]
    k2 = [5, 9, 3, 9, 5, 1]
    perm = _bufs(a, [a["ir_series"]("a", k1), a["ir_series"]("b", k2),
                     a["ir_sort"]("p", "a", "b",
                                  descending=[False, True])])["p"]
    rows = [(k1[int(i)], k2[int(i)]) for i in perm]
    assert rows == [(0, 9), (0, 9), (0, 1), (1, 5), (1, 5), (1, 3)]
    assert [int(i) for i in perm] == [1, 3, 5, 0, 4, 2]


@pytest.mark.fast
def test_sort_random_lexicographic_vs_reference(kernel):
    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 10_000
    k1 = rng.integers(0, 50, size=n).astype(np.int32)
    k2 = rng.integers(0, 50, size=n).astype(np.int32)
    s = time.perf_counter()
    perm = _bufs(a, [a["ir_series"]("a", k1), a["ir_series"]("b", k2),
                     a["ir_sort"]("p", "a", "b")])["p"]
    t_exec = (time.perf_counter() - s) * 1000
    ref = np.lexsort((k2, k1))
    assert [int(i) for i in perm] == [int(i) for i in ref]
    assert (k1[perm][:-1] <= k1[perm][1:]).all()
    print(f"\nstages ms: sort_10k_total={t_exec:.3f}")


@pytest.mark.fast
def test_sort_invalid_rows_last_both_directions(kernel):
    a = kernel.alias
    vals = [3, 1, 2, 0]
    validity = [1, 0, 1, 0]
    for desc in (False, True):
        perm = _bufs(a, [a["ir_series"]("v", vals, validity=validity),
                         a["ir_sort"]("p", "v", descending=desc)])["p"]
        got = [int(i) for i in perm]
        assert got[2:] == ([1, 3] if not desc else [1, 3])
        assert [vals[i] for i in got[:2]] == ([2, 3] if not desc else [3, 2])


@pytest.mark.fast
def test_sort_text_rank_via_dictionary(kernel):
    a = kernel.alias
    words = ["pear", "apple", "fig", "apple", "date"]
    enc = a["dictionary_encode"](words)
    perm = _bufs(a, [a["ir_series"]("c", enc["codes"]),
                     a["ir_sort"]("p", "c")])["p"]
    assert [words[int(i)] for i in perm] == ["apple", "apple", "date", "fig", "pear"]


@pytest.mark.fast
def test_sort_rejects_raw_strings(kernel):
    a = kernel.alias
    # TEXT never reaches sort as strings: series ingest already refuses them,
    # so the only TEXT path is dictionary_encode -> numeric rank (above).
    with pytest.raises(ValueError):
        _bufs(a, [a["ir_series"]("t", ["b", "a"]), a["ir_sort"]("p", "t")])


@pytest.mark.fast
def test_sort_accepts_int64_and_float_keys(kernel):
    a = kernel.alias
    big = [2 ** 40, 1, 2 ** 40 - 1]
    perm = _bufs(a, [a["ir_series"]("k", big, "int64"),
                     a["ir_sort"]("p", "k")])["p"]
    assert [big[int(i)] for i in perm] == [1, 2 ** 40 - 1, 2 ** 40]
    fl = [0.5, -1.25, 0.5, 2.0]
    perm = _bufs(a, [a["ir_series"]("f", fl, "float32"),
                     a["ir_sort"]("p", "f", descending=True)])["p"]
    assert [fl[int(i)] for i in perm] == [2.0, 0.5, 0.5, -1.25]


@pytest.mark.fast
def test_sort_empty_and_single(kernel):
    a = kernel.alias
    assert _bufs(a, [a["ir_series"]("e", []),
                     a["ir_sort"]("p", "e")])["p"].size == 0
    assert _bufs(a, [a["ir_series"]("e", []),
                     a["ir_sort"]("p", "e", descending=True)])["p"].size == 0
    assert [int(i) for i in _bufs(a, [a["ir_series"]("s", [7]),
                                      a["ir_sort"]("p", "s")])["p"]] == [0]


@pytest.mark.fast
def test_slice_limit_offset_exact(kernel):
    a = kernel.alias
    vals = [10, 20, 30, 40, 50]
    out = _bufs(a, [a["ir_series"]("v", vals),
                    a["ir_slice"]("s", "v", limit=2, offset=1)])["s"]
    assert [int(v) for v in out] == [20, 30]
    out = _bufs(a, [a["ir_series"]("v", vals),
                    a["ir_slice"]("s", "v", offset=3)])["s"]
    assert [int(v) for v in out] == [40, 50]
    out = _bufs(a, [a["ir_series"]("v", vals),
                    a["ir_slice"]("s", "v", limit=99)])["s"]
    assert [int(v) for v in out] == vals
    out = _bufs(a, [a["ir_series"]("v", vals),
                    a["ir_slice"]("s", "v", limit=10, offset=5)])["s"]
    assert out.size == 0 and str(out.dtype) == "int32"


@pytest.mark.fast
def test_slice_rejects_bad_bounds(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="slice"):
        a["ir_slice"]("s", "v", limit=-1)
    with pytest.raises(ValueError, match="slice"):
        a["ir_slice"]("s", "v", offset=-2)
    with pytest.raises(ValueError, match="slice"):
        a["ir_slice"]("s", "v", limit=True)


@pytest.mark.fast
def test_topk_sort_slice_gather_composition(kernel):
    a = kernel.alias
    vals = [5, 1, 4, 1, 5, 9, 2, 6]
    bufs = _bufs(a, [a["ir_series"]("v", vals),
                     a["ir_sort"]("p", "v", descending=True),
                     a["ir_slice"]("t", "p", limit=3),
                     a["ir_gather"]("g", "v", "t")])
    assert [int(v) for v in bufs["g"]] == [9, 6, 5]


@pytest.mark.fast
def test_rank_is_inverse_permutation(kernel):
    a = kernel.alias
    vals = [30, 10, 20]
    bufs = _bufs(a, [a["ir_series"]("v", vals),
                     a["ir_sort"]("p", "v"),
                     a["ir_sort"]("r", "p")])
    assert [int(i) for i in bufs["p"]] == [1, 2, 0]
    assert [int(i) for i in bufs["r"]] == [2, 0, 1]


@pytest.mark.fast
def test_sort_slice_planner_capability(kernel):
    a = kernel.alias
    cap = a["cpu_capability"]()
    assert "sort" in cap["ops"] and "slice" in cap["ops"]
    assert cap["chunkable_hints"]["sort"] is False
    assert cap["chunkable_hints"]["slice"] is False
    graph = a["compile"]([a["ir_series"]("v", [2, 1]),
                          a["ir_sort"]("p", "v"),
                          a["ir_slice"]("s", "p", limit=1)])
    assert [n["kernel_id"] for n in graph["nodes"]] == ["series", "sort", "slice"]
    assert a["chunk_plan"]("sort", 10)["num_chunks"] == 1
