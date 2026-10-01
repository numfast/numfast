# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: BoolMask as 1-bit packed enum + generic width pack (DELTA-5).

Series stores bool/enum:W as BitPack (8 masks/byte); mask and/or/not run on
packed bytes (never int32); filter unpacks bits->bool once (proven-necessary
for the compacting take). Bit-exact vs numpy reference, seed 42.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from harness import load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()


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


def _bitpack_cls(a):
    bufs = _bufs(a, [a["ir_series"]("v", [True], "bool")])
    return type(bufs["v"])


@pytest.mark.fast
def test_series_bool_roundtrip(kernel):
    a = kernel.alias
    BitPack = _bitpack_cls(a)
    b = _bufs(a, [a["ir_series"]("v", [True, False, True, True], "bool")])
    assert isinstance(b["v"], BitPack) and b["v"].width == 1
    assert b["v"].nbytes == 1  # 4 masks -> 1 byte (was 16B as int32)
    assert list(b["v"]) == [True, False, True, True]
    b = _bufs(a, [a["ir_series"]("v", [1, 0, 1], "bool")])
    assert list(b["v"]) == [True, False, True]


@pytest.mark.fast
def test_series_enum_width_pack(kernel):
    a = kernel.alias
    BitPack = _bitpack_cls(a)
    b = _bufs(a, [a["ir_series"]("e", [0, 1, 3, 2, 3, 0, 1, 2], "enum:2")])
    assert isinstance(b["e"], BitPack) and b["e"].width == 2
    assert b["e"].nbytes == 2  # 8x2bit = 2 bytes (was 32B as int32)
    assert [int(v) for v in b["e"]] == [0, 1, 3, 2, 3, 0, 1, 2]
    b = _bufs(a, [a["ir_series"]("e", [0, 15, 7], "enum:4")])
    assert b["e"].nbytes == 2 and [int(v) for v in b["e"]] == [0, 15, 7]


@pytest.mark.fast
def test_mask_bit_exact_vs_reference(kernel):
    """Packed and/or/not bit-equal to the numpy bool reference (100k, seed 42)."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    n = 100_000
    x = rng.integers(0, 2, size=n).astype(bool)
    y = rng.integers(0, 2, size=n).astype(bool)
    BitPack = _bitpack_cls(a)
    base = [a["ir_series"]("x", x, "bool"), a["ir_series"]("y", y, "bool")]
    for op, ref in (("and", x & y), ("or", x | y)):
        got = _bufs(a, base + [a["ir_mask"]("m", "x", "y", op)])["m"]
        assert isinstance(got, BitPack)
        assert bool((np.asarray(got) == ref).all())
    got = _bufs(a, base + [a["ir_mask"]("m", "x", op="not")])["m"]
    assert isinstance(got, BitPack)
    assert bool((np.asarray(got) == ~x).all())


@pytest.mark.fast
def test_mask_empty_all_true_all_false_tail_zero(kernel):
    a = kernel.alias
    b = _bufs(a, [a["ir_series"]("x", [], "bool"),
                  a["ir_series"]("y", [], "bool"),
                  a["ir_mask"]("m", "x", "y", "and")])
    assert list(b["m"]) == [] and b["m"].nbytes == 0
    for vals, op, want in [([True] * 10, "not", [False] * 10),
                           ([False] * 10, "not", [True] * 10),
                           ([True] * 3, "not", [False] * 3)]:
        b = _bufs(a, [a["ir_series"]("x", vals, "bool"),
                      a["ir_mask"]("m", "x", op=op)])
        assert list(b["m"]) == want
        tail = 8 - (len(vals) % 8) if len(vals) % 8 else 0
        if tail:
            assert int(b["m"].packed[-1]) >> (8 - tail) == 0  # padding stays 0


@pytest.mark.fast
def test_filter_packed_matches_bool_and_3vl(kernel):
    a = kernel.alias
    v = [10, 20, 30, 40]
    jobs_bool = [a["ir_series"]("v", v),
                 a["ir_compare"]("m", "v", 15, ">"),
                 a["ir_filter"]("f", "v", "m")]
    ref = list(_bufs(a, jobs_bool)["f"])
    jobs_packed = [a["ir_series"]("v", v),
                   a["ir_series"]("s", [False, True, True, True], "bool"),
                   a["ir_filter"]("f", "v", "s")]
    assert list(_bufs(a, jobs_packed)["f"]) == ref == [20, 30, 40]
    # 3VL: invalid mask rows excluded, values validity kept on survivors.
    b = _bufs(a, [a["ir_series"]("v", [10, 20, 30], "int32", [1, 0, 1]),
                  a["ir_series"]("s", [True, True, True], "bool", [1, 0, 1]),
                  a["ir_filter"]("f", "v", "s")])
    assert list(b["f"]) == [10, 30]
    assert list(b["f#validity"]) == [True, True]


@pytest.mark.fast
def test_packed_10M_memory(kernel):
    """10M masks: 1_250_000B packed (was 40MB as int32 Series, 10MB bool)."""
    a = kernel.alias
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, size=10_000_000).astype(bool)
    b = _bufs(a, [a["ir_series"]("m", x, "bool")])
    assert b["m"].nbytes == 1_250_000
    assert b["m"].n == 10_000_000


@pytest.mark.fast
def test_packed_invalid_rejected(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="bool series needs 0/1"):
        _bufs(a, [a["ir_series"]("v", [0, 2], "bool")])
    with pytest.raises(ValueError, match="out of range"):
        _bufs(a, [a["ir_series"]("v", [0, 4], "enum:2")])
    with pytest.raises(ValueError, match="enum width"):
        _bufs(a, [a["ir_series"]("v", [0, 1], "enum:3")])
    with pytest.raises(ValueError, match="filter"):
        _bufs(a, [a["ir_series"]("v", [1, 2]),
                  a["ir_series"]("m", [1, 0], "int32"),
                  a["ir_filter"]("f", "v", "m")])
    with pytest.raises(ValueError, match="mask"):
        _bufs(a, [a["ir_series"]("x", [True, False], "bool"),
                  a["ir_series"]("y", [True], "bool"),
                  a["ir_mask"]("m", "x", "y", "and")])
