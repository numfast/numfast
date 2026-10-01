# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: BoolMask полноправный тип — Series[bool] end-to-end (DELTA-5 gate-2).

Цепочка compare()->BoolMask->AND/OR/NOT->Filter->Series идёт по битам без
int32-посредника (промежутки — BitPack w=1); unpack только в compacting take.
Плюс: Series bool/enum с validity sidecar, filter порядок/границы (первая/
последняя строка, хвост не кратный 8, обратный порядок значений), пустые/
все-true через packed-путь. Seed 42.
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
    return type(_bufs(a, [a["ir_series"]("v", [True], "bool")])["v"])


@pytest.mark.fast
def test_chain_compare_mask_filter_no_int32(kernel):
    """compare->mask(and/or/not)->filter: промежутки BitPack, результат точен."""
    a = kernel.alias
    BitPack = _bitpack_cls(a)
    rng = np.random.default_rng(42)
    n = 10_000
    v = rng.integers(0, 100, size=n).astype("int32")
    w = rng.integers(0, 100, size=n).astype("int32")
    base = [a["ir_series"]("v", v), a["ir_series"]("w", w),
            a["ir_compare"]("m1", "v", 50, ">"),
            a["ir_compare"]("m2", "w", 50, "<")]
    for op in ("and", "or"):
        b = _bufs(a, base + [a["ir_mask"]("m", "m1", "m2", op),
                             a["ir_filter"]("f", "v", "m")])
        assert isinstance(b["m1"], BitPack) and isinstance(b["m2"], BitPack)
        assert isinstance(b["m"], BitPack)  # int32-посредника нет
        ref = v[(v > 50) & (w < 50)] if op == "and" else v[(v > 50) | (w < 50)]
        assert bool((np.asarray(b["f"]) == ref).all()) and len(b["f"]) == len(ref)
    b = _bufs(a, base + [a["ir_mask"]("m", "m1", op="not"),
                         a["ir_filter"]("f", "v", "m")])
    assert isinstance(b["m"], BitPack)
    assert bool((np.asarray(b["f"]) == v[~(v > 50)]).all())


@pytest.mark.fast
def test_packed_series_with_validity(kernel):
    """Series[bool]/enum с validity: sidecar bool, mismatch отклоняется."""
    a = kernel.alias
    BitPack = _bitpack_cls(a)
    b = _bufs(a, [a["ir_series"]("v", [True, False, True], "bool", [1, 0, 1])])
    assert isinstance(b["v"], BitPack) and list(b["v"]) == [True, False, True]
    assert list(b["v#validity"]) == [True, False, True]
    b = _bufs(a, [a["ir_series"]("e", [0, 1, 3], "enum:2", [1, 1, 0])])
    assert isinstance(b["e"], BitPack) and [int(x) for x in b["e"]] == [0, 1, 3]
    # 3VL end-to-end: invalid mask row исключается фильтром.
    b = _bufs(a, [a["ir_series"]("v", [10, 20, 30], "int32"),
                  a["ir_series"]("s", [True, True, True], "bool", [1, 0, 1]),
                  a["ir_filter"]("f", "v", "s")])
    assert list(b["f"]) == [10, 30]
    with pytest.raises(ValueError, match="validity size"):
        _bufs(a, [a["ir_series"]("v", [True, False], "bool", [1])])


@pytest.mark.fast
def test_filter_packed_order_boundaries(kernel):
    """Порядок значений сохранён; границы: первая/последняя, хвост %8 != 0."""
    a = kernel.alias
    vals = list(range(11))  # 11 строк: хвост 3 бита, padding нулевой
    sel = [10, 0, 5]  # обратный порядок значений, включая границы
    mask = [i in sel for i in vals]
    b = _bufs(a, [a["ir_series"]("v", vals),
                  a["ir_series"]("m", mask, "bool"),
                  a["ir_filter"]("f", "v", "m")])
    assert list(b["f"]) == [0, 5, 10]  # порядок строк, не порядок выбора
    b = _bufs(a, [a["ir_series"]("v", vals),
                  a["ir_compare"]("c", "v", 100, ">"),
                  a["ir_filter"]("f", "v", "c")])
    assert list(b["f"]) == []
    b = _bufs(a, [a["ir_series"]("v", vals),
                  a["ir_compare"]("c", "v", -1, ">"),
                  a["ir_filter"]("f", "v", "c")])
    assert list(b["f"]) == vals
