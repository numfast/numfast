# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: filter (boolean selection) + mask (and/or/not) + gather (spec 02).

WHERE = column -> compare -> BoolMask -> filter. No SQL-specific nodes.
int — exact; bool — exact. Empty / all-true / all-false valid, order kept.
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


@pytest.mark.fast
def test_filter_basic_order_kept(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [10, 20, 30, 40]),
        a["ir_compare"]("m", "v", 25, ">"),
        a["ir_filter"]("f", "v", "m"),
    ]
    assert list(_bufs(a, jobs)["f"]) == [30, 40]


@pytest.mark.fast
def test_filter_empty_all_true_all_false(kernel):
    a = kernel.alias
    base = [a["ir_series"]("v", [1, 2, 3])]
    assert list(_bufs(a, base + [a["ir_compare"]("m", "v", 0, ">"),
                                 a["ir_filter"]("f", "v", "m")])["f"]) == [1, 2, 3]
    assert list(_bufs(a, base + [a["ir_compare"]("m", "v", 99, ">"),
                                 a["ir_filter"]("f", "v", "m")])["f"]) == []
    e = _bufs(a, [a["ir_series"]("v", [], "int32"),
                  a["ir_series"]("m", [], "int32"),
                  a["ir_compare"]("c", "m", "m", "=="),
                  a["ir_filter"]("f", "v", "c")])["f"]
    assert list(e) == []


@pytest.mark.fast
def test_filter_null_mask_excludes(kernel):
    """3VL: invalid mask rows excluded, values validity kept on survivors."""
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [10, 20, 30], "int32", [1, 0, 1]),
        a["ir_compare"]("m", "v", 5, ">"),
        a["ir_filter"]("f", "v", "m"),
    ]
    bufs = _bufs(a, jobs)
    assert list(bufs["m"]) == [True, True, True]
    assert list(bufs["m#validity"]) == [True, False, True]
    assert list(bufs["f"]) == [10, 30]
    assert list(bufs["f#validity"]) == [True, True]


@pytest.mark.fast
def test_filter_size_mismatch_errors(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [1, 2, 3]),
        a["ir_series"]("w", [1, 2]),
        a["ir_compare"]("m", "w", 0, ">"),
        a["ir_filter"]("f", "v", "m"),
    ]
    with pytest.raises(ValueError, match="filter"):
        _bufs(a, jobs)


@pytest.mark.fast
def test_mask_and_or_not(kernel):
    a = kernel.alias
    base = [
        a["ir_series"]("v", [1, 2, 3, 4]),
        a["ir_compare"]("hi", "v", 1, ">"),
        a["ir_compare"]("lo", "v", 4, "<"),
    ]
    assert list(_bufs(a, base + [a["ir_mask"]("m", "hi", "lo", "and")])["m"]) == [
        False, True, True, False]
    assert list(_bufs(a, base + [a["ir_mask"]("m", "hi", "lo", "or")])["m"]) == [
        True, True, True, True]
    assert list(_bufs(a, base + [a["ir_mask"]("m", "hi", op="not")])["m"]) == [
        True, False, False, False]


@pytest.mark.fast
def test_mask_bad_op_rejected_at_ir(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="mask"):
        a["ir_mask"]("m", "x", "y", "xor")


@pytest.mark.fast
def test_gather_order_from_indices(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [10, 20, 30]),
        a["ir_series"]("i", [2, 0, 2]),
        a["ir_gather"]("g", "v", "i"),
    ]
    assert list(_bufs(a, jobs)["g"]) == [30, 10, 30]


@pytest.mark.fast
def test_gather_oob_errors(kernel):
    a = kernel.alias
    jobs = [
        a["ir_series"]("v", [10, 20]),
        a["ir_series"]("i", [0, 5]),
        a["ir_gather"]("g", "v", "i"),
    ]
    with pytest.raises(ValueError, match="gather"):
        _bufs(a, jobs)


@pytest.mark.fast
def test_where_chain_q2_shape(kernel):
    """ClickBench Q2 shape: AdvEngineID<>0 -> filter -> count (synthetic)."""
    a = kernel.alias
    adv = [0, 1, 0, 3, 0, 2]
    jobs = [
        a["ir_series"]("adv", adv),
        a["ir_compare"]("m", "adv", 0, "!="),
        a["ir_filter"]("f", "adv", "m"),
        a["ir_reduce"]("c", "f", "count"),
    ]
    assert _bufs(a, jobs)["c"] == 3


@pytest.mark.fast
def test_differential_native_vs_reference(kernel):
    """Native select/mask (when ABI present) bit-equal to numpy reference."""
    import importlib.util
    import sys

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    cpu_dir = Path(APP_DIR) / "src" / "Drivers" / "CPU" / "_lib"
    nat = _load("nf_native_cpu", cpu_dir / "native_cpu.py")
    # cpu.py uses same-extension absolute imports (builder mounts ext root):
    ext_root = str(cpu_dir.parent)
    sys.path.insert(0, ext_root)
    try:
        cpu = _load("nf_cpu_ref", cpu_dir / "cpu.py")
    finally:
        sys.path.remove(ext_root)

    rng = np.random.default_rng(42)
    for dt in (np.int32, np.int64, np.float32, np.float64, np.bool_):
        v = rng.integers(0, 100, size=1000).astype(dt)
        if dt is np.bool_:
            v = rng.integers(0, 2, size=1000).astype(bool)
        m = rng.integers(0, 2, size=1000).astype(bool)
        ref, eff = cpu._filter_ref(v, m)
        assert list(nat.select_scatter(v, eff)) == list(ref)
    x = rng.integers(0, 2, size=1000).astype(bool)
    y = rng.integers(0, 2, size=1000).astype(bool)
    assert list(nat.mask_and(x, y)) == list(cpu._mask_ref(x, y, "and"))
    assert list(nat.mask_or(x, y)) == list(cpu._mask_ref(x, y, "or"))
    assert list(nat.mask_not(x)) == list(cpu._mask_ref(x, None, "not"))
