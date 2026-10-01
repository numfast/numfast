# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: ir_text_startswith/endswith/equals CPU (affix + equality).

Semantics (frozen): anchored affix, case-sensitive, empty affix matches
all valid rows; equals = full-row == (empty key matches only empty
valid rows). None/non-str rows -> invalid (DELTA-3), never an error,
never a match. Oracle: independent pure-Python (no numfast imports
inside oracle fns). Seed 42 where RNG is used.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
SEED = 42


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


def _rand_texts(rng, n):
    alpha = ["a", "ab", "abc", "Hello", "hello", "", " ", "x" * 40,
             "CJK", "日本語テスト", "emoji", "a😀b", "🎉🎉", "café",
             "naïve", "Ω≈ç√", "id123", "CITY#9", "MiXeD CaSe"]
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.05:
            out.append(None)
        else:
            out.append(alpha[int(rng.integers(len(alpha)))])
    return out


def oracle_sw(values, prefix):
    hit, valid = [], []
    for v in values:
        if isinstance(v, str):
            hit.append(v.startswith(prefix))
            valid.append(True)
        else:
            hit.append(False)
            valid.append(False)
    return np.array(hit, dtype=bool), np.array(valid, dtype=bool)


def oracle_ew(values, suffix):
    hit, valid = [], []
    for v in values:
        if isinstance(v, str):
            hit.append(v.endswith(suffix))
            valid.append(True)
        else:
            hit.append(False)
            valid.append(False)
    return np.array(hit, dtype=bool), np.array(valid, dtype=bool)


def oracle_eq(values, key):
    hit, valid = [], []
    for v in values:
        if isinstance(v, str):
            hit.append(v == key)
            valid.append(True)
        else:
            hit.append(False)
            valid.append(False)
    return np.array(hit, dtype=bool), np.array(valid, dtype=bool)


# ---------- parity vs Python reference (seed 42) ----------

@pytest.mark.fast
def test_parity_affix_seed42(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = _rand_texts(rng, 5000)
    for pre in ("a", "CITY", "", "日本", "😀", "MiXeD"):
        bufs = _bufs(a, [a["ir_text_startswith"]("S", np.asarray(vals, dtype=object), pre)])
        exp_hit, exp_valid = oracle_sw(vals, pre)
        assert np.array_equal(np.asarray(bufs["S"], dtype=bool), exp_hit), pre
        assert np.array_equal(np.asarray(bufs["S#validity"], dtype=bool), exp_valid), pre
    for suf in ("o", "9", "", "テスト", "😀", "eSe"):
        bufs = _bufs(a, [a["ir_text_endswith"]("E", np.asarray(vals, dtype=object), suf)])
        exp_hit, exp_valid = oracle_ew(vals, suf)
        assert np.array_equal(np.asarray(bufs["E"], dtype=bool), exp_hit), suf
        assert np.array_equal(np.asarray(bufs["E#validity"], dtype=bool), exp_valid), suf
    for key in ("a", "", "hello", "Hello", "日本語テスト", "a😀b"):
        bufs = _bufs(a, [a["ir_text_equals"]("Q", np.asarray(vals, dtype=object), key)])
        exp_hit, exp_valid = oracle_eq(vals, key)
        assert np.array_equal(np.asarray(bufs["Q"], dtype=bool), exp_hit), key
        assert np.array_equal(np.asarray(bufs["Q#validity"], dtype=bool), exp_valid), key


@pytest.mark.fast
def test_native_vs_fallback_parity(kernel):
    """Native first == numpy fallback: disable backend, rerun, bit-exact."""
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = np.asarray(_rand_texts(rng, 2000), dtype=object)
    refs = [
        _bufs(a, [a["ir_text_startswith"]("S", vals, "a")]),
        _bufs(a, [a["ir_text_endswith"]("E", vals, "o")]),
        _bufs(a, [a["ir_text_equals"]("Q", vals, "hello")]),
    ]
    old = os.environ.get("NUMFAST_NATIVE_DISABLE")
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    try:
        fbs = [
            _bufs(a, [a["ir_text_startswith"]("S", vals, "a")]),
            _bufs(a, [a["ir_text_endswith"]("E", vals, "o")]),
            _bufs(a, [a["ir_text_equals"]("Q", vals, "hello")]),
        ]
    finally:
        if old is None:
            del os.environ["NUMFAST_NATIVE_DISABLE"]
        else:
            os.environ["NUMFAST_NATIVE_DISABLE"] = old
    for r, f, k in zip(refs, fbs, ("S", "E", "Q")):
        assert np.array_equal(np.asarray(r[k], dtype=bool), np.asarray(f[k], dtype=bool)), k
        assert np.array_equal(np.asarray(r[k + "#validity"], dtype=bool),
                              np.asarray(f[k + "#validity"], dtype=bool)), k


# ---------- edge ----------

@pytest.mark.fast
def test_edge_empty_nullable_unicode(kernel):
    a = kernel.alias
    b = _bufs(a, [a["ir_text_startswith"]("S", [], "a")])
    assert b["S"].size == 0
    b = _bufs(a, [a["ir_text_endswith"]("E", [], "a")])
    assert b["E"].size == 0
    b = _bufs(a, [a["ir_text_equals"]("Q", [], "a")])
    assert b["Q"].size == 0
    vals = [None, None]
    for op, kw in (("ir_text_startswith", "a"), ("ir_text_endswith", "a"), ("ir_text_equals", "")):
        b = _bufs(a, [a[op]("X", vals, kw)])
        assert not np.asarray(b["X"], dtype=bool).any()
        assert not np.asarray(b["X#validity"], dtype=bool).any()
    # empty affix matches every VALID row, never invalid
    b = _bufs(a, [a["ir_text_startswith"]("S", ["x", None, ""], "")])
    assert list(np.asarray(b["S"], dtype=bool)) == [True, False, True]
    b = _bufs(a, [a["ir_text_endswith"]("E", ["x", None, ""], "")])
    assert list(np.asarray(b["E"], dtype=bool)) == [True, False, True]
    # equals: empty key matches only empty valid rows
    b = _bufs(a, [a["ir_text_equals"]("Q", ["", "x", None], "")])
    assert list(np.asarray(b["Q"], dtype=bool)) == [True, False, False]
    # anchored + case-sensitive + unicode
    b = _bufs(a, [a["ir_text_startswith"]("S", ["Hello", "hello", "HELLO"], "He")])
    assert list(np.asarray(b["S"], dtype=bool)) == [True, False, False]
    b = _bufs(a, [a["ir_text_endswith"]("E", ["日本語テスト", "テスト", "日本語"], "テスト")])
    assert list(np.asarray(b["E"], dtype=bool)) == [True, True, False]
    b = _bufs(a, [a["ir_text_equals"]("Q", ["a😀b", "a😀b", "a😀c"], "a😀b")])
    assert list(np.asarray(b["Q"], dtype=bool)) == [True, True, False]
    # non-str needle -> explicit error, never silent
    with pytest.raises(ValueError):
        _bufs(a, [a["ir_text_startswith"]("S", ["a"], 123)])
    with pytest.raises(ValueError):
        _bufs(a, [a["ir_text_equals"]("Q", ["a", 123, None], "a")])


# ---------- benchmark (kernel vs prep share, honest) ----------

@pytest.mark.fast
def test_benchmark_kernel_e2e(kernel):
    import sys
    sys.path.insert(0, str(Path(APP_DIR) / "src" / "Drivers" / "CPU" / "_lib"))
    import native_cpu as nc

    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = np.asarray(_rand_texts(rng, 200_000), dtype=object)
    U = [v if isinstance(v, str) else "" for v in vals]
    # prep share: python encode (data+offs) outside kernel timing
    s = time.perf_counter()
    data, offs = nc.build_text_buffers(U)
    t_prep = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    got = nc.text_startswith_buffers(data, offs, "a")
    t_kern = (time.perf_counter() - s) * 1000
    exp, _ = oracle_sw(vals, "a")
    assert np.array_equal(got.astype(bool), exp)
    # e2e via IR (includes validity + dispatch)
    s = time.perf_counter()
    b = _bufs(a, [a["ir_text_startswith"]("S", vals, "a")])
    t_e2e = (time.perf_counter() - s) * 1000
    assert np.array_equal(np.asarray(b["S"], dtype=bool), exp)
    print(f"\nbench N=200k startswith: prep={t_prep:.1f}ms kernel={t_kern:.1f}ms "
          f"e2e={t_e2e:.1f}ms prep_share={100*t_prep/max(t_prep+t_kern,1e-9):.1f}%")
    s = time.perf_counter()
    got = nc.text_equals_buffers(data, offs, "hello")
    t_kern2 = (time.perf_counter() - s) * 1000
    exp2, _ = oracle_eq(vals, "hello")
    assert np.array_equal(got.astype(bool), exp2)
    print(f"bench N=200k equals: kernel={t_kern2:.1f}ms (same prep={t_prep:.1f}ms)")


# ---------- WASM parity (seed 42, via Node) ----------

@pytest.mark.fast
def test_wasm_parity_seed42():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH")
    tools = Path(APP_DIR) / "numfast-native" / "tools"
    wasm = Path(APP_DIR) / "numfast-native" / "target" / "wasm32-unknown-unknown" \
        / "release" / "numfast_native.wasm"
    if not wasm.exists():
        pytest.skip("wasm artifact missing")
    rng = np.random.default_rng(SEED)
    strs = [s if isinstance(s, str) else "" for s in _rand_texts(rng, 2000)]
    data = "".join(strs).encode("utf-8")
    offs = np.zeros(len(strs) + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(s.encode("utf-8")) for s in strs]).astype(np.int64)
    cases = [("startswith", "a"), ("endswith", "o"), ("equals", "hello"),
             ("startswith", ""), ("equals", "")]
    with tempfile.TemporaryDirectory() as td:
        dp, op = os.path.join(td, "d.bin"), os.path.join(td, "o.i32")
        open(dp, "wb").write(bytes(data))
        offs.tofile(op)
        for mode, needle in cases:
            r = subprocess.run([node, str(tools / "wasm_text.mjs"), str(wasm),
                                dp, op, str(len(strs)), f"{mode}-parity", needle, td],
                               capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stderr
            got = np.fromfile(os.path.join(td, "hits.u8"), dtype=np.uint8)
            arr = np.asarray(strs, dtype=str)
            if mode == "startswith":
                exp = np.char.startswith(arr, needle).astype(np.uint8)
            elif mode == "endswith":
                exp = np.char.endswith(arr, needle).astype(np.uint8)
            else:
                exp = (arr == needle).astype(np.uint8)
            assert np.array_equal(got, exp), (mode, needle)
