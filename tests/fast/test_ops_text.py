# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: ir_text_length / ir_text_contains CPU (length+contains only).

Semantics (frozen): length = Unicode code points (NOT bytes);
contains = UTF-8 substring, case-sensitive, empty needle matches all
valid rows. None/non-str rows -> invalid (DELTA-3), never an error,
never a match. Oracle: independent pure-Python (no numfast imports
inside oracle fns). Seed 42 where RNG is used.
"""

import json
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


def oracle_length(values):
    """Independent oracle: ([lens int32], [valid bool]); len() = code points."""
    lens, valid = [], []
    for v in values:
        if isinstance(v, str):
            lens.append(len(v))
            valid.append(True)
        else:
            lens.append(0)
            valid.append(False)
    return np.array(lens, dtype=np.int32), np.array(valid, dtype=bool)


def oracle_contains(values, substr):
    """Independent oracle: ([hit bool], [valid bool]); `in` semantics."""
    hit, valid = [], []
    for v in values:
        if isinstance(v, str):
            hit.append(substr in v)
            valid.append(True)
        else:
            hit.append(False)
            valid.append(False)
    return np.array(hit, dtype=bool), np.array(valid, dtype=bool)


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


# ---------- parity vs Python reference (seed 42) ----------

@pytest.mark.fast
def test_parity_length_seed42(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = _rand_texts(rng, 5000)
    # object column keeps the non-str scalars (str-ness contract probe)
    bufs = _bufs(a, [a["ir_text_length"]("L", np.asarray(vals, dtype=object))])
    exp_lens, exp_valid = oracle_length(vals)
    assert np.array_equal(bufs["L"], exp_lens)
    assert np.array_equal(np.asarray(bufs["L#validity"], dtype=bool), exp_valid)


@pytest.mark.fast
def test_parity_contains_seed42(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = _rand_texts(rng, 5000)
    for sub in ("a", "日本", "😀", "", "CITY", "city", " "):
        bufs = _bufs(a, [a["ir_text_contains"]("H", np.asarray(vals, dtype=object), sub)])
        exp_hit, exp_valid = oracle_contains(vals, sub)
        assert np.array_equal(np.asarray(bufs["H"], dtype=bool), exp_hit), sub
        assert np.array_equal(np.asarray(bufs["H#validity"], dtype=bool), exp_valid), sub


@pytest.mark.fast
def test_native_vs_fallback_parity(kernel):
    """Native first == numpy fallback: disable backend, rerun, bit-exact."""
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = np.asarray(_rand_texts(rng, 2000), dtype=object)
    ref_l = _bufs(a, [a["ir_text_length"]("L", vals)])
    ref_h = _bufs(a, [a["ir_text_contains"]("H", vals, "a")])
    old = os.environ.get("NUMFAST_NATIVE_DISABLE")
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    try:
        fb_l = _bufs(a, [a["ir_text_length"]("L", vals)])
        fb_h = _bufs(a, [a["ir_text_contains"]("H", vals, "a")])
    finally:
        if old is None:
            del os.environ["NUMFAST_NATIVE_DISABLE"]
        else:
            os.environ["NUMFAST_NATIVE_DISABLE"] = old
    assert np.array_equal(ref_l["L"], fb_l["L"])
    assert np.array_equal(ref_h["H"], fb_h["H"])


@pytest.mark.fast
def test_dict_domain_parity(kernel):
    """Dict-TEXT row ops == domain ops over values[D] (codes path)."""
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = _rand_texts(rng, 1000)
    enc = a["dictionary_encode"]([v if isinstance(v, str) else None for v in vals])
    codes, vocab, validity = enc["codes"], enc["values"], enc["validity"]
    lut = np.asarray(a["dict_len_lut"](vocab), dtype=np.int32)
    row_lens = lut[np.asarray(codes, dtype=np.int32)]
    exp_lens, _ = oracle_length(vals)
    eff = np.ones(len(vals), dtype=bool) if validity is None \
        else np.asarray(validity, dtype=bool)
    assert np.array_equal(row_lens[eff], exp_lens[eff])
    allowed = np.asarray(a["dict_contains"](vocab, "a"), dtype=np.int32)
    exp_codes = np.array([i for i, s in enumerate(vocab) if "a" in s], dtype=np.int32)
    assert np.array_equal(allowed, exp_codes)


# ---------- edge ----------

@pytest.mark.fast
def test_edge_empty_nullable_unicode(kernel):
    a = kernel.alias
    # empty column
    b = _bufs(a, [a["ir_text_length"]("L", [])])
    assert b["L"].size == 0 and np.asarray(b["L#validity"], dtype=bool).size == 0
    b = _bufs(a, [a["ir_text_contains"]("H", [], "a")])
    assert b["H"].size == 0
    # all-None
    vals = [None, None]
    b = _bufs(a, [a["ir_text_length"]("L", vals)])
    assert list(b["L"]) == [0, 0] and not np.asarray(b["L#validity"], dtype=bool).any()
    b = _bufs(a, [a["ir_text_contains"]("H", vals, "")])
    assert not np.asarray(b["H"], dtype=bool).any()  # invalid never match, even ""
    # code points, NOT bytes: emoji/CJK/cafe == char count
    vals = ["a😀b", "日本語", "café", "", "🎉🎉"]
    b = _bufs(a, [a["ir_text_length"]("L", vals)])
    assert list(b["L"]) == [3, 3, 4, 0, 2]
    # case-sensitive + empty substr
    b = _bufs(a, [a["ir_text_contains"]("H", ["Hello", "hello", "HELLO"], "hello")])
    assert list(np.asarray(b["H"], dtype=bool)) == [False, True, False]
    b = _bufs(a, [a["ir_text_contains"]("H", ["x", None, ""], "")])
    assert list(np.asarray(b["H"], dtype=bool)) == [True, False, True]
    # non-str scalar -> explicit error, never silent (incl. mixed columns)
    with pytest.raises(ValueError):
        _bufs(a, [a["ir_text_length"]("L", [1, 2])])
    with pytest.raises(ValueError):
        _bufs(a, [a["ir_text_length"]("L", ["a", 123, None])])
    with pytest.raises(ValueError):
        _bufs(a, [a["ir_text_contains"]("H", ["a"], 123)])


# ---------- benchmark (stage breakdown, correctness first) ----------

@pytest.mark.fast
def test_benchmark_native_vs_fallback(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    vals = np.asarray(_rand_texts(rng, 200_000), dtype=object)
    s = time.perf_counter()
    b = _bufs(a, [a["ir_text_length"]("L", vals)])
    t_nat_l = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    _bufs(a, [a["ir_text_contains"]("H", vals, "a")])
    t_nat_c = (time.perf_counter() - s) * 1000
    exp_lens, _ = oracle_length(vals)
    assert np.array_equal(b["L"], exp_lens)  # correctness first
    old = os.environ.get("NUMFAST_NATIVE_DISABLE")
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    try:
        s = time.perf_counter()
        _bufs(a, [a["ir_text_length"]("L", vals)])
        t_fb_l = (time.perf_counter() - s) * 1000
        s = time.perf_counter()
        _bufs(a, [a["ir_text_contains"]("H", vals, "a")])
        t_fb_c = (time.perf_counter() - s) * 1000
    finally:
        if old is None:
            del os.environ["NUMFAST_NATIVE_DISABLE"]
        else:
            os.environ["NUMFAST_NATIVE_DISABLE"] = old
    print(f"\nbench N=200k length: native={t_nat_l:.1f}ms fallback={t_fb_l:.1f}ms "
          f"contains: native={t_nat_c:.1f}ms fallback={t_fb_c:.1f}ms")


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
    with tempfile.TemporaryDirectory() as td:
        dp, op = os.path.join(td, "d.bin"), os.path.join(td, "o.i32")
        open(dp, "wb").write(bytes(data))
        offs.tofile(op)
        r = subprocess.run([node, str(tools / "wasm_text.mjs"), str(wasm),
                            dp, op, str(len(strs)), "len-parity", td],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        got = np.fromfile(os.path.join(td, "lens.i32"), dtype=np.int32)
        assert np.array_equal(got, np.char.str_len(np.asarray(strs, dtype=str)).astype(np.int32))
        for sub in ("a", "日本", ""):
            r = subprocess.run([node, str(tools / "wasm_text.mjs"), str(wasm),
                                dp, op, str(len(strs)), "contains-parity", sub, td],
                               capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stderr
            got = np.fromfile(os.path.join(td, "hits.u8"), dtype=np.uint8)
            exp = np.ones(len(strs), dtype=np.uint8) if sub == "" else \
                (np.char.find(np.asarray(strs, dtype=str), sub) != -1).astype(np.uint8)
            assert np.array_equal(got, exp), sub
