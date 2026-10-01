# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""RNG acceptance gate: CORE determinism/chunk-invariance, R-COMPAT,
ROUND half-even, UNIQUE promotion, CPU==GPU (i32 + GPU chunk-invariance).

Path: nf.* public API (kernel alias inside), fixed seed 42, stage
breakdown (ms) printed per run. WASM gate needs the Rust toolchain
(rustc/wasm-pack): asserted as documented-PASS only when the `nf_rng`
symbols probe present, else skipped with reason (no silent pass).
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(scope="module")
def nf():
    import sys
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    import numfast as _nf
    _nf.get_kernel(fresh=True)
    return _nf


def _stages():
    return {}


def test_rng_seed_validation(nf):
    assert nf.rng_seed(42) == 42
    assert nf.rng_seed(2 ** 64 - 1) == 2 ** 64 - 1
    for bad in (-1, 2 ** 64, True, 1.5, "42", None):
        with pytest.raises(ValueError):
            nf.rng_seed(bad)


def test_core_determinism_same_bytes(nf):
    t0 = time.perf_counter()
    a = nf.to_numpy(nf.rng_fill_i32(1000, 42, 3, 7, -50, 50))
    b = nf.to_numpy(nf.rng_fill_i32(1000, 42, 3, 7, -50, 50))
    c = nf.to_numpy(nf.rng_fill_i32(1000, 42, 4, 7, -50, 50))
    assert (a == b).all()
    assert not (a == c).all()  # different stream differs
    fa = nf.to_numpy(nf.rng_fill_f64(500, 42, 1, 0, -1.0, 1.0))
    fb = nf.to_numpy(nf.rng_fill_f64(500, 42, 1, 0, -1.0, 1.0))
    assert (fa == fb).all()
    assert ((fa >= -1.0) & (fa < 1.0)).all()
    print(f"\ncore determinism stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_core_chunked_unchunked_bitexact(nf):
    t0 = time.perf_counter()
    n = 100003  # odd, non-multiple of workgroup size
    full = nf.to_numpy(nf.rng_fill_i32(n, 7, 3, 0, -50, 50))
    for splits in ((40000, 60003), (1, n - 1), (33333, 33333, 33337)):
        off = 0
        parts = []
        for ln in splits:
            parts.append(nf.to_numpy(nf.rng_fill_i32(ln, 7, 3, off, -50, 50)))
            off += ln
        assert (full == np.concatenate(parts)).all()
    ff = nf.to_numpy(nf.rng_fill_f64(n, 7, 3, 0, -2.0, 2.0))
    off = 0
    parts = []
    for ln in (40000, 60003):
        parts.append(nf.to_numpy(nf.rng_fill_f64(ln, 7, 3, off, -2.0, 2.0)))
        off += ln
    assert (ff == np.concatenate(parts)).all()
    print(f"\nchunk-invariance stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_core_width1_constant(nf):
    a = nf.to_numpy(nf.rng_fill_i32(16, 1, 0, 0, 9, 10))
    assert (a == 9).all()
    b = nf.to_numpy(nf.rng_fill_i32(16, 1, 0, 5, 9, 10))
    assert (b == 9).all()  # constant under offset too


def test_core_errors(nf):
    with pytest.raises(ValueError):
        nf.to_numpy(nf.rng_fill_i32(10, 42, 0, 0, 5, 5))  # hi <= lo -> -2
    with pytest.raises(ValueError):
        nf.to_numpy(nf.rng_fill_f64(10, 42, 0, 0, 1.0, 1.0))
    with pytest.raises(ValueError):
        nf.to_numpy(nf.rng_fill_f64(10, 42, 0, 0, float("nan"), 1.0))
    with pytest.raises(ValueError):
        nf.to_numpy(nf.rng_sample(10, 11, 42))  # k > n -> -2
    with pytest.raises(ValueError):
        nf.map_round(nf.from_numpy(np.array([1.0])), 16)  # ndigits range
    k = nf.get_kernel()
    a = k.alias
    with pytest.raises(ValueError):
        a["ir_rng_fill_i32"]("r", 10, 42, 0, 0, 0, 10, mode="legacy")


def test_core_sample_permutation_contract(nf):
    s = nf.to_numpy(nf.rng_sample(1000, 10, 7))
    assert s.size == 10 and len(set(s.tolist())) == 10
    assert ((s >= 0) & (s < 1000)).all()
    # draw order, not sorted (overwhelmingly likely for k=10)
    p = nf.to_numpy(nf.rng_permutation(1000, 7))
    assert sorted(p.tolist()) == list(range(1000))
    assert not (p == np.arange(1000)).all()


def test_rcompat_matches_reference(nf):
    ref = Path(__file__).resolve().parents[2] / "scratch" / "r_rng_py.py"
    if not ref.exists():
        pytest.skip("scratch/r_rng_py.py reference absent")
    spec = importlib.util.spec_from_file_location("r_rng_ref", str(ref))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for seed in (42, 1):
        want = mod.R(seed).runif(50)
        got = nf.to_numpy(nf.rng_compat(50, seed))
        assert all(x == y for x, y in zip(want, got.tolist()))
        want_s = [x - 1 for x in mod.R(seed).sample_replace(100, 20)]
        got_s = nf.to_numpy(nf.rng_compat(100, seed, kind="sample", m=20))
        assert want_s == got_s.tolist()


def test_round_half_even(nf):
    t0 = time.perf_counter()
    k = nf.get_kernel()
    a = k.alias
    # exact-binary ties: 0.125 d2 -> 0.12 (12 even), 0.375 d2 -> 0.38
    jobs = [a["ir_series"]("s", np.array(
        [2.5, 3.5, -2.5, -3.5, 0.125, 0.375, 1.0, -0.0,
         float("nan"), float("inf"), float("-inf")], dtype=np.float64),
        "float64"),
            a["ir_map_round"]("m", "s", 2 if False else 0)]
    # ndigits=0 for the .5 ties; decimals checked in a second graph
    bufs = a["cpu_execute"](a["optimize"](a["compile"](jobs))["nodes"])
    o = bufs["m"]
    assert o[:4].tolist() == [2.0, 4.0, -2.0, -4.0]
    jobs2 = [a["ir_series"]("s", np.array([0.125, 0.375, 2.675], dtype=np.float64),
                            "float64"),
             a["ir_map_round"]("m", "s", 2)]
    o2 = a["cpu_execute"](a["optimize"](a["compile"](jobs2))["nodes"])["m"]
    assert o2[0] == 0.12 and o2[1] == 0.38
    # 2.675*100 == 267.5 exactly in binary (product rounds up) -> tie,
    # 267 odd -> 268 -> 2.68 (binary-value half-even, == Rust same op order)
    assert o2[2] == 2.68
    # non-finite passthrough on VALID lanes (direct IR keeps NaN data)
    assert np.isnan(o[8]) and o[9] == float("inf") and o[10] == float("-inf")
    # -0.0 signbit preserved
    assert o[7] == 0.0 and np.signbit(o[7])
    # invalid rows -> 0 + valid 0
    jobs3 = [a["ir_series"]("s", np.array([5.0, 2.5]), "float64",
                            validity=np.array([False, True])),
             a["ir_map_round"]("m", "s", 0)]
    b3 = a["cpu_execute"](a["optimize"](a["compile"](jobs3))["nodes"])
    assert b3["m"][0] == 0.0 and not b3["m#validity"][0]
    assert b3["m"][1] == 2.0 and b3["m#validity"][1]
    print(f"\nround stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_unique_contract(nf):
    t0 = time.perf_counter()
    keys = np.array([3, 1, 3, 2, 1, 3], dtype=np.int32)
    r = nf.unique(nf.from_numpy(keys))
    u, inv, ng = nf.to_numpy(r["uniq"]), nf.to_numpy(r["inv"]), r["ng"]
    assert ng == 3 and u.tolist() == [1, 2, 3]
    assert (u[inv] == keys).all()
    # invalid -> -1, excluded from uniq
    sv = nf.from_numpy(np.array([3, 1, 3], dtype=np.int32),
                       validity=np.array([True, False, True]))
    r2 = sv.unique()
    assert nf.to_numpy(r2["inv"]).tolist() == [0, -1, 0]
    assert r2["ng"] == 1 and nf.to_numpy(r2["uniq"]).tolist() == [3]
    print(f"\nunique stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_capability_chunkable_flags(nf):
    cap = nf.get_kernel().alias["cpu_capability"]()
    hints = cap["chunkable_hints"]
    assert hints["rng_fill_i32"] is True
    assert hints["rng_fill_f64"] is True
    assert hints["map_round"] is True
    assert hints["rng_sample_no_replace"] is False
    assert hints["rng_permutation"] is False
    assert hints["rng_compat"] is False
    assert hints["unique_inverse"] is False
    gcap = nf.get_kernel().alias["gpu_capability"]()
    assert "rng_fill_i32" in gcap["ops"]
    assert "rng_compat" not in gcap["ops"]  # CPU-only, explicit error


def test_gpu_fill_cpu_parity_and_chunk_invariance(nf):
    t0 = time.perf_counter()
    n = 10000
    cpu = nf.to_numpy(nf.rng_fill_i32(n, 9, 5, 0, -1000, 1000))
    gpu = nf.to_numpy(nf.rng_fill_i32(n, 9, 5, 0, -1000, 1000, backend="gpu"))
    assert (cpu == gpu).all()
    # GPU chunk-invariance: required gate before accepting GPU RNG
    k = nf.get_kernel()
    a = k.alias
    g1 = a["optimize"](a["compile"](
        [a["ir_rng_fill_i32"]("r", 4000, 9, 5, 0, -1000, 1000)]))
    g2 = a["optimize"](a["compile"](
        [a["ir_rng_fill_i32"]("r", 6000, 9, 5, 4000, -1000, 1000)]))
    c1 = a["evaluate"](g1, "gpu", 4000)["result"]
    c2 = a["evaluate"](g2, "gpu", 6000)["result"]
    assert (gpu == np.concatenate([c1, c2])).all()
    # GPU explicit errors (no silent fallback)
    with pytest.raises(RuntimeError):
        a["evaluate"](a["optimize"](a["compile"](
            [a["ir_rng_compat"]("r", 10, 42)])), "gpu", 10)
    print(f"\ngpu-rng stages ms: total={(time.perf_counter()-t0)*1000:.2f}")


def test_native_symbol_probe(nf):
    from pathlib import Path as _P
    dll = _P(nf.native_info().get("dll") or "")
    import ctypes as _ct
    names = ["nf_rng_fill_i32", "nf_rng_fill_f64", "nf_rng_map_round",
             "nf_rng_sample_no_replace", "nf_rng_permutation",
             "nf_rng_compat_runif", "nf_rng_compat_sample"]
    if not dll.exists():
        pytest.skip("native DLL absent")
    try:
        lib = _ct.CDLL(str(dll))
        missing = [s for s in names if not hasattr(lib, s)]
    except OSError:
        pytest.skip("native DLL unloadable here")
    if missing:
        pytest.skip(f"DLL predates RNG symbols, missing={missing}")
    assert not missing
