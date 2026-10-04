# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Interleaved A/B for two native DLLs across all 6 kernels (seed 42).

Usage: python tools/ab_compare.py <dll_A> <dll_B> [--n N] [--reps R]
Builds no code; loads both via ctypes, interleaves A/B reps per kernel,
reports median ms + B/A ratio. Integrity (exact) checked first; any
mismatch = STOP. New-file rule: no existing bench touched.
"""
# SCOPE, STATED: an A/B TIMING harness for two native DLLs. It exercises no
# WASM artefact at all, so it establishes no parity and reports no
# correctness claim beyond the exactness check it runs first. It is not a
# parity script and must not be counted as one.
import ctypes
import sys
import time

import numpy as np

rng = np.random.default_rng(42)


def load(dll):
    d = ctypes.CDLL(dll)
    f = {}
    f["dense"] = d.nf_group_sum_count
    f["dense"].argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
    f["dense"].restype = ctypes.c_int32
    f["multi"] = d.nf_group_multi_sum_count
    f["multi"].argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
    f["multi"].restype = ctypes.c_int32
    f["pack"] = d.nf_pack_i32_direct
    f["pack"].argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
    f["pack"].restype = ctypes.c_int32
    f["pat"] = d.nf_pattern_encode
    f["pat"].argtypes = ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
                         + [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_void_p] * 2
                         + [ctypes.c_void_p] * 2)
    f["pat"].restype = ctypes.c_int32
    f["s64"] = d.nf_sorted_run_i64
    f["s64"].argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
    f["s64"].restype = ctypes.c_int64
    f["c64"] = d.nf_carry_build_i64
    f["c64"].argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] + [ctypes.c_void_p] * 3
    f["c64"].restype = ctypes.c_int64
    return f


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def bench_call(fn, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return med(ts), ts


def main():
    dll_a, dll_b = sys.argv[1], sys.argv[2]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1_000_000
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 15
    g = 256
    A, B = load(dll_a), load(dll_b)

    keys = rng.integers(0, g, size=n).astype(np.int32)
    vals = rng.standard_normal(n)
    s = np.zeros(g)
    c = np.zeros(g, dtype=np.int64)

    def mk_dense(F):
        def call():
            rc = F["dense"](keys.ctypes.data, vals.ctypes.data, n, s.ctypes.data, c.ctypes.data, g)
            assert rc == 0, rc
        return call

    # integrity dense
    s0 = np.zeros(g)
    c0 = np.zeros(g, dtype=np.int64)
    mk_dense(A)()
    ref_s, ref_c = s.copy(), c.copy()
    mk_dense(B)()
    assert np.max(np.abs(s - ref_s)) == 0.0 and np.sum(c != ref_c) == 0, "DENSE INTEGRITY FAIL"

    ncols = 3
    V = np.ascontiguousarray(rng.standard_normal((ncols, n)))
    S = np.zeros(ncols * g)
    C = np.zeros(g, dtype=np.int64)

    def mk_multi(F):
        def call():
            rc = F["multi"](keys.ctypes.data, V.ctypes.data, n, ncols, S.ctypes.data, C.ctypes.data, g)
            assert rc == 0, rc
        return call

    mk_multi(A)()
    ref_S, ref_C = S.copy(), C.copy()
    mk_multi(B)()
    assert np.max(np.abs(S - ref_S)) == 0.0 and np.sum(C != ref_C) == 0, "MULTI INTEGRITY FAIL"

    k1 = rng.integers(0, 100, size=n).astype(np.int32)
    k2 = rng.integers(0, 1000, size=n).astype(np.int32)
    out = np.zeros(n, dtype=np.int32)

    def mk_pack(F):
        def call():
            rc = F["pack"](k1.ctypes.data, k2.ctypes.data, 1000, n, out.ctypes.data)
            assert rc == 0, rc
        return call

    mk_pack(A)()
    ref_o = out.copy()
    mk_pack(B)()
    assert np.sum(out != ref_o) == 0, "PACK INTEGRITY FAIL"

    pn = 500_000
    widths = rng.integers(1, 8, size=pn)
    mags = np.array([rng.integers(0, 10 ** int(w)) for w in widths])
    strs = np.array(["P%0*d" % (int(w), int(m)) for w, m in zip(widths, mags)], dtype="S12")
    raw = b"".join(bytes(x).rstrip(b"\x00") for x in strs)
    data = np.frombuffer(raw, dtype=np.uint8).copy()
    offs = np.zeros(pn + 1, dtype=np.int32)
    pos = 0
    for i, x in enumerate(strs):
        L = len(bytes(x).rstrip(b"\x00"))
        pos += L
        offs[i + 1] = pos
    codes = np.zeros(pn, dtype=np.int32)
    valid = np.zeros(pn, dtype=np.uint8)
    w_out = np.zeros(1, dtype=np.int32)
    e_out = np.zeros(1, dtype=np.int32)
    pfx = np.frombuffer(b"P", dtype=np.uint8)

    def mk_pat(F):
        def call():
            rc = F["pat"](data.ctypes.data, len(data), offs.ctypes.data, pn,
                           pfx.ctypes.data, 1, codes.ctypes.data, valid.ctypes.data,
                           w_out.ctypes.data, e_out.ctypes.data)
            assert rc == 0, rc
        return call

    mk_pat(A)()
    ref_codes, ref_valid = codes.copy(), valid.copy()
    mk_pat(B)()
    assert np.sum(codes != ref_codes) == 0 and np.sum(valid != ref_valid) == 0, "PATTERN INTEGRITY FAIL"

    n2 = (n // g) * g
    sk = np.repeat(np.arange(g, dtype=np.int32), n2 // g)
    sv = rng.standard_normal(n2)
    uk = np.zeros(n2, dtype=np.int64)
    ss = np.zeros(n2, dtype=np.int64)
    cc = np.zeros(n2, dtype=np.int64)
    svi = (sv * 100).astype(np.int64)

    def mk_srt(F):
        def call():
            ng = F["s64"](sk.ctypes.data, svi.ctypes.data, n2, uk.ctypes.data, ss.ctypes.data, cc.ctypes.data)
            assert ng == g, ng
        return call

    mk_srt(A)()
    ref_uk, ref_ss, ref_cc = uk.copy(), ss.copy(), cc.copy()
    mk_srt(B)()
    assert np.sum(uk != ref_uk) == 0 and np.sum(ss != ref_ss) == 0 and np.sum(cc != ref_cc) == 0, "SORTED INTEGRITY FAIL"

    m = 1_000_000
    cm = (rng.random(m) < 0.1).astype(np.int64) * rng.integers(1, 5, size=m).astype(np.int64)
    sm = (rng.standard_normal(m) * 100).astype(np.int64) * np.sign(cm).astype(np.int64)
    ou, oc, os_ = np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)

    def mk_cry(F):
        def call():
            ng = F["c64"](cm.ctypes.data, sm.ctypes.data, m, ou.ctypes.data, oc.ctypes.data, os_.ctypes.data)
            assert ng > 0, ng
            return ng
        return call

    ng_a = mk_cry(A)()
    ref = (ou.copy(), oc.copy(), os_.copy())
    ng_b = mk_cry(B)()
    assert ng_a == ng_b and np.sum(ou != ref[0]) == 0, "CARRY INTEGRITY FAIL"

    print("integrity: ALL EXACT (dense/multi/pack/pattern/sorted/carry)")
    cases = [("dense1M", mk_dense), ("multi3c", mk_multi), ("pack1M", mk_pack),
             ("pat500k", mk_pat), ("sorted1M", mk_srt), ("carry1M", mk_cry)]
    print("%-9s %10s %10s %8s %8s %8s" % ("kernel", "A_med", "B_med", "B/Amd", "B/Amin", " determin"))
    verdict = {}
    for name, mk in cases:
        fa, fb = mk(A), mk(B)
        # interleave at rep level for fairness
        ta, tb = [], []
        for _ in range(2):
            fa()
            fb()
        for r in range(reps):
            order = ((fa, ta), (fb, tb)) if r % 2 == 0 else ((fb, tb), (fa, ta))
            for fn, acc in order:
                t = time.perf_counter()
                fn()
                acc.append((time.perf_counter() - t) * 1e3)
        ma, mb = med(ta), med(tb)
        ra = min(ta) / ma if ma else float("nan")
        print("%-9s %10.3f %10.3f %8.3f %8.3f %8.3f" % (name, ma, mb, mb / ma if ma else float("nan"),
              min(tb) / min(ta) if min(ta) else float("nan"), ra))
        verdict[name] = mb / ma if ma else float("nan")


main()
