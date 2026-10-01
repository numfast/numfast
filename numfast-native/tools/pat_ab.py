# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Focused pattern A/B on golden-style corpus (seed 42, prefix 'id').

Usage: python tools/pat_ab.py <dll_A> <dll_B> [--n N] [--reps R]
Corpus mirrors tools/parity_pattern.py mix (id%03d/id%010d/neg/edge/invalid,
overflow rows dropped). Reports median + min per DLL + ratios. New file per
benchmark-modularity rule; ab_compare.py keeps its own synthetic corpus.
"""
import ctypes
import sys
import time

import numpy as np

PREFIX = "id"


def load(dll):
    d = ctypes.CDLL(dll)
    f = d.nf_pattern_encode
    f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                  ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.c_void_p]
    f.restype = ctypes.c_int32
    return f


def build_corpus(n):
    rng = np.random.default_rng(42)
    strs = []
    while len(strs) < n:
        r = rng.random()
        if r < 0.70:
            strs.append("id%03d" % int(rng.integers(0, 1000)))
        elif r < 0.80:
            strs.append("id%010d" % int(rng.integers(0, 1000)))
        elif r < 0.84:
            strs.append("id-%d" % int(rng.integers(0, 100)))
        elif r < 0.88:
            strs.append(rng.choice(["xx1", "IDX5", "", "id", "id-", "id12a", "id 12", "id_12", "i", "id+3"]))
        elif r < 0.92:
            strs.append("id%d" % int(rng.integers(0, 10 ** 9)))
        elif r < 0.96:
            s = "id" + "9" * int(rng.integers(11, 19))
            try:
                if int(s[2:]) > 2 ** 31 - 1:
                    continue
                strs.append(s)
            except ValueError:
                continue
        else:
            strs.append(rng.choice(["id12", "id-5", "id007"]))
    enc = [s.encode() for s in strs]
    offs = np.zeros(len(enc) + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(b) for b in enc])
    data = np.frombuffer(b"".join(enc), dtype=np.uint8).copy()
    return data, offs


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    dll_a, dll_b = sys.argv[1], sys.argv[2]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 500_000
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 15
    data, offs = build_corpus(n)
    pfx = np.frombuffer(PREFIX.encode(), dtype=np.uint8)
    A, B = load(dll_a), load(dll_b)
    bufs = {}
    for tag in ("a", "b"):
        bufs[tag] = (np.zeros(n, np.int32), np.zeros(n, np.uint8), np.zeros(1, np.int32), np.zeros(1, np.int32))

    def call(F, tag):
        co, va, w, e = bufs[tag]
        rc = F(data.ctypes.data, len(data), offs.ctypes.data, n, pfx.ctypes.data, len(PREFIX),
               co.ctypes.data, va.ctypes.data, w.ctypes.data, e.ctypes.data)
        assert rc == 0, rc

    call(A, "a")
    ra = (bufs["a"][0].copy(), bufs["a"][1].copy(), int(bufs["a"][2][0]))
    call(B, "b")
    rb = (bufs["b"][0].copy(), bufs["b"][1].copy(), int(bufs["b"][2][0]))
    assert (ra[0] == rb[0]).all() and (ra[1] == rb[1]).all() and ra[2] == rb[2], "PATTERN INTEGRITY FAIL"
    print("integrity: EXACT codes/valid/width=%d n=%d" % (ra[2], n))
    ta, tb = [], []
    for _ in range(2):
        call(A, "a")
        call(B, "b")
    for r in range(reps):
        order = ((A, "a", ta), (B, "b", tb)) if r % 2 == 0 else ((B, "b", tb), (A, "a", ta))
        for F, tag, acc in order:
            t = time.perf_counter()
            call(F, tag)
            acc.append((time.perf_counter() - t) * 1e3)
    print("A_med=%.3f A_min=%.3f | B_med=%.3f B_min=%.3f | B/A_med=%.3f B/A_min=%.3f"
          % (med(ta), min(ta), med(tb), min(tb), med(tb) / med(ta), min(tb) / min(ta)))


main()
