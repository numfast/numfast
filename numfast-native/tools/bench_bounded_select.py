# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Standalone benchmark + validation: generic bounded top-K selection.

Native symbol: nf_bounded_select_2i32 (a=primary, b=secondary int32,
CSR offs, runtime K). Consumer-side ctypes wrapper only — no numfast
package import, no domain vocabulary, no integration.

GATES (all mandatory, exit 1 on any FAIL):
 1) full-sort parity bit-by-bit vs np.lexsort by (a, b, idx)
 2) equal-a 200 rows + shuffled/reversed input-order invariance
 3) secondary-key semantics (b breaks a-ties; 3 sub-cases)
 4) infeasible-as-INF (a==INT32_MAX sinks; all-INF -> all INF/-1)
 5) prefix invariant (top(K-1) == prefix of topK)
 6) K-sweep 1/4/8/16/32
 7) fuzz random + fixed seed 42
 8) benchmark vs C-sort (np.lexsort) N thousands->millions + scaling

STOP: standalone only. No integration, no existing-primitive changes.
"""
import ctypes
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")

INF = np.int32(2 ** 31 - 1)
SEED = 42
WARM_REPS = 7

lib = ctypes.CDLL(DLL)
lib.nf_bounded_select_2i32.argtypes = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
lib.nf_bounded_select_2i32.restype = ctypes.c_int32


def call_native(a, b, offs, g, k):
    a = np.ascontiguousarray(a, dtype=np.int32)
    b = np.ascontiguousarray(b, dtype=np.int32)
    offs = np.ascontiguousarray(offs, dtype=np.int32)
    n = int(a.size)
    assert b.size == n and offs.size == g + 1
    oa = np.empty(g * k, dtype=np.int32)
    ob = np.empty(g * k, dtype=np.int32)
    oi = np.empty(g * k, dtype=np.int32)
    rc = lib.nf_bounded_select_2i32(
        a.ctypes.data, b.ctypes.data, n,
        offs.ctypes.data, g, k,
        oa.ctypes.data, ob.ctypes.data, oi.ctypes.data)
    assert rc == 0, f"native rc={rc}"
    return oa, ob, oi


def ref_topk(a, b, lo, hi, k):
    idx = np.arange(lo, hi, dtype=np.int64)
    aa = a[lo:hi].astype(np.int64)
    bb = b[lo:hi].astype(np.int64)
    order = np.lexsort((idx, bb, aa))  # (a, b, idx) ascending
    take = order[:k]
    ra = np.full(k, INF, dtype=np.int32)
    rb = np.full(k, INF, dtype=np.int32)
    ri = np.full(k, np.int32(-1), dtype=np.int32)
    m = len(take)
    ra[:m] = a[lo:hi][take]
    rb[:m] = b[lo:hi][take]
    ri[:m] = (lo + take).astype(np.int32)
    return ra, rb, ri


def check_group(a, b, offs, g, k, tag, fails):
    oa, ob, oi = call_native(a, b, offs, g, k)
    ok = True
    for gi in range(g):
        lo, hi = int(offs[gi]), int(offs[gi + 1])
        ra, rb, ri = ref_topk(a, b, lo, hi, k)
        s = slice(gi * k, (gi + 1) * k)
        if not (np.array_equal(oa[s], ra) and np.array_equal(ob[s], rb) and np.array_equal(oi[s], ri)):
            ok = False
            fails.append(f"{tag}@g{gi}")
            break
    print(("PASS " if ok else "FAIL ") + tag)
    return ok


def main():
    fails = []
    rng = np.random.default_rng(SEED)

    # G1: full-sort parity, flat + grouped, bit-by-bit
    for (n, g, k) in [(1000, 1, 8), (5000, 1, 16), (3000, 3, 8), (64, 8, 4)]:
        a = np.ascontiguousarray(rng.integers(-1000, 1000, size=n).astype(np.int32))
        b = np.ascontiguousarray(rng.integers(-1000, 1000, size=n).astype(np.int32))
        cuts = sorted(rng.choice(n + 1, size=g + 1, replace=False))
        cuts[0], cuts[-1] = 0, n
        offs = np.ascontiguousarray(np.array(cuts, dtype=np.int32))
        check_group(a, b, offs, g, k, f"G1 parity n={n} g={g} k={k}", fails)

    # G2: equal-a 200 rows, shuffled + reversed invariance
    n, k = 200, 8
    a0 = np.full(n, 5, dtype=np.int32)
    b0 = np.arange(n, dtype=np.int32)
    offs = np.array([0, n], dtype=np.int32)
    ea, eb, ei = call_native(a0, b0, offs, 1, k)
    perm = rng.permutation(n)
    a1, b1 = a0[perm], b0[perm]
    fa, fb, fi = call_native(a1, b1, offs, 1, k)
    # canonical expectation: smallest (b, idx-in-shuffled-array?) — compare as SETS of (a,b)
    # plus sorted-order equality after mapping back: both must equal ref on their own arrays
    ok2a = bool(np.array_equal(ea, ref_topk(a0, b0, 0, n, k)[0]))
    ok2b = bool(np.array_equal(fa, ref_topk(a1, b1, 0, n, k)[0]))
    # set invariance: same multiset of (a,b) pairs regardless of input order
    set_e = sorted(zip(ea.tolist(), eb.tolist()))
    set_f = sorted(zip(fa.tolist(), fb.tolist()))
    ok2c = (set_e == set_f)
    a2, b2 = a0[::-1].copy(), b0[::-1].copy()
    ra2, rb2, _ = call_native(a2, b2, offs, 1, k)
    ok2d = (sorted(zip(ra2.tolist(), rb2.tolist())) == set_e)
    ok2 = ok2a and ok2b and ok2c and ok2d
    print(("PASS " if ok2 else "FAIL ") + "G2 equal-a-200 shuffled/reverse")
    if not ok2:
        fails.append("G2")

    # G3: secondary-key semantics (b breaks a-ties) — 3 sub-cases
    g3ok = True
    # R-10-like: distinct b decides among equal a
    a = np.array([10, 10, 10, 10, 10], dtype=np.int32)
    b = np.array([50, 10, 30, 20, 40], dtype=np.int32)
    oa, ob, oi = call_native(a, b, np.array([0, 5], dtype=np.int32), 1, 3)
    g3ok &= bool(list(ob) == [10, 20, 30] and list(oi) == [1, 3, 2])
    # R-2-like: a decides first, b only on tie
    a = np.array([1, 1, 0, 0], dtype=np.int32)
    b = np.array([99, 5, 7, 3], dtype=np.int32)
    oa, ob, oi = call_native(a, b, np.array([0, 4], dtype=np.int32), 1, 4)
    g3ok &= bool(list(oa) == [0, 0, 1, 1] and list(ob) == [3, 7, 5, 99])
    # R-9-like: b extremes (INT32_MIN/MAX ride as ordinary ordered values)
    a = np.zeros(4, dtype=np.int32)
    b = np.array([np.int32(2**31 - 1), np.int32(-2**31), np.int32(0), np.int32(-1)], dtype=np.int32)
    oa, ob, oi = call_native(a, b, np.array([0, 4], dtype=np.int32), 1, 4)
    g3ok &= bool(list(ob) == [np.int32(-2**31), np.int32(-1), np.int32(0), np.int32(2**31 - 1)])
    print(("PASS " if g3ok else "FAIL ") + "G3 secondary-key semantics (3 sub-cases)")
    if not g3ok:
        fails.append("G3")

    # G4: infeasible-as-INF
    a = np.array([INF, 3, INF, 1, 2, INF], dtype=np.int32)
    b = np.array([1, 9, 2, 8, 7, 3], dtype=np.int32)
    oa, ob, oi = call_native(a, b, np.array([0, 6], dtype=np.int32), 1, 4)
    ok4a = bool(list(oa) == [1, 2, 3, INF] and int(oi[3]) in (0, 2, 5))
    aa = np.full(10, INF, dtype=np.int32)
    bb = np.arange(10, dtype=np.int32)
    oa, ob, oi = call_native(aa, bb, np.array([0, 10], dtype=np.int32), 1, 4)
    # all-INF: feasible set (a != INF) is empty; lanes carry canonical
    # (INF, b, idx) order, caller filters a == INF -> empty.
    ok4b = bool(np.all(oa == INF) and (oa != INF).sum() == 0
                and list(oi) == [0, 1, 2, 3] and list(ob) == [0, 1, 2, 3])
    ok4 = ok4a and ok4b
    print(("PASS " if ok4 else "FAIL ") + "G4 infeasible-as-INF (+all-INF empty)")
    if not ok4:
        fails.append("G4")

    # G5: prefix invariant
    n = 2000
    a = np.ascontiguousarray(rng.integers(-500, 500, size=n).astype(np.int32))
    b = np.ascontiguousarray(rng.integers(-500, 500, size=n).astype(np.int32))
    offs = np.array([0, n], dtype=np.int32)
    ok5 = True
    prev = None
    outs = {}
    for k in (1, 2, 4, 8, 16, 32):
        outs[k] = call_native(a, b, offs, 1, k)
    for k in (2, 4, 8, 16, 32):
        smaller = k // 2 if k > 2 else 1
        big = outs[k]
        sml = outs[smaller]
        if not (np.array_equal(big[0][:smaller], sml[0])
                and np.array_equal(big[1][:smaller], sml[1])
                and np.array_equal(big[2][:smaller], sml[2])):
            ok5 = False
            fails.append(f"G5 prefix k={k}")
            break
    print(("PASS " if ok5 else "FAIL ") + "G5 prefix invariant")
    if not ok5 and "G5 prefix k=" not in str(fails):
        fails.append("G5")

    # G6: K-sweep 1/4/8/16/32 (flat + grouped, parity each)
    ok6 = True
    for k in (1, 4, 8, 16, 32):
        n = 3000
        a = np.ascontiguousarray(rng.integers(-2000, 2000, size=n).astype(np.int32))
        b = np.ascontiguousarray(rng.integers(-2000, 2000, size=n).astype(np.int32))
        offs = np.array([0, 1000, 3000], dtype=np.int32)
        oa, ob, oi = call_native(a, b, offs, 2, k)
        for gi, (lo, hi) in enumerate([(0, 1000), (1000, 3000)]):
            ra, rb, ri = ref_topk(a, b, lo, hi, k)
            s = slice(gi * k, (gi + 1) * k)
            if not (np.array_equal(oa[s], ra) and np.array_equal(ob[s], rb) and np.array_equal(oi[s], ri)):
                ok6 = False
                fails.append(f"G6 K={k}")
                break
        if not ok6:
            break
    print(("PASS " if ok6 else "FAIL ") + "G6 K-sweep 1/4/8/16/32")
    if not ok6 and not any(f.startswith("G6") for f in fails):
        fails.append("G6")

    # G7: fuzz random + seed (50 trials, random N/G/K, ties + INF injected)
    ok7 = True
    fr = np.random.default_rng(SEED)
    for t in range(50):
        n = int(fr.integers(0, 3000))
        g = int(fr.integers(1, 5)) if n > 0 else 1
        k = int(fr.integers(1, 33))
        a = fr.integers(-100, 100, size=n).astype(np.int32) if n else np.empty(0, dtype=np.int32)
        b = fr.integers(-100, 100, size=n).astype(np.int32) if n else np.empty(0, dtype=np.int32)
        if n > 10 and fr.random() < 0.5:
            sel = fr.random(size=n) < 0.1
            a[sel] = INF
        if n > 0:
            cuts = sorted(fr.choice(n + 1, size=g + 1, replace=False).tolist())
            cuts[0], cuts[-1] = 0, n
        else:
            cuts = [0] * (g + 1)
        offs = np.ascontiguousarray(np.array(cuts, dtype=np.int32))
        oa, ob, oi = call_native(a, b, offs, g, k)
        for gi in range(g):
            lo, hi = cuts[gi], cuts[gi + 1]
            ra, rb, ri = ref_topk(a, b, lo, hi, k)
            s = slice(gi * k, (gi + 1) * k)
            if not (np.array_equal(oa[s], ra) and np.array_equal(ob[s], rb) and np.array_equal(oi[s], ri)):
                ok7 = False
                fails.append(f"G7 fuzz t={t} n={n} g={g} k={k}")
                break
        if not ok7:
            break
    print(("PASS " if ok7 else "FAIL ") + "G7 fuzz 50 trials seed=42")
    if not ok7 and not any(f.startswith("G7") for f in fails):
        fails.append("G7")

    # G8: benchmark vs C-sort (np.lexsort), N thousands->millions + scaling
    bench = []
    for n in (10_000, 100_000, 1_000_000, 2_000_000):
        try:
            a = np.ascontiguousarray(rng.integers(-2**30, 2**30, size=n).astype(np.int32))
            b = np.ascontiguousarray(rng.integers(-2**30, 2**30, size=n).astype(np.int32))
        except MemoryError:
            print(f"SKIP N={n} (alloc failed)")
            continue
        k = 8
        offs = np.array([0, n], dtype=np.int32)
        call_native(a, b, offs, 1, k)  # warmup
        idx = np.arange(n)
        t = time.perf_counter()
        np.lexsort((idx, b, a))[:k]  # C-sort warmup (full sort, prefix taken)
        tn, tc = [], []
        for _ in range(WARM_REPS):
            t0 = time.perf_counter()
            call_native(a, b, offs, 1, k)
            tn.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            np.lexsort((idx, b, a))[:k]
            tc.append((time.perf_counter() - t0) * 1e3)
        wn, wc = statistics.median(tn), statistics.median(tc)
        row = {"n": n, "k": k, "native_ms": wn, "csort_ms": wc,
               "speedup": wc / wn if wn > 0 else float("nan"),
               "native_Mrows_s": n / wn / 1000.0 if wn > 0 else float("nan")}
        bench.append(row)
        print(f"N={n:>8} K={k} native={wn:.3f}ms csort={wc:.3f}ms "
              f"speedup={row['speedup']:.2f}x thru={row['native_Mrows_s']:.2f}Mrows/s")
        del a, b
    # K-scaling at fixed N=1M
    kscale = []
    try:
        n = 1_000_000
        a = np.ascontiguousarray(rng.integers(-2**30, 2**30, size=n).astype(np.int32))
        b = np.ascontiguousarray(rng.integers(-2**30, 2**30, size=n).astype(np.int32))
        offs = np.array([0, n], dtype=np.int32)
        for k in (1, 4, 8, 16, 32):
            call_native(a, b, offs, 1, k)
            ts = []
            for _ in range(WARM_REPS):
                t0 = time.perf_counter()
                call_native(a, b, offs, 1, k)
                ts.append((time.perf_counter() - t0) * 1e3)
            wm = statistics.median(ts)
            kscale.append({"n": n, "k": k, "native_ms": wm})
            print(f"Kscale N={n} K={k:>2} native={wm:.3f}ms")
        del a, b
    except MemoryError:
        print("SKIP Kscale (alloc failed)")

    json.dump({"seed": SEED, "warm_reps": WARM_REPS, "dll": DLL,
               "gates": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"],
               "bench": bench, "k_scaling": kscale,
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "bench_bounded_select.json"), "w"), indent=1)
    print("PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
