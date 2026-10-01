# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P1/P2 research: single-pass mixed kernel + sum-only kernel, 100M/16T.

Methodology mirrors bench_opt_phase3_100M (verbatim boundary): NFS stream
blocks loaded ONCE (file IO outside all timers), per-block MT aggregate +
global dense merge, best-of-3, exact chk vs service refs. Bench-only file;
production modules untouched (experimental kernels live in the TEST dll,
loaded here by explicit path; production uses the working dll).

Variants (strictly sequential sections):
  BASE Q5/Q3 : production _native_mt (working dll, per-column passes)
  P1-kernel   : kernel-only ST on 1 block (per-col vs mixed vs sum-only)
  P1-e2e Q5   : single-pass mixed with counts (1 key pass, shared counts)
  P2-e2e Q5   : mixed sum-only + keys-only count pass (2 passes, no bump)
  P1-e2e Q3 / P2-e2e Q3 (same, (1,1) lane shape)
Plus: bounds/branch unit probes (BAD_RANGE/OVERFLOW), MT scaling (T=1/16).

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_p12_100M.py
"""
import ctypes
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

_CPU_LIB = str(FORK / "src" / "Drivers" / "CPU")
if _CPU_LIB not in sys.path:
    sys.path.insert(0, _CPU_LIB)
import _lib.cpu as _C  # noqa: E402 (production module, read-only here)

TEST_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
               / "release" / "numfast_native.dll")

_tlib = ctypes.CDLL(TEST_DLL)
# P1 (single-pass WITH counts) was measured then REJECTED and removed
# from production sources (marginal at 16T, regresses at low T; evidence
# in scratch/bench_p12_100M.json). P1 sections below run only when the
# loaded DLL still exports the P1 symbol (old test DLL); P2 sections
# always run against the working DLL.
try:
    _tlib.nf_group_mixed_i32_f64.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 3
        + [ctypes.c_size_t])
    _tlib.nf_group_mixed_i32_f64.restype = ctypes.c_int32
    _ = _tlib.nf_group_mixed_i32_f64
    _HAS_P1 = True
except (OSError, AttributeError, ValueError):
    _HAS_P1 = False
_tlib.nf_group_mixed_sum_only.argtypes = (
    [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 2
    + [ctypes.c_size_t])
_tlib.nf_group_mixed_sum_only.restype = ctypes.c_int32
_tlib.nf_group_sum_only_f64.argtypes = (
    [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t])
_tlib.nf_group_sum_only_f64.restype = ctypes.c_int32
_tlib.nf_group_count_only.argtypes = (
    [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t])
_tlib.nf_group_count_only.restype = ctypes.c_int32


def _split_cols(cols):
    """Partition into (i32_list, f64_list); None if any col not servable."""
    i32c, f64c = [], []
    for c in cols:
        c = np.ascontiguousarray(c)
        if c.dtype == np.dtype(np.int32):
            i32c.append(c)
        elif c.dtype.kind == "f" and c.dtype != np.dtype(np.float64):
            f64c.append(np.ascontiguousarray(c, dtype=np.float64))
        elif c.dtype == np.dtype(np.float64):
            f64c.append(c)
        else:
            return None  # int64/other -> proven path owns it (fallback)
    return i32c, f64c


def _x_mixed_1(keys, i32c, f64c, g):
    """ONE mixed kernel call (ST): ([sums...], counts) or (None, rc)."""
    if not _HAS_P1:
        return None, -99
    n = int(keys.size)
    ni, nf = len(i32c), len(f64c)
    # NOTE: stacked SoA buffers must stay referenced until the call
    # returns (ctypes.data is a bare address; temporaries would dangle).
    i32a = (np.ascontiguousarray(np.stack(i32c).reshape(-1)) if ni else None)
    f64a = (np.ascontiguousarray(np.stack(f64c).reshape(-1)) if nf else None)
    i32flat = i32a.ctypes.data if ni else None
    f64flat = f64a.ctypes.data if nf else None
    s32 = np.zeros(ni * g, dtype=np.int32)
    s64 = np.zeros(nf * g, dtype=np.float64)
    cc = np.zeros(g, dtype=np.int64)
    rc = _tlib.nf_group_mixed_i32_f64(
        keys.ctypes.data, i32flat, f64flat, n, ni, nf,
        s32.ctypes.data if ni else None, s64.ctypes.data if nf else None,
        cc.ctypes.data, g)
    if rc != 0:
        return None, rc
    outs = ([s32[i * g:(i + 1) * g] for i in range(ni)]
            + [s64[i * g:(i + 1) * g] for i in range(nf)])
    return outs, cc


def _x_sumonly_1(keys, i32c, f64c, g):
    """ONE sum-only kernel call (ST): ([sums...], rc). No counts lane."""
    n = int(keys.size)
    ni, nf = len(i32c), len(f64c)
    # NOTE: stacked SoA buffers must stay referenced until the call
    # returns (ctypes.data is a bare address; temporaries would dangle).
    i32a = (np.ascontiguousarray(np.stack(i32c).reshape(-1)) if ni else None)
    f64a = (np.ascontiguousarray(np.stack(f64c).reshape(-1)) if nf else None)
    i32flat = i32a.ctypes.data if ni else None
    f64flat = f64a.ctypes.data if nf else None
    s32 = np.zeros(ni * g, dtype=np.int32)
    s64 = np.zeros(nf * g, dtype=np.float64)
    rc = _tlib.nf_group_mixed_sum_only(
        keys.ctypes.data, i32flat, f64flat, n, ni, nf,
        s32.ctypes.data if ni else None, s64.ctypes.data if nf else None, g)
    if rc != 0:
        return None, rc
    outs = ([s32[i * g:(i + 1) * g] for i in range(ni)]
            + [s64[i * g:(i + 1) * g] for i in range(nf)])
    return outs, rc


def _x_mixed_flat(keys, i32a, f64a, ni, nf, g):
    """Mixed kernel on PRE-BUILT SoA flats (no assembly copy in timer)."""
    assert _HAS_P1, "P1 symbol absent (rejected/removed)"
    s32 = np.zeros(ni * g, dtype=np.int32)
    s64 = np.zeros(nf * g, dtype=np.float64)
    cc = np.zeros(g, dtype=np.int64)
    rc = _tlib.nf_group_mixed_i32_f64(
        keys.ctypes.data,
        i32a.ctypes.data if ni else None, f64a.ctypes.data if nf else None,
        int(keys.size), ni, nf,
        s32.ctypes.data if ni else None, s64.ctypes.data if nf else None,
        cc.ctypes.data, g)
    assert rc == 0, rc
    return s32, s64, cc


def _x_sumonly_flat(keys, i32a, f64a, ni, nf, g):
    """Sum-only kernel on PRE-BUILT SoA flats (no assembly copy in timer)."""
    s32 = np.zeros(ni * g, dtype=np.int32)
    s64 = np.zeros(nf * g, dtype=np.float64)
    rc = _tlib.nf_group_mixed_sum_only(
        keys.ctypes.data,
        i32a.ctypes.data if ni else None, f64a.ctypes.data if nf else None,
        int(keys.size), ni, nf,
        s32.ctypes.data if ni else None, s64.ctypes.data if nf else None, g)
    assert rc == 0, rc
    return s32, s64


def _x_count_1(keys, g):
    cc = np.zeros(g, dtype=np.int64)
    rc = _tlib.nf_group_count_only(keys.ctypes.data, keys.size,
                                   cc.ctypes.data, g)
    assert rc == 0, rc
    return cc


def _x_mt(keys, vcols, m, t, mode):
    """Host MT mirror of _native_mt with experimental inner (P1/P2).

    mode 'mixed' (P1): one mixed kernel per worker (counts once).
    mode 'sumonly' (P2): sum-only kernel + keys-only count per worker.
    Returns (ukeys, counts, [sums...], agg_ms, carry_ms, merge_ms) or None.
    """
    try:
        keys = np.ascontiguousarray(keys)
        cols = [np.ascontiguousarray(v) for v in vcols]
        sp = _split_cols(cols)
        if sp is None:
            return None
        i32c, f64c = sp
        # i32 gate per column (same predicate as production mixed path)
        n0 = int(keys.size)
        for c in i32c:
            mn, mx = int(c.min()), int(c.max())
            if n0 * max(abs(mn), abs(mx)) >= 2 ** 31 - 1:
                return None  # beyond i32 lanes -> proven path owns it
        n = int(keys.size)
        t = max(1, min(int(t), n))
        k32 = np.ascontiguousarray(keys, dtype=np.int32)
        ex = _C._pool(t)
        t0 = time.perf_counter()

        def _work(kk, a, b):
            ki = i32c and [c[a:b] for c in i32c]
            kf = f64c and [c[a:b] for c in f64c]
            if mode == "mixed":
                r = _x_mixed_1(kk, ki or [], kf or [], m)
                return r if r[0] is not None else None
            s = _x_sumonly_1(kk, ki or [], kf or [], m)
            if s[0] is None:
                return None
            return s[0], _x_count_1(kk, m)

        bounds = np.linspace(0, n, t + 1).astype(np.int64)
        futs = [ex.submit(_work, k32[int(bounds[w]):int(bounds[w + 1])],
                          int(bounds[w]), int(bounds[w + 1]))
                for w in range(t)]
        parts = [f.result() for f in futs]
        if any(p is None for p in parts):
            return None
        agg_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        ncols = len(parts[0][0])
        _mdt = [np.int64 if s.dtype == np.dtype(np.int32) else s.dtype
                for s in parts[0][0]]
        sums_m = [np.zeros(m, dtype=dt) for dt in _mdt]
        counts_m = np.zeros(m, dtype=np.int64)
        _ns = max(1, min(int(t), (int(m) // 65536) + 1))
        _mb = np.linspace(0, int(m), _ns + 1).astype(np.int64)

        def _mshard(dst, outs, s0, s1):
            sl = slice(int(s0), int(s1))
            if sl.start >= sl.stop:
                return
            x = dst[sl]
            for o in outs:
                x += o[sl]

        _mf = []
        for j in range(ncols):
            _outs = [ss[j] for ss, _ in parts]
            for w in range(_ns):
                _mf.append(ex.submit(_mshard, sums_m[j], _outs,
                                     _mb[w], _mb[w + 1]))
        _cout = [cc for _, cc in parts]
        for w in range(_ns):
            _mf.append(ex.submit(_mshard, counts_m, _cout,
                                 _mb[w], _mb[w + 1]))
        for _f in _mf:
            _f.result()
        merge_ms = (time.perf_counter() - t1) * 1000
    except Exception:  # noqa: BLE001
        return None
    try:
        ukeys, counts, out, cms = None, None, [], 0.0
        for s_m in sums_m:
            uk, c, s, cm = _C._native_compact(counts_m, s_m)
            if ukeys is None:
                ukeys, counts = uk, c
            cms += cm
            out.append(s)
        return ukeys, counts, out, agg_ms, cms, merge_ms
    except Exception:  # noqa: BLE001
        return None


def main():
    global _C
    t = max(1, min(64, int(os.environ.get("NUMFAST_THREADS", "1"))))
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    # Builder exec() purges _lib.* from sys.modules: the top-level _C is a
    # STALE pre-build instance (virgin CDLL handle -> racy MT first-touch).
    # Rebind to a FRESH post-build instance; warm it single-threaded below.
    import importlib as _il
    for _m in [m for m in sys.modules if m == "_lib" or m.startswith("_lib.")]:
        del sys.modules[_m]
    _C = _il.import_module("_lib.cpu")
    a = kernel.alias
    plan_groupby = a["plan_groupby"]
    nfs = str(FORK / "scratch" / "nfs_stream" / "G1_1e8.stream.nfs")
    h = a["nfs_stream_open"](nfs, force_lazy=True)
    idx = h["index"]
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    M3 = int(max(m["maxs"]["k3"] for m in idx)) + 1
    M5 = int(max(m["maxs"]["id6"] for m in idx)) + 1
    t0 = time.perf_counter()
    blocks = []
    with open(nfs, "rb") as fh:
        for m in idx:
            fh.seek(int(m["file_offset"]))
            raw = fh.read(int(m["blob_len"]))
            z = np.load(io.BytesIO(raw))
            blocks.append({k: np.asarray(z[k]) for k in
                           ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")})
            z.close()
            del raw
    print(f"resident rows={sum(len(b['k1']) for b in blocks)} "
          f"load={(time.perf_counter()-t0)*1000:.0f}ms T={t} M3={M3} M5={M5} "
          f"testdll={TEST_DLL}", flush=True)
    out = {"N": N, "T": t, "rows": {}}

    # ---- bounds/branch unit probes (tiny, exact return codes) ----
    kk = np.array([0, 1, 0, 2], dtype=np.int32)
    vv = np.array([1, 2, 3, 4], dtype=np.int32)
    vf = np.array([1.0, 2.0, 3.0, 4.0])
    if _HAS_P1:
        outs, _ = _x_mixed_1(kk, [vv], [vf], 3)
        assert [int(x.sum()) for x in outs] == [10, 10], outs
        _, rc = _x_mixed_1(np.array([0, 9], dtype=np.int32), [vv[:2]],
                           [vf[:2]], 3)
        assert rc == -2, rc  # BAD_RANGE, no panic
        big = np.array([2 ** 30, 2 ** 30], dtype=np.int32)
        _, rc = _x_mixed_1(np.array([0, 0], dtype=np.int32), [big],
                           [vf[:2]], 1)
        assert rc == -4, rc  # OVERFLOW, never wraps
        print("probes: P1 mixed exact + BAD_RANGE(-2) + OVERFLOW(-4) OK",
              flush=True)
    else:
        print("probes: P1 symbols absent (rejected/removed), skipped",
              flush=True)
    s, rc = _x_sumonly_1(kk, [vv], [vf], 3)
    assert rc == 0 and [int(x.sum()) for x in s] == [10, 10], (s, rc)
    assert int(_x_count_1(kk, 3).sum()) == 4
    print("probes: sumonly/countonly OK", flush=True)

    # ---- native warmup, single-threaded, OUTSIDE all timers ----
    # Pre-existing ctypes lazy-config race in native_cpu (_req_variant /
    # _req_select set argtypes on first touch; racing MT first-touch can
    # lock in an unconfigured cached symbol -> deterministic ArgumentError).
    # Production files untouched (frozen); warmup here only. Tiny arrays.
    from _lib.native_cpu import fused_sum_count as _nf1
    from _lib.native_cpu import sum_count_i32 as _ni32
    _wk = np.ascontiguousarray(np.array([0, 1, 0], dtype=np.int32))
    _wv = np.ascontiguousarray(np.array([1, 2, 3], dtype=np.int32))
    _wf = np.ascontiguousarray(np.array([1.0, 2.0, 3.0]))
    assert [int(x) for x in _ni32(_wk, _wv, 2)[0]] == [4, 2]
    _nf1(_wk, _wf, 2)
    print("warmup: native symbols configured ST (race dodged)", flush=True)

    # ---- kernel-only ST microbench on block 0 (~10M rows) ----
    b0 = blocks[0]
    k5 = np.ascontiguousarray(b0["id6"], dtype=np.int32)
    c1 = np.ascontiguousarray(b0["v1"], dtype=np.int32)
    c2 = np.ascontiguousarray(b0["v2"], dtype=np.int32)
    c3 = np.ascontiguousarray(b0["v3"], dtype=np.float64)
    # SoA flats pre-built ONCE (assembly copy outside kernel timers)
    f32 = np.ascontiguousarray(np.stack([c1, c2]).reshape(-1))
    f64 = np.ascontiguousarray(np.stack([c3]).reshape(-1))

    def _best(fn, reps=3):
        best = None
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            dt = (time.perf_counter() - s) * 1000
            best = dt if best is None else min(best, dt)
        return best

    r1 = _best(lambda: _ni32(k5, c1, M5), 3)
    r2 = _best(lambda: (_ni32(k5, c2, M5), _nf1(k5, c3, M5)), 1)
    # current production kernel cost for Q5 lanes: 2x i32 + 1x f64
    cur = _best(lambda: (_ni32(k5, c1, M5), _ni32(k5, c2, M5),
                         _nf1(k5, c3, M5)), 3)
    mx = (_best(lambda: _x_mixed_flat(k5, f32, f64, 2, 1, M5), 3)
          if _HAS_P1 else float("nan"))
    so = _best(lambda: _x_sumonly_flat(k5, f32, f64, 2, 1, M5), 3)
    co = _best(lambda: _x_count_1(k5, M5), 3)
    # SoA assembly cost itself (one-time per worker slice in e2e)
    asm = _best(lambda: (np.ascontiguousarray(np.stack([c1, c2]).reshape(-1)),
                         np.ascontiguousarray(np.stack([c3]).reshape(-1))), 3)
    # correctness kernel-only (bit-exact f64 expected: same i order)
    e1, _ = _ni32(k5, c1, M5)
    e2, _ = _ni32(k5, c2, M5)
    e3, _ = _nf1(k5, c3, M5)
    if _HAS_P1:
        mouts, mcc = _x_mixed_1(k5, [c1, c2], [c3], M5)
        assert np.array_equal(mouts[0], e1) and np.array_equal(mouts[1], e2), \
            "i32 lane mismatch"
        assert np.array_equal(mouts[2], e3), \
            float(np.max(np.abs(mouts[2] - e3)))
    souts, _ = _x_sumonly_1(k5, [c1, c2], [c3], M5)
    assert np.array_equal(souts[2], e3), "sumonly f64 mismatch"
    assert np.array_equal(souts[0], e1) and np.array_equal(souts[1], e2), \
        "sumonly i32 mismatch"
    print(f"KERNEL-ST blk0 n={k5.size} M={M5}: cur-per-col={cur:.1f}ms "
          f"(i32={r1:.1f}) mixed={mx:.1f} sumonly={so:.1f} count={co:.1f} "
          f"asm={asm:.1f} exact=OK", flush=True)
    out["kernel_st"] = {"cur_per_col_ms": round(cur, 1),
                        "mixed_ms": (round(mx, 1) if _HAS_P1 else None),
                        "sumonly_ms": round(so, 1),
                        "count_ms": round(co, 1), "asm_ms": round(asm, 1),
                        "exact": True}

    def run_query(tag, keyname, colnames, M, check, mode):
        if mode == "mixed" and not _HAS_P1:
            print(f"{tag} SKIPPED (P1 symbols absent: rejected/removed)",
                  flush=True)
            out["rows"][tag] = {"skipped": True,
                                "note": "P1 rejected/removed; see JSON history"}
            return
        res = {}
        for variant in ("BASE", "X"):
            best, chk = None, None
            for _ in range(3):
                gc.collect()
                accs, cc = None, None
                s = time.perf_counter()
                for bi, b in enumerate(blocks):
                    k = np.asarray(b[keyname])
                    vs = [np.asarray(b[c]) for c in colnames]
                    kb, vb = _C._probe_all(k, vs, t)
                    dec, mm, _ = _C._dense_gate(k, plan_groupby,
                                                naggs=len(vs), _kb=kb)
                    assert dec is not None, (tag, variant, "gate")
                    if variant == "BASE":
                        r = _C._native_mt(k, vs, int(mm), t, _vb=vb)
                    else:
                        r = _x_mt(k, vs, int(mm), t, mode)
                    assert r is not None, (tag, variant, "agg", f"block{bi}")
                    uk, cnt, sums, _, _, _ = r
                    if accs is None:
                        accs = [np.zeros(M, dtype=np.asarray(x).dtype)
                                for x in sums]
                        cc = np.zeros(M, dtype=np.int64)
                    ui = np.asarray(uk, dtype=np.int64)
                    for d, sl in zip(accs, sums):
                        d[ui] += np.asarray(sl, dtype=d.dtype)
                    cc[ui] += np.asarray(cnt, dtype=np.int64)
                    del r, uk, cnt, sums
                dt = (time.perf_counter() - s) * 1000
                chk, ok = check(accs, cc)
                assert ok, (tag, variant, chk)
                best = dt if best is None else min(best, dt)
                del accs, cc
            print(f"{tag} {variant} best3={best:.1f}ms chk={chk}",
                  flush=True)
            res[variant] = {"best3_ms": round(best, 1), "chk": chk}
        res["speedup"] = round(res["BASE"]["best3_ms"]
                               / res["X"]["best3_ms"], 3)
        print(f"{tag} SPEEDUP {res['speedup']}x", flush=True)
        out["rows"][tag] = res

    q5chk = lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                           "s2": int(np.asarray(a[1], dtype=np.int64).sum()),
                           "v3": float(np.asarray(a[2],
                                                  dtype=np.float64).sum())},
                          int(np.asarray(a[0], dtype=np.int64).sum())
                          == refs["v1"] and int(np.asarray(a[1],
                                                           dtype=np.int64).sum())
                          == refs["v2"] and abs(float(np.asarray(
                              a[2], dtype=np.float64).sum()) - refs["v3"])
                          <= 1e-6 * abs(refs["v3"]) + 1.0)
    q3chk = lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                           "v3": float(np.asarray(a[1],
                                                  dtype=np.float64).sum())},
                          int(np.asarray(a[0], dtype=np.int64).sum())
                          == refs["v1"] and abs(float(np.asarray(
                              a[1], dtype=np.float64).sum()) - refs["v3"])
                          <= 1e-6 * abs(refs["v3"]) + 1.0)
    run_query("Q5-P1-mixed", "id6", ["v1", "v2", "v3"], M5, q5chk, "mixed")
    run_query("Q5-P2-sumonly", "id6", ["v1", "v2", "v3"], M5, q5chk,
              "sumonly")
    run_query("Q3-P1-mixed", "k3", ["v1", "v3"], M3, q3chk, "mixed")
    run_query("Q3-P2-sumonly", "k3", ["v1", "v3"], M3, q3chk, "sumonly")

    o = FORK / "scratch" / "bench_p12_100M.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o}", flush=True)


if __name__ == "__main__":
    main()
