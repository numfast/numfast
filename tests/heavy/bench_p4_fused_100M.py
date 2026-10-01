# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 research: fused pack+aggregate (one pass, no packed-key buffer), 100M/16T.

Methodology mirrors bench_p2_accept_100M (verbatim boundary): NFS stream
blocks loaded ONCE (file IO outside all timers), per-block MT aggregate +
global dense merge, best-of-3, exact chk vs service refs. Bench-only file;
production modules untouched (experimental kernels live in the TEST dll,
loaded here by explicit path; production uses the working dll).

Sections (strictly sequential):
  S0 synthetic cardinalities (N=10M, seed 42, T=16): M=100 / 10K / 100K.
     For each M, 6 пунктов: (1) pack ST ms, (2) aggregate MT ms,
     (3) sum ms, (4) fused MT ms, (5) saved bytes, (6) e2e note (real Q2
     in S1; synthetic e2e = sum vs fused on the same N).
  S1 e2e Q2 real 100M/16T: BASE (ST pack + MT agg per block + global
     merge) vs FUSED (MT fused per block + global merge), best-of-3.
Q1/Q3/Q4/Q5 run only if FUSED wins >=5% on the production path (accept
gate); otherwise the experiment is discarded (no production change).

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_p4_fused_100M.py
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

WORK_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
               / "release" / "numfast_native.dll")
TEST_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
               / "release" / "numfast_native_p4test.dll")


def _load_test():
    lib = ctypes.CDLL(TEST_DLL)
    lib.nf_pack_sum_count_i32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.nf_pack_sum_count_i32.restype = ctypes.c_int32
    lib.nf_pack_sum_count_f64.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.nf_pack_sum_count_f64.restype = ctypes.c_int32
    _ = lib.nf_pack_sum_count_i32  # force Windows lookup now
    _ = lib.nf_pack_sum_count_f64
    return lib


def _best(fn, reps=3):
    best = None
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        dt = (time.perf_counter() - s) * 1000
        best = dt if best is None else min(best, dt)
    return best


def main():
    global _C
    t = max(1, min(64, int(os.environ.get("NUMFAST_THREADS", "1"))))
    assert os.path.exists(TEST_DLL), TEST_DLL
    assert os.path.exists(WORK_DLL), WORK_DLL
    _tlib = _load_test()
    print(f"P4 testdll={TEST_DLL} T={t}", flush=True)

    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    import importlib as _il
    for _m in [m for m in sys.modules if m == "_lib" or m.startswith("_lib.")]:
        del sys.modules[_m]
    import _lib.cpu as _Cf
    _C = _Cf
    a = kernel.alias
    plan_groupby = a["plan_groupby"]
    from _lib.native_cpu import pack_i32_direct as _nfpack

    # ST warmup outside timers (working + test symbols, tiny arrays).
    _wk1 = np.ascontiguousarray(np.array([0, 1, 0], dtype=np.int32))
    _wk2 = np.ascontiguousarray(np.array([2, 1, 0], dtype=np.int32))
    _wv = np.ascontiguousarray(np.array([1, 2, 3], dtype=np.int32))
    _wf = np.ascontiguousarray(np.array([1.0, 2.0, 3.0]))
    assert (_nfpack(_wk1, _wk2, 10) == _wk1 * 10 + _wk2).all()
    _s = np.zeros(100, dtype=np.int32)
    _c = np.zeros(100, dtype=np.int64)
    rc = _tlib.nf_pack_sum_count_i32(
        _wk1.ctypes.data, _wk2.ctypes.data, 10, _wv.ctypes.data, 3,
        _s.ctypes.data, _c.ctypes.data, 100)
    assert rc == 0 and int(_c.sum()) == 3, rc
    _sf = np.zeros(100)
    _cf = np.zeros(100, dtype=np.int64)
    rc = _tlib.nf_pack_sum_count_f64(
        _wk1.ctypes.data, _wk2.ctypes.data, 10, _wf.ctypes.data, 3,
        _sf.ctypes.data, _cf.ctypes.data, 100)
    assert rc == 0 and int(_cf.sum()) == 3, rc
    print("warmup: ST working pack + P4 test fused i32/f64 OK", flush=True)

    # ---- fused MT worker (mirror of _native_mt merge, test inner) ----
    ex = _C._pool(t)

    def _fused_mt(k1, k2, m2, vals, m):
        """Row-sharded fused pack+agg: private i32 states, int64 merge."""
        k1 = np.ascontiguousarray(k1, dtype=np.int32)
        k2 = np.ascontiguousarray(k2, dtype=np.int32)
        vals = np.ascontiguousarray(vals, dtype=np.int32)
        n = int(k1.size)
        tt = max(1, min(int(t), n))
        bounds = np.linspace(0, n, tt + 1).astype(np.int64)

        def _work(aa, bb):
            s = np.zeros(m, dtype=np.int32)
            c = np.zeros(m, dtype=np.int64)
            rc = _tlib.nf_pack_sum_count_i32(
                k1[aa:bb].ctypes.data, k2[aa:bb].ctypes.data, int(m2),
                vals[aa:bb].ctypes.data, int(bb - aa),
                s.ctypes.data, c.ctypes.data, int(m))
            assert rc == 0, rc
            return s, c

        t0 = time.perf_counter()
        futs = [ex.submit(_work, int(bounds[w]), int(bounds[w + 1]))
                for w in range(tt)]
        parts = [f.result() for f in futs]
        agg_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        sums_m = np.zeros(m, dtype=np.int64)
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
        _outs = [ss for ss, _ in parts]
        for w in range(_ns):
            _mf.append(ex.submit(_mshard, sums_m, _outs, _mb[w], _mb[w + 1]))
        _cout = [cc for _, cc in parts]
        for w in range(_ns):
            _mf.append(ex.submit(_mshard, counts_m, _cout, _mb[w], _mb[w + 1]))
        for _f in _mf:
            _f.result()
        merge_ms = (time.perf_counter() - t1) * 1000
        return sums_m, counts_m, agg_ms, merge_ms

    out = {"T": t, "synthetic": {}, "note": "seed=42"}
    t_all = time.perf_counter()

    # ---- S0: synthetic cardinalities (N=10M, resident once per M) ----
    for tag, m1c, m2c in (("M=100", 10, 10), ("M=10K", 100, 100),
                          ("M=100K", 100, 1000)):
        N = 10_000_000
        m2 = int(m2c)
        g = int(m1c * m2c)
        rng = np.random.default_rng(42)
        k1 = np.ascontiguousarray(
            rng.integers(0, m1c, size=N).astype(np.int32))
        k2 = np.ascontiguousarray(
            rng.integers(0, m2c, size=N).astype(np.int32))
        v1 = np.ascontiguousarray(
            rng.integers(1, 6, size=N).astype(np.int32))
        del rng
        gc.collect()
        saved_bytes = int(N) * 4  # packed-key buffer that disappears
        # reference (numpy, outside timers)
        pref = (k1.astype(np.int64) * np.int64(m2)
                + k2.astype(np.int64)).astype(np.int32)
        eref = np.bincount(pref, weights=v1.astype(np.int64),
                           minlength=g).astype(np.int64)
        # (1) current native pack ST
        pack_ms = _best(lambda: _nfpack(k1, k2, m2), 3)
        packed = _nfpack(k1, k2, m2)
        assert packed.dtype == np.dtype(np.int32)
        assert bool((packed == pref).all()), "pack exact"
        # (2) current aggregate MT on packed keys
        def _agg():
            r = _C._native_mt(packed, [v1], g, t)
            assert r is not None
            return r
        _r = _agg()  # warm pool paths
        del _r
        agg_ms = _best(_agg, 3)
        r = _agg()
        uk, cnt, sums, _ams, _cms, _mms = r
        # _native_mt int lane widens merge to int64 (see cpu.py)
        base_dense = np.zeros(g, dtype=np.int64)
        base_dense[np.asarray(uk, dtype=np.int64)] += np.asarray(
            sums[0], dtype=np.int64)
        assert bool((base_dense == eref).all()), "baseline exact"
        del r, uk, cnt, sums
        # (4) fused MT
        _fr = _fused_mt(k1, k2, m2, v1, g)  # warm
        del _fr
        fused_ms = _best(lambda: _fused_mt(k1, k2, m2, v1, g), 3)
        fsums, fcounts, _, _ = _fused_mt(k1, k2, m2, v1, g)
        assert bool((fsums == eref).all()), "fused exact"
        assert int(fcounts.sum()) == N
        total_ms = pack_ms + agg_ms
        ratio = total_ms / fused_ms
        print(f"S0 {tag} N={N} g={g}: (1)pack={pack_ms:.1f}ms "
              f"(2)agg={agg_ms:.1f}ms (3)sum={total_ms:.1f}ms "
              f"(4)fused={fused_ms:.1f}ms ratio={ratio:.3f}x "
              f"(5)saved={saved_bytes}B exact=OK", flush=True)
        out["synthetic"][tag] = {
            "N": N, "g": g, "pack_ms": round(pack_ms, 1),
            "agg_ms": round(agg_ms, 1), "sum_ms": round(total_ms, 1),
            "fused_ms": round(fused_ms, 1), "ratio": round(ratio, 3),
            "saved_bytes": saved_bytes, "exact": True}
        del k1, k2, v1, packed, pref, eref, base_dense, fsums, fcounts
        gc.collect()

    # ---- S1: e2e Q2 real 100M/16T (NFS resident once, variants batched) ----
    nfs = str(FORK / "scratch" / "nfs_stream" / "G1_1e8.stream.nfs")
    h = a["nfs_stream_open"](nfs, force_lazy=True)
    idx = h["index"]
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2k2 = int(max(m["maxs"]["k2"] for m in idx)) + 1
    M2 = M1 * M2k2
    t0 = time.perf_counter()
    blocks = []
    with open(nfs, "rb") as fh:
        for m in idx:
            fh.seek(int(m["file_offset"]))
            raw = fh.read(int(m["blob_len"]))
            z = np.load(io.BytesIO(raw))
            blocks.append({k: np.asarray(z[k]) for k in
                           ("k1", "k2", "v1")})
            z.close()
            del raw
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"S1 resident rows={sum(len(b['k1']) for b in blocks)} "
          f"load={load_ms:.0f}ms (outside timers) N={N} M2={M2} T={t}",
          flush=True)

    def _e2e_base():
        acc = np.zeros(M2, dtype=np.int64)
        cc = np.zeros(M2, dtype=np.int64)
        for b in blocks:
            k = _nfpack(np.ascontiguousarray(b["k1"], dtype=np.int32),
                        np.ascontiguousarray(b["k2"], dtype=np.int32), M2k2)
            vs = [np.asarray(b["v1"])]
            kb, vb = _C._probe_all(k, vs, t)
            dec, mm, _ = _C._dense_gate(k, plan_groupby, naggs=1, _kb=kb)
            assert dec is not None
            r = _C._native_mt(k, vs, int(mm), t, _vb=vb)
            assert r is not None
            uk, cnt, sums, _, _, _ = r
            ui = np.asarray(uk, dtype=np.int64)
            acc[ui] += np.asarray(sums[0], dtype=np.int64)
            cc[ui] += np.asarray(cnt, dtype=np.int64)
            del r, uk, cnt, sums, k
        return acc, cc

    def _e2e_fused():
        acc = np.zeros(M2, dtype=np.int64)
        cc = np.zeros(M2, dtype=np.int64)
        for b in blocks:
            s_m, c_m, _, _ = _fused_mt(b["k1"], b["k2"], M2k2,
                                       b["v1"], M2)
            nz = np.flatnonzero(c_m)
            acc[nz] += s_m[nz]
            cc[nz] += c_m[nz]
            del s_m, c_m, nz
        return acc, cc

    _a, _c = _e2e_base()  # warm (pool + dll paths)
    assert int(_a.sum()) == refs["v1"] and int((_a != 0).sum()) == 10000
    del _a, _c
    _a, _c = _e2e_fused()
    assert int(_a.sum()) == refs["v1"] and int((_a != 0).sum()) == 10000
    del _a, _c
    gc.collect()

    base_ms, fused_ms, chk = None, None, {}
    for _ in range(3):
        gc.collect()
        s = time.perf_counter()
        a0, c0 = _e2e_base()
        dt = (time.perf_counter() - s) * 1000
        assert int(a0.sum()) == refs["v1"]
        assert int((a0 != 0).sum()) == 10000
        base_ms = dt if base_ms is None else min(base_ms, dt)
        del a0, c0
    for _ in range(3):
        gc.collect()
        s = time.perf_counter()
        a1, c1 = _e2e_fused()
        dt = (time.perf_counter() - s) * 1000
        assert int(a1.sum()) == refs["v1"]
        assert int((a1 != 0).sum()) == 10000
        fused_ms = dt if fused_ms is None else min(fused_ms, dt)
        chk = {"total": int(a1.sum()), "ng": int((a1 != 0).sum())}
        del a1, c1
    ratio = base_ms / fused_ms
    saved_100M = int(N) * 4
    print(f"S1 (6)e2e Q2 100M/16T: BASE={base_ms:.1f}ms FUSED={fused_ms:.1f}ms "
          f"ratio={ratio:.3f}x chk={chk} "
          f"(5)saved={saved_100M}B+{saved_100M}B-traffic exact=OK",
          flush=True)
    out["e2e_q2"] = {"N": N, "M2": M2, "base_ms": round(base_ms, 1),
                     "fused_ms": round(fused_ms, 1),
                     "ratio": round(ratio, 3), "chk": chk,
                     "saved_bytes": saved_100M, "exact": True}
    out["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    o = FORK / "scratch" / "bench_p4_fused_100M.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o} elapsed={out['elapsed_s']}s", flush=True)


_C = None

if __name__ == "__main__":
    main()
