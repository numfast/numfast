# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase2: int32-direct / int64 / ST-vs-MT / bincount controls, 100M/16T.

NFS load ONCE (resident blocks), all variants batched on same data.
Bench-only, no core change. Correctness exact vs service refs.
No GPU. No new deps. Format NFS untouched.

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_opt_phase2_100M.py
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
from _lib.native_cpu import available as _native_ok  # noqa: E402
from _lib.native_cpu import fused_sum_count as _nf1  # noqa: E402
from _lib.native_cpu import multi_sum_count as _nfN  # noqa: E402
from _lib.native_cpu import pack_i32_direct as _nfpack  # noqa: E402

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
           / "release" / "numfast_native.dll")
_lib = ctypes.CDLL(_DLL)
_lib.nf_group_variant_i64.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_size_t, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_size_t]
_lib.nf_group_variant_i64.restype = ctypes.c_int32
_lib.nf_group_variant_i32.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_size_t, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_size_t]
_lib.nf_group_variant_i32.restype = ctypes.c_int32

_MT = {}


def _threads():
    return max(1, min(64, int(os.environ.get("NUMFAST_THREADS", "1"))))


def _pool(t):
    p = _MT.get(t)
    if p is None:
        p = ThreadPoolExecutor(max_workers=t)
        _MT[t] = p
    return p


def v_i64(keys, ticks, g):
    """Single-col int64 dense (V_DENSE=0). ticks int64 zero-copy view ok."""
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    ticks = np.ascontiguousarray(ticks, dtype=np.int64)
    sums = np.zeros(g, dtype=np.int64)
    counts = np.zeros(g, dtype=np.int64)
    rc = _lib.nf_group_variant_i64(keys.ctypes.data, ticks.ctypes.data,
                                   keys.size, 0, sums.ctypes.data,
                                   counts.ctypes.data, g)
    assert rc == 0, rc
    return sums, counts


def v_i32(keys, vals, g):
    """Single-col int32-direct (V_DENSE=0). Zero-copy when vals int32 contig."""
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    vals = np.ascontiguousarray(vals, dtype=np.int32)
    sums = np.zeros(g, dtype=np.int32)
    counts = np.zeros(g, dtype=np.int64)
    rc = _lib.nf_group_variant_i32(keys.ctypes.data, vals.ctypes.data,
                                   keys.size, 0, sums.ctypes.data,
                                   counts.ctypes.data, g)
    assert rc == 0, rc
    return sums, counts


def mt_map(t, fn, parts):
    ex = _pool(t)
    return [f.result() for f in [ex.submit(fn, *p) for p in parts]]


def main():
    t = _threads()
    assert _native_ok()
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    h = a["nfs_stream_open"](
        str(FORK / "scratch" / "nfs_stream" / "G1_1e8.stream.nfs"), force_lazy=True)
    idx = h["index"]
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2k2 = int(max(m["maxs"]["k2"] for m in idx)) + 1
    M2 = M1 * M2k2
    M4 = int(max(m["maxs"]["id4"] for m in idx)) + 1
    M5 = int(max(m["maxs"]["id6"] for m in idx)) + 1
    t0 = time.perf_counter()
    blocks = []
    with open(str(FORK / "scratch" / "nfs_stream" / "G1_1e8.stream.nfs"), "rb") as fh:
        for m in idx:
            fh.seek(int(m["file_offset"]))
            raw = fh.read(int(m["blob_len"]))
            z = np.load(io.BytesIO(raw))
            blocks.append({k: np.asarray(z[k]) for k in
                           ("k1", "k2", "id4", "id6", "v1", "v2", "v3")})
            z.close()
            del raw
    print(f"resident rows={sum(len(b['k1']) for b in blocks)} "
          f"load={(time.perf_counter()-t0)*1000:.0f}ms T={t}", flush=True)
    out = {"N": N, "T": t, "variants": {}}

    def rep(tag, fn, expect):
        best = None
        chk = None
        for _ in range(3):
            gc.collect()
            s = time.perf_counter()
            chk, ok = fn()
            dt = (time.perf_counter() - s) * 1000
            assert ok, (tag, chk)
            best = dt if best is None else min(best, dt)
        print(f"{tag} best3={best:.1f}ms chk={chk}", flush=True)
        out["variants"][tag] = {"best3_ms": round(best, 1), "chk": chk}

    K1 = [np.ascontiguousarray(b["k1"], dtype=np.int32) for b in blocks]
    V1 = [b["v1"] for b in blocks]  # int32 views, zero-copy
    V2 = [b["v2"] for b in blocks]
    V3 = [b["v3"] for b in blocks]  # f64
    ID4 = [np.ascontiguousarray(b["id4"], dtype=np.int32) for b in blocks]
    ID6 = [np.ascontiguousarray(b["id6"], dtype=np.int32) for b in blocks]

    def q1_f64_mt():
        acc = np.zeros(M1)
        for k, v in zip(K1, V1):
            kk = np.ascontiguousarray(k, dtype=np.int32)
            cc = np.ascontiguousarray(v, dtype=np.float64)
            n = kk.size
            bounds = np.linspace(0, n, t + 1).astype(np.int64)
            ps = mt_map(t, lambda a, b: _nf1(kk[a:b], cc[a:b], M1),
                        [(int(bounds[w]), int(bounds[w + 1])) for w in range(t)])
            s = np.zeros(M1)
            for sl, _ in ps:
                s += sl
            acc += s
        return {"total": int(acc.astype(np.int64).sum())}, True

    def q1_i32_mt():
        acc = np.zeros(M1, dtype=np.int64)
        for k, v in zip(K1, V1):
            n = k.size
            bounds = np.linspace(0, n, t + 1).astype(np.int64)
            ps = mt_map(t, lambda a, b: v_i32(k[a:b], v[a:b], M1),
                        [(int(bounds[w]), int(bounds[w + 1])) for w in range(t)])
            for sl, _ in ps:
                acc += sl.astype(np.int64)
        return {"total": int(acc.sum())}, True

    def q1_i32_st():
        acc = np.zeros(M1, dtype=np.int64)
        for k, v in zip(K1, V1):
            sl, _ = v_i32(k, v, M1)
            acc += sl.astype(np.int64)
        return {"total": int(acc.sum())}, True

    def q1_bincount():
        acc = np.zeros(M1)
        for k, v in zip(K1, V1):
            acc += np.bincount(k, weights=v.astype(np.float64), minlength=M1)
        return {"total": int(acc.astype(np.int64).sum())}, True

    def q1_i32_bincount():
        acc = np.zeros(M1, dtype=np.int64)
        for k, v in zip(K1, V1):
            acc += np.bincount(k, weights=v.astype(np.int64), minlength=M1).astype(np.int64)
        return {"total": int(acc.sum())}, True

    rep("Q1_f64_MT16", q1_f64_mt, refs["v1"])
    rep("Q1_i32_MT16", q1_i32_mt, refs["v1"])
    rep("Q1_i32_ST", q1_i32_st, refs["v1"])
    rep("Q1_bincount_f64_ST", q1_bincount, refs["v1"])
    rep("Q1_bincount_i32_ST", q1_i32_bincount, refs["v1"])

    _K2 = []

    def get_k2():
        if not _K2:
            _K2.append([_nfpack(np.ascontiguousarray(b["k1"], dtype=np.int32),
                                np.ascontiguousarray(b["k2"], dtype=np.int32), M2k2)
                        for b in blocks])
        return _K2[0]

    def q2_i32_mt():
        kk = get_k2()
        acc = np.zeros(M2, dtype=np.int64)
        for k, v in zip(kk, V1):
            n = k.size
            bounds = np.linspace(0, n, t + 1).astype(np.int64)
            ps = mt_map(t, lambda a, b: v_i32(k[a:b], v[a:b], M2),
                        [(int(bounds[w]), int(bounds[w + 1])) for w in range(t)])
            for sl, _ in ps:
                acc += sl.astype(np.int64)
        ng = int((acc > 0).sum())
        return {"total": int(acc.sum()), "ng": ng}, int(acc.sum()) == refs["v1"] and ng == 10000

    def q2_f64_mt():
        kk = get_k2()
        acc = np.zeros(M2)
        for k, v in zip(kk, V1):
            cc = np.ascontiguousarray(v, dtype=np.float64)
            n = k.size
            bounds = np.linspace(0, n, t + 1).astype(np.int64)
            ps = mt_map(t, lambda a, b: _nf1(k[a:b], cc[a:b], M2),
                        [(int(bounds[w]), int(bounds[w + 1])) for w in range(t)])
            s = np.zeros(M2)
            for sl, _ in ps:
                s += sl
            acc += s
        ng = int((acc > 0).sum())
        return {"total": int(acc.astype(np.int64).sum()), "ng": ng}, \
            int(acc.astype(np.int64).sum()) == refs["v1"] and ng == 10000

    rep("Q2_i32_MT16_reusePack", q2_i32_mt, refs["v1"])
    rep("Q2_f64_MT16_reusePack", q2_f64_mt, refs["v1"])

    def q4_mix_mt():
        a1 = np.zeros(M4, dtype=np.int64)
        a2 = np.zeros(M4, dtype=np.int64)
        a3 = np.zeros(M4)
        for k, u, w, f in zip(ID4, V1, V2, V3):
            n = k.size
            bounds = np.linspace(0, n, t + 1).astype(np.int64)
            idxs = [(int(bounds[q]), int(bounds[q + 1])) for q in range(t)]
            for sl, _ in mt_map(t, lambda a, b: v_i32(k[a:b], u[a:b], M4), idxs):
                a1 += sl.astype(np.int64)
            for sl, _ in mt_map(t, lambda a, b: v_i32(k[a:b], w[a:b], M4), idxs):
                a2 += sl.astype(np.int64)
            cc = f  # f64 view, zero copy
            for sl, _ in mt_map(t, lambda a, b: _nf1(k[a:b], cc[a:b], M4), idxs):
                a3 += sl
        return {"s1": int(a1.sum()), "s2": int(a2.sum())}, \
            int(a1.sum()) == refs["v1"] and int(a2.sum()) == refs["v2"]

    rep("Q4_mix3x1_MT16", q4_mix_mt, refs["v1"])

    def q5_mix_mt():
        a1 = np.zeros(M5, dtype=np.int64)
        a2 = np.zeros(M5, dtype=np.int64)
        a3 = np.zeros(M5)
        for k, u, w, f in zip(ID6, V1, V2, V3):
            n = k.size
            bounds = np.linspace(0, n, t + 1).astype(np.int64)
            idxs = [(int(bounds[q]), int(bounds[q + 1])) for q in range(t)]
            for sl, _ in mt_map(t, lambda a, b: v_i32(k[a:b], u[a:b], M5), idxs):
                a1 += sl.astype(np.int64)
            for sl, _ in mt_map(t, lambda a, b: v_i32(k[a:b], w[a:b], M5), idxs):
                a2 += sl.astype(np.int64)
            for sl, _ in mt_map(t, lambda a, b: _nf1(k[a:b], f[a:b], M5), idxs):
                a3 += sl
        return {"s1": int(a1.sum()), "s2": int(a2.sum()), "v3": float(a3.sum())}, \
            int(a1.sum()) == refs["v1"] and int(a2.sum()) == refs["v2"] and \
            abs(float(a3.sum()) - refs["v3"]) <= 1e-6 * abs(refs["v3"]) + 1.0

    rep("Q5_mix3x1_MT16", q5_mix_mt, refs["v1"])

    o = FORK / "tests" / "heavy" / "bench_opt_phase2_100M.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o}", flush=True)


if __name__ == "__main__":
    main()
