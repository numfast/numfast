# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Batch operator profiler 100M/16T: NFS load ONCE, all variants on resident blocks.

Methodology: read 10 blocks once (hybrid views, file IO outside all timers),
then run Q1/Q2/Q4/Q5 compute variants back-to-back on the same resident
blocks (per-block MT16 dense, same compute boundary as bench_reader_abc:
per-block aggregate + global merge, file load/mat excluded from compute).
Correctness exact vs service refs every variant. Bench-only, no core change.
No GPU. No DuckDB/Polars/Numba. Format NFS untouched.

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_opt_batch_100M.py
"""
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

_ZV = str(FORK / "src" / "Storage" / "NfsZview" / "_lib")
if _ZV not in sys.path:
    sys.path.insert(0, _ZV)
import zview as _Z  # noqa: E402

SCR = FORK / "scratch" / "nfs_stream"
F100 = SCR / "G1_1e8.stream.nfs"
COLS = ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")

_MT = {}


def _threads():
    return max(1, min(64, int(os.environ.get("NUMFAST_THREADS", "1"))))


def _pool(t):
    p = _MT.get(t)
    if p is None:
        p = ThreadPoolExecutor(max_workers=t)
        _MT[t] = p
    return p


def _dense_timed(keys, vcols, M, t):
    """Mirror bench_reader_abc._dense + operator breakdown timers."""
    br = {}
    s = time.perf_counter()
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    br["key_prep"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    cols = [np.ascontiguousarray(c, dtype=np.float64) for c in vcols]
    br["vprep"] = (time.perf_counter() - s) * 1000
    n = keys.size
    t = max(1, min(t, n))
    if t == 1:
        s = time.perf_counter()
        out = _nfN(keys, cols, M) if len(cols) > 1 else (_nf1(keys, cols[0], M), None)
        if len(cols) == 1 and not isinstance(out, tuple):
            pass
        br["kernel"] = (time.perf_counter() - s) * 1000
        br["merge_mt"] = 0.0
        if len(cols) == 1:
            sl, c = (out[0], out[1]) if isinstance(out, tuple) else out
            return (sl,), c, br
        return out[0], out[1], br
    bounds = np.linspace(0, n, t + 1).astype(np.int64)
    ex = _pool(t)
    s = time.perf_counter()
    futs = [ex.submit(_nfN if len(cols) > 1 else None, keys[int(bounds[w]):int(bounds[w + 1])],
                      [c[int(bounds[w]):int(bounds[w + 1])] for c in cols], M)
            if len(cols) > 1 else ex.submit(_nf1, keys[int(bounds[w]):int(bounds[w + 1])],
                                            cols[0][int(bounds[w]):int(bounds[w + 1])], M)
            for w in range(t)]
    parts = [f.result() for f in futs]
    br["kernel"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    if len(cols) > 1:
        nc = len(parts[0][0])
        sm = [np.zeros(M, dtype=np.float64) for _ in range(nc)]
        cc = np.zeros(M, dtype=np.int64)
        for sl, c in parts:
            for j in range(nc):
                sm[j] += sl[j]
            cc += c
    else:
        sm = [np.zeros(M, dtype=np.float64)]
        cc = np.zeros(M, dtype=np.int64)
        for sl, c in parts:
            sm[0] += sl
            cc += c
    br["merge_mt"] = (time.perf_counter() - s) * 1000
    return sm, cc, br


def main():
    t = _threads()
    assert F100.exists(), f"missing {F100}"
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    h = a["nfs_stream_open"](str(F100), force_lazy=True)
    idx = h["index"]
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2k2 = int(max(m["maxs"]["k2"] for m in idx)) + 1
    M2 = M1 * M2k2
    M4 = int(max(m["maxs"]["id4"] for m in idx)) + 1
    M5 = int(max(m["maxs"]["id6"] for m in idx)) + 1
    print(f"load-once {len(idx)} blocks N={N} T={t} native={_native_ok()} "
          f"M1={M1} M2={M2} M4={M4} M5={M5}", flush=True)
    # ---- LOAD ONCE (file IO, outside all compute timers) ----
    t0 = time.perf_counter()
    blocks = []
    with open(str(F100), "rb") as fh:
        for m in idx:
            fh.seek(int(m["file_offset"]))
            raw = fh.read(int(m["blob_len"]))
            z = np.load(io.BytesIO(raw))
            b = {k: np.asarray(z[k]) for k in COLS}
            z.close()
            del raw
            blocks.append(b)
    load_ms = (time.perf_counter() - t0) * 1000
    nrows = sum(len(b["k1"]) for b in blocks)
    print(f"resident {len(blocks)} blocks rows={nrows} load_ms={load_ms:.0f}", flush=True)
    assert nrows == N

    out = {"N": N, "T": t, "load_once_ms": load_ms, "native": bool(_native_ok()),
           "doms": [M1, M2, M4, M5], "variants": {}}

    def run_q(tag, fn):
        tc = {"key_prep": 0.0, "vprep": 0.0, "kernel": 0.0, "merge_mt": 0.0,
              "merge_global": 0.0, "pack": 0.0, "extra": 0.0}
        t0 = time.perf_counter()
        chk, ok = fn(tc)
        wall = (time.perf_counter() - t0) * 1000
        compute = tc["key_prep"] + tc["vprep"] + tc["pack"] + tc["extra"] + tc["kernel"] + tc["merge_mt"] + tc["merge_global"]
        print(f"{tag} compute={compute:.1f} key={tc['key_prep']:.1f} vprep={tc['vprep']:.1f} "
              f"pack={tc['pack']:.1f} extra={tc['extra']:.1f} kernel={tc['kernel']:.1f} "
              f"mtmerge={tc['merge_mt']:.1f} gmerge={tc['merge_global']:.1f} wall={wall:.1f} "
              f"chk={chk} ok={ok}", flush=True)
        assert ok, (tag, chk)
        out["variants"][tag] = {"stages": {k: round(v, 1) for k, v in tc.items()},
                                "compute_ms": round(compute, 1), "chk": chk, "ok": ok}
        gc.collect()
        return compute

    # Q1 baseline: k1, v1, M1
    def q1_base(tc):
        acc = np.zeros(M1, np.float64)
        for b in blocks:
            sl, _, br = _dense_timed(np.asarray(b["k1"]), [np.asarray(b["v1"])], M1, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            s = time.perf_counter()
            acc += sl[0]
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl
        chk = {"total": int(acc.astype(np.int64).sum()), "ngroups": int((acc > 0).sum())}
        return chk, chk["total"] == refs["v1"] and chk["ngroups"] == 100

    # Q2 baseline: pack k1*k2, v1, M2
    def q2_base(tc):
        acc = np.zeros(M2, np.float64)
        for b in blocks:
            s = time.perf_counter()
            kk = np.asarray(b["k1"]).astype(np.int64) * np.int64(M2k2) + np.asarray(b["k2"]).astype(np.int64)
            kk = kk.astype(np.int32)
            tc["pack"] += (time.perf_counter() - s) * 1000
            sl, _, br = _dense_timed(kk, [np.asarray(b["v1"])], M2, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            del kk
            s = time.perf_counter()
            acc += sl[0]
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl
        chk = {"total": int(acc.astype(np.int64).sum()), "ngroups": int((acc > 0).sum())}
        return chk, chk["total"] == refs["v1"] and chk["ngroups"] == 10000

    # Q2 native-pack variant: nf_pack_i32_direct instead of int64 astype chain
    def q2_npack(tc):
        acc = np.zeros(M2, np.float64)
        for b in blocks:
            s = time.perf_counter()
            kk = _nfpack(np.ascontiguousarray(b["k1"], dtype=np.int32),
                         np.ascontiguousarray(b["k2"], dtype=np.int32), M2k2)
            tc["pack"] += (time.perf_counter() - s) * 1000
            sl, _, br = _dense_timed(kk, [np.asarray(b["v1"])], M2, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            del kk
            s = time.perf_counter()
            acc += sl[0]
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl
        chk = {"total": int(acc.astype(np.int64).sum()), "ngroups": int((acc > 0).sum())}
        return chk, chk["total"] == refs["v1"] and chk["ngroups"] == 10000

    # Q4 baseline: id4, v1+v2+v3, M4 (3-col fused)
    def q4_base(tc):
        s1 = np.zeros(M4, np.float64)
        s2 = np.zeros(M4, np.float64)
        s3 = np.zeros(M4, np.float64)
        cc = np.zeros(M4, np.int64)
        for b in blocks:
            sl, c, br = _dense_timed(np.asarray(b["id4"]),
                                     [np.asarray(b["v1"]), np.asarray(b["v2"]), np.asarray(b["v3"])], M4, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            s = time.perf_counter()
            s1 += sl[0]
            s2 += sl[1]
            s3 += sl[2]
            cc += c
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl, c
        chk = {"cnt": int(cc.sum()), "s1": int(round(float(s1.sum()))), "s2": int(round(float(s2.sum())))}
        return chk, chk["cnt"] == N and chk["s1"] == refs["v1"] and chk["s2"] == refs["v2"]

    # Q4 single-col control: id4, v1 only (isolates multi-col + mean cost vs Q1 shape)
    def q4_1col(tc):
        acc = np.zeros(M4, np.float64)
        for b in blocks:
            sl, _, br = _dense_timed(np.asarray(b["id4"]), [np.asarray(b["v1"])], M4, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            s = time.perf_counter()
            acc += sl[0]
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl
        chk = {"total": int(acc.astype(np.int64).sum())}
        return chk, chk["total"] == refs["v1"]

    # Q4 3x-single control: three 1-col passes (fused vs separate cost)
    def q4_3x1(tc):
        accs = [np.zeros(M4, np.float64) for _ in range(3)]
        cols = ("v1", "v2", "v3")
        for b in blocks:
            for j, cn in enumerate(cols):
                sl, _, br = _dense_timed(np.asarray(b["id4"]), [np.asarray(b[cn])], M4, t)
                for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                    tc[k] += br[k]
                s = time.perf_counter()
                accs[j] += sl[0]
                tc["merge_global"] += (time.perf_counter() - s) * 1000
                del sl
        chk = {"s1": int(accs[0].astype(np.int64).sum()), "s2": int(accs[1].astype(np.int64).sum())}
        return chk, chk["s1"] == refs["v1"] and chk["s2"] == refs["v2"]

    # Q4 v3-only control: id4, v3 only (no int->f64 conversion on values)
    def q4_v3only(tc):
        acc = np.zeros(M4, np.float64)
        for b in blocks:
            sl, _, br = _dense_timed(np.asarray(b["id4"]), [np.asarray(b["v3"])], M4, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            s = time.perf_counter()
            acc += sl[0]
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl
        chk = {"v3": float(acc.sum())}
        return chk, abs(chk["v3"] - refs["v3"]) <= 1e-6 * abs(refs["v3"]) + 1.0

    # Q5 baseline: id6, v1+v2+v3, M5
    def q5_base(tc):
        s1 = np.zeros(M5, np.float64)
        s2 = np.zeros(M5, np.float64)
        s3 = np.zeros(M5, np.float64)
        for b in blocks:
            sl, _, br = _dense_timed(np.asarray(b["id6"]),
                                     [np.asarray(b["v1"]), np.asarray(b["v2"]), np.asarray(b["v3"])], M5, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            s = time.perf_counter()
            s1 += sl[0]
            s2 += sl[1]
            s3 += sl[2]
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl
        chk = {"s1": int(round(float(s1.sum()))), "s2": int(round(float(s2.sum()))), "v3": float(s3.sum())}
        return chk, chk["s1"] == refs["v1"] and chk["s2"] == refs["v2"] and abs(chk["v3"] - refs["v3"]) <= 1e-6 * abs(refs["v3"]) + 1.0

    # Q5 v3-only control
    def q5_v3only(tc):
        acc = np.zeros(M5, np.float64)
        for b in blocks:
            sl, _, br = _dense_timed(np.asarray(b["id6"]), [np.asarray(b["v3"])], M5, t)
            for k in ("key_prep", "vprep", "kernel", "merge_mt"):
                tc[k] += br[k]
            s = time.perf_counter()
            acc += sl[0]
            tc["merge_global"] += (time.perf_counter() - s) * 1000
            del sl
        chk = {"v3": float(acc.sum())}
        return chk, abs(chk["v3"] - refs["v3"]) <= 1e-6 * abs(refs["v3"]) + 1.0

    run_q("Q1_base", q1_base)
    run_q("Q2_base", q2_base)
    run_q("Q2_npack", q2_npack)
    run_q("Q4_base_3col", q4_base)
    run_q("Q4_1col_ctrl", q4_1col)
    run_q("Q4_3x1_ctrl", q4_3x1)
    run_q("Q4_v3only_ctrl", q4_v3only)
    run_q("Q5_base_3col", q5_base)
    run_q("Q5_v3only_ctrl", q5_v3only)

    o = FORK / "tests" / "heavy" / "bench_opt_batch_100M.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o}", flush=True)


if __name__ == "__main__":
    main()
