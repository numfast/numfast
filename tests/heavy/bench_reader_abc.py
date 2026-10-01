# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reader-path control rebenchmark: copy vs hybrid vs zero (bench-only).

Scales: 10M (existing 1-block NFS), 100M (built from G1_1e8 CSV if missing).
Modes: A=copy (np.load+asarray), B=hybrid (zview int views + v3 copy),
C=zero (zview all views). Format NFS untouched. No Core/Planner/kernel change.
No GPU. No ClickBench.

Per (scale, mode, Q): 1 cold pass + 3 warm passes, warm-best reported
(same best-of-3 idea as bench_engine_carry). Each pass = full streaming scan
(load all blocks, compute that Q only, merge into global dense acc).
Stages per pass: load (seek+read blob bytes), mat (bytes->ready arrays),
compute (dense MT aggregate, NUMFAST_THREADS), merge (global accum add),
wall (end-to-end), peak RSS. Correctness exact every pass vs service refs.
Compute boundary: per-block dense MT (native_mt/bincount, T) + global merge.
NOT same boundary as historical carry-ms (in-memory cpu_execute, no NFS load,
no per-block merge) -> reference only.

Usage (Git Bash, strictly sequential, timeout 550 each):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_reader_abc.py 10M
  .../python.exe tests/heavy/bench_reader_abc.py 100M
  .../python.exe tests/heavy/bench_reader_abc.py build100M
"""
import gc
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import psutil

PROC = psutil.Process()
_CPU_LIB = str(FORK / "src" / "Drivers" / "CPU")
if _CPU_LIB not in sys.path:
    sys.path.insert(0, _CPU_LIB)
from _lib.native_cpu import available as _native_ok  # noqa: E402
from _lib.native_cpu import multi_sum_count as _nfN  # noqa: E402

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

_ZV = str(FORK / "src" / "Storage" / "NfsZview" / "_lib")
if _ZV not in sys.path:
    sys.path.insert(0, _ZV)
import zview as _Z  # noqa: E402

SCR = FORK / "scratch" / "nfs_stream"
F10 = SCR / "G1_1e7.stream.nfs"
F100 = SCR / "G1_1e8.stream.nfs"
CSV100 = "C:/App/competitions/H2O/data/G1_1e8_1e2_0_0.csv"
BR = 10_000_000
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


def _dense(keys, vcols, M, t):
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    cols = [np.ascontiguousarray(c, dtype=np.float64) for c in vcols]
    n = keys.size
    t = max(1, min(t, n))
    if t == 1 or not _native_ok():
        if _native_ok():
            return _nfN(keys, cols, M)
        sl = [np.bincount(keys, weights=c, minlength=M) for c in cols]
        return sl, np.bincount(keys, minlength=M).astype(np.int64)
    bounds = np.linspace(0, n, t + 1).astype(np.int64)
    ex = _pool(t)
    futs = [ex.submit(_nfN, keys[int(bounds[w]):int(bounds[w + 1])],
                      [c[int(bounds[w]):int(bounds[w + 1])] for c in cols], M)
            for w in range(t)]
    parts = [f.result() for f in futs]
    nc = len(parts[0][0])
    sm = [np.zeros(M, dtype=np.float64) for _ in range(nc)]
    cc = np.zeros(M, dtype=np.int64)
    for sl, c in parts:
        for j in range(nc):
            sm[j] += sl[j]
        cc += c
    return sm, cc


def rss():
    return PROC.memory_info().rss / 1e9


class Peak:
    def __init__(self):
        self.v, self._run = -1.0, False

    def start(self):
        self.v = rss()
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self._run:
            try:
                r = rss()
                if r > self.v:
                    self.v = r
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self):
        self._run = False
        self.t.join()
        return self.v


def build_kernel():
    from builder import MAIN
    return MAIN["build"](str(FORK))


def read_copy(path, m):
    off, blen = int(m["file_offset"]), int(m["blob_len"])
    s = time.perf_counter()
    with open(str(path), "rb") as fh:
        fh.seek(off)
        raw = fh.read(blen)
    t_load = (time.perf_counter() - s) * 1000
    assert len(raw) == blen
    s = time.perf_counter()
    z = np.load(io.BytesIO(raw))
    b = {k: np.asarray(z[k]) for k in COLS}
    z.close()
    del raw
    t_mat = (time.perf_counter() - s) * 1000
    return b, t_load, t_mat


def read_zview(path, m, mode):
    off, blen = int(m["file_offset"]), int(m["blob_len"])
    s = time.perf_counter()
    with open(str(path), "rb") as fh:
        fh.seek(off)
        raw = fh.read(blen)
    t_load = (time.perf_counter() - s) * 1000
    assert len(raw) == blen
    s = time.perf_counter()
    found = _Z._entries(raw)
    assert [n for n, _ in found] == [c + ".npy" for c in COLS]
    b = {}
    for name, lho in found:
        key = name[:-4]
        arr = _Z._view(raw, lho)
        if key == "v3":
            b[key] = arr if mode == "zero" else np.array(arr, dtype=np.float64, copy=True)
        else:
            b[key] = arr
    b["_raw"] = raw
    t_mat = (time.perf_counter() - s) * 1000
    return b, t_load, t_mat


def _parity(a, path, idx):
    m = idx[0]
    bc, _, _ = read_copy(path, m)
    bh, _, _ = read_zview(path, m, "hybrid")
    bz, _, _ = read_zview(path, m, "zero")
    for k in COLS:
        assert np.array_equal(np.asarray(bc[k]), np.asarray(bh[k])), k
        assert np.array_equal(np.asarray(bc[k]), np.asarray(bz[k])), k
    del bc, bh, bz
    gc.collect()


def one_pass(path, idx, mode, q, doms, t, refs, N):
    M1, M2, M2k2, M3, M4, M5 = doms
    pk = Peak()
    pk.start()
    tl = tm = tc = tg = 0.0
    t0 = time.perf_counter()
    if q == "Q1":
        acc = np.zeros(M1, np.float64)
        for m in idx:
            b, dl, dm = read_copy(path, m) if mode == "copy" else read_zview(path, m, mode)
            tl += dl
            tm += dm
            s = time.perf_counter()
            sl, _ = _dense(np.asarray(b["k1"]), [np.asarray(b["v1"]).astype(np.float64)], M1, t)
            tc += (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            acc += sl[0]
            tg += (time.perf_counter() - s) * 1000
            del b, sl
            gc.collect()
        chk = {"total": int(acc.astype(np.int64).sum()), "ngroups": int((acc > 0).sum())}
        ok = chk["total"] == refs["v1"] and chk["ngroups"] == 100
    elif q == "Q2":
        acc = np.zeros(M2, np.float64)
        for m in idx:
            b, dl, dm = read_copy(path, m) if mode == "copy" else read_zview(path, m, mode)
            tl += dl
            tm += dm
            s = time.perf_counter()
            kk = np.asarray(b["k1"]).astype(np.int64) * np.int64(M2k2) + np.asarray(b["k2"]).astype(np.int64)
            sl, _ = _dense(kk.astype(np.int32), [np.asarray(b["v1"]).astype(np.float64)], M2, t)
            tc += (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            acc += sl[0]
            tg += (time.perf_counter() - s) * 1000
            del b, sl, kk
            gc.collect()
        chk = {"total": int(acc.astype(np.int64).sum()), "ngroups": int((acc > 0).sum())}
        ok = chk["total"] == refs["v1"] and chk["ngroups"] == 10000
    elif q == "Q3":
        s1 = np.zeros(M3, np.float64)
        cc = np.zeros(M3, np.int64)
        s3 = np.zeros(M3, np.float64)
        for m in idx:
            b, dl, dm = read_copy(path, m) if mode == "copy" else read_zview(path, m, mode)
            tl += dl
            tm += dm
            s = time.perf_counter()
            sl, c = _dense(np.asarray(b["k3"]), [np.asarray(b["v1"]).astype(np.float64), np.asarray(b["v3"])], M3, t)
            tc += (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            s1 += sl[0]
            cc += c
            s3 += sl[1]
            tg += (time.perf_counter() - s) * 1000
            del b, sl, c
            gc.collect()
        chk = {"v1": int(s1.astype(np.int64).sum()), "cnt": int(cc.sum()), "v3": float(s3.sum())}
        ok = chk["v1"] == refs["v1"] and chk["cnt"] == N and abs(chk["v3"] - refs["v3"]) <= 1e-6 * abs(refs["v3"]) + 1.0
    elif q == "Q4":
        s1 = np.zeros(M4, np.float64)
        s2 = np.zeros(M4, np.float64)
        s3 = np.zeros(M4, np.float64)
        cc = np.zeros(M4, np.int64)
        for m in idx:
            b, dl, dm = read_copy(path, m) if mode == "copy" else read_zview(path, m, mode)
            tl += dl
            tm += dm
            s = time.perf_counter()
            sl, c = _dense(np.asarray(b["id4"]), [np.asarray(b["v1"]).astype(np.float64),
                           np.asarray(b["v2"]).astype(np.float64), np.asarray(b["v3"])], M4, t)
            tc += (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            s1 += sl[0]
            s2 += sl[1]
            s3 += sl[2]
            cc += c
            tg += (time.perf_counter() - s) * 1000
            del b, sl, c
            gc.collect()
        chk = {"cnt": int(cc.sum()), "s1": int(round(float(s1.sum()))), "s2": int(round(float(s2.sum())))}
        ok = chk["cnt"] == N and chk["s1"] == refs["v1"] and chk["s2"] == refs["v2"]
    else:
        s1 = np.zeros(M5, np.float64)
        s2 = np.zeros(M5, np.float64)
        s3 = np.zeros(M5, np.float64)
        for m in idx:
            b, dl, dm = read_copy(path, m) if mode == "copy" else read_zview(path, m, mode)
            tl += dl
            tm += dm
            s = time.perf_counter()
            sl, _ = _dense(np.asarray(b["id6"]), [np.asarray(b["v1"]).astype(np.float64),
                           np.asarray(b["v2"]).astype(np.float64), np.asarray(b["v3"])], M5, t)
            tc += (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            s1 += sl[0]
            s2 += sl[1]
            s3 += sl[2]
            tg += (time.perf_counter() - s) * 1000
            del b, sl
            gc.collect()
        chk = {"s1": int(round(float(s1.sum()))), "s2": int(round(float(s2.sum()))), "v3": float(s3.sum())}
        ok = chk["s1"] == refs["v1"] and chk["s2"] == refs["v2"] and abs(chk["v3"] - refs["v3"]) <= 1e-6 * abs(refs["v3"]) + 1.0
    wall = (time.perf_counter() - t0) * 1000
    peak = pk.stop()
    gc.collect()
    return {"load_ms": tl, "mat_ms": tm, "compute_ms": tc, "merge_ms": tg,
            "wall_ms": wall, "peak_rss_gb": peak, "chk": chk, "ok": ok}


def cmd_build100():
    kernel = build_kernel()
    a = kernel.alias
    print(f"=== build100M {CSV100} -> {F100} ===", flush=True)
    t0 = time.perf_counter()
    st = a["nfs_stream_build"](CSV100, str(F100), BR, resume=True)
    st["wall_ms"] = (time.perf_counter() - t0) * 1000
    st["rss_gb"] = rss()
    with open(FORK / "tests" / "heavy" / "bench_stream_100M.json", "w") as fh:
        json.dump(st, fh, indent=1, default=float)
    print(f"build100M N={st['N']} blocks={st['nblocks']} v1={st['refs']['v1']} "
          f"wall={st['wall_ms']:.0f}ms RSS {rss():.2f}GB", flush=True)


def cmd_scale(tag):
    t = _threads()
    path = F10 if tag == "10M" else F100
    assert path.exists(), f"missing {path}"
    kernel = build_kernel()
    a = kernel.alias
    t0 = time.perf_counter()
    h = a["nfs_stream_open"](str(path), force_lazy=True)
    open_ms = (time.perf_counter() - t0) * 1000
    idx = h["index"]
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2k2 = int(max(m["maxs"]["k2"] for m in idx)) + 1
    M2 = M1 * M2k2
    M3 = int(max(m["maxs"]["k3"] for m in idx)) + 1
    M4 = int(max(m["maxs"]["id4"] for m in idx)) + 1
    M5 = int(max(m["maxs"]["id6"] for m in idx)) + 1
    doms = (M1, M2, M2k2, M3, M4, M5)
    print(f"=== reader-abc {tag} N={N} blocks={len(idx)} T={t} native={_native_ok()} "
          f"open={open_ms:.1f}ms M1={M1} M2={M2} M3={M3} M4={M4} M5={M5} RSS {rss():.2f}GB", flush=True)
    _parity(a, str(path), idx)
    print("parity copy==hybrid==zero block0 EXACT", flush=True)
    out = {"scale": tag, "N": N, "nblocks": len(idx), "T": t, "open_ms": open_ms,
           "doms": doms, "native": bool(_native_ok()), "modes": {}}
    for mode in ("copy", "hybrid", "zero"):
        out["modes"][mode] = {}
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            cold = one_pass(str(path), idx, mode, q, doms, t, refs, N)
            assert cold["ok"], (tag, mode, q, cold["chk"])
            print(f"{tag} {mode} {q} COLD wall={cold['wall_ms']:.1f}ms "
                  f"load={cold['load_ms']:.1f} mat={cold['mat_ms']:.1f} comp={cold['compute_ms']:.1f} "
                  f"merge={cold['merge_ms']:.1f} peak={cold['peak_rss_gb']:.2f}GB chk={cold['chk']}", flush=True)
            warms = []
            for _ in range(3):
                w = one_pass(str(path), idx, mode, q, doms, t, refs, N)
                assert w["ok"], (tag, mode, q, w["chk"])
                warms.append(w)
            best = min(warms, key=lambda r: r["wall_ms"])
            print(f"{tag} {mode} {q} WARM-BEST wall={best['wall_ms']:.1f}ms "
                  f"load={best['load_ms']:.1f} mat={best['mat_ms']:.1f} comp={best['compute_ms']:.1f} "
                  f"merge={best['merge_ms']:.1f} peak={best['peak_rss_gb']:.2f}GB", flush=True)
            out["modes"][mode][q] = {"cold": cold, "warm_best": best,
                                     "warm_all": [w["wall_ms"] for w in warms]}
    out["correctness"] = "exact"
    o = FORK / "tests" / "heavy" / f"bench_reader_abc_{tag}_T{t}.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "10M"
    if arg == "build100M":
        cmd_build100()
    else:
        cmd_scale(arg)
