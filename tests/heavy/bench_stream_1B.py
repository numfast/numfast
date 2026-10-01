# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFS single-file stream proof: build (one pass) + Q1-Q5 through blocks.

SPEC-DELTA: nfs-stream-v1 file ([HEADER stub][BLOCKS][SERVICE][patch]).
No Planner/kernel/calibration change. No ClickBench. Bench-only script
(precedent: tests/heavy/bench_resident_1B.py).

Pipeline per query (producer-consumer):
  loader-thread: nfs_stream_read_block sequential -> q_in (maxsize 2)
  compute (main): per-block graph/dense partial -> q_out
  merger-thread: partials -> global acc (exact int add; f64 tol for v3)
Chunking: nfs_stream_plan (RAM/VRAM budget formula, no hardcode).
Queries: q1/q2 Runtime graphs per block (byte-identical to bench_resident_1B);
q3/q4/q5 dense bincount on bounded domains (k3/id4/id6 max from index,
exact int64 sums/counts, means = sums/counts). All chk vs build refs.

Usage (Git Bash, strictly sequential, timeout 550 per stage):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/competitions/H2O/python310/python.exe tests/heavy/bench_stream_1B.py build10
  .../python.exe tests/heavy/bench_stream_1B.py q10      # GREEN gate
  .../python.exe tests/heavy/bench_stream_1B.py build1B  # resume-safe slices
  .../python.exe tests/heavy/bench_stream_1B.py q1|q2|q3|q4|q5
"""

import gc
import json
import queue
import sys
import threading
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CSV_10M = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
CSV_1B = "C:/App/competitions/H2O/data/G1_1e9_1e2_0_0.csv"
SCR = FORK / "scratch" / "nfs_stream"
F10 = SCR / "G1_1e7.stream.nfs"
F1B = SCR / "G1_1e9.stream.nfs"
BR = 10_000_000
STOP = object()

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

PROC = psutil.Process()

# Native MT primitives (standalone module, no Builder deps)
_CPU_LIB = str(FORK / "src" / "Drivers" / "CPU")
if _CPU_LIB not in sys.path:
    sys.path.insert(0, _CPU_LIB)
from _lib.native_cpu import multi_sum_count as _nfN, available as _native_ok

_MT_POOLS = {}


def _mt_threads():
    return max(1, min(64, int(os.environ.get("NUMFAST_THREADS", "1"))))


def _pool_mt(t):
    p = _MT_POOLS.get(t)
    if p is None:
        p = ThreadPoolExecutor(max_workers=t)
        _MT_POOLS[t] = p
    return p


def _native_mt_dense(keys, vcols, M, t):
    """MT native dense aggregate returning dense M-arrays (for streaming merge).

    Same mechanism as cpu.py::_native_mt but returns raw dense M-arrays
    instead of ColumnCarry compact — needed for cross-block associative merge.
    """
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    cols = [np.ascontiguousarray(c, dtype=np.float64) for c in vcols]
    n = keys.size
    t = max(1, min(t, n))
    if t == 1 or not _native_ok():
        if _native_ok():
            sums_list, counts = _nfN(keys, cols, M)
            return sums_list, counts
        # numpy fallback
        sums_list = [np.bincount(keys, weights=c, minlength=M) for c in cols]
        counts = np.bincount(keys, minlength=M).astype(np.int64)
        return sums_list, counts
    bounds = np.linspace(0, n, t + 1).astype(np.int64)
    ex = _pool_mt(t)
    futs = []
    for w in range(t):
        a, b = int(bounds[w]), int(bounds[w + 1])
        kk = keys[a:b]
        vv = [c[a:b] for c in cols]
        futs.append(ex.submit(_nfN, kk, vv, M))
    parts = [f.result() for f in futs]
    ncols = len(parts[0][0])
    sums_m = [np.zeros(M, dtype=np.float64) for _ in range(ncols)]
    counts_m = np.zeros(M, dtype=np.int64)
    for sums_list, cc in parts:
        for j in range(ncols):
            sums_m[j] += sums_list[j]
        counts_m += cc
    return sums_m, counts_m


def rss():
    return PROC.memory_info().rss / 1e9


def build_kernel():
    from builder import MAIN
    return MAIN["build"](str(FORK))


def run_graph(a, jobs, n):
    st = {}
    s = time.perf_counter()
    g = a["compile"](jobs)
    st["compile"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    st["optimize"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    r = a["evaluate"](g, "cpu", n)
    st["execute"] = (time.perf_counter() - s) * 1000
    st["total"] = st["compile"] + st["optimize"] + st["execute"]
    return r["result"], st


class Peak:
    def __init__(self):
        self.v, self._run = -1.0, False

    def start(self):
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self._run:
            try:
                r = PROC.memory_info().rss / 1e9
                if r > self.v:
                    self.v = r
            except Exception:
                pass
            time.sleep(0.2)

    def stop(self):
        self._run = False
        self.t.join()
        return self.v


def pipeline(a, h, compute_fn, merge_fn, acc0):
    """loader-thread -> compute(main) -> merger-thread. Returns acc + times."""
    n = len(h["index"])
    q_in, q_out = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    t_load = [0.0]

    def loader():
        s = time.perf_counter()
        for i in range(n):
            b = a["nfs_stream_read_block"](h, i)
            q_in.put((i, b))
        t_load[0] = (time.perf_counter() - s) * 1000
        q_in.put(STOP)

    lt = threading.Thread(target=loader, daemon=True)
    acc, t_merge = [acc0], [0.0]

    def merger():
        while True:
            p = q_out.get()
            if p is STOP:
                return
            s = time.perf_counter()
            acc[0] = merge_fn(acc[0], p)
            t_merge[0] += (time.perf_counter() - s) * 1000

    mt = threading.Thread(target=merger, daemon=True)
    t0 = time.perf_counter()
    lt.start()
    mt.start()
    t_comp = 0.0
    nrows = 0
    while True:
        item = q_in.get()
        if item is STOP:
            q_out.put(STOP)
            break
        _, b = item
        nrows += int(b["k1"].size)
        s = time.perf_counter()
        q_out.put(compute_fn(b))
        t_comp += (time.perf_counter() - s) * 1000
        del b
        gc.collect()
    lt.join()
    mt.join()
    total = (time.perf_counter() - t0) * 1000
    return acc[0], {"load_ms": t_load[0], "compute_ms": t_comp,
                    "merge_ms": t_merge[0], "total_ms": total, "rows": nrows}


def cmd_build(csv, out, tag, max_blocks=None):
    kernel = build_kernel()
    a = kernel.alias
    print(f"=== stream {tag}: {csv} -> {out} ===", flush=True)
    print(f"RSS start {rss():.2f}GB", flush=True)
    t0 = time.perf_counter()
    st = a["nfs_stream_build"](csv, str(out), BR, resume=True,
                               max_blocks=max_blocks)
    st["wall_ms"] = (time.perf_counter() - t0) * 1000
    st["rss_gb"] = rss()
    st["throughput_Mrows_s"] = st["N"] / max(1e-9, st["wall_ms"] / 1000) / 1e6
    with open(FORK / "tests" / "heavy" / f"bench_stream_{tag}.json", "w") as fh:
        json.dump(st, fh, indent=1, default=float)
    print(f"build {tag}: N={st['N']} blocks={st['nblocks']} "
          f"v1={st['refs']['v1']} wall={st['wall_ms']:.0f}ms "
          f"{st['throughput_Mrows_s']:.2f}Mrows/s RSS {rss():.2f}GB", flush=True)


def open_lazy(a, path):
    h = a["nfs_stream_open"](str(path), force_lazy=True)
    plan = a["nfs_stream_plan"](h)
    print(f"open mode={h['mode']} blocks={len(h['index'])} "
          f"blob={h['blob_bytes']/1e9:.2f}GB plan={plan}", flush=True)
    return h


def cmd_q10():
    """GREEN gate on 10M: q1 via graphs + full colsum chk vs build refs."""
    kernel = build_kernel()
    a = kernel.alias
    h = open_lazy(a, F10)
    refs = h["service"]["refs"]
    n = h["header"]["N"]

    def comp(b):
        jobs = [a["ir_series"]("k", np.asarray(b["k1"])),
                a["ir_series"]("v", np.asarray(b["v1"])),
                a["ir_groupby"]("g", "v", "k", "sum")]
        r, st = run_graph(a, jobs, int(b["k1"].size))
        return (dict(r), st["total"])

    acc, tm = pipeline(a, h, comp,
                       lambda acc, p: (acc[0].update(
                           {k: acc[0].get(k, 0) + int(v)
                            for k, v in p[0].items()}) or acc[0], acc[1] + p[1]),
                       ({}, 0.0))
    d, cold = acc
    tot = int(sum(d.values()))
    # direct colsum cross-check (zero-transform proof: stored == memory)
    b = a["nfs_stream_read_block"](h, 0)
    direct = {c: (float(np.asarray(b[c]).sum()) if c == "v3"
                  else int(np.asarray(b[c], dtype=np.int64).sum()))
              for c in ("v1", "v2", "v3")}
    ok = (tot == refs["v1"] == direct["v1"] and len(d) == 100
          and direct["v2"] == refs["v2"]
          and abs(direct["v3"] - refs["v3"]) <= 1e-6 * abs(refs["v3"]))
    print(f"q10 GREEN={ok} groups={len(d)} chk_v1={tot} refs_v1={refs['v1']} "
          f"direct={direct} cold={cold:.0f}ms pipe={tm} RSS {rss():.2f}GB",
          flush=True)
    assert ok, (tot, refs, direct)
    # --- Q3 GREEN gate (native MT dense) ---
    M3 = int(max(m["maxs"]["k3"] for m in h["index"])) + 1
    N3 = h["header"]["N"]

    def _comp_q3(b):
        k = np.asarray(b["k3"])
        v1 = np.asarray(b["v1"])
        v3 = np.asarray(b["v3"])
        t = _mt_threads()
        sums_list, counts = _native_mt_dense(
            k, [v1.astype(np.float64), v3], M3, t)
        return (sums_list[0], counts, sums_list[1])

    (gs1_3, gcc_3, gs3_3), _ = pipeline(
        a, h, _comp_q3,
        lambda acc, p: (acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]),
        (np.zeros(M3, np.float64), np.zeros(M3, np.float64),
         np.zeros(M3, np.float64)))
    chk3 = int(gs1_3.astype(np.int64).sum())
    cnt3 = int(gcc_3.astype(np.int64).sum())
    v3x3 = float(gs3_3.sum())
    ok3 = (chk3 == refs["v1"] and cnt3 == N3
           and abs(v3x3 - refs["v3"]) <= 1e-9 * abs(refs["v3"])
           * max(1.0, abs(refs["v3"]) / 1e6))
    print(f"q3 GREEN={ok3} chk_v1={chk3} cnt={cnt3} v3={v3x3:.6f} "
          f"refs_v1={refs['v1']}", flush=True)
    assert ok3, (chk3, cnt3, v3x3, refs)
    # --- Q5 GREEN gate (native MT dense) ---
    M5 = int(max(m["maxs"]["id6"] for m in h["index"])) + 1

    def _comp_q5(b):
        k = np.asarray(b["id6"])
        v1 = np.asarray(b["v1"])
        v2 = np.asarray(b["v2"])
        v3 = np.asarray(b["v3"])
        t = _mt_threads()
        sums_list, counts = _native_mt_dense(
            k, [v1.astype(np.float64), v2.astype(np.float64), v3],
            M5, t)
        return (sums_list[0], sums_list[1], sums_list[2])

    (gs1_5, gs2_5, gs3_5), _ = pipeline(
        a, h, _comp_q5,
        lambda acc, p: tuple(x + y for x, y in zip(acc, p)),
        (np.zeros(M5, np.float64), np.zeros(M5, np.float64),
         np.zeros(M5, np.float64)))
    t1_5 = int(round(gs1_5.sum()))
    t2_5 = int(round(gs2_5.sum()))
    s3f_5 = float(gs3_5.sum())
    ok5 = ((t1_5, t2_5) == (refs["v1"], refs["v2"])
           and abs(s3f_5 - refs["v3"]) <= 1e-9 * abs(refs["v3"])
           * max(1.0, abs(refs["v3"]) / 1e6))
    print(f"q5 GREEN={ok5} s1={t1_5} s2={t2_5} v3={s3f_5:.6f} "
          f"refs_v1={refs['v1']} refs_v2={refs['v2']}", flush=True)
    assert ok5, (t1_5, t2_5, s3f_5, refs)
    with open(FORK / "tests" / "heavy" / "bench_stream_10M_q.json", "w") as fh:
        json.dump({"green": ok, "chk_v1": tot, "refs": refs, "direct": direct,
                   "cold_ms": cold, "pipe": tm, "rss_gb": rss()}, fh, indent=1,
                  default=float)


def cmd_q(q):
    kernel = build_kernel()
    a = kernel.alias
    print(f"=== stream 1B {q} (lazy blocks, CPU) === RSS {rss():.2f}GB", flush=True)
    h = open_lazy(a, F1B)
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    idx = h["index"]
    pk = Peak()
    pk.start()
    out = {"mode": "lazy-blocks-cpu"}
    if q == "q1":
        def comp(b):
            jobs = [a["ir_series"]("k", np.asarray(b["k1"])),
                    a["ir_series"]("v", np.asarray(b["v1"])),
                    a["ir_groupby"]("g", "v", "k", "sum")]
            r, st = run_graph(a, jobs, int(b["k1"].size))
            return (r, st["total"])

        def merge(acc, p):
            r, c = p
            d, cold = acc
            for k, v in r.items():
                d[k] = d.get(k, 0) + int(v)
            return (d, cold + c)

        (d, cold), tm = pipeline(a, h, comp, merge, ({}, 0.0))
        tot = int(sum(d.values()))
        assert len(d) == 100 and tot == refs["v1"], (len(d), tot)
        out.update({"cold_ms": cold, "chk": tot, "ngroups": len(d)})
    elif q == "q2":
        def comp(b):
            jobs = [a["ir_series"]("c1", np.asarray(b["k1"])),
                    a["ir_series"]("c2", np.asarray(b["k2"])),
                    a["ir_series"]("v", np.asarray(b["v1"])),
                    a["ir_pack_keys"]("k", "c1", "c2"),
                    a["ir_groupby"]("g", "v", "k", "sum")]
            r, st = run_graph(a, jobs, int(b["k1"].size))
            return (r, st["total"])

        def merge(acc, p):
            r, c = p
            d, cold = acc
            for k, v in r.items():
                d[k] = d.get(k, 0) + int(v)
            return (d, cold + c)

        (d, cold), tm = pipeline(a, h, comp, merge, ({}, 0.0))
        tot = int(sum(d.values()))
        assert len(d) == 10000 and tot == refs["v1"], (len(d), tot)
        out.update({"cold_ms": cold, "chk": tot, "ngroups": len(d)})
    elif q == "q3":
        M = int(max(m["maxs"]["k3"] for m in idx)) + 1
        print(f"q3 domain k3 M={M}", flush=True)

        def comp(b):
            k = np.asarray(b["k3"])
            v1 = np.asarray(b["v1"])
            v3 = np.asarray(b["v3"])
            t = _mt_threads()
            sums_list, counts = _native_mt_dense(
                k, [v1.astype(np.float64), v3], M, t)
            return (sums_list[0], counts, sums_list[1])

        def merge(acc, p):
            s1, c, s3 = acc
            a1, a2, a3 = p
            return (s1 + a1, c + a2, s3 + a3)

        zf3 = (np.zeros(M, np.float64), np.zeros(M, np.float64),
               np.zeros(M, np.float64))
        (gs1, gcc, gs3), tm = pipeline(a, h, comp, merge, zf3)
        gs1 = gs1.astype(np.int64)
        gcc = gcc.astype(np.int64)
        chk = int(gs1.sum())
        assert chk == refs["v1"] and int(gcc.sum()) == N, (chk, gcc.sum())
        v3x = float(gs3.sum())
        assert abs(v3x - refs["v3"]) <= 1e-9 * abs(refs["v3"]) * max(
            1.0, abs(refs["v3"]) / 1e6), (v3x, refs["v3"])
        nz = int((gcc > 0).sum())
        out.update({"chk_v1": chk, "ngroups": nz,
                    "mean3_sum": float((gs3 / np.maximum(gcc, 1))[gcc > 0].sum())})
    elif q == "q4":
        M = int(max(m["maxs"]["id4"] for m in idx)) + 1
        print(f"q4 domain id4 M={M}", flush=True)

        def comp(b):
            k = np.asarray(b["id4"])
            return (np.bincount(k, weights=np.asarray(b["v1"]).astype(np.float64),
                                minlength=M),
                    np.bincount(k, weights=np.asarray(b["v2"]).astype(np.float64),
                                minlength=M),
                    np.bincount(k, weights=np.asarray(b["v3"]), minlength=M),
                    np.bincount(k, minlength=M))

        def merge(acc, p):
            return tuple(x + y for x, y in zip(acc, p))

        zf = (np.zeros(M, np.float64),) * 3 + (np.zeros(M, np.float64),)
        (s1, s2, s3, cc), tm = pipeline(a, h, comp, merge, zf)
        cc = cc.astype(np.int64)
        assert int(cc.sum()) == N, cc.sum()
        assert int(round(s1.sum())) == refs["v1"], (s1.sum(), refs["v1"])
        assert int(round(s2.sum())) == refs["v2"], (s2.sum(), refs["v2"])
        gm = {c: (s / np.maximum(cc, 1)) for c, s in
              zip(("v1", "v2", "v3"), (s1, s2, s3))}
        out.update({"sums": {c: float(gm[c][cc > 0].sum()) for c in gm},
                    "ngroups": int((cc > 0).sum())})
    elif q == "q5":
        M = int(max(m["maxs"]["id6"] for m in idx)) + 1
        print(f"q5 domain id6 M={M}", flush=True)

        def comp(b):
            k = np.asarray(b["id6"])
            v1 = np.asarray(b["v1"])
            v2 = np.asarray(b["v2"])
            v3 = np.asarray(b["v3"])
            t = _mt_threads()
            sums_list, counts = _native_mt_dense(
                k, [v1.astype(np.float64), v2.astype(np.float64), v3],
                M, t)
            return (sums_list[0], sums_list[1], sums_list[2])

        def merge(acc, p):
            return tuple(x + y for x, y in zip(acc, p))

        zf = (np.zeros(M, np.float64),) * 3
        (s1, s2, s3), tm = pipeline(a, h, comp, merge, zf)
        t1, t2 = int(round(s1.sum())), int(round(s2.sum()))
        assert (t1, t2) == (refs["v1"], refs["v2"]), (t1, t2)
        s3f = float(s3.sum())
        assert abs(s3f - refs["v3"]) <= 1e-9 * abs(refs["v3"]) * max(
            1.0, abs(refs["v3"]) / 1e6), (s3f, refs["v3"])
        out.update({"s1": t1, "s2": t2, "s3_scaled": int(round(s3f * 1e6)),
                    "ngroups": int((s1 != 0).sum() + ((s1 == 0) & (s3 != 0)).sum())})
    else:
        raise SystemExit(f"unknown q {q}")
    peak = pk.stop()
    out.update({"pipe": tm, "rss_gb": rss(), "peak_gb": peak,
                "refs": refs, "N": N})
    print(f"{q}: {json.dumps({k: v for k, v in out.items() if k != 'refs'}, default=float)} "
          f"RSS {rss():.2f}GB PEAK {peak:.2f}GB", flush=True)
    with open(FORK / "tests" / "heavy" / f"bench_stream_1B_{q}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("JSON written", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build10":
        cmd_build(CSV_10M, F10, "10M")
    elif cmd == "q10":
        cmd_q10()
    elif cmd == "build1B":
        mb = int(sys.argv[2]) if len(sys.argv) > 2 else None
        cmd_build(CSV_1B, F1B, "1B", max_blocks=mb)
    elif cmd in ("q1", "q2", "q3", "q4", "q5"):
        cmd_q(cmd)
    else:
        raise SystemExit("usage: bench_stream_1B.py build10|q10|build1B|q1..q5")
