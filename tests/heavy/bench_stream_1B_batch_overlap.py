# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFS 1B double-buffer pipeline: loader thread prefetches next block
while CPU computes Q1..Q5 on current block. Bounded queue = 2 slots.

Usage (Git Bash, sequential, timeout 550 per stage):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    timeout 550 /c/App/numfast/.venv/Scripts/python.exe \
    tests/heavy/bench_stream_1B_batch_overlap.py
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

import numpy as np

CPU_LIB = str(FORK / "src" / "Drivers" / "CPU")
if CPU_LIB not in sys.path:
    sys.path.insert(0, CPU_LIB)
from _lib.native_cpu import multi_sum_count as _nfN, available as _native_ok

import os
from concurrent.futures import ThreadPoolExecutor

PROC = None
try:
    import psutil
    PROC = psutil.Process()
except Exception:
    pass

F1B = FORK / "scratch" / "nfs_stream" / "G1_1e9.stream.nfs"
OUT = FORK / "tests" / "heavy" / "bench_stream_1B_batch_overlap.json"

QUEUE_CAP = 2  # bounded queue: max 2 prefetched blocks


def rss():
    if PROC is None:
        return -1.0
    return PROC.memory_info().rss / 1e9


def build_kernel():
    from builder import MAIN
    return MAIN["build"](str(FORK))


def run_graph(a, jobs, n):
    s = time.perf_counter()
    g = a["compile"](jobs)
    compile_ms = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g = a["optimize"](g)
    optimize_ms = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    r = a["evaluate"](g, "cpu", n)
    execute_ms = (time.perf_counter() - s) * 1000
    return r["result"], compile_ms + optimize_ms + execute_ms


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
    keys = np.ascontiguousarray(keys, dtype=np.int32)
    cols = [np.ascontiguousarray(c, dtype=np.float64) for c in vcols]
    n = keys.size
    t = max(1, min(t, n))
    if t == 1 or not _native_ok():
        if _native_ok():
            return _nfN(keys, cols, M)
        sums_list = [np.bincount(keys, weights=c, minlength=M) for c in cols]
        counts = np.bincount(keys, minlength=M).astype(np.int64)
        return sums_list, counts
    bounds = np.linspace(0, n, t + 1).astype(np.int64)
    ex = _pool_mt(t)
    futs = []
    for w in range(t):
        a, b = int(bounds[w]), int(bounds[w + 1])
        futs.append(ex.submit(_nfN, keys[a:b], [c[a:b] for c in cols], M))
    parts = [f.result() for f in futs]
    ncols = len(parts[0][0])
    sums_m = [np.zeros(M, dtype=np.float64) for _ in range(ncols)]
    counts_m = np.zeros(M, dtype=np.int64)
    for sums_list, cc in parts:
        for j in range(ncols):
            sums_m[j] += sums_list[j]
        counts_m += cc
    return sums_m, counts_m


def main():
    print(f"=== NFS 1B double-buffer overlap Q1..Q5 === RSS {rss():.2f}GB",
          flush=True)

    kernel = build_kernel()
    a = kernel.alias

    t0 = time.perf_counter()
    h = a["nfs_stream_open"](str(F1B), force_lazy=True)
    open_ms = (time.perf_counter() - t0) * 1000
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    idx = h["index"]
    nblocks = len(idx)
    print(f"open {open_ms:.0f}ms blocks={nblocks} N={N} RSS {rss():.2f}GB",
          flush=True)

    # domains
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2_k2 = int(max(m["maxs"]["k2"] for m in idx)) + 1
    M2 = M1 * M2_k2
    M3 = int(max(m["maxs"]["k3"] for m in idx)) + 1
    M4 = int(max(m["maxs"]["id4"] for m in idx)) + 1
    M5 = int(max(m["maxs"]["id6"] for m in idx)) + 1
    print(f"domains M1={M1} M2={M2} M3={M3} M4={M4} M5={M5}", flush=True)

    # merge accumulators
    q1_d = {}
    q1_cold = 0.0
    q2_d = {}
    q2_cold = 0.0
    q3_s1 = np.zeros(M3, np.float64)
    q3_cc = np.zeros(M3, np.float64)
    q3_s3 = np.zeros(M3, np.float64)
    q4_s1 = np.zeros(M4, np.float64)
    q4_s2 = np.zeros(M4, np.float64)
    q4_s3 = np.zeros(M4, np.float64)
    q4_cc = np.zeros(M4, np.float64)
    q5_s1 = np.zeros(M5, np.float64)
    q5_s2 = np.zeros(M5, np.float64)
    q5_s3 = np.zeros(M5, np.float64)

    # --- loader thread ---
    load_q = queue.Queue(maxsize=QUEUE_CAP)
    load_total = 0.0
    load_lock = threading.Lock()
    stop_event = threading.Event()

    def loader_fn():
        nonlocal load_total
        for bi in range(nblocks):
            s = time.perf_counter()
            b = a["nfs_stream_read_block"](h, bi)
            block_ms = (time.perf_counter() - s) * 1000
            with load_lock:
                load_total += block_ms
            load_q.put((bi, b, block_ms))
        load_q.put(None)  # sentinel

    t_wall = time.perf_counter()
    loader = threading.Thread(target=loader_fn, daemon=True)
    loader.start()

    comp_q1 = 0.0
    comp_q2 = 0.0
    comp_q3 = 0.0
    comp_q4 = 0.0
    comp_q5 = 0.0
    merge_total = 0.0
    nrows = 0

    # queue stats
    q_put_wait = 0.0
    q_get_wait = 0.0
    q_occupancy_sum = 0.0
    q_occupancy_n = 0

    t_put_wait_start = None
    t_get_wait_start = None
    qsize = 0

    for bi in range(nblocks):
        t_get = time.perf_counter()
        item = load_q.get()
        q_get_wait += (time.perf_counter() - t_get) * 1000

        if item is None:
            break
        bi_got, b, block_load_ms = item
        qsize = load_q.qsize()
        q_occupancy_sum += qsize
        q_occupancy_n += 1

        n_b = int(b["k1"].size)
        nrows += n_b

        # --- Q1 (graph groupby k1 sum v1) ---
        s = time.perf_counter()
        jobs = [a["ir_series"]("k", np.asarray(b["k1"])),
                a["ir_series"]("v", np.asarray(b["v1"])),
                a["ir_groupby"]("g", "v", "k", "sum")]
        r1, cold1 = run_graph(a, jobs, n_b)
        comp_q1 += cold1
        s_m = time.perf_counter()
        for k, v in r1.items():
            q1_d[k] = q1_d.get(k, 0) + int(v)
        q1_cold += cold1
        merge_total += (time.perf_counter() - s_m) * 1000

        # --- Q2 (graph pack k1*k2 sum v1) ---
        s = time.perf_counter()
        jobs = [a["ir_series"]("c1", np.asarray(b["k1"])),
                a["ir_series"]("c2", np.asarray(b["k2"])),
                a["ir_series"]("v", np.asarray(b["v1"])),
                a["ir_pack_keys"]("k", "c1", "c2"),
                a["ir_groupby"]("g", "v", "k", "sum")]
        r2, cold2 = run_graph(a, jobs, n_b)
        comp_q2 += cold2
        s_m = time.perf_counter()
        for k, v in r2.items():
            q2_d[k] = q2_d.get(k, 0) + int(v)
        q2_cold += cold2
        merge_total += (time.perf_counter() - s_m) * 1000

        # --- Q3 (k3 groupby: v1 sum + v3 mean via native_mt_dense) ---
        s = time.perf_counter()
        k3 = np.asarray(b["k3"])
        v1 = np.asarray(b["v1"])
        v3 = np.asarray(b["v3"])
        t_mt = _mt_threads()
        sums_list, counts = _native_mt_dense(k3, [v1.astype(np.float64), v3], M3, t_mt)
        comp_q3 += (time.perf_counter() - s) * 1000
        s_m = time.perf_counter()
        q3_s1 += sums_list[0]
        q3_cc += counts
        q3_s3 += sums_list[1]
        merge_total += (time.perf_counter() - s_m) * 1000

        # --- Q4 (id4 groupby: v1,v2,v3 sums + count via bincount) ---
        s = time.perf_counter()
        k4 = np.asarray(b["id4"])
        v1_4 = np.asarray(b["v1"])
        v2_4 = np.asarray(b["v2"])
        v3_4 = np.asarray(b["v3"])
        s1_4 = np.bincount(k4, weights=v1_4.astype(np.float64), minlength=M4)
        s2_4 = np.bincount(k4, weights=v2_4.astype(np.float64), minlength=M4)
        s3_4 = np.bincount(k4, weights=v3_4, minlength=M4)
        cc_4 = np.bincount(k4, minlength=M4)
        comp_q4 += (time.perf_counter() - s) * 1000
        s_m = time.perf_counter()
        q4_s1 += s1_4
        q4_s2 += s2_4
        q4_s3 += s3_4
        q4_cc += cc_4
        merge_total += (time.perf_counter() - s_m) * 1000

        # --- Q5 (id6 groupby: v1,v2,v3 via native_mt_dense) ---
        s = time.perf_counter()
        k6 = np.asarray(b["id6"])
        v1_5 = np.asarray(b["v1"])
        v2_5 = np.asarray(b["v2"])
        v3_5 = np.asarray(b["v3"])
        sums5, _ = _native_mt_dense(k6, [v1_5.astype(np.float64), v2_5.astype(np.float64), v3_5], M5, t_mt)
        comp_q5 += (time.perf_counter() - s) * 1000
        s_m = time.perf_counter()
        q5_s1 += sums5[0]
        q5_s2 += sums5[1]
        q5_s3 += sums5[2]
        merge_total += (time.perf_counter() - s_m) * 1000

        del b, r1, r2, k3, v1, v3, k4, v1_4, v2_4, v3_4, s1_4, s2_4, s3_4, cc_4
        del k6, v1_5, v2_5, v3_5, sums5
        gc.collect()

        if (bi + 1) % 10 == 0 or bi == nblocks - 1:
            print(f"  block {bi+1}/{nblocks} load={load_total:.0f}ms "
                  f"comp={comp_q1+comp_q2+comp_q3+comp_q4+comp_q5:.0f}ms "
                  f"merge={merge_total:.0f}ms RSS {rss():.2f}GB", flush=True)

    loader.join()
    wall_ms = (time.perf_counter() - t_wall) * 1000

    avg_occ = q_occupancy_sum / max(1, q_occupancy_n)

    # correctness checks
    q1_tot = int(sum(q1_d.values()))
    q2_tot = int(sum(q2_d.values()))
    ok1 = (len(q1_d) == 100 and q1_tot == refs["v1"])
    ok2 = (len(q2_d) == 10000 and q2_tot == refs["v1"])
    q3_chk = int(q3_s1.astype(np.int64).sum())
    q3_cnt = int(q3_cc.astype(np.int64).sum())
    q3_v3x = float(q3_s3.sum())
    ok3 = (q3_chk == refs["v1"] and q3_cnt == N
           and abs(q3_v3x - refs["v3"]) <= 1e-9 * abs(refs["v3"]) * max(1.0, abs(refs["v3"]) / 1e6))
    q4_cnt = int(q4_cc.astype(np.int64).sum())
    ok4 = (q4_cnt == N and int(round(q4_s1.sum())) == refs["v1"]
           and int(round(q4_s2.sum())) == refs["v2"])
    q5_t1 = int(round(q5_s1.sum()))
    q5_t2 = int(round(q5_s2.sum()))
    q5_v3f = float(q5_s3.sum())
    ok5 = ((q5_t1, q5_t2) == (refs["v1"], refs["v2"])
           and abs(q5_v3f - refs["v3"]) <= 1e-9 * abs(refs["v3"]) * max(1.0, abs(refs["v3"]) / 1e6))

    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    print(f"correctness: Q1={ok1} Q2={ok2} Q3={ok3} Q4={ok4} Q5={ok5} ALL={all_ok}",
          flush=True)
    assert all_ok, f"FAIL: Q1={ok1} Q2={ok2} Q3={ok3} Q4={ok4} Q5={ok5}"

    total_comp = comp_q1 + comp_q2 + comp_q3 + comp_q4 + comp_q5

    result = {
        "N": N,
        "nblocks": nblocks,
        "open_ms": open_ms,
        "total_load_ms": load_total,
        "q1_ms": comp_q1,
        "q2_ms": comp_q2,
        "q3_ms": comp_q3,
        "q4_ms": comp_q4,
        "q5_ms": comp_q5,
        "total_compute_ms": total_comp,
        "merge_ms": merge_total,
        "wall_ms": wall_ms,
        "refs": refs,
        "correctness": "exact",
        "queue": {
            "capacity": QUEUE_CAP,
            "avg_occupancy": round(avg_occ, 2),
            "get_wait_ms": round(q_get_wait, 1),
        },
        "q1_groups": len(q1_d),
        "q1_chk": q1_tot,
        "q2_groups": len(q2_d),
        "q2_chk": q2_tot,
        "q3_chk_v1": q3_chk,
        "q4_ngroups": int((q4_cc > 0).sum()),
        "q5_s1": q5_t1,
        "q5_s2": q5_t2,
        "rss_gb": rss(),
    }
    OUT.write_text(json.dumps(json.loads(json.dumps(result, default=float)),
                              indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "refs"},
                     default=float, indent=1))
    print(f"JSON {OUT}", flush=True)


if __name__ == "__main__":
    main()
