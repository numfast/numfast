# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Lab drivers: serial/threaded local phase, hierarchical L0->L1->L2,
bounded-queue pipeline, pareto helpers. All timings ms, stage breakdown."""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def run_serial(mod, blocks):
    t0 = time.perf_counter()
    states, stats = [], []
    for b in blocks:
        st, s = mod.local(b["keys"], b["vals"])
        states.append(st)
        stats.append(s)
    wall = (time.perf_counter() - t0) * 1000.0
    return states, stats, wall


def run_threaded(mod, blocks, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda b: mod.local(b["keys"], b["vals"]), blocks))
    wall = (time.perf_counter() - t0) * 1000.0
    states = [o[0] for o in out]
    stats = [o[1] for o in out]
    return states, stats, wall


def merge_all(mod, states):
    t0 = time.perf_counter()
    st, info = mod.merge(states)
    wall = (time.perf_counter() - t0) * 1000.0
    info = dict(info, wall_ms=wall)
    return st, info


def cpu_sum_ms(stats):
    return float(sum(s.get("local_ms", 0.0) for s in stats))


def hierarchical(mod, blocks, fanin=4):
    """L0 blocks -> L1 partials (+meta) -> L2 fanin merges -> final.
    Returns stage breakdown + state-byte funnel for the reduction hypothesis.
    """
    from .metadata import block_meta, state_meta

    t0 = time.perf_counter()
    l0_meta = [block_meta(b["keys"]) for b in blocks]
    raw_bytes = sum(m["raw_bytes"] for m in l0_meta)
    t_meta = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    l1 = [mod.local(b["keys"], b["vals"]) for b in blocks]
    l1_ms = (time.perf_counter() - t0) * 1000.0
    l1_states = [s for s, _ in l1]
    l1_bytes = sum(s.state_bytes() for s in l1_states)

    t0 = time.perf_counter()
    l2_states, l2_infos = [], []
    for i in range(0, len(l1_states), fanin):
        st, info = merge_all(mod, l1_states[i:i + fanin])
        l2_states.append(st)
        l2_infos.append(info)
    l2_ms = (time.perf_counter() - t0) * 1000.0
    l2_bytes = sum(s.state_bytes() for s in l2_states)

    t0 = time.perf_counter()
    final, finfo = merge_all(mod, l2_states)
    f_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "final": final, "finfo": finfo,
        "l0_meta_ms": t_meta, "l1_ms": l1_ms, "l2_ms": l2_ms, "final_ms": f_ms,
        "raw_bytes": raw_bytes, "l1_bytes": l1_bytes, "l2_bytes": l2_bytes,
        "final_bytes": int(final.state_bytes()),
        "l1_reduction": 1.0 - l1_bytes / max(raw_bytes, 1),
        "l2_reduction": 1.0 - l2_bytes / max(l1_bytes, 1),
        "total_reduction": 1.0 - final.state_bytes() / max(raw_bytes, 1),
        "l1_meta": [state_meta(s) for s in l1_states],
    }


def pipelined(mod, blocks, nworkers=4, qmax=4):
    """Bounded-queue pipeline: producer -> workers -> merger.

    Blocking Queue.get/put only (no busy wait); sync via Queue sentinel +
    Thread.join (no register flags). Overlap = serial_sum - wall.
    """
    in_q = queue.Queue(maxsize=qmax)
    out_q = queue.Queue(maxsize=qmax)
    acc = {"put_ms": 0.0, "get_wait_ms": 0.0, "compute_ms": 0.0}
    lock = threading.Lock()

    def producer():
        for i, b in enumerate(blocks):
            t = time.perf_counter()
            in_q.put((i, b))  # blocks when full: backpressure wait included
            with lock:
                acc["put_ms"] += (time.perf_counter() - t) * 1000.0
        for _ in range(nworkers):
            in_q.put(None)

    def worker():
        while True:
            t = time.perf_counter()
            item = in_q.get()  # blocks when empty: consumer idle included
            wait = (time.perf_counter() - t) * 1000.0
            with lock:
                acc["get_wait_ms"] += wait
            if item is None:
                out_q.put(None)
                in_q.task_done()
                return
            i, b = item
            t = time.perf_counter()
            st, s = mod.local(b["keys"], b["vals"])
            comp = (time.perf_counter() - t) * 1000.0
            with lock:
                acc["compute_ms"] += comp
            out_q.put((i, st, s))
            in_q.task_done()

    t0 = time.perf_counter()
    pt = threading.Thread(target=producer)
    pt.start()
    wts = [threading.Thread(target=worker) for _ in range(nworkers)]
    for w in wts:
        w.start()
    got = {}
    gstats = {}
    done = 0
    t = time.perf_counter()
    while done < nworkers:
        item = out_q.get()  # merger blocks: overlap with tail compute
        if item is None:
            done += 1
            continue
        i, st, s = item
        got[i] = st
        gstats[i] = s
    merge_wait_ms = (time.perf_counter() - t) * 1000.0
    pt.join()
    for w in wts:
        w.join()
    t = time.perf_counter()
    ordered = [got[i] for i in sorted(got)]
    final, finfo = merge_all(mod, ordered)
    merge_ms = (time.perf_counter() - t) * 1000.0
    wall = (time.perf_counter() - t0) * 1000.0
    serial_sum = acc["compute_ms"] + merge_ms
    return {
        "final": final, "finfo": finfo,
        "wall_ms": wall, "put_ms": acc["put_ms"],
        "worker_idle_ms": acc["get_wait_ms"], "compute_sum_ms": acc["compute_ms"],
        "merge_collect_ms": merge_wait_ms, "merge_ms": merge_ms,
        "overlap_ms": max(0.0, serial_sum - wall),
        "overlap_ratio": max(0.0, serial_sum - wall) / max(serial_sum, 1e-9),
        "throughput_rows_s": sum(b["keys"].size for b in blocks) / max(wall / 1e3, 1e-9),
        "qmax": qmax, "nworkers": nworkers,
    }


def pareto_front(rows):
    """Skyline on (ms, bytes): row kept iff no other row beats it on both."""
    front = []
    for r in rows:
        dominated = any(
            o["ms"] <= r["ms"] and o["bytes"] <= r["bytes"]
            and (o["ms"] < r["ms"] or o["bytes"] < r["bytes"]) for o in rows)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: (r["ms"], r["bytes"]))


def best_per_regime(rows):
    best = {}
    for r in rows:
        k = r["regime"]
        if k not in best or (r["ms"], r["bytes"]) < (best[k]["ms"], best[k]["bytes"]):
            best[k] = r
    return best
