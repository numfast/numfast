# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reader breakdown + pipeline overlap probe (bench-only, no core change).

Splits existing loader path (seek+read / np.load / asarray mat) per block,
measures queue occupancy + put/get waits + merge + overlap on Q3 compute.
Format NFS untouched. 1B CSV never read. First 10 blocks of 1B stream NFS.

Usage (Git Bash, sequential, timeout 550):
  PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_stream_reader_profile.py
"""
import gc
import io
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

import numpy as np  # noqa: E402

F1B = FORK / "scratch" / "nfs_stream" / "G1_1e9.stream.nfs"
COLS = ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")
NPROBE = 10
STOP = object()
SEED = 42
np.random.seed(SEED)


def build_kernel():
    from builder import MAIN
    return MAIN["build"](str(FORK))


def manual_read(path, m):
    """Same bytes as read_block_impl, split into 3 timed stages."""
    off, blen = int(m["file_offset"]), int(m["blob_len"])
    s = time.perf_counter()
    with open(str(path), "rb") as fh:
        fh.seek(off)
        raw = fh.read(blen)
    t_disk = (time.perf_counter() - s) * 1000
    assert len(raw) == blen, (len(raw), blen)
    s = time.perf_counter()
    z = np.load(io.BytesIO(raw))
    t_decode = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    b = {k: np.asarray(z[k]) for k in COLS}
    z.close()
    t_mat = (time.perf_counter() - s) * 1000
    return b, t_disk, t_decode, t_mat, len(raw)


def main():
    kernel = build_kernel()
    a = kernel.alias
    h = a["nfs_stream_open"](str(F1B), force_lazy=True)
    idx = h["index"]
    n = min(NPROBE, len(idx))
    print(f"open mode={h['mode']} blocks={len(idx)} "
          f"probe={n} blob/block~{idx[0]['blob_len']/1e6:.0f}MB", flush=True)

    # integrity: manual path == kernel read_block on block 0 (exact)
    b_ref = a["nfs_stream_read_block"](h, 0)
    b_man, _, _, _, _ = manual_read(str(F1B), idx[0])
    for k in COLS:
        assert np.array_equal(np.asarray(b_ref[k]), b_man[k]), k
    assert int(b_man["k1"].size) == idx[0]["n"]
    print(f"parity block0 OK n={idx[0]['n']} "
          f"v1sum={int(b_man['v1'].astype(np.int64).sum())}", flush=True)
    del b_ref, b_man
    gc.collect()

    # Stage A: reader breakdown per block
    rows = []
    for i in range(n):
        b, td, tz, tm, nbytes = manual_read(str(F1B), idx[i])
        arr_mb = sum(np.asarray(b[k]).nbytes for k in COLS) / 1e6
        rows.append({"i": i, "n": int(b["k1"].size), "blob_MB": nbytes / 1e6,
                     "arr_MB": arr_mb, "disk_ms": td, "decode_ms": tz,
                     "mat_ms": tm, "tot_ms": td + tz + tm})
        print(f"block {i}: blob={nbytes/1e6:.0f}MB arr={arr_mb:.0f}MB "
              f"disk={td:.0f}ms decode={tz:.0f}ms mat={tm:.0f}ms "
              f"tot={td+tz+tm:.0f}ms", flush=True)
        del b
        gc.collect()
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("blob_MB", "arr_MB", "disk_ms", "decode_ms",
                     "mat_ms", "tot_ms")}
    print(f"AVG/block ({n}): " + " ".join(
        f"{k}={v:.1f}" for k, v in agg.items()), flush=True)

    # Stage B: instrumented Q3 pipeline over same n blocks
    M = int(max(m["maxs"]["k3"] for m in idx)) + 1
    print(f"q3 domain k3 M={M}", flush=True)
    q_in, q_out = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    st = {"load_ms": 0.0, "disk_ms": 0.0, "decode_ms": 0.0, "mat_ms": 0.0,
          "put_wait_ms": 0.0, "get_wait_ms": 0.0, "occ_sum": 0.0,
          "occ_n": 0, "occ_max": 0}
    occ_lock = threading.Lock()

    def occ_sample():
        try:
            q = q_in.qsize()
        except Exception:
            return
        with occ_lock:
            st["occ_sum"] += q
            st["occ_n"] += 1
            st["occ_max"] = max(st["occ_max"], q)

    def loader():
        s_all = time.perf_counter()
        for i in range(n):
            m = idx[i]
            off, blen = int(m["file_offset"]), int(m["blob_len"])
            s = time.perf_counter()
            with open(str(F1B), "rb") as fh:
                fh.seek(off)
                raw = fh.read(blen)
            st["disk_ms"] += (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            z = np.load(io.BytesIO(raw))
            st["decode_ms"] += (time.perf_counter() - s) * 1000
            del raw
            s = time.perf_counter()
            b = {k: np.asarray(z[k]) for k in COLS}
            z.close()
            st["mat_ms"] += (time.perf_counter() - s) * 1000
            occ_sample()
            s = time.perf_counter()
            q_in.put((i, b))
            st["put_wait_ms"] += (time.perf_counter() - s) * 1000
            occ_sample()
            del b
            gc.collect()
        st["load_ms"] = (time.perf_counter() - s_all) * 1000
        q_in.put(STOP)

    acc = [(np.zeros(M, np.float64), np.zeros(M, np.float64),
            np.zeros(M, np.float64))]
    t_merge = [0.0]

    def merger():
        while True:
            p = q_out.get()
            if p is STOP:
                return
            s = time.perf_counter()
            s1, c, s3 = acc[0]
            a1, a2, a3 = p
            acc[0] = (s1 + a1, c + a2, s3 + a3)
            t_merge[0] += (time.perf_counter() - s) * 1000

    lt = threading.Thread(target=loader, daemon=True)
    mt = threading.Thread(target=merger, daemon=True)
    t0 = time.perf_counter()
    lt.start()
    mt.start()
    t_comp = 0.0
    nrows = 0
    while True:
        s = time.perf_counter()
        item = q_in.get()
        st["get_wait_ms"] += (time.perf_counter() - s) * 1000
        occ_sample()
        if item is STOP:
            q_out.put(STOP)
            break
        _, b = item
        nrows += int(b["k1"].size)
        s = time.perf_counter()
        k = np.asarray(b["k3"])
        v1 = np.asarray(b["v1"])
        v3 = np.asarray(b["v3"])
        p = (np.bincount(k, weights=v1.astype(np.float64), minlength=M),
             np.bincount(k, minlength=M),
             np.bincount(k, weights=v3, minlength=M))
        del b
        gc.collect()
        q_out.put(p)
        t_comp += (time.perf_counter() - s) * 1000
        occ_sample()
        del p
        gc.collect()
    lt.join()
    mt.join()
    wall = (time.perf_counter() - t0) * 1000
    occ_avg = st["occ_sum"] / max(1, st["occ_n"])
    load = st["disk_ms"] + st["decode_ms"] + st["mat_ms"]
    ssum = load + t_comp
    overlap = ssum - (wall - t_merge[0])
    # wall includes merge overlapped too; report both bases
    overlap_full = (load + t_comp + t_merge[0]) - wall
    chk = int(acc[0][0].sum())
    print(f"PIPE n={n} rows={nrows} chk_v1={chk}", flush=True)
    print(f"load stages: disk={st['disk_ms']:.0f} decode={st['decode_ms']:.0f} "
          f"mat={st['mat_ms']:.0f} sum={load:.0f} loader_wall={st['load_ms']:.0f}",
          flush=True)
    print(f"queue: occ_avg={occ_avg:.2f} occ_max={st['occ_max']} "
          f"put_wait={st['put_wait_ms']:.0f}ms get_wait={st['get_wait_ms']:.0f}ms",
          flush=True)
    print(f"compute={t_comp:.0f}ms merge={t_merge[0]:.0f}ms wall={wall:.0f}ms",
          flush=True)
    print(f"overlap(load+comp-wall+merge)={overlap:.0f}ms "
          f"({100*overlap/max(1, ssum):.1f}% of load+comp скрыто); "
          f"overlap_full={overlap_full:.0f}ms", flush=True)
    out = {"seed": SEED, "nblocks_probe": n, "M": M,
           "breakdown_avg": agg, "breakdown_rows": rows,
           "pipe": {"disk_ms": st["disk_ms"], "decode_ms": st["decode_ms"],
                    "mat_ms": st["mat_ms"], "load_sum_ms": load,
                    "loader_wall_ms": st["load_ms"],
                    "put_wait_ms": st["put_wait_ms"],
                    "get_wait_ms": st["get_wait_ms"],
                    "occ_avg": occ_avg, "occ_max": st["occ_max"],
                    "compute_ms": t_comp, "merge_ms": t_merge[0],
                    "wall_ms": wall, "rows": nrows, "chk_v1": chk,
                    "overlap_ms": overlap,
                    "overlap_pct": 100 * overlap / max(1, ssum),
                    "overlap_full_ms": overlap_full}}
    with open(FORK / "tests" / "heavy" / "bench_stream_reader_profile.json",
              "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("JSON written", flush=True)


if __name__ == "__main__":
    main()
