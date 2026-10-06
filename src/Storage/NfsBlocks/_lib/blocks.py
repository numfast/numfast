# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""NFS block-container v1 (SPEC-DELTA storage, beside npz-staging-v0).

Layout (directory container `<name>.nfs/`):
  header.json      {format, schema, N, columns, block_rows, nblocks, csv}
  dictionary.json  {id1/id2/id3: {prefix, width}} (pattern sidecars, resident)
  index.json       [{block_id,row_start,row_end,n,file,mins,maxs,refs}]
  blocks/block_XXXXX.npz  {k1,k2,k3,id4,id6,v1,v2,v3} int32/float64

Rules: header+dictionary+index load ONCE (resident, KBs for G1 pattern
dataset); data blocks lazy via LRU (default 3). Encode via existing
resident_prepare (pattern row-local => per-block codes == global codes,
keys comparable across blocks). No Planner/Kernel/ClickBench change.
"""
import collections
import gc
import json
import time
from pathlib import Path

import numpy as np

FORMAT = "nfs-block-v1"
COLS = ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")


def _err(format_error, what, fix="", doc=""):
    if format_error is not None:
        raise format_error(what, fix=fix, doc=doc)
    msg = str(what)
    if fix:
        msg += f" Fix: {fix}."
    return ValueError(msg + (f" See {doc}" if doc else ""))


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return -1.0


def build_impl(csv_path, out_dir, block_rows, resident_prepare,
               format_error=None, limit_rows=None, resume=True):
    """Stream CSV -> NFS blocks (one pass, resumable). Returns stats dict."""
    err = format_error
    csv_path, out_dir = str(csv_path), Path(str(out_dir))
    br = int(block_rows)
    if br <= 0:
        raise _err(err, f"nfs_build: block_rows must be >0, got {br}",
                   fix="pass block_rows=10000000")
    import pandas as pd
    import pyarrow as pa
    blocks_d = out_dir / "blocks"
    blocks_d.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    # resume: collect existing block metas
    metas, widths = {}, {}
    if resume:
        for mf in sorted(blocks_d.glob("block_*.json")):
            try:
                with open(mf) as fh:
                    m = json.load(fh)
                metas[int(m["block_id"])] = m
                for k in ("id1", "id2", "id3"):
                    w = (m.get("pattern") or {}).get(k, {}).get("width")
                    if w is not None:
                        widths[k] = max(widths.get(k, 0), int(w))
            except Exception:
                pass
    row_start, block_id, done_rows = 0, 0, 0
    # skip already-done prefix: find contiguous done blocks from 0
    while block_id in metas:
        row_start += int(metas[block_id]["n"])
        done_rows += int(metas[block_id]["n"])
        block_id += 1
    skip_blocks = block_id
    if skip_blocks:
        print(f"nfs_build: resume skip {skip_blocks} blocks ({done_rows} rows)",
              flush=True)
    # stream reader; skip rows already done WITHOUT huge range list:
    # header row + done_rows body rows skipped via int skiprows + header=None.
    usecols = ["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"]
    if done_rows or limit_rows is not None:
        reader = pd.read_csv(csv_path, header=None, names=usecols,
                             usecols=[0, 1, 2, 3, 5, 6, 7, 8], chunksize=br,
                             skiprows=int(done_rows) + 1,
                             nrows=(int(limit_rows) if limit_rows is not None else None))
    else:
        reader = pd.read_csv(csv_path, usecols=usecols, chunksize=br)
    n_total, v1_tot, v2_tot, v3_tot = done_rows, 0, 0, 0.0
    for m in metas.values():
        v1_tot += int(m["refs"]["v1"])
        v2_tot += int(m["refs"]["v2"])
        v3_tot += float(m["refs"]["v3"])
    blocks_done, csv_ms, prep_ms, save_ms = skip_blocks, 0.0, 0.0, 0.0
    hrs_start = time.perf_counter()
    for chunk in reader:
        if limit_rows is not None and n_total >= done_rows + int(limit_rows):
            break
        t0 = time.perf_counter()
        n = len(chunk)
        csv_ms += (time.perf_counter() - t0) * 1000  # read done by iterator
        t0 = time.perf_counter()
        s1 = pa.array(chunk["id1"].to_numpy(), type=pa.string())
        s2 = pa.array(chunk["id2"].to_numpy(), type=pa.string())
        s3 = pa.array(chunk["id3"].to_numpy(), type=pa.string())
        id4 = chunk["id4"].to_numpy().astype(np.int32)
        id6 = chunk["id6"].to_numpy().astype(np.int32)
        v1 = chunk["v1"].to_numpy().astype(np.int32)
        v2 = chunk["v2"].to_numpy().astype(np.int32)
        v3 = chunk["v3"].to_numpy().astype(np.float64)
        t1 = time.perf_counter()
        r1 = resident_prepare({"id1": {"values": s1, "prefix": "id"}})
        del s1
        gc.collect()
        r2 = resident_prepare({"id2": {"values": s2, "prefix": "id"}})
        del s2
        gc.collect()
        r3 = resident_prepare({"id3": {"values": s3, "prefix": "id"}})
        del s3
        gc.collect()
        rn = resident_prepare({
            "id4": {"values": id4, "dtype": "int32"},
            "id6": {"values": id6, "dtype": "int32"},
            "v1": {"values": v1, "dtype": "int32"},
            "v2": {"values": v2, "dtype": "int32"},
            "v3": {"values": v3, "dtype": "float64"}})
        # zero-copy contract (numeric views, like bench_resident_1B)
        assert np.shares_memory(rn["v1"]["codes"], v1)
        k1 = np.asarray(r1["id1"]["codes"])
        k2 = np.asarray(r2["id2"]["codes"])
        k3 = np.asarray(r3["id3"]["codes"])
        for k in ("id1", "id2", "id3"):
            w = {"id1": r1, "id2": r2, "id3": r3}[k][k].get("pattern", {}).get("width")
            if w is not None:
                widths[k] = max(widths.get(k, 0), int(w))
        prep_ms += (time.perf_counter() - t1) * 1000
        v1s, v2s, v3s = int(v1.astype(np.int64).sum()), int(v2.astype(np.int64).sum()), float(v3.sum())
        t2 = time.perf_counter()
        bf = blocks_d / f"block_{block_id:05d}.npz"
        np.savez(str(bf), k1=k1, k2=k2, k3=k3,
                 id4=np.asarray(rn["id4"]["codes"]), id6=np.asarray(rn["id6"]["codes"]),
                 v1=np.asarray(rn["v1"]["codes"]), v2=np.asarray(rn["v2"]["codes"]),
                 v3=np.asarray(rn["v3"]["codes"]))
        # per-block min/max (C-speed, for METADATA/INDEX)
        mins = {c: int(np.asarray(a).min()) for c, a in
                (("k1", k1), ("k2", k2), ("k3", k3),
                 ("id4", id4), ("id6", id6), ("v1", v1), ("v2", v2))}
        maxs = {c: int(np.asarray(a).max()) for c, a in
                (("k1", k1), ("k2", k2), ("k3", k3),
                 ("id4", id4), ("id6", id6), ("v1", v1), ("v2", v2))}
        mins["v3"], maxs["v3"] = float(np.asarray(v3).min()), float(np.asarray(v3).max())
        meta = {"block_id": block_id, "row_start": row_start,
                "row_end": row_start + n, "n": n, "file": bf.name,
                "mins": mins, "maxs": maxs,
                "pattern": {k: {"prefix": "id", "width": widths.get(k)} for k in ("id1", "id2", "id3")},
                "refs": {"v1": v1s, "v2": v2s, "v3": v3s}}
        with open(blocks_d / f"block_{block_id:05d}.json", "w") as fh:
            json.dump(meta, fh)
        metas[block_id] = meta
        save_ms += (time.perf_counter() - t2) * 1000
        row_start += n
        n_total += n
        v1_tot += v1s
        v2_tot += v2s
        v3_tot += v3s
        blocks_done += 1
        print(f"nfs_build block {block_id}: n={n} v1={v1s} rss={_rss_gb():.2f}GB", flush=True)
        block_id += 1
        del chunk, k1, k2, k3, id4, id6, v1, v2, v3, r1, r2, r3, rn
        gc.collect()
        # per-stage time cap courtesy: caller uses timeout 550 externally
    nblocks = len(metas)
    header = {"format": FORMAT, "schema": "G1",
              "columns": [{"name": c, "dtype": ("float64" if c == "v3" else "int32"),
                           "physical": ("float64" if c == "v3" else "int32")} for c in COLS],
              "N": int(n_total), "block_rows": br, "nblocks": int(nblocks),
              "csv": csv_path}
    dictionary = {k: {"prefix": "id", "width": widths.get(k),
                      "method": "pattern", "ids_stable": True} for k in ("id1", "id2", "id3")}
    index = [metas[i] for i in sorted(metas)]
    with open(out_dir / "header.json", "w") as fh:
        json.dump(header, fh, indent=1)
    with open(out_dir / "dictionary.json", "w") as fh:
        json.dump({"resident_dictionaries": dictionary,
                   "note": "pattern columns: O(1) formula, no materialized codebook"}, fh, indent=1)
    with open(out_dir / "index.json", "w") as fh:
        json.dump(index, fh, indent=1)
    stats = {"out": str(out_dir), "format": FORMAT, "N": int(n_total),
             "nblocks": int(nblocks), "block_rows": br,
             "refs": {"v1": int(v1_tot), "v2": int(v2_tot), "v3": float(v3_tot)},
             "csv_ms": csv_ms, "prep_ms": prep_ms, "save_ms": save_ms,
             "total_ms": (time.perf_counter() - t_all) * 1000,
             "stream_ms": (time.perf_counter() - hrs_start) * 1000,
             "rss_gb": _rss_gb()}
    with open(out_dir / "build.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"nfs_build done N={n_total} blocks={nblocks} "
          f"v1={v1_tot} total={(time.perf_counter()-t_all)*1000:.0f}ms rss={_rss_gb():.2f}GB",
          flush=True)
    return stats


def open_impl(out_dir, cache_blocks=3):
    """Load header+dictionary+index ONCE (resident). Data blocks lazy LRU."""
    root = Path(str(out_dir))
    with open(root / "header.json") as fh:
        header = json.load(fh)
    with open(root / "dictionary.json") as fh:
        dictionary = json.load(fh)
    with open(root / "index.json") as fh:
        index = json.load(fh)
    return {"root": root, "header": header, "dictionary": dictionary,
            "index": index, "cache": collections.OrderedDict(),
            "cache_max": max(1, int(cache_blocks)),
            "hits": 0, "misses": 0}


def read_block_impl(handle, idx):
    """LRU block read (2-3 blocks resident). Returns {col: np.ndarray}."""
    idx = int(idx)
    c = handle["cache"]
    if idx in c:
        c.move_to_end(idx)
        handle["hits"] += 1
        return c[idx]
    meta = handle["index"][idx]
    z = np.load(str(handle["root"] / "blocks" / meta["file"]))
    block = {k: np.asarray(z[k]) for k in COLS}
    c[idx] = block
    handle["misses"] += 1
    while len(c) > handle["cache_max"]:
        c.popitem(last=False)
    return block
