# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""NFS single-file stream v1 (SPEC-DELTA storage, beside nfs-block-v1 dir).

BLOCK LAYOUT (as-is in memory, zero transforms on save/load):
  k1,k2,k3 : int32 pattern codes (schema prefix-trim "id" + int body,
             encode_pattern row-local; G1: id1/id2 width 3 -> 1..100,
             id3 width 10 -> 1..10000000; validity all-true, no nulls)
  id4,id6,v1,v2 : int32 codes == values (schema scale=1/offset=0 identity,
             series zero-copy views; NO bitpack active on this path)
  v3 : float64 codes == values (scale=1/offset=0 identity; NO per-block
       float scale/offset active on this path)
  SPEC-DELTA note: plan assumed per-block float scale+offset + int bitpack;
  existing in-memory code has neither active (raw int32/float64). Saved
  as-is: read -> ready, zero transforms. Bitpack/scale stay future work.
Per-block meta: n, mins/maxs per col, pattern widths, refs (v1/v2 int sums,
v3 float sum). Blob = np.savez bytes of the 8 arrays (exact dtypes).

FILE FORMAT:
  [HEADER 128B stub][BLOCK...][SERVICE][patch HEADER]
  HEADER: magic b'NFS1', version u32=1, block_rows u64, nblocks u64,
          N u64, ncols u32, service_offset u64, service_len u64, pad 0.
  BLOCK: u32 blob_len + blob(npz k1..v3) + u32 meta_len + meta(json).
  SERVICE: u64 json_len + json{format,schema,columns,N,block_rows,nblocks,
           csv,index,dictionary,refs,stats} + trailer magic b'NFS1'.
  Writer appends blocks one pass (RAM constant), writes SERVICE at end,
  then seeks to 0 and patches nblocks/N/service_offset/service_len.
  Resume: scan len-prefixed blocks from offset 128 (no SERVICE needed),
  skip done prefix in CSV, append.
"""
import io
import json
import struct
import time
from pathlib import Path

import numpy as np

FORMAT = "nfs-stream-v1"
MAGIC = b"NFS1"
VERSION = 1
HEADER_LEN = 128
COLS = ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")
_HDR = struct.Struct("<4sIQQQIQ Q")
assert _HDR.size <= HEADER_LEN

_INT32 = ("k1", "k2", "k3", "id4", "id6", "v1", "v2")


def _err(format_error, what, fix="", doc="specs/09-serialization-nfs.md"):
    if format_error is not None:
        raise format_error(what, fix=fix, doc=doc)
    msg = str(what)
    if fix:
        msg += f" Fix: {fix}."
    return ValueError(msg + f" See {doc}")


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return -1.0


def _pack_header(block_rows, nblocks, n, service_offset, service_len):
    raw = _HDR.pack(MAGIC, VERSION, int(block_rows), int(nblocks), int(n),
                    len(COLS), int(service_offset), int(service_len))
    return raw + bytes(HEADER_LEN - len(raw))


def _read_header(fh):
    fh.seek(0)
    raw = fh.read(HEADER_LEN)
    if len(raw) < _HDR.size:
        raise ValueError("nfs_stream: truncated header")
    magic, ver, br, nb, n, nc, soff, slen = _HDR.unpack(raw[:_HDR.size])
    if magic != MAGIC:
        raise ValueError(f"nfs_stream: bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"nfs_stream: version {ver} != {VERSION}")
    return {"block_rows": br, "nblocks": nb, "N": n, "ncols": nc,
            "service_offset": soff, "service_len": slen}


def _byte_offset(csv_path, body_rows_to_skip):
    """Byte offset of body row #body_rows_to_skip (header + rows skipped).

    Buffered b'\\n' count (C-speed, ~GB/s, O(MBs) RAM) instead of pandas
    skiprows (which costs ~0.7us + ~100B retained per skipped row).
    """
    need = int(body_rows_to_skip) + 1  # header line + body rows
    off = 0
    with open(str(csv_path), "rb") as fh:
        while need > 0:
            buf = fh.read(1 << 24)
            if not buf:
                break
            c = buf.count(b"\n")
            if c >= need:
                idx = -1
                for _ in range(need):
                    idx = buf.index(b"\n", idx + 1)
                off += idx + 1
                need = 0
            else:
                off += len(buf)
                need -= c
    return off


def _scan_blocks(fh):
    """Resume scan: sequential (offset, blob, meta) list; stops at SERVICE."""
    fh.seek(0, 2)
    end = fh.tell()
    out, pos = [], HEADER_LEN
    import struct as _st
    while pos + 4 <= end:
        fh.seek(pos)
        (blen,) = _st.unpack("<I", fh.read(4))
        if blen == 0 or pos + 4 + blen + 4 > end:
            break
        blob_off = pos + 4
        fh.seek(blob_off + blen)
        (mlen,) = _st.unpack("<I", fh.read(4))
        if pos + 4 + blen + 4 + mlen > end:
            break
        meta = json.loads(fh.read(mlen).decode("utf-8"))
        out.append({"offset": blob_off, "blob_len": blen, "meta": meta})
        pos = pos + 4 + blen + 4 + mlen
    return out


def _encode_block(chunk, resident_prepare):
    """DataFrame chunk -> (arrays dict, meta core). Same path as NfsBlocks."""
    import gc
    import pyarrow as pa
    n = len(chunk)
    s1 = pa.array(chunk["id1"].to_numpy(), type=pa.string())
    s2 = pa.array(chunk["id2"].to_numpy(), type=pa.string())
    s3 = pa.array(chunk["id3"].to_numpy(), type=pa.string())
    id4 = chunk["id4"].to_numpy().astype(np.int32)
    id6 = chunk["id6"].to_numpy().astype(np.int32)
    v1 = chunk["v1"].to_numpy().astype(np.int32)
    v2 = chunk["v2"].to_numpy().astype(np.int32)
    v3 = chunk["v3"].to_numpy().astype(np.float64)
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
    assert np.shares_memory(rn["v1"]["codes"], v1)
    arr = {"k1": np.asarray(r1["id1"]["codes"]),
           "k2": np.asarray(r2["id2"]["codes"]),
           "k3": np.asarray(r3["id3"]["codes"]),
           "id4": np.asarray(rn["id4"]["codes"]),
           "id6": np.asarray(rn["id6"]["codes"]),
           "v1": np.asarray(rn["v1"]["codes"]),
           "v2": np.asarray(rn["v2"]["codes"]),
           "v3": np.asarray(rn["v3"]["codes"])}
    widths = {}
    for k, r in (("id1", r1), ("id2", r2), ("id3", r3)):
        w = r[k].get("pattern", {}).get("width")
        if w is not None:
            widths[k] = int(w)
    mins = {c: int(arr[c].min()) for c in _INT32}
    maxs = {c: int(arr[c].max()) for c in _INT32}
    mins["v3"], maxs["v3"] = float(arr["v3"].min()), float(arr["v3"].max())
    core = {"n": int(n), "mins": mins, "maxs": maxs,
            "pattern": {k: {"prefix": "id", "width": widths.get(k)}
                        for k in ("id1", "id2", "id3")},
            "refs": {"v1": int(v1.astype(np.int64).sum()),
                     "v2": int(v2.astype(np.int64).sum()),
                     "v3": float(v3.sum())}}
    return arr, core, widths


def build_impl(csv_path, out_path, block_rows, resident_prepare,
               format_error=None, limit_rows=None, resume=True, max_blocks=None):
    """Stream CSV -> single NFS file, one pass, constant RAM. Returns stats."""
    import gc
    import pandas as pd
    err = format_error
    br = int(block_rows)
    if br <= 0:
        raise _err(err, f"nfs_stream_build: block_rows must be >0, got {br}",
                   fix="pass block_rows=10000000")
    out = Path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    done_blocks, done_rows = [], 0
    if resume and out.exists() and out.stat().st_size >= HEADER_LEN:
        with open(out, "rb") as fh:
            try:
                done_blocks = _scan_blocks(fh)
                done_rows = sum(b["meta"]["n"] for b in done_blocks)
                print(f"nfs_stream_build: resume {len(done_blocks)} blocks "
                      f"({done_rows} rows)", flush=True)
            except Exception:
                done_blocks, done_rows = [], 0
    mode = "r+b" if (resume and out.exists()) else "w+b"
    if mode == "w+b" or out.stat().st_size < HEADER_LEN:
        done_blocks, done_rows = [], 0
    # recompute exact append offset from scan (no SERVICE survives resume)
    pos = HEADER_LEN
    index, widths = [], {}
    for b in done_blocks:
        m = dict(b["meta"])
        m["file_offset"] = b["offset"]
        m["blob_len"] = b["blob_len"]
        index.append(m)
        for k in ("id1", "id2", "id3"):
            w = (m.get("pattern") or {}).get(k, {}).get("width")
            if w is not None:
                widths[k] = max(widths.get(k, 0), int(w))
    # simpler exact append offset: walk file again counting bytes
    omode = "w+b" if mode == "w+b" else "r+b"
    with open(out, omode) as fh:
        if mode == "w+b" or out.stat().st_size < HEADER_LEN:
            fh.seek(0)
            fh.write(_pack_header(br, 0, 0, 0, 0))
            fh.flush()
            pos = HEADER_LEN
        else:
            fh.seek(HEADER_LEN)
            pos = HEADER_LEN
            import struct as _st
            for b in done_blocks:
                fh.seek(pos)
                (blen,) = _st.unpack("<I", fh.read(4))
                fh.seek(pos + 4 + blen)
                (mlen,) = _st.unpack("<I", fh.read(4))
                fh.read(mlen)
                pos = pos + 4 + blen + 4 + mlen
            fh.seek(pos)
            fh.truncate(pos)  # drop stale SERVICE / partial tail
        usecols = ["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"]
        csv_fh = None
        if done_rows or limit_rows is not None:
            if done_rows:
                # fast byte-seek (no pandas skiprows transient); pandas reads
                # body rows from the positioned binary object, header=None.
                t_skip = time.perf_counter()
                csv_fh = open(str(csv_path), "rb")
                csv_fh.seek(_byte_offset(str(csv_path), done_rows))
                print(f"nfs_stream_build: byte-skip {done_rows} rows "
                      f"{(time.perf_counter()-t_skip)*1000:.0f}ms "
                      f"rss={_rss_gb():.2f}GB", flush=True)
                reader = pd.read_csv(csv_fh, header=None, names=usecols,
                                     usecols=[0, 1, 2, 3, 5, 6, 7, 8],
                                     chunksize=br,
                                     nrows=(int(limit_rows) if limit_rows is not None else None))
            else:
                reader = pd.read_csv(str(csv_path), header=None, names=usecols,
                                     usecols=[0, 1, 2, 3, 5, 6, 7, 8],
                                     chunksize=br,
                                     nrows=(int(limit_rows) if limit_rows is not None else None))
        else:
            reader = pd.read_csv(str(csv_path), usecols=usecols, chunksize=br)
        n_total = done_rows
        v1_tot = sum(int(b["meta"]["refs"]["v1"]) for b in done_blocks)
        v2_tot = sum(int(b["meta"]["refs"]["v2"]) for b in done_blocks)
        v3_tot = sum(float(b["meta"]["refs"]["v3"]) for b in done_blocks)
        block_id = len(done_blocks)
        row_start = done_rows
        prep_ms, save_ms = 0.0, 0.0
        made = 0
        for chunk in reader:
            if limit_rows is not None and n_total >= done_rows + int(limit_rows):
                break
            if max_blocks is not None and made >= int(max_blocks):
                break
            n = len(chunk)
            t1 = time.perf_counter()
            arr, core, w = _encode_block(chunk, resident_prepare)
            for k, v in w.items():
                widths[k] = max(widths.get(k, 0), int(v))
            prep_ms += (time.perf_counter() - t1) * 1000
            t2 = time.perf_counter()
            buf = io.BytesIO()
            np.savez(buf, **{k: arr[k] for k in COLS})
            blob = buf.getvalue()
            meta = {"block_id": block_id, "row_start": row_start,
                    "row_end": row_start + n, "n": n,
                    "file_offset": pos + 4, "blob_len": len(blob),
                    "mins": core["mins"], "maxs": core["maxs"],
                    "pattern": core["pattern"], "refs": core["refs"]}
            mb = json.dumps(meta).encode("utf-8")
            fh.write(struct.pack("<I", len(blob)))
            fh.write(blob)
            fh.write(struct.pack("<I", len(mb)))
            fh.write(mb)
            fh.flush()
            save_ms += (time.perf_counter() - t2) * 1000
            pos += 4 + len(blob) + 4 + len(mb)
            index.append(meta)
            row_start += n
            n_total += n
            v1_tot += core["refs"]["v1"]
            v2_tot += core["refs"]["v2"]
            v3_tot += core["refs"]["v3"]
            print(f"nfs_stream_build block {block_id}: n={n} "
                  f"v1={core['refs']['v1']} rss={_rss_gb():.2f}GB", flush=True)
            block_id += 1
            made += 1
            del chunk, arr, core, blob, mb
            gc.collect()
        if csv_fh is not None:
            csv_fh.close()
        nblocks = len(index)
        dictionary = {k: {"prefix": "id", "width": widths.get(k),
                          "method": "pattern", "ids_stable": True}
                      for k in ("id1", "id2", "id3")}
        service = {"format": FORMAT, "schema": "G1",
                   "columns": [{"name": c, "dtype": ("float64" if c == "v3" else "int32"),
                                "physical": ("float64" if c == "v3" else "int32")}
                               for c in COLS],
                   "N": int(n_total), "block_rows": br, "nblocks": int(nblocks),
                   "csv": str(csv_path), "index": index,
                   "dictionary": {"resident_dictionaries": dictionary,
                                  "note": "pattern columns: O(1) formula, no codebook"},
                   "refs": {"v1": int(v1_tot), "v2": int(v2_tot),
                            "v3": float(v3_tot)}}
        sb = json.dumps(service).encode("utf-8")
        service_offset = pos
        fh.write(struct.pack("<Q", len(sb)))
        fh.write(sb)
        fh.write(MAGIC)
        fh.flush()
        fh.seek(0)
        fh.write(_pack_header(br, nblocks, n_total, service_offset, len(sb)))
        fh.close()
    stats = {"out": str(out), "format": FORMAT, "N": int(n_total),
             "nblocks": int(nblocks), "block_rows": br,
             "refs": {"v1": int(v1_tot), "v2": int(v2_tot),
                      "v3": float(v3_tot)},
             "bytes": int(service_offset + 8 + len(sb) + 4),
             "prep_ms": prep_ms, "save_ms": save_ms,
             "total_ms": (time.perf_counter() - t_all) * 1000,
             "rss_gb": _rss_gb()}
    print(f"nfs_stream_build done N={n_total} blocks={nblocks} v1={v1_tot} "
          f"total={(time.perf_counter()-t_all)*1000:.0f}ms rss={_rss_gb():.2f}GB",
          flush=True)
    return stats


def open_impl(path, budget_frac=0.25, force_lazy=False):
    """Header+SERVICE resident; blocks full (fits budget) or lazy (index)."""
    import struct as _st
    with open(str(path), "rb") as fh:
        hdr = _read_header(fh)
        fh.seek(hdr["service_offset"])
        (slen,) = _st.unpack("<Q", fh.read(8))
        service = json.loads(fh.read(slen).decode("utf-8"))
        tail = fh.read(4)
        if tail != MAGIC:
            raise ValueError("nfs_stream: bad service trailer")
    index = service["index"]
    blob_bytes = sum(int(m["blob_len"]) for m in index)
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
    except Exception:
        avail = 4 * 1024 ** 3
    budget = int(avail * float(budget_frac))
    full = (not force_lazy) and (blob_bytes <= budget)
    handle = {"path": str(path), "header": hdr, "service": service,
              "index": index, "mode": "full" if full else "lazy",
              "blocks": {}, "budget_frac": float(budget_frac),
              "blob_bytes": int(blob_bytes), "avail_bytes": int(avail)}
    if full:
        with open(str(path), "rb") as fh:
            for i, m in enumerate(index):
                fh.seek(int(m["file_offset"]))
                z = np.load(io.BytesIO(fh.read(int(m["blob_len"]))))
                handle["blocks"][i] = {k: np.asarray(z[k]) for k in COLS}
    return handle


def read_block_impl(handle, idx):
    """Block arrays {col: ndarray}; zero transforms (as stored)."""
    idx = int(idx)
    if idx in handle["blocks"]:
        return handle["blocks"][idx]
    m = handle["index"][idx]
    with open(handle["path"], "rb") as fh:
        fh.seek(int(m["file_offset"]))
        z = np.load(io.BytesIO(fh.read(int(m["blob_len"]))))
        return {k: np.asarray(z[k]) for k in COLS}


def plan_impl(handle, budget_frac=0.10):
    """Chunk plan from RAM budget (formula, no hardcode)."""
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
    except Exception:
        avail = 4 * 1024 ** 3
    n = len(handle["index"])
    avg_b = max(1, int(handle["blob_bytes"] // max(1, n)))
    budget = int(avail * float(budget_frac))
    per = max(1, budget // avg_b)
    per = min(per, n)
    nchunks = (n + per - 1) // per
    return {"mode": handle["mode"], "avail_bytes": int(avail),
            "budget_bytes": int(budget), "avg_block_bytes": int(avg_b),
            "blocks_per_chunk": int(per), "nchunks": int(nchunks),
            "nblocks": int(n)}
