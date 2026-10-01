# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Zero-copy views over one raw npz blob buffer (no per-array alloc+memcpy).

Blob is ZIP_STORED (uncompressed): payload bytes already lie contiguous
in `raw`. Central directory (KBs at blob tail) gives each entry offset;
npy header gives dtype/shape; columns become np.frombuffer views into
the single `raw` buffer. `raw` stays alive via view .base chain.

Modes: hybrid (default: int32 views + aligned v3 copy, no compute tax),
zero (all 8 views, read-only, int keys exact, v3 unaligned tax ~+80ms
per 10M block on x86), copy (old NpzFile path via kernel alias).
Format NFS untouched (read as-is). No Core/Planner/kernel change.
"""
import ast
import struct

import numpy as np

COLS = ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")
_INT32_COLS = ("k1", "k2", "k3", "id4", "id6", "v1", "v2")
_EOCD = struct.Struct("<IHHHHIIH")
_EOCD_SIG = b"PK\x05\x06"
_CDIR_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"
_NPY_MAGIC = b"\x93NUMPY"


def _entries(raw):
    """Central-directory scan -> [(name, local_header_off)] (no data copy)."""
    eocd = raw.rfind(_EOCD_SIG)
    if eocd < 0 or len(raw) - eocd < _EOCD.size:
        raise ValueError("zview: EOCD not found (not np.savez blob?)")
    (_, _, _, n1, n2, _, cdoff, _) = _EOCD.unpack(raw[eocd:eocd + _EOCD.size])
    if not 0 < n2 <= 64 or n1 != n2:
        raise ValueError(f"zview: bad central-dir count {n1}/{n2}")
    pos, out = int(cdoff), []
    for _ in range(n2):
        if raw[pos:pos + 4] != _CDIR_SIG:
            raise ValueError("zview: central-dir entry sig mismatch")
        (fnlen, extralen, comlen) = struct.unpack("<HHH", raw[pos + 28:pos + 34])
        lho = struct.unpack("<I", raw[pos + 42:pos + 46])[0]
        out.append((raw[pos + 46:pos + 46 + fnlen].decode("ascii"), lho))
        pos += 46 + fnlen + extralen + comlen
    return out


def _view(raw, lho):
    """One entry -> (key, array view into raw)."""
    if raw[lho:lho + 4] != _LOCAL_SIG:
        raise ValueError("zview: local header sig mismatch")
    (fnlen, extralen) = struct.unpack("<HH", raw[lho + 26:lho + 30])
    start = lho + 30 + fnlen + extralen
    if raw[start:start + 6] != _NPY_MAGIC:
        raise ValueError("zview: npy magic mismatch (format changed?)")
    hlen = struct.unpack("<H", raw[start + 8:start + 10])[0]
    info = ast.literal_eval(raw[start + 10:start + 10 + hlen].decode("latin1").strip())
    if info.get("fortran_order"):
        raise ValueError("zview: fortran order unsupported (no silent copy)")
    dt = np.dtype(info["descr"])
    shape = tuple(int(s) for s in info["shape"])
    if len(shape) != 1:
        raise ValueError(f"zview: non-1D shape {shape}")
    off = start + 10 + hlen
    return np.frombuffer(raw, dtype=dt, count=shape[0], offset=off).reshape(shape)


def read_block_impl(handle, idx, mode="hybrid", fallback=None):
    """Block arrays {col: ndarray}; old path kept under mode='copy'."""
    if mode == "copy":
        if fallback is None:
            raise ValueError("zview: mode='copy' needs kernel nfs_stream_read_block")
        return fallback(handle, int(idx))
    if mode not in ("hybrid", "zero"):
        raise ValueError(f"zview: mode must be hybrid/zero/copy, got {mode!r}")
    idx = int(idx)
    if idx in handle.get("blocks", {}):
        return handle["blocks"][idx]
    m = handle["index"][idx]
    with open(handle["path"], "rb") as fh:
        fh.seek(int(m["file_offset"]))
        raw = fh.read(int(m["blob_len"]))
    if len(raw) != int(m["blob_len"]):
        raise ValueError(f"zview: short read block {idx}")
    found = _entries(raw)
    if [n for n, _ in found] != [c + ".npy" for c in COLS]:
        raise ValueError(f"zview: names {[n for n, _ in found]} != {list(COLS)}")
    block = {}
    for name, lho in found:
        key = name[:-4]
        arr = _view(raw, lho)
        if key == "v3":
            if arr.dtype != np.dtype("<f8"):
                raise ValueError(f"zview: v3 dtype {arr.dtype}")
            block[key] = arr if mode == "zero" else np.array(arr, dtype=np.float64, copy=True)
        elif key in _INT32_COLS:
            if arr.dtype != np.dtype("<i4"):
                raise ValueError(f"zview: {key} dtype {arr.dtype}")
            block[key] = arr
        else:
            raise ValueError(f"zview: unexpected column {key}")
    expect = int(m["n"])
    for c in COLS:
        if block[c].shape != (expect,):
            raise ValueError(f"zview: {c} shape {block[c].shape} != ({expect},)")
    return block
