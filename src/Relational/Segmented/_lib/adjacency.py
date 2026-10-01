# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P2 adjacency_slice: thin wrapper over Rust canonical (native-first).

Вход: indptr[V+1] u32 монотонные + indices[E] u32 + query[k] u32.
Выход: begins[k] + ends[k] (u32); flat собирается существующим
gather-путём (native gather при u32, иначе take/concatenate).
UINT32_MAX reserved INF/INVALID -> ValueError. V/E/k=0 safe.

Канон: Rust adjacency_slice/gather. Python — только совместимость:
вызов native при наличии, иначе bit-exact fallback. Коэрсия первой
(reserved-проверка до k==0 early return — паритет с Rust).
"""

import importlib.util
import numpy as np
from pathlib import Path

_INF = np.int64(4294967295)

_NATIVE = None
_NATIVE_OK = None


def _native_cpu():
    """Canonical Rust backend (native_cpu.py by file path, cached)."""
    global _NATIVE, _NATIVE_OK
    if _NATIVE_OK is not None:
        return _NATIVE
    _NATIVE, _NATIVE_OK = None, False
    try:
        here = Path(__file__).resolve()
        cand = (here.parents[3] / "Drivers" / "CPU" / "_lib"
                / "native_cpu.py")
        if cand.is_file():
            spec = importlib.util.spec_from_file_location(
                "_adj_native_cpu", str(cand))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _NATIVE, _NATIVE_OK = mod, True
    except Exception:
        _NATIVE, _NATIVE_OK = None, False
    return _NATIVE


def _as_u32_ids(a, name):
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError(f"adjacency {name} ndim != 1")
    if x.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if x.dtype.kind not in "iu":
        raise ValueError(f"adjacency {name} dtype {x.dtype} not u32")
    x64 = x.astype(np.int64)
    if bool(np.any(x64 == _INF)):
        raise ValueError(f"adjacency {name} holds UINT32_MAX (INF/INVALID)")
    if bool(np.any((x64 < 0) | (x64 > 4294967294))):
        raise ValueError(f"adjacency {name} out of u32 range")
    return x64


def _fb_slice(ip, ix, q):
    if ip.shape[0] == 0:
        v = 0
        if ix.shape[0] != 0:
            raise ValueError("adjacency indptr empty but indices non-empty")
    else:
        v = int(ip.shape[0]) - 1
    e = int(ix.shape[0])
    k = int(q.shape[0])
    if k == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32)
    if v == 0:
        raise ValueError("adjacency query on empty graph (V=0)")
    if bool(np.any(np.diff(ip) < 0)):
        raise ValueError("adjacency indptr not monotone")
    if bool(np.any((ip < 0) | (ip > e))):
        raise ValueError("adjacency indptr outside [0, E]")
    if int(ip[-1]) != e:
        raise ValueError(f"adjacency indptr[-1]={int(ip[-1])} != E={e}")
    if bool(np.any((q < 0) | (q >= v))):
        raise ValueError("adjacency query vertex out of range")
    begins = ip[q].astype(np.uint32)
    ends = ip[q + 1].astype(np.uint32)
    return begins, ends


def adjacency_slice(indptr, indices, query):
    """Return (begins, ends) uint32 for each query vertex."""
    ip = _as_u32_ids(indptr, "indptr")
    ix = _as_u32_ids(indices, "indices")
    q = _as_u32_ids(query, "query")
    mod = _native_cpu()
    if mod is not None:
        try:
            if mod.adjacency_available():
                (be, en), _kind = mod.adjacency_slice_native(
                    indptr, indices, query)
                return be, en
        except (ValueError, RuntimeError):
            raise
        except Exception:
            pass
    return _fb_slice(ip, ix, q)


def adjacency_flat(indices, begins, ends):
    """Assemble flat neighbour ids using existing gather semantics (take).

    Native gather при u32-индексах, иначе take/concatenate. Guards
    begins/ends (shape, begins<=ends, ends<=E) — паритет с Rust
    (ValueError на тех же входах).
    """
    ix = np.asarray(indices)
    b = np.asarray(begins).astype(np.int64)
    en = np.asarray(ends).astype(np.int64)
    if b.shape != en.shape:
        raise ValueError("adjacency_flat begins/ends shape mismatch")
    if b.size == 0:
        return np.zeros(0, dtype=ix.dtype if ix.size else np.uint32)
    if bool(np.any(en < b)):
        raise ValueError("adjacency_flat begins > ends")
    if bool(np.any(en > ix.size)):
        raise ValueError("adjacency_flat ends > E")
    if bool(np.any(b < 0)):
        raise ValueError("adjacency_flat begins < 0")
    mod = _native_cpu()
    if mod is not None and ix.dtype == np.dtype(np.uint32):
        try:
            if mod.adjacency_available():
                flat, _kind = mod.adjacency_gather_native(ix, b, en)
                return flat
        except (ValueError, RuntimeError):
            raise
        except Exception:
            pass
    parts = [np.take(ix, np.arange(int(s), int(t))) for s, t in zip(
        b.tolist(), en.tolist()) if int(t) > int(s)]
    if not parts:
        return np.zeros(0, dtype=ix.dtype)
    return np.concatenate(parts)
