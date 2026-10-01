# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P1 segmented_reduce: thin wrapper over Rust canonical (native-first).

Вход: values[n] (int32 / float32) + bounds[M+1] (u32 монотонные).
Выход: out[M]. ops: sum | count | min | max (mean НЕ op -> ValueError).
count -> uint32; sum f32 -> float32 IEEE-propagate (NaN через add);
sum i32 -> int32 saturating (host-аккумулятор int64, clamp);
min/max пустого сегмента -> ValueError; n=0 -> M=0;
лимит n<=4194240 на dispatch (иначе ValueError, chunkable снаружи).
Вариант B (ключи) — через существующий Sort-путь, не здесь.

Канон: Rust sequential lanes (strided order). Python — только
совместимость: вызов native при наличии, иначе bit-exact fallback
(sequential python-циклы; reduceat только как последний ресорт,
т.к. reduceat pairwise расходится с sequential до ~1ulp на длинных
сегментах). Trailing-empty bounds ([0,5,5]) — pad values одним zero
lane для sum-проб (иначе reduceat IndexError).
"""

import importlib.util
import numpy as np
from pathlib import Path

MAX_DISPATCH_N = 4194240
_I32MIN = np.int64(-2147483648)
_I32MAX = np.int64(2147483647)

def _seg_sum_i32_np(vals, bounds, out):
    for g in range(bounds.shape[0] - 1):
        s = bounds[g]
        e = bounds[g + 1]
        acc = np.int64(0)
        for i in range(s, e):
            acc += np.int64(vals[i])
        if acc > _I32MAX:
            acc = _I32MAX
        elif acc < _I32MIN:
            acc = _I32MIN
        out[g] = np.int32(acc)

def _seg_sum_f32_np(vals, bounds, out):
    m = bounds.shape[0] - 1
    for g in range(m):
            s = bounds[g]
            e = bounds[g + 1]
            acc = np.float32(0.0)
            for i in range(s, e):
                acc += vals[i]
            out[g] = acc


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
                "_seg_native_cpu", str(cand))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _NATIVE, _NATIVE_OK = mod, True
    except Exception:
        _NATIVE, _NATIVE_OK = None, False
    return _NATIVE


def _as_values(v):
    a = np.asarray(v)
    if a.ndim != 1:
        raise ValueError(f"segmented values ndim != 1 ({a.ndim})")
    if a.dtype == np.int32 or a.dtype == np.float32:
        return a
    if a.dtype == np.int64:
        if a.size and (int(a.min()) < -2147483648 or int(a.max()) > 2147483647):
            raise ValueError("segmented values int64 out of int32 range")
        return a.astype(np.int32)
    if a.dtype == np.float64:
        return a.astype(np.float32)
    if a.dtype.kind in "iu":
        info = np.iinfo(a.dtype)
        if info.min < -2147483648 or info.max > 2147483647:
            tmp = a.astype(np.int64)
            if tmp.size and (int(tmp.min()) < -2147483648
                             or int(tmp.max()) > 2147483647):
                raise ValueError("segmented values out of int32 range")
            return tmp.astype(np.int32)
        return a.astype(np.int32)
    if a.dtype.kind == "f":
        return a.astype(np.float32)
    raise ValueError(f"segmented values dtype {a.dtype} not f32/i32")


def _as_bounds(b, n):
    a = np.asarray(b)
    if a.ndim != 1:
        raise ValueError("segmented bounds ndim != 1")
    if a.size == 0:
        if n == 0:
            return np.zeros(1, dtype=np.int64)
        raise ValueError("segmented bounds empty with n>0")
    if a.dtype.kind not in "iu":
        raise ValueError(f"segmented bounds dtype {a.dtype} not u32")
    if a.dtype == np.uint32:
        bad = a.astype(np.int64)
    else:
        bad = a.astype(np.int64)
        if bad.size and (int(bad.min()) < 0 or int(bad.max()) > 4294967294):
            raise ValueError("segmented bounds out of u32 range")
    b64 = bad
    if b64.size and int(b64[0]) < 0:
        raise ValueError("segmented bounds[0] < 0")
    if b64.size and int(b64[-1]) != int(n):
        raise ValueError(
            f"segmented bounds[-1]={int(b64[-1])} != n={int(n)}")
    if b64.size and bool(np.any(np.diff(b64) < 0)):
        raise ValueError("segmented bounds not monotone")
    if b64.size and bool(np.any((b64 < 0) | (b64 > n))):
        raise ValueError("segmented bounds outside [0, n]")
    return b64


def _fb_sum_i32(v, b64, m):
    out = np.empty(m, dtype=np.int32)
    if m == 0:
        return out
    _seg_sum_i32_np(v, b64, out)
    return out


def _fb_sum_f32(v, b64, m):
    if m == 0:
        return np.empty(0, dtype=np.float32)
    out = np.empty(m, dtype=np.float32)
    _seg_sum_f32_np(v, b64, out)
    return out


def segmented_reduce(values, bounds, op):
    """Reduce segments [bounds[i], bounds[i+1]) of values with op."""
    if op not in ("sum", "count", "min", "max"):
        raise ValueError(f"segmented op {op!r} not in sum/count/min/max")
    vals = _as_values(values)
    n = int(vals.shape[0])
    if n > MAX_DISPATCH_N:
        raise ValueError(
            f"segmented n={n} > {MAX_DISPATCH_N}: chunkable wrapper outside")
    b64 = _as_bounds(bounds, n)
    m = int(b64.shape[0]) - 1
    if n == 0:
        if m != 0:
            raise ValueError(f"segmented n=0 requires M=0, got M={m}")
        if op == "count":
            return np.zeros(0, dtype=np.uint32)
        if vals.dtype == np.int32 or vals.dtype.kind in "iu":
            dt = vals.dtype if vals.dtype == np.int32 else np.int32
            return np.zeros(0, dtype=dt)
        return np.zeros(0, dtype=np.float32)
    # Native-first (Rust canonical); validation above owns error messages.
    mod = _native_cpu()
    if mod is not None:
        try:
            if op == "count" and mod.segment_available():
                out, _be = mod.segmented_reduce_native(vals, bounds, op)
                return out
            elif op != "count" and mod.segment_available():
                out, _be = mod.segmented_reduce_native(vals, bounds, op)
                return out
        except (ValueError, RuntimeError):
            raise
        except Exception:
            pass
    if op == "count":
        return np.diff(b64).astype(np.uint32)
    if vals.dtype == np.int32 or vals.dtype.kind in "iu":
        v = vals if vals.dtype == np.int32 else vals.astype(np.int32)
        if op == "sum":
            return _fb_sum_i32(v, b64, m)
        if op in ("min", "max"):
            if m == 0:
                return np.empty(0, dtype=np.int32)
            if bool(np.any(np.diff(b64) == 0)):
                raise ValueError("segmented min/max of empty group")
            fn = np.minimum if op == "min" else np.maximum
            return fn.reduceat(v, b64[:-1].astype(np.int64)).astype(np.int32)
    else:
        v = vals if vals.dtype == np.float32 else vals.astype(np.float32)
        if op == "sum":
            return _fb_sum_f32(v, b64, m)
        if op in ("min", "max"):
            if m == 0:
                return np.empty(0, dtype=np.float32)
            if bool(np.any(np.diff(b64) == 0)):
                raise ValueError("segmented min/max of empty group")
            fn = np.minimum if op == "min" else np.maximum
            return fn.reduceat(v, b64[:-1].astype(np.int64)).astype(np.float32)
    raise ValueError(f"segmented unreachable op={op!r}")
