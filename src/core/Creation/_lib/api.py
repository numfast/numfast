"""Creation public API (S200): zeros/ones/full/arange/linspace/tile/repeat/random_*.

Все операции идут через Runtime execute (canonical path B: Runtime singleton,
Compute.register_all не более одного раза на процесс). Kernel'и: IndexKernel
(mode: 0=const, 1=arange, 2=linspace, 3=tile, 4=repeat), RandomKernel
(mode: 0=uniform, 1=int_range, 2=normal).

v1 упрощения (помечено):
- N > 4_194_240 -> ValueError: chunking (S209) planned.
- только rank-1 shape (int или 1-tuple); N-D planned.
"""

import math

import numpy as np

from .array import CreatedArray

_NP_DTYPE = {"float32": np.float32, "int32": np.int32}
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_MAX_DISPATCH = 65535 * 64  # 4_194_240; лимит принадлежит драйверу (wgpu_driver)

_out_name = "creation_out"
_registered = False


def _ensure_runtime():
    """Runtime singleton (path B) + однократная регистрация Compute."""
    global _registered
    from Runtime.Runtime import _get_runtime
    from Compute import register_all

    rt = _get_runtime()
    if not _registered:
        if len(rt.kernel_table) == 0:
            register_all(rt)
        _registered = True
    return rt


def _check_dtype(dtype):
    if dtype not in _NP_DTYPE:
        raise ValueError(
            f"Creation: dtype {dtype!r} not supported in v1 "
            f"(allowed: {sorted(_NP_DTYPE)})")


def _n_of(shape):
    """shape -> rank-1 длина N. int или 1-tuple."""
    if isinstance(shape, (int, np.integer)):
        n = int(shape)
        if n < 0:
            raise ValueError(f"Creation: shape must be >= 0, got {n}")
        return n
    if (isinstance(shape, tuple) and len(shape) == 1
            and isinstance(shape[0], (int, np.integer))):
        return int(shape[0])
    raise NotImplementedError("Creation: N-D shape planned (rank-1 in v1)")


def _check_n(n):
    if n > _MAX_DISPATCH:
        raise ValueError(
            f"Creation: n={n} > single-dispatch limit {_MAX_DISPATCH}; "
            f"chunking (S209) planned -- v1 simplification")


def _empty(n, dtype):
    """Пустой CreatedArray без dispatch (N=0)."""
    return CreatedArray(np.empty(n, dtype=_NP_DTYPE[dtype]), None, (n,), dtype)


def _run_index_kernel(n, mode, p0, p1, p2, dtype, pat=None, rep=None):
    """IndexKernel через Runtime execute -> CreatedArray.

    dtype="int32" -> нативный i32-путь (op=IndexKernelI32): выходной буфер
    array<i32>, значения точны до 2^31 (лимит f32 2^24 не применяется).
    dtype="float32" -> op=IndexKernel (f32 буферы).

    Pattern (modes 3/4) — обычный GPU-resident ВХОД job'а (любой k,
    лимита 64 нет): inputs=["pat"], данные через source_data execute
    (канонический путь B, как в Operations.scan).
    """
    rt = _ensure_runtime()
    op = "IndexKernelI32" if dtype == "int32" else "IndexKernel"
    params = {"n": n, "mode": mode, "p0": p0, "p1": p1, "p2": p2,
              "k": 1, "rep": 1, "dtype": dtype}
    src_data = {}
    if pat is not None:
        if len(pat) < 1:
            raise ValueError("Creation: pattern must contain at least 1 element")
        pat_arr = np.asarray(pat, dtype=np.float32)
        params["k"] = int(len(pat_arr))
        params["rep"] = 1 if rep is None else int(rep)
        src_data["pat"] = pat_arr
    jobs = [{"op": op,
             "inputs": ["pat"] if pat is not None else [],
             "params": params,
             "out": _out_name}]
    rt.execute(rt.compile(jobs), src_data)
    raw = rt.driver.resolve_output(_out_name)
    arr = np.asarray(raw, dtype=_NP_DTYPE[dtype])
    return CreatedArray(arr, None, (n,), dtype)


def _run_random_kernel(n, mode, seed, p0, p1, dtype):
    """RandomKernel через Runtime execute -> CreatedArray."""
    if seed is None:
        raise ValueError(
            "Creation: seed is REQUIRED keyword (stateless RNG, S205)")
    rt = _ensure_runtime()
    jobs = [{"op": "RandomKernel", "inputs": [],
             "params": {"n": n, "mode": mode, "seed": int(seed),
                        "p0": p0, "p1": p1},
             "out": _out_name}]
    rt.execute(rt.compile(jobs), {})
    raw = rt.driver.resolve_output(_out_name)
    arr = np.asarray(raw, dtype=_NP_DTYPE[dtype])
    return CreatedArray(arr, None, (n,), dtype)


def zeros(shape, dtype="float32"):
    """Массив из нулей."""
    _check_dtype(dtype)
    n = _n_of(shape)
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    return _run_index_kernel(n, 0, 0.0, 0.0, 1.0, dtype)


def ones(shape, dtype="float32"):
    """Массив из единиц."""
    _check_dtype(dtype)
    n = _n_of(shape)
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    return _run_index_kernel(n, 0, 1.0, 0.0, 1.0, dtype)


def full(shape, value, dtype="float32"):
    """Массив, заполненный value."""
    _check_dtype(dtype)
    n = _n_of(shape)
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    return _run_index_kernel(n, 0, float(value), 0.0, 1.0, dtype)


def arange(start=0, stop=None, step=1, dtype="float32"):
    """Равномерная сетка start + step*i (IndexKernel mode=1)."""
    _check_dtype(dtype)
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise ValueError("Creation: arange step must be != 0")
    n = max(0, math.ceil((stop - start) / step))
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    if dtype == "int32":
        last = start + step * (n - 1)
        lo, hi = min(start, last), max(start, last)
        if lo < _INT32_MIN or hi > _INT32_MAX:
            raise ValueError(
                "Creation: arange domain outside int32 "
                "(deterministic pre-dispatch check)")
    return _run_index_kernel(n, 1, float(start), float(step), 1.0, dtype)


def index(n, dtype="int32"):
    """Индексная серия 0..n-1 (IndexKernel mode=1, arange(start=0, step=1)).

    Виртуальный индекс — база для построения остальных паттернов
    через существующие операции (mod, mul, ...).

    НАСТОЯЩАЯ integer семантика (решение владельца):

    - logical dtype: int32 (дефолт СМЕНЁН с float32); physical: i32 буфер
      (op=IndexKernelI32), значения точны до 2^31 — лимит f32 2^24 НЕ
      применяется.
    - upper bound: n <= 2**31-1, иначе ValueError. Для GPU single-dispatch
      дополнительно действует лимит драйвера 4_194_240 элементов
      (wgpu_driver; CPU-путь работает во всём int32-домене).
    - %, //, сравнения — в integer domain: чисто-int цепочки
      (MapBinaryI32 op=add/sub/mul WRAP, mod=C-trunc) остаются int32.
    - Overflow: арифметика i32 WRAP (two's complement). Например i*i для
      i > ~46340 переполняется и оборачивается — документированное
      поведение, не ошибка.
    - Promotion: int32 Series при операции с float АВТОМАТИЧЕСКИ
      приводится к f32 (значения >2^24 теряют точность — например
      sin(idx*0.01)); осознанный вызов. Обратного сужения нет.

    dtype="float32" -> прежний f32 путь (arange-семантика).
    """
    _check_dtype(dtype)
    n = int(n)
    if n < 0:
        raise ValueError(f"Creation: index n must be >= 0, got {n}")
    if n > _INT32_MAX:
        raise ValueError(
            f"Creation: index n={n} exceeds int32 domain "
            f"(upper bound {_INT32_MAX} = 2**31-1)")
    if n == 0:
        return _empty(n, dtype)
    return _run_index_kernel(n, 1, 0, 1, 1, dtype)


def linspace(start, stop, num, endpoint=True, dtype="float32"):
    """num точек от start до stop (IndexKernel mode=2, denom=(num-1)|num)."""
    _check_dtype(dtype)
    n = int(num)
    if n < 0:
        raise ValueError(f"Creation: linspace num must be >= 0, got {num}")
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    denom = (n - 1) if endpoint else n
    p2 = float(denom) if denom != 0 else 1.0
    return _run_index_kernel(n, 2, float(start), float(stop), p2, dtype)


def tile(pattern, n, dtype="float32"):
    """Повторение паттерна до длины n (IndexKernel mode=3): out[i] = pat[i % k]."""
    _check_dtype(dtype)
    n = int(n)
    if n < 0:
        raise ValueError(f"Creation: tile n must be >= 0, got {n}")
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    return _run_index_kernel(n, 3, 0.0, 1.0, 1.0, dtype,
                             pat=list(pattern), rep=1)


def repeat(pattern, repeats, n=None, dtype="float32"):
    """Поэлементное повторение паттерна (IndexKernel mode=4): out[i] = pat[i // r].

    n = repeats*len(pattern) если n is None.
    """
    _check_dtype(dtype)
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError(f"Creation: repeat requires repeats >= 1, got {repeats}")
    if n is None:
        n = repeats * len(pattern)
    else:
        n = int(n)
    if n < 0:
        raise ValueError(f"Creation: repeat n must be >= 0, got {n}")
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    return _run_index_kernel(n, 4, 0.0, 1.0, 1.0, dtype,
                             pat=list(pattern), rep=repeats)


def random_uniform(low=0.0, high=1.0, shape=(0,), *, seed):
    """seed-required uniform [low, high) (RandomKernel mode=0)."""
    n = _n_of(shape)
    if n == 0:
        return _empty(n, "float32")
    _check_n(n)
    return _run_random_kernel(n, 0, seed, float(low), float(high), "float32")


def random_normal(loc=0.0, scale=1.0, shape=(0,), *, seed):
    """seed-required normal(loc, scale) (RandomKernel mode=2, всегда f32)."""
    n = _n_of(shape)
    if n == 0:
        return _empty(n, "float32")
    _check_n(n)
    return _run_random_kernel(n, 2, seed, float(loc), float(scale), "float32")


def random_integers(low, high, shape=(0,), *, seed, dtype="int32"):
    """seed-required integers [low, high) (RandomKernel mode=1)."""
    _check_dtype(dtype)
    n = _n_of(shape)
    if n == 0:
        return _empty(n, dtype)
    _check_n(n)
    return _run_random_kernel(n, 1, seed, float(low), float(high), dtype)
