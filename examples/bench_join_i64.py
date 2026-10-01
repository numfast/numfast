# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Bench + validate the int64 join lane (additive; int32 lane untouched).

Covers: bit-exact parity native vs sorted-searchsorted fallback on
SF1 Q2-Q5 id domains, fuzz (empty/single/dupes/INT64_MIN/MAX/wide
ids/non-contiguous/int32 widening/determinism/thread-invariance),
and the Q2-Q5 join-stage benchmark before (numpy path) -> after
(native lane): cold/warm, memory accounting, exact parity.

Run:  C:/App/numfast/.venv/Scripts/python.exe bench_join_i64.py
Seed fixed at 42. Stages timed with perf_counter (ms per stage).
"""

import importlib.util as _ilu
import json as _json
import time as _time
from pathlib import Path as _Path

import numpy as _np

_HERE = _Path(__file__).resolve().parent
_MOD = _HERE / "src" / "Relational" / "Join" / "_lib" / "native_i64.py"
_spec = _ilu.spec_from_file_location("_nf_i64lane", str(_MOD))
_lane = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lane)

SEED = 42
BASE = _np.int64(2 ** 40)  # LDBC-like ids: exceed int32
THREADS = 16


def _pair(seed, n, s, hit_rate):
    rng = _np.random.default_rng(seed)
    rk = (BASE + rng.permutation(s)).astype(_np.int64)
    rv = rng.integers(1, 101, s).astype(_np.int32)
    n_hit = int(n * hit_rate)
    xk = _np.empty(n, dtype=_np.int64)
    if n_hit:
        xk[:n_hit] = rk[rng.integers(0, max(s, 1), n_hit)]
    xk[n_hit:] = BASE + s + rng.integers(0, max(1, s // 2), n - n_hit)
    rng.shuffle(xk)
    xv = rng.integers(1, 101, n).astype(_np.int32)
    return xk, xv, rk, rv


def _assert_close(a, b, what):
    a, b = _np.asarray(a), _np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or not bool((a == b).all()):
        raise AssertionError(
            f"parity FAIL [{what}]: shape {a.shape}/{b.shape} "
            f"dtype {a.dtype}/{b.dtype}")


def validate():
    checks = []
    # empty/empty
    b_nat = _lane.build(_np.zeros(0, _np.int64), _np.zeros(0, _np.int32))
    b_np = _lane.NumpyJoinBuildI64(_np.zeros(0, _np.int64),
                                   _np.zeros(0, _np.int32))
    for b in (b_nat, b_np):
        pos, hit, k = _lane.lookup_positions(b, _np.zeros(0, _np.int64))
        assert pos.size == 0 and hit.size == 0 and k == 0
    checks.append("empty: PASS")
    # single hit + single miss
    rk = _np.array([7], dtype=_np.int64)
    rv = _np.array([42], dtype=_np.int32)
    for mk in (True, False):
        xk = _np.array([7 if mk else 8], dtype=_np.int64)
        pn, hn, kn = _lane.lookup_positions(_lane.build(rk, rv), xk)
        pf, hf, kf = _lane.lookup_positions(
            _lane.NumpyJoinBuildI64(rk, rv), xk)
        _assert_close(pn, pf, "single-pos")
        _assert_close(hn, hf, "single-hit")
        assert kn == kf == 1
    checks.append("single: PASS")
    # duplicates in build -> ValueError on both paths
    dup_k = _np.array([5, 5], dtype=_np.int64)
    dup_v = _np.array([1, 2], dtype=_np.int32)
    for fn in (_lane.build, _lane.NumpyJoinBuildI64):
        try:
            fn(dup_k, dup_v)
        except ValueError:
            pass
        else:
            raise AssertionError("dupe build must raise")
    checks.append("duplicates: PASS")
    # INT64_MIN / MAX / -1 / 0 / INT32 edges, dupes in probe ok
    edge = _np.array([_np.int64(-2 ** 63), _np.int64(-1), _np.int64(0),
                      _np.int64(2 ** 31 - 1), _np.int64(2 ** 31),
                      _np.int64(2 ** 63 - 1)], dtype=_np.int64)
    ev = _np.arange(1, 7, dtype=_np.int32)
    probe = _np.array([_np.int64(2 ** 63 - 1), _np.int64(2 ** 63 - 1),
                       _np.int64(-2 ** 63), _np.int64(1234567890123),
                       _np.int64(0)], dtype=_np.int64)
    pn, hn, kn = _lane.lookup_positions(_lane.build(edge, ev), probe)
    pf, hf, kf = _lane.lookup_positions(
        _lane.NumpyJoinBuildI64(edge, ev), probe)
    _assert_close(pn, pf, "edge-pos")
    _assert_close(hn, hf, "edge-hit")
    assert kn == kf == 6 and bool((hn == [True, True, True, False, True]).all())
    checks.append("int64-min-max: PASS")
    # non-contiguous + int32 widening parity
    xk, xv, rk, rv = _pair(SEED, 20_000, 2_000, 0.9)
    xk_nc = xk[::2]
    xv_nc = xv[::2]
    assert not xk_nc.flags["C_CONTIGUOUS"]
    pn, hn, _ = _lane.lookup_positions(_lane.build(rk, rv), xk_nc)
    pf, hf, _ = _lane.lookup_positions(
        _lane.NumpyJoinBuildI64(rk, rv), xk_nc)
    _assert_close(pn, pf, "noncontig-pos")
    _assert_close(hn, hf, "noncontig-hit")
    rk32 = (rk - BASE).astype(_np.int32)
    xk32 = _np.where(xk < BASE + 2_000, (xk - BASE).astype(_np.int64),
                     xk).astype(_np.int64)
    pn2, hn2, _ = _lane.lookup_positions(
        _lane.build(rk32.astype(_np.int64), rv), xk32)
    assert pn2.dtype == _np.int64 and hn2.dtype == bool
    checks.append("noncontig+ widen: PASS")
    # determinism + thread invariance (1 vs 16)
    xk, xv, rk, rv = _pair(SEED, 100_000, 10_000, 0.85)
    b = _lane.build(rk, rv)
    p1, h1, _ = _lane.lookup_positions(b, xk, 1)
    p2, h2, _ = _lane.lookup_positions(b, xk, 16)
    p3, h3, _ = _lane.lookup_positions(b, xk, 16)
    _assert_close(p1, p2, "threads-pos")
    _assert_close(h1, h2, "threads-hit")
    _assert_close(p2, p3, "deterministic-pos")
    _assert_close(h2, h3, "deterministic-hit")
    checks.append("deterministic: PASS")
    # join materialize parity inner + left on fuzz domain
    bn = _lane.build(rk, rv)
    bf = _lane.NumpyJoinBuildI64(rk, rv)
    on, o1n, o2n = _lane.join_inner(xk, xv, bn)
    of_, o1f, o2f = _lane.join_inner(xk, xv, bf)
    _assert_close(on, of_, "inner-keys")
    _assert_close(o1n, o1f, "inner-v1")
    _assert_close(o2n, o2f, "inner-v2")
    assert _lane.chk_inner(o1n, o2n) == _lane.chk_inner(o1f, o2f)
    ln = _lane.join_left(xk, xv, bn)
    lf = _lane.join_left(xk, xv, bf)
    for a, c, w in zip(ln, lf, ("left-k", "left-v1", "left-v2", "left-v")):
        _assert_close(a, c, w)
    assert _lane.chk_left(*ln[1:]) == _lane.chk_left(*lf[1:])
    checks.append("join-parity: PASS")
    return checks


DOMAINS = [
    ("Q2", 5_000, 50_000, 0.90),
    ("Q3", 25_000, 200_000, 0.80),
    ("Q4", 100_000, 500_000, 0.70),
    ("Q5", 400_000, 1_000_000, 0.60),
]


def _mem_bytes(build_obj, n):
    m = int(build_obj.u.nbytes + build_obj.pu.nbytes)
    if hasattr(build_obj, "t_keys"):
        m += int(build_obj.t_keys.nbytes + build_obj.t_pos.nbytes
                 + build_obj.t_occ.nbytes + build_obj.rank.nbytes)
    m += int(n * (8 + 1))  # pos int64 + hit per probe row
    return m


def benchmark():
    rows = []
    for name, s, n, hr in DOMAINS:
        xk, xv, rk, rv = _pair(SEED, n, s, hr)
        # BEFORE: numpy (sorted-searchsorted) join stage
        t0 = _time.perf_counter()
        bf = _lane.NumpyJoinBuildI64(rk, rv)
        t_build_np = (_time.perf_counter() - t0) * 1000
        t0 = _time.perf_counter()
        of_, o1f, o2f = _lane.join_inner(xk, xv, bf)
        t_probe_np = (_time.perf_counter() - t0) * 1000
        chk_np = _lane.chk_inner(o1f, o2f)
        mem_np = _mem_bytes(bf, n)
        # AFTER: native lane (cold = fresh build+probe, warm = resident)
        t0 = _time.perf_counter()
        bn = _lane.build(rk, rv)
        t_build_cold = (_time.perf_counter() - t0) * 1000
        t0 = _time.perf_counter()
        on, o1n, o2n = _lane.join_inner(xk, xv, bn)
        t_probe_cold = (_time.perf_counter() - t0) * 1000
        chk_nat = _lane.chk_inner(o1n, o2n)
        mem_nat = _mem_bytes(bn, n)
        assert chk_np == chk_nat
        _assert_close(on, of_, f"{name}-keys")
        _assert_close(o1n, o1f, f"{name}-v1")
        _assert_close(o2n, o2f, f"{name}-v2")
        uncached = []
        for _ in range(5):
            t0 = _time.perf_counter()
            _lane.join_inner(xk, xv, bn)
            uncached.append((_time.perf_counter() - t0) * 1000)
        t_warm = sorted(uncached)[2]
        rows.append({
            "q": name, "s": s, "n": n,
            "numpy_build_ms": round(t_build_np, 3),
            "numpy_probe_ms": round(t_probe_np, 3),
            "native_build_cold_ms": round(t_build_cold, 3),
            "native_probe_cold_ms": round(t_probe_cold, 3),
            "native_probe_warm_ms": round(t_warm, 3),
            "speedup_probe_cold": round(t_probe_np / max(t_probe_cold, 1e-9), 3),
            "speedup_probe_warm": round(t_probe_np / max(t_warm, 1e-9), 3),
            "mem_numpy_b": mem_np, "mem_native_b": mem_nat,
            "chk": list(chk_nat), "parity": "exact",
        })
    return rows


def main():
    t0 = _time.perf_counter()
    checks = validate()
    t_val = (_time.perf_counter() - t0) * 1000
    t0 = _time.perf_counter()
    rows = benchmark()
    t_bench = (_time.perf_counter() - t0) * 1000
    out = {"seed": SEED, "threads": THREADS,
           "backend": _lane.why(), "native_available": _lane.available(),
           "validation": checks, "validation_ms": round(t_val, 3),
           "benchmark_ms": round(t_bench, 3), "rows": rows}
    print(_json.dumps(out, indent=2))
    with open(_HERE / "bench_join_i64_results.json", "w") as f:
        _json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
