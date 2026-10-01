# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""JOIN M3 bench: fused native output hypothesis (BEFORE/AFTER).

BEFORE (M2): Rust hash build + Rust hash probe + Rust gather
  (nf_join_gather_i32) + NumPy materialize (pos/pos[hit]/fancy-indexing).
AFTER  (M3): Rust hash build + Rust fused probe->output in one call
  (nf_join_fused_inner_i32 two-pass count+fill / nf_join_fused_left_i32
  single-pass), no pos/pos[hit]/fancy-indexing temps, no materialize.
join.py semantics untouched.

Shapes (same synthetic seed-42 parquet as bench_join_cpu, IO outside timers):
  J2: x[100M] INNER JOIN medium[S=100K] ON id2 (90% hits)
  J3: x[100M] LEFT  JOIN medium[S=100K] ON id2 (miss -> NULL payload)

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_join_m3.py
"""

import gc as _gc
import os as _os
import sys as _sys
import time as _time
from pathlib import Path as _Path

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
_os.environ.setdefault("MKL_NUM_THREADS", "16")
_os.environ.setdefault("OMP_NUM_THREADS", "16")
_os.environ.setdefault("NUMFAST_THREADS", "16")

FORK = _Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as _np

_LIB = str(FORK / "src" / "Relational" / "Join")
if _LIB not in _sys.path:
    _sys.path.insert(0, _LIB)
import _lib.join as OJ  # noqa: E402 (oracle semantics, read-only here)
import _lib.native as N  # noqa: E402 (M2/M3 extension point, additive use)

DATA = FORK / "scratch" / "join_data"
T = int(_os.environ.get("NUMFAST_THREADS", "16"))
M2_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
             / "release" / "numfast_native_join_m2.dll")
M3_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
             / "release" / "numfast_native_join_m3.dll")


def use_dll(path):
    N._DLL_PATH = path
    N._dll = None


def symbols(path):
    import ctypes as _ct
    lib = _ct.CDLL(path)
    return {s: hasattr(lib, s) for s in (
        "nf_join_build", "nf_join_probe", "nf_join_gather_i32",
        "nf_join_fused_inner_i32", "nf_join_fused_left_i32")}


def fuzz(n_cases=200):
    """Oracle + fuzz: fused vs join.py semantics on small random pairs."""
    use_dll(M3_DLL)
    rng = _np.random.default_rng(42)
    bad = 0
    for c in range(n_cases):
        n = int(rng.integers(0, 3000))
        s = int(rng.integers(1, 300))
        hr = float(rng.uniform(0.0, 1.0))
        xk, xv, rk, rv = OJ.make_pair(seed=1000 + c, n=max(n, 1),
                                      s=s, hit_rate=hr)
        if n == 0:
            xk = xk[:0]
            xv = xv[:0]
        b, _ = N.build_timed(rk, rv)
        (ok, o1, o2), _ = N.fused_inner(xk, xv, b, threads=4)
        (ek, e1, e2), _ = OJ.join_inner(xk, xv, OJ.JoinBuild(rk, rv))
        if not (len(ok) == len(ek) and bool((ok == ek).all())
                and bool((o1 == e1).all()) and bool((o2 == e2).all())):
            bad += 1
            print(f"  FUZZ inner MISMATCH case={c} n={n} s={s}", flush=True)
        (fk, f1, f2, fv), _ = N.fused_left(xk, xv, b, threads=4)
        (lk, l1, l2, lv), _ = OJ.join_left(xk, xv, OJ.JoinBuild(rk, rv))
        if not (bool((fk == lk).all()) and bool((f1 == l1).all())
                and bool((f2 == l2).all()) and bool((fv == lv).all())):
            bad += 1
            print(f"  FUZZ left MISMATCH case={c} n={n} s={s}", flush=True)
    print(f"FUZZ oracle {n_cases} cases {'GREEN' if bad == 0 else f'RED bad={bad}'}",
          flush=True)
    return bad == 0


def load(tag):
    import pyarrow.parquet as _pq
    tx = _pq.read_table(DATA / f"{tag}_x.parquet")
    tr = _pq.read_table(DATA / f"{tag}_r.parquet")
    xk = _np.ascontiguousarray(tx.column("k").to_numpy(), dtype=_np.int32)
    xv = _np.ascontiguousarray(tx.column("v1").to_numpy(), dtype=_np.int32)
    rk = _np.ascontiguousarray(tr.column("k").to_numpy(), dtype=_np.int32)
    rv = _np.ascontiguousarray(tr.column("v2").to_numpy(), dtype=_np.int32)
    del tx, tr
    _gc.collect()
    return xk, xv, rk, rv


def bench_tag(tag, how):
    print(f"LOAD {tag} (IO outside timers)", flush=True)
    xk, xv, rk, rv = load(tag)
    run_b = N.join_inner_g if how == "inner" else N.join_left_g
    run_a = N.fused_inner if how == "inner" else N.fused_left

    use_dll(M2_DLL)
    b1, bms1 = N.build_timed(rk, rv)
    best1, out1 = None, None
    for r in range(3):
        out, t = run_b(xk, xv, b1, threads=T)
        print(f"  BEFORE run{r}: probe={t['probe']:.1f} gather={t['gather']:.1f} "
              f"materialize={t['materialize']:.1f} e2e={t['e2e']:.1f}ms",
              flush=True)
        if best1 is None or t["e2e"] < best1["e2e"]:
            best1, out1 = t, out

    use_dll(M3_DLL)
    b2, bms2 = N.build_timed(rk, rv)
    best2, out2 = None, None
    for r in range(3):
        out, t = run_a(xk, xv, b2, threads=T)
        print(f"  AFTER  run{r}: fused={t['fused']:.1f} e2e={t['e2e']:.1f}ms",
              flush=True)
        if best2 is None or t["e2e"] < best2["e2e"]:
            best2, out2 = t, out

    if how == "inner":
        c1 = N.chk_inner(out1[1], out1[2])
        c2 = N.chk_inner(out2[1], out2[2])
        exact = (len(out1[0]) == len(out2[0])
                 and bool((out1[0] == out2[0]).all())
                 and bool((out1[1] == out2[1]).all())
                 and bool((out1[2] == out2[2]).all()))
        rows, nulls = len(out2[0]), 0
    else:
        c1 = N.chk_left(out1[1], out1[2], out1[3])
        c2 = N.chk_left(out2[1], out2[2], out2[3])
        exact = (bool((out2[0] == out1[0]).all())
                 and bool((out2[1] == out1[1]).all())
                 and bool((out2[2] == out1[2]).all())
                 and bool((out2[3] == out1[3]).all()))
        rows = len(out2[0])
        nulls = int((~_np.asarray(out2[3], dtype=bool)).sum())
    stage1 = best1["probe"] + best1["gather"] + best1["materialize"]
    e2e_ratio = best1["e2e"] / best2["e2e"] if best2["e2e"] else 0.0
    e2e_save = best1["e2e"] - best2["e2e"]
    print(f"M3 {tag} {how} rows={rows} nulls={nulls} chk={c2} "
          f"PARITY={'GREEN exact-bit' if (exact and c1 == c2) else 'RED'}",
          flush=True)
    print(f"  BEFORE build={bms1:.1f} probe={best1['probe']:.1f} "
          f"gather={best1['gather']:.1f} materialize={best1['materialize']:.1f} "
          f"stage={stage1:.1f} E2E={best1['e2e']:.1f}ms", flush=True)
    print(f"  AFTER  build={bms2:.1f} fused={best2['fused']:.1f} "
          f"E2E={best2['e2e']:.1f}ms", flush=True)
    print(f"  E2E {best1['e2e']:.1f}->{best2['e2e']:.1f}ms "
          f"x{e2e_ratio:.2f} (save {e2e_save:.1f}ms) (best-of-3, T={T})",
          flush=True)
    del xk, xv, rk, rv, b1, b2, out1, out2, best1, best2
    _gc.collect()
    _sys.stdout.flush()
    return e2e_ratio


if __name__ == "__main__":
    t0 = _time.perf_counter()
    print(f"M2 symbols: {symbols(M2_DLL)}", flush=True)
    print(f"M3 symbols: {symbols(M3_DLL)}", flush=True)
    ok = fuzz(200)
    r2 = bench_tag("J2_100M", "inner")
    r3 = bench_tag("J3_100M", "left")
    print(f"fuzz={'GREEN' if ok else 'RED'} J2x={r2:.2f} J3x={r3:.2f} "
          f"elapsed={(_time.perf_counter() - t0):.1f}s", flush=True)
