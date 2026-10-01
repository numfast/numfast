# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU JOIN production bench (CpuJoin Extension) + DuckDB/Polars competitors.

Shapes (synthetic seed 42, honest SYNTHETIC: no J*.csv on disk):
  J1: x[N] INNER JOIN small[S=N/1M] ON id1 (10M->S=10, 100M->S=100)
  J2: x[N] INNER JOIN medium[S=N/1K] ON id2 (10M->S=10K, 100M->S=100K)
  J3: x[N] LEFT JOIN medium[S=N/1K] ON id2 (miss -> NULL payload)
Keys unique on right, 90% overlap; v1/v2 randint(1,101) int32.
Datasets generated ONCE as parquet (scratch/join_data/), reused by all
three engines. IO outside all timers; query-only best-of-3; chk exact
(rows + sum(v1) + sum(v2) int64).

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_join_cpu.py --mode gen10m
  ... --mode bench10m | bench100m | comp10m | comp100m (gen100m before bench100m)
"""

import argparse as _ap
import gc as _gc
import os as _os
import sys as _sys
import time as _time
from pathlib import Path as _Path

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
_os.environ.setdefault("MKL_NUM_THREADS", "16")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
_os.environ.setdefault("OMP_NUM_THREADS", "16")
_os.environ.setdefault("POLARS_MAX_THREADS", "16")
_os.environ.setdefault("NUMFAST_THREADS", "16")

FORK = _Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as _np

_LIB = str(FORK / "src" / "Compute" / "CpuJoin")
if _LIB not in _sys.path:
    _sys.path.insert(0, _LIB)
import _lib.join as J  # noqa: E402 (production module, read-only here)

DATA = FORK / "scratch" / "join_data"
T = int(_os.environ.get("NUMFAST_THREADS", "16"))

SHAPES = {
    # tag: (N, S)
    "J1_10M": (10_000_000, 10),
    "J2_10M": (10_000_000, 10_000),
    "J1_100M": (100_000_000, 100),
    "J2_100M": (100_000_000, 100_000),
}


def _pq():
    import pyarrow.parquet as _pq
    return _pq


def gen(tag):
    import pyarrow as _pa
    n, s = SHAPES[tag]
    print(f"GEN {tag} n={n} s={s} seed=42 SYNTHETIC", flush=True)
    s0 = _time.perf_counter()
    xk, xv, rk, rv = J.make_pair(seed=42, n=n, s=s)
    DATA.mkdir(parents=True, exist_ok=True)
    _pq().write_table(_pa.table({"k": xk, "v1": xv}),
                      DATA / f"{tag}_x.parquet")
    _pq().write_table(_pa.table({"k": rk, "v2": rv}),
                      DATA / f"{tag}_r.parquet")
    del xk, xv, rk, rv
    _gc.collect()
    print(f"gen {tag} elapsed={(_time.perf_counter() - s0):.1f}s", flush=True)


def load(tag):
    import pyarrow.compute as _pc
    pq = _pq()
    tx = pq.read_table(DATA / f"{tag}_x.parquet")
    tr = pq.read_table(DATA / f"{tag}_r.parquet")
    xk = _np.ascontiguousarray(tx.column("k").to_numpy(), dtype=_np.int32)
    xv = _np.ascontiguousarray(tx.column("v1").to_numpy(), dtype=_np.int32)
    rk = _np.ascontiguousarray(tr.column("k").to_numpy(), dtype=_np.int32)
    rv = _np.ascontiguousarray(tr.column("v2").to_numpy(), dtype=_np.int32)
    del tx, tr
    _gc.collect()
    return xk, xv, rk, rv


def pandas_parity(xk, xv, rk, rv, how):
    import pandas as _pd
    s = _time.perf_counter()
    m = _pd.DataFrame({"k": _np.asarray(xk), "v1": _np.asarray(xv)}).merge(
        _pd.DataFrame({"k": _np.asarray(rk), "v2": _np.asarray(rv)}),
        on="k", how=how)
    e = (_time.perf_counter() - s) * 1000
    if how == "inner":
        chk = (int(m["v1"].sum()), int(m["v2"].sum()))
    else:
        chk = (int(m["v1"].sum()), int(m["v2"].fillna(0).sum()))
    return len(m), chk, e


def bench(tags, pandas=True):
    for tag in tags:
        kind = "inner" if tag.startswith("J1") or tag == "J2_10M" or \
            tag == "J2_100M" else "left"
        print(f"LOAD {tag} (IO outside timers)", flush=True)
        xk, xv, rk, rv = load(tag)
        b, build_ms = J.build_timed(rk, rv)
        print(f"build {tag} S={b.k} build_ms={build_ms:.1f}", flush=True)
        run = J.join_inner if not tag.startswith("J3") else J.join_left
        best, bt = None, None
        for r in range(3):
            out, t = run(xk, xv, b, threads=T)
            print(f"  run{r}: probe={t['probe']:.1f} gather={t['gather']:.1f} "
                  f"materialize={t['materialize']:.1f} e2e={t['e2e']:.1f}ms",
                  flush=True)
            if best is None or t["e2e"] < best["e2e"]:
                best, bt = t, out
        if tag.startswith("J3"):
            ok, o1, o2, valid = bt
            rows = len(ok)
            chk = J.chk_left(o1, o2, valid)
            nulls = int((~_np.asarray(valid, dtype=bool)).sum())
        else:
            ok, o1, o2 = bt
            rows = len(ok)
            chk = J.chk_inner(o1, o2)
            nulls = 0
        print(f"NUMFAST {tag} rows={rows} nulls={nulls} chk={chk} "
              f"build={build_ms:.1f} probe={best['probe']:.1f} "
              f"gather={best['gather']:.1f} materialize={best['materialize']:.1f} "
              f"E2E={best['e2e']:.1f}ms (best-of-3, T={T})", flush=True)
        if pandas:
            try:
                pn, pchk, pms = pandas_parity(xk, xv, rk, rv,
                                             "left" if tag.startswith("J3")
                                             else "inner")
                print(f"PANDAS {tag} rows={pn} chk={pchk} e2e={pms:.1f}ms "
                      f"PARITY={'GREEN' if (pn == rows and pchk == chk) else 'RED'}",
                      flush=True)
            except Exception as e:
                print(f"PANDAS {tag} SKIP/FAIL: {type(e).__name__}: {e}",
                      flush=True)
        del xk, xv, rk, rv, b, bt, best
        _gc.collect()
        _sys.stdout.flush()


def competitors(tags):
    import duckdb as _ddb
    import polars as _pl
    print(f"duckdb {_ddb.__version__} threads=16 | polars {_pl.__version__} "
          f"POLARS_MAX_THREADS=16", flush=True)
    for tag in tags:
        xp, rp = str(DATA / f"{tag}_x.parquet"), str(DATA / f"{tag}_r.parquet")
        is_left = tag.startswith("J3")
        how = "left" if is_left else "inner"
        jt = "LEFT JOIN" if is_left else "JOIN"
        # DuckDB best-of-3 query-only
        con = _ddb.connect(database=":memory:")
        con.execute("PRAGMA THREADS=16")
        con.execute(f"CREATE VIEW x AS SELECT * FROM read_parquet('{xp}')")
        con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{rp}')")
        q = ("SELECT COUNT(*) n, SUM(v1) s1, SUM(v2) s2 FROM "
             f"(SELECT v1, v2 FROM x {jt} r USING (k))")
        d_best, d_chk = None, None
        for r in range(3):
            s = _time.perf_counter()
            n, s1, s2 = con.execute(q).fetchone()
            e = (_time.perf_counter() - s) * 1000
            d_chk = (int(n), int(s1), int(s2 or 0))
            if d_best is None or e < d_best:
                d_best = e
            print(f"  duckdb {tag} run{r}: {e:.1f}ms chk={d_chk}", flush=True)
        con.close()
        # Polars best-of-3 query-only (lazy scan, collect timed)
        p_best, p_chk = None, None
        for r in range(3):
            s = _time.perf_counter()
            m = (_pl.scan_parquet(xp).join(_pl.scan_parquet(rp), on="k",
                                           how=how).select(
                _pl.len().alias("n"),
                # int64 accumulators (i32 sums wrap past 2**31 at 100M)
                _pl.col("v1").cast(_pl.Int64).sum().alias("s1"),
                _pl.col("v2").cast(_pl.Int64).sum().alias("s2")).collect())
            e = (_time.perf_counter() - s) * 1000
            p_chk = (int(m["n"][0]), int(m["s1"][0]), int(m["s2"][0] or 0))
            if p_best is None or e < p_best:
                p_best = e
            print(f"  polars {tag} run{r}: {e:.1f}ms chk={p_chk}", flush=True)
        print(f"COMP {tag} duckdb16T={d_best:.1f}ms chk={d_chk} | "
              f"polars16T={p_best:.1f}ms chk={p_chk} "
              f"MATCH={'GREEN' if d_chk == p_chk else 'RED'}", flush=True)
        _gc.collect()
        _sys.stdout.flush()


if __name__ == "__main__":
    p = _ap.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["gen10m", "bench10m", "gen100m", "bench100m",
                            "comp10m", "comp100m", "bench10m_nopandas"])
    a = p.parse_args()
    t0 = _time.perf_counter()
    if a.mode == "gen10m":
        gen("J1_10M")
        gen("J2_10M")
    elif a.mode == "bench10m":
        bench(["J1_10M", "J2_10M"], pandas=True)
        # J3 reuses J2 datasets (same medium shape, LEFT semantics)
        xk, xv, rk, rv = load("J2_10M")
        b, build_ms = J.build_timed(rk, rv)
        best, bt = None, None
        for r in range(3):
            out, t = J.join_left(xk, xv, b, threads=T)
            print(f"  run{r}: probe={t['probe']:.1f} gather={t['gather']:.1f} "
                  f"materialize={t['materialize']:.1f} e2e={t['e2e']:.1f}ms",
                  flush=True)
            if best is None or t["e2e"] < best["e2e"]:
                best, bt = t, out
        ok, o1, o2, valid = bt
        chk = J.chk_left(o1, o2, valid)
        print(f"NUMFAST J3_10M rows={len(ok)} "
              f"nulls={int((~_np.asarray(valid, dtype=bool)).sum())} chk={chk} "
              f"build={build_ms:.1f} probe={best['probe']:.1f} "
              f"gather={best['gather']:.1f} materialize={best['materialize']:.1f} "
              f"E2E={best['e2e']:.1f}ms (best-of-3, T={T})", flush=True)
        try:
            pn, pchk, pms = pandas_parity(xk, xv, rk, rv, "left")
            print(f"PANDAS J3_10M rows={pn} chk={pchk} e2e={pms:.1f}ms "
                  f"PARITY={'GREEN' if (pn == len(ok) and pchk == chk) else 'RED'}",
                  flush=True)
        except Exception as e:
            print(f"PANDAS J3_10M SKIP/FAIL: {type(e).__name__}: {e}",
                  flush=True)
    elif a.mode == "gen100m":
        gen("J1_100M")
        gen("J2_100M")
    elif a.mode == "bench100m":
        bench(["J1_100M", "J2_100M"], pandas=False)
        xk, xv, rk, rv = load("J2_100M")
        b, build_ms = J.build_timed(rk, rv)
        best, bt = None, None
        for r in range(3):
            out, t = J.join_left(xk, xv, b, threads=T)
            print(f"  run{r}: probe={t['probe']:.1f} gather={t['gather']:.1f} "
                  f"materialize={t['materialize']:.1f} e2e={t['e2e']:.1f}ms",
                  flush=True)
            if best is None or t["e2e"] < best["e2e"]:
                best, bt = t, out
        ok, o1, o2, valid = bt
        chk = J.chk_left(o1, o2, valid)
        print(f"NUMFAST J3_100M rows={len(ok)} "
              f"nulls={int((~_np.asarray(valid, dtype=bool)).sum())} chk={chk} "
              f"build={build_ms:.1f} probe={best['probe']:.1f} "
              f"gather={best['gather']:.1f} materialize={best['materialize']:.1f} "
              f"E2E={best['e2e']:.1f}ms (best-of-3, T={T})", flush=True)
        del xk, xv, rk, rv, b, bt, best
        _gc.collect()
    elif a.mode == "comp10m":
        competitors(["J1_10M", "J2_10M"])
        # J3 competitors reuse J2 files with LEFT semantics: emulate via
        # tag alias (competitors() keys J3 by file J2_10M)
        import shutil as _sh
        _sh.copy(DATA / "J2_10M_x.parquet", DATA / "J3_10M_x.parquet")
        _sh.copy(DATA / "J2_10M_r.parquet", DATA / "J3_10M_r.parquet")
        competitors(["J3_10M"])
    elif a.mode == "comp100m":
        competitors(["J1_100M", "J2_100M"])
        import shutil as _sh
        _sh.copy(DATA / "J2_100M_x.parquet", DATA / "J3_100M_x.parquet")
        _sh.copy(DATA / "J2_100M_r.parquet", DATA / "J3_100M_r.parquet")
        competitors(["J3_100M"])
    elif a.mode == "bench10m_nopandas":
        bench(["J1_10M", "J2_10M"], pandas=False)
    print(f"elapsed={(_time.perf_counter() - t0):.1f}s", flush=True)
