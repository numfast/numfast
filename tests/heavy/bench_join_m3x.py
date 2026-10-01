# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""JOIN M3 portability: J1 (small build) / J4 (string key id5) / J5 (big build).

M3 backend (nf_join_fused_inner_i32, M3 DLL) reused as-is on new shapes.
join.py semantics untouched. No shuffle, no new kernels.

Shapes (synthetic seed 42; IO outside timers; query-only best-of-3; T=16):
  J1: x[N] INNER JOIN small[S=N/1M] ON id1 (parquet scratch/join_data/, same
      files as bench_join_cpu: J1_10M S=10, J1_100M S=100). Stages on J1 data:
      OJ-baseline + M1 + M2 + M3 (existing DLLs) + DuckDB/Polars E2E.
  J4: string key id5: shared-encode BOTH sides via CPU-driver
      _encode_pattern_vec (prefix 'id5_', absolute codes, no dict-sync risk),
      then fused inner on codes. 10M always; 100M time-guarded.
  J5: big build S=N int keys (broadcast-model boundary): 10M->10M full
      (exact-bit vs OJ); 100M->100M guarded attempt (chk + slice parity).

Usage (Git Bash, strictly sequential, timeout 550, one mode per process):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_join_m3x.py --mode j1
  ... --mode j4 | j5_10m | j5_100m
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

_JLIB = str(FORK / "src" / "Relational" / "Join")
if _JLIB not in _sys.path:
    _sys.path.insert(0, _JLIB)
import _lib.join as OJ  # noqa: E402 (oracle semantics, read-only here)
import _lib.native as N  # noqa: E402 (M1/M2/M3 extension point, additive use)

DATA = FORK / "scratch" / "join_data"
T = int(_os.environ.get("NUMFAST_THREADS", "16"))
M1_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
             / "release" / "numfast_native_join_m1.dll")
M2_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
             / "release" / "numfast_native_join_m2.dll")
M3_DLL = str(FORK / "numfast-native" / "target" / "x86_64-pc-windows-gnu"
             / "release" / "numfast_native_join_m3.dll")


def use_dll(path):
    N._DLL_PATH = path
    N._dll = None


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


def best3(tag, label, run, xk, xv, b):
    best, out = None, None
    for r in range(3):
        o, t = run(xk, xv, b, threads=T)
        e2e = t["e2e"]
        det = " ".join(f"{k}={v:.1f}" for k, v in t.items() if k != "e2e")
        print(f"  {label} run{r}: {det} e2e={e2e:.1f}ms", flush=True)
        if best is None or e2e < best["e2e"]:
            best, out = t, o
    return best, out


def comp_parquet(tags, how="inner"):
    """DuckDB + Polars E2E, query-only best-of-3 on the same parquet files."""
    import duckdb as _ddb
    import polars as _pl
    print(f"duckdb {_ddb.__version__} threads=16 | polars {_pl.__version__} "
          f"POLARS_MAX_THREADS=16", flush=True)
    res = {}
    for tag in tags:
        xp, rp = str(DATA / f"{tag}_x.parquet"), str(DATA / f"{tag}_r.parquet")
        jt = "LEFT JOIN" if how == "left" else "JOIN"
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
        p_best, p_chk = None, None
        for r in range(3):
            s = _time.perf_counter()
            m = (_pl.scan_parquet(xp).join(_pl.scan_parquet(rp), on="k",
                                           how=how).select(
                _pl.len().alias("n"),
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
        res[tag] = (d_best, d_chk, p_best, p_chk)
        _gc.collect()
    return res


def mode_j1():
    """J1 small-build: OJ + M1 + M2 + M3 on J1 parquet + competitors."""
    for tag in ("J1_10M", "J1_100M"):
        print(f"LOAD {tag} (IO outside timers)", flush=True)
        xk, xv, rk, rv = load(tag)
        print(f"  n={xk.size} s={rk.size}", flush=True)
        ob, obms = OJ.build_timed(rk, rv)
        bb, bbms = best3(tag, "OJ-baseline", OJ.join_inner, xk, xv, ob)
        (ek, e1, e2), _ = OJ.join_inner(xk, xv, ob, threads=T)
        echk = OJ.chk_inner(e1, e2)
        print(f"OJ {tag} rows={len(ek)} chk={echk} build={obms:.1f} "
              f"E2E={bb['e2e']:.1f}ms", flush=True)
        use_dll(M1_DLL)
        b1, bm1 = N.build_timed(rk, rv)
        t1, _ = best3(tag, "M1", N.join_inner, xk, xv, b1)
        use_dll(M2_DLL)
        b2, bm2 = N.build_timed(rk, rv)
        t2, _ = best3(tag, "M2", N.join_inner_g, xk, xv, b2)
        use_dll(M3_DLL)
        b3, bm3 = N.build_timed(rk, rv)
        t3, o3 = best3(tag, "M3-fused", N.fused_inner, xk, xv, b3)
        c3 = N.chk_inner(o3[1], o3[2])
        exact = (len(o3[0]) == len(ek) and bool((o3[0] == ek).all())
                 and bool((o3[1] == e1).all()) and bool((o3[2] == e2).all()))
        print(f"M3 {tag} rows={len(o3[0])} chk={c3} "
              f"PARITY={'GREEN exact-bit' if (exact and c3 == echk) else 'RED'}",
              flush=True)
        print(f"J1 {tag} OJ={bb['e2e']:.1f} M1={t1['e2e']:.1f} M2={t2['e2e']:.1f} "
              f"M3={t3['e2e']:.1f}ms (best-of-3, T={T}; builds "
              f"{obms:.1f}/{bm1:.1f}/{bm2:.1f}/{bm3:.1f}ms)", flush=True)
        del xk, xv, rk, rv, ob, b1, b2, b3, ek, e1, e2, o3
        _gc.collect()
    comp_parquet(["J1_10M", "J1_100M"], how="inner")

def _dict_encode():
    """Storage Dictionary shared encoder (reuse, no copy).

    dictionary.py imports numpy only -> safe file-location load
    (no _lib intra-package imports, no shadowing issue).
    Shared-encode = ONE encode over the UNION of both sides' strings,
    so codes are consistent by construction (no dict-sync risk).
    """
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "nf_dict", str(FORK / "src" / "Storage" / "Dictionary"
                       / "_lib" / "dictionary.py"))
    D = _ilu.module_from_spec(spec)
    spec.loader.exec_module(D)
    return D.dictionary_encode_impl


def j4_pair(n, s, seed=42):
    """Synthetic J4: right S unique id5_%08d strings; left 90% hits + novel.

    Strings built vectorized (np.char.mod, no Python per-row loop);
    integer bodies kept alongside for the encode contract check.
    """
    rng = _np.random.default_rng(seed)
    rb = rng.permutation(s).astype(_np.int64)
    rstr = _np.char.mod("id5_%08d", rb)
    rv = rng.integers(1, 101, s).astype(_np.int32)
    n_hit = int(n * 0.9)
    lb = _np.empty(n, dtype=_np.int64)
    lb[:n_hit] = rb[rng.integers(0, s, n_hit)]
    lb[n_hit:] = s + rng.integers(0, max(1, s // 2), n - n_hit)
    rng.shuffle(lb)
    xstr = _np.char.mod("id5_%08d", lb)
    xv = rng.integers(1, 101, n).astype(_np.int32)
    return xstr, xv, rstr, rv


def mode_j4():
    dict_encode = _dict_encode()
    use_dll(M3_DLL)
    for n, s in ((10_000_000, 10_000), (100_000_000, 100_000)):
        t0 = _time.perf_counter()
        print(f"GEN J4 n={n} s={s} seed=42 SYNTHETIC(str id5_%08d)", flush=True)
        xstr, xv, rstr, rv = j4_pair(n, s)
        gen_ms = (_time.perf_counter() - t0) * 1000
        s0 = _time.perf_counter()
        both = _np.concatenate([xstr, rstr])
        d = dict_encode(both)
        del both
        xk = _np.ascontiguousarray(d["codes"][:n], dtype=_np.int32)
        rk = _np.ascontiguousarray(d["codes"][n:], dtype=_np.int32)
        enc_ms = (_time.perf_counter() - s0) * 1000
        print(f"  gen={gen_ms:.0f}ms shared-union-encode={enc_ms:.0f}ms "
              f"D={d['metadata']['d']} nulls={d['metadata']['nulls']} "
              f"runique={len(_np.unique(rk)) == s}", flush=True)
        assert d["validity"] is None and len(_np.unique(rk)) == s
        ob, obms = OJ.build_timed(rk, rv)
        bb, bout = best3("J4", "OJ-codes", OJ.join_inner, xk, xv, ob)
        (ek, e1, e2) = bout
        echk = OJ.chk_inner(e1, e2)
        b3, bm3 = N.build_timed(rk, rv)
        t3, o3 = best3("J4", "M3-fused", N.fused_inner, xk, xv, b3)
        c3 = N.chk_inner(o3[1], o3[2])
        exact = (len(o3[0]) == len(ek) and bool((o3[0] == ek).all())
                 and bool((o3[1] == e1).all()) and bool((o3[2] == e2).all()))
        print(f"J4 n={n} s={s} rows={len(o3[0])} chk={c3} "
              f"PARITY={'GREEN exact-bit' if (exact and c3 == echk) else 'RED'} "
              f"OJ={bb['e2e']:.1f} M3={t3['e2e']:.1f}ms "
              f"(encode {enc_ms:.0f}ms outside join; builds "
              f"{obms:.1f}/{bm3:.1f}ms)", flush=True)
        # pandas slice parity on RAW strings (200K rows, exact rows+sums)
        import pandas as _pd
        sl = slice(0, 200_000)
        m = _pd.DataFrame({"k": xstr[sl], "v1": xv[sl]}).merge(
            _pd.DataFrame({"k": rstr, "v2": rv}), on="k", how="inner")
        (sk, s1, s2), _ = OJ.join_inner(
            xk[sl], xv[sl], OJ.JoinBuild(rk, rv), threads=T)
        okp = (len(m) == len(sk) and int(m["v1"].sum()) == int(s1.sum())
               and int(m["v2"].sum()) == int(s2.sum()))
        print(f"  pandas-str-slice200K rows={len(m)} "
              f"PARITY={'GREEN' if okp else 'RED'}", flush=True)
        el = _time.perf_counter() - t0
        # in-memory competitors on arrow strings (query-only best-of-3)
        import pyarrow as _pa
        import duckdb as _ddb
        import polars as _pl
        xa = _pa.table({"k": xstr, "v1": xv})
        ra = _pa.table({"k": rstr, "v2": rv})
        con = _ddb.connect(database=":memory:")
        con.execute("PRAGMA THREADS=16")
        q = ("SELECT COUNT(*) n, SUM(v1) s1, SUM(v2) s2 FROM "
             "(SELECT v1, v2 FROM xa JOIN ra USING (k))")
        d_best, d_chk = None, None
        for r in range(3):
            st = _time.perf_counter()
            nn, s1, s2 = con.execute(q).fetchone()
            e = (_time.perf_counter() - st) * 1000
            d_chk = (int(nn), int(s1), int(s2 or 0))
            if d_best is None or e < d_best:
                d_best = e
        con.close()
        xp, rp = _pl.from_arrow(xa), _pl.from_arrow(ra)
        p_best, p_chk = None, None
        for r in range(3):
            st = _time.perf_counter()
            mm = (xp.join(rp, on="k", how="inner").select(
                _pl.len().alias("n"),
                _pl.col("v1").cast(_pl.Int64).sum().alias("s1"),
                _pl.col("v2").cast(_pl.Int64).sum().alias("s2")))
            e = (_time.perf_counter() - st) * 1000
            p_chk = (int(mm["n"][0]), int(mm["s1"][0]), int(mm["s2"][0] or 0))
            if p_best is None or e < p_best:
                p_best = e
        match = (d_chk[0] == len(o3[0]) and (d_chk[1], d_chk[2]) == c3
                 and d_chk == p_chk)
        print(f"COMP J4 n={n} duckdb16T={d_best:.1f}ms chk={d_chk} | "
              f"polars16T={p_best:.1f}ms chk={p_chk} "
              f"vs-M3={'GREEN' if match else 'RED'} elapsed={el:.0f}s",
              flush=True)
        del xstr, xv, rstr, rv, xk, rk, ob, b3, ek, e1, e2, o3, xa, ra, xp, rp
        _gc.collect()
        if n == 10_000_000 and (_time.perf_counter() - t0) > 400:
            print("J4_100M SKIPPED (budget guard)", flush=True)
            break


def j5_pair(n, s, seed=42):
    rng = _np.random.default_rng(seed)
    rk = rng.permutation(s).astype(_np.int32)
    rv = rng.integers(1, 101, s).astype(_np.int32)
    n_hit = int(n * 0.9)
    xk = _np.empty(n, dtype=_np.int32)
    xk[:n_hit] = rk[rng.integers(0, s, n_hit)]
    xk[n_hit:] = (s + rng.integers(0, max(1, s // 2),
                                   n - n_hit)).astype(_np.int32)
    rng.shuffle(xk)
    xv = rng.integers(1, 101, n).astype(_np.int32)
    return xk, xv, rk, rv


def mode_j5(n, full_exact):
    print(f"GEN J5 n={n} s={n} seed=42 SYNTHETIC(big build S=N)", flush=True)
    t0 = _time.perf_counter()
    xk, xv, rk, rv = j5_pair(n, n)
    print(f"  gen={(_time.perf_counter() - t0):.1f}s "
          f"cap={N.cap_for(n)} lanes (~{N.cap_for(n) * 9 / 2**30:.2f}GiB table)",
          flush=True)
    use_dll(M3_DLL)
    s0 = _time.perf_counter()
    b3, bm3 = N.build_timed(rk, rv)
    print(f"  M3 build={bm3:.1f}ms", flush=True)
    t3, o3 = best3("J5", "M3-fused", N.fused_inner, xk, xv, b3)
    c3 = N.chk_inner(o3[1], o3[2])
    if full_exact:
        ob, obms = OJ.build_timed(rk, rv)
        bb, bout = best3("J5", "OJ-baseline", OJ.join_inner, xk, xv, ob)
        (ek, e1, e2) = bout
        echk = OJ.chk_inner(e1, e2)
        exact = (len(o3[0]) == len(ek) and bool((o3[0] == ek).all())
                 and bool((o3[1] == e1).all()) and bool((o3[2] == e2).all()))
        print(f"J5 n={n} s={n} rows={len(o3[0])} chk={c3} "
              f"PARITY={'GREEN exact-bit' if (exact and c3 == echk) else 'RED'} "
              f"OJ={bb['e2e']:.1f} M3={t3['e2e']:.1f}ms "
              f"(builds {obms:.1f}/{bm3:.1f}ms)", flush=True)
        del ob, ek, e1, e2, bout
    else:
        ob, obms = OJ.build_timed(rk, rv)
        s0 = _time.perf_counter()
        (ek, e1, e2), bt = OJ.join_inner(xk, xv, ob, threads=T)
        oj_ms = (_time.perf_counter() - s0) * 1000
        echk = OJ.chk_inner(e1, e2)
        fullchk = (len(ek) == len(o3[0]) and echk == c3)
        b3s, _ = N.build_timed(rk, rv)
        (fk, f1, f2), _ = N.fused_inner(
            xk[:1_000_000], xv[:1_000_000], b3s, threads=T)
        (sk, s1, s2), _ = OJ.join_inner(
            xk[:1_000_000], xv[:1_000_000], OJ.JoinBuild(rk, rv), threads=T)
        exact = (len(fk) == len(sk) and bool((fk == sk).all())
                 and bool((f1 == s1).all()) and bool((f2 == s2).all()))
        print(f"J5 n={n} s={n} rows={len(o3[0])} chk={c3} "
              f"slice1M-exact-bit={'GREEN' if exact else 'RED'} "
              f"full-chk-vs-OJ={'GREEN' if fullchk else 'RED'} "
              f"OJ E2E={oj_ms:.1f}ms M3 E2E={t3['e2e']:.1f}ms "
              f"(builds {obms:.1f}/{bm3:.1f}ms)", flush=True)
        del sk, s1, s2, fk, f1, f2, b3s, ek, e1, e2
    print(f"J5 elapsed={(_time.perf_counter() - t0):.1f}s", flush=True)
    del xk, xv, rk, rv, b3, o3
    _gc.collect()


if __name__ == "__main__":
    p = _ap.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["j1", "j4", "j5_10m", "j5_100m", "comp23"])
    a = p.parse_args()
    t0 = _time.perf_counter()
    if a.mode == "j1":
        mode_j1()
    elif a.mode == "j4":
        mode_j4()
    elif a.mode == "j5_10m":
        mode_j5(10_000_000, full_exact=True)
    elif a.mode == "j5_100m":
        mode_j5(100_000_000, full_exact=False)
    elif a.mode == "comp23":
        comp_parquet(["J2_10M", "J2_100M"], how="inner")
        comp_parquet(["J3_10M", "J3_100M"], how="left")
    print(f"elapsed={(_time.perf_counter() - t0):.1f}s", flush=True)
