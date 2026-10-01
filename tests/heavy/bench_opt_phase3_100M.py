# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase3: OLD vs NEW production aggregate path, 100M/16T, load-once batch.

OLD = verbatim HEAD logic (int-exact gate + f64 fused kernels), inlined here
from untouched functions; NEW = modified _lib.cpu (_native_st/_native_mt with
int-lane routing). Same resident blocks, same T=16 pool, same probe+gate
inputs, per-block aggregate + global dense merge, best-of-3, exact chk.
Bench-only. No GPU. No new deps. Format NFS untouched.

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_opt_phase3_100M.py
"""
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
for _p in (str(FORK.parent / "app-builder-ponytail"), str(FORK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

_CPU_LIB = str(FORK / "src" / "Drivers" / "CPU")
if _CPU_LIB not in sys.path:
    sys.path.insert(0, _CPU_LIB)
import _lib.cpu as _C  # noqa: E402 (modified production module)
from _lib.native_cpu import fused_sum_count as _nf1  # noqa: E402
from _lib.native_cpu import multi_sum_count as _nfN  # noqa: E402
from _lib.native_cpu import pack_i32_direct as _nfpack  # noqa: E402


# ---- OLD (verbatim HEAD _native_st/_native_mt bodies, untouched callees) ----
def _old_st(keys, vcols, m, _vb=_C._SENT):
    try:
        keys = np.ascontiguousarray(keys, dtype=np.int32)
        cols = [np.ascontiguousarray(v) for v in vcols]
        if not _C._int_exact_ok(cols, _vb):
            return None
        t0 = time.perf_counter()
        if len(cols) == 1:
            sums_dense, counts_m = _nf1(keys, cols[0], m)
            sums_list = [sums_dense]
        else:
            sums_list, counts_m = _nfN(keys, cols, m)
        agg_ms = (time.perf_counter() - t0) * 1000
    except Exception:  # noqa: BLE001
        return None
    try:
        ukeys, counts, out, cms = None, None, [], 0.0
        for s_m in sums_list:
            uk, c, s, cm = _C._native_compact(counts_m, s_m)
            if ukeys is None:
                ukeys, counts = uk, c
            cms += cm
            out.append(s)
        return ukeys, counts, out, agg_ms, cms
    except Exception:  # noqa: BLE001
        return None


def _old_mt(keys, vcols, m, t, _vb=_C._SENT):
    try:
        keys = np.ascontiguousarray(keys)
        cols = [np.ascontiguousarray(v) for v in vcols]
        if not _C._int_exact_ok(cols, _vb):
            return None
        n = int(keys.size)
        t = max(1, min(int(t), n))
        if t == 1:
            r = _old_st(keys, cols, m)
            return (*r, 0.0) if r else None
        bounds = np.linspace(0, n, t + 1).astype(np.int64)
        k32 = np.ascontiguousarray(keys, dtype=np.int32)
        ex = _C._pool(t)
        t0 = time.perf_counter()
        futs = []
        for w in range(t):
            a, b = int(bounds[w]), int(bounds[w + 1])
            kk = k32[a:b]
            vv = [c[a:b] for c in cols]

            def _work(kk=kk, vv=vv):
                if len(vv) == 1:
                    sums_dense, cc = _nf1(kk, vv[0], m)
                    return [sums_dense], cc
                return _nfN(kk, vv, m)

            futs.append(ex.submit(_work))
        parts = [f.result() for f in futs]
        agg_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        ncols = len(parts[0][0])
        sums_m = [np.zeros(m, dtype=s.dtype) for s in parts[0][0]]
        counts_m = np.zeros(m, dtype=np.int64)
        if ncols > 1:
            def _mcol(j):
                a = sums_m[j]
                for ss, _cc in parts:
                    a += ss[j]
            list(ex.map(_mcol, range(ncols)))
            for _ss, cc in parts:
                counts_m += cc
        else:
            for ss, cc in parts:
                for a, b in zip(sums_m, ss):
                    a += b
                counts_m += cc
        merge_ms = (time.perf_counter() - t1) * 1000
    except Exception:  # noqa: BLE001
        return None
    try:
        ukeys, counts, out, cms = None, None, [], 0.0
        for s_m in sums_m:
            uk, c, s, cm = _C._native_compact(counts_m, s_m)
            if ukeys is None:
                ukeys, counts = uk, c
            cms += cm
            out.append(s)
        return ukeys, counts, out, agg_ms, cms, merge_ms
    except Exception:  # noqa: BLE001
        return None


def main():
    t = max(1, min(64, int(os.environ.get("NUMFAST_THREADS", "1"))))
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    plan_groupby = a["plan_groupby"]
    h = a["nfs_stream_open"](
        str(FORK / "scratch" / "nfs_stream" / "G1_1e8.stream.nfs"), force_lazy=True)
    idx = h["index"]
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2k2 = int(max(m["maxs"]["k2"] for m in idx)) + 1
    M2 = M1 * M2k2
    M3 = int(max(m["maxs"]["k3"] for m in idx)) + 1
    M4 = int(max(m["maxs"]["id4"] for m in idx)) + 1
    M5 = int(max(m["maxs"]["id6"] for m in idx)) + 1
    t0 = time.perf_counter()
    blocks = []
    with open(str(FORK / "scratch" / "nfs_stream" / "G1_1e8.stream.nfs"), "rb") as fh:
        for m in idx:
            fh.seek(int(m["file_offset"]))
            raw = fh.read(int(m["blob_len"]))
            z = np.load(io.BytesIO(raw))
            blocks.append({k: np.asarray(z[k]) for k in
                           ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")})
            z.close()
            del raw
    print(f"resident rows={sum(len(b['k1']) for b in blocks)} "
          f"load={(time.perf_counter()-t0)*1000:.0f}ms T={t} "
          f"M1={M1} M2={M2} M3={M3} M4={M4} M5={M5}", flush=True)
    out = {"N": N, "T": t, "rows": {}}

    def run_query(tag, keyname, colnames, M, check, pack=None):
        """pack: None or ('old'|'new') composite builder for Q2."""
        res = {}
        for mode in ("OLD", "NEW"):
            best = None
            chk = None
            for _ in range(3):
                gc.collect()
                accs = None
                cc = None
                s = time.perf_counter()
                for b in blocks:
                    k = np.asarray(b[keyname])
                    if pack == "old":
                        k = (np.asarray(b["k1"]).astype(np.int64) * np.int64(M2k2)
                             + np.asarray(b["k2"]).astype(np.int64)).astype(np.int32)
                    elif pack == "new":
                        k = _nfpack(np.ascontiguousarray(b["k1"], dtype=np.int32),
                                    np.ascontiguousarray(b["k2"], dtype=np.int32), M2k2)
                    vs = [np.asarray(b[c]) for c in colnames]
                    kb, vb = _C._probe_all(k, vs, t)
                    dec, mm, _ = _C._dense_gate(k, plan_groupby, naggs=len(vs), _kb=kb)
                    assert dec is not None, (tag, mode, "gate")
                    fn = _old_mt if mode == "OLD" else _C._native_mt
                    r = fn(k, vs, int(mm), t, _vb=vb)
                    assert r is not None, (tag, mode, "agg")
                    uk, cnt, sums, _, _, _ = r
                    if accs is None:
                        accs = [np.zeros(M, dtype=np.asarray(s).dtype) for s in sums]
                        cc = np.zeros(M, dtype=np.int64)
                    ui = np.asarray(uk, dtype=np.int64)
                    for d, sl in zip(accs, sums):
                        d[ui] += np.asarray(sl, dtype=d.dtype)
                    cc[ui] += np.asarray(cnt, dtype=np.int64)
                    del r, uk, cnt, sums
                dt = (time.perf_counter() - s) * 1000
                chk, ok = check(accs, cc)
                assert ok, (tag, mode, chk)
                best = dt if best is None else min(best, dt)
                del accs, cc
            print(f"{tag} {mode} best3={best:.1f}ms chk={chk}", flush=True)
            res[mode] = {"best3_ms": round(best, 1), "chk": chk}
        res["speedup"] = round(res["OLD"]["best3_ms"] / res["NEW"]["best3_ms"], 2)
        print(f"{tag} SPEEDUP {res['speedup']}x", flush=True)
        out["rows"][tag] = res

    run_query("Q1", "k1", ["v1"], M1,
              lambda a, c: ({"total": int(a[0].sum())},
                            int(a[0].sum()) == refs["v1"]))
    run_query("Q2", "k1", ["v1"], M2,
              lambda a, c: ({"total": int(a[0].sum()),
                             "ng": int((a[0] > 0).sum())},
                            int(a[0].sum()) == refs["v1"] and int((a[0] > 0).sum()) == 10000),
              pack="old")
    run_query("Q2np", "k1", ["v1"], M2,
              lambda a, c: ({"total": int(a[0].sum()),
                             "ng": int((a[0] > 0).sum())},
                            int(a[0].sum()) == refs["v1"] and int((a[0] > 0).sum()) == 10000),
              pack="new")
    run_query("Q3", "k3", ["v1", "v3"], M3,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "v3": float(np.asarray(a[1], dtype=np.float64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum()) == refs["v1"] and
                            abs(float(np.asarray(a[1], dtype=np.float64).sum()) - refs["v3"])
                            <= 1e-6 * abs(refs["v3"]) + 1.0))
    run_query("Q4", "id4", ["v1", "v2", "v3"], M4,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "s2": int(np.asarray(a[1], dtype=np.int64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum()) == refs["v1"] and
                            int(np.asarray(a[1], dtype=np.int64).sum()) == refs["v2"] and
                            int(c.sum()) == N))
    run_query("Q5", "id6", ["v1", "v2", "v3"], M5,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "s2": int(np.asarray(a[1], dtype=np.int64).sum()),
                             "v3": float(np.asarray(a[2], dtype=np.float64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum()) == refs["v1"] and
                            int(np.asarray(a[1], dtype=np.int64).sum()) == refs["v2"] and
                            abs(float(np.asarray(a[2], dtype=np.float64).sum()) - refs["v3"])
                            <= 1e-6 * abs(refs["v3"]) + 1.0))

    o = FORK / "tests" / "heavy" / "bench_opt_phase3_100M.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o}", flush=True)


if __name__ == "__main__":
    main()
