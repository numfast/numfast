# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Q4-B acceptance: cached-bounds direct MT vs verbatim, 100M/16T + gate.

Q4 (M=101, mean v1/v2/v3 by id4): per-block cached kb=(0,M-1) + vb from
NFS index-maxs (zero scans) + _dense_gate (Planner untouched) +
_C._native_mt_direct (fail-fast: checked i32 lane validates keys before
any unchecked lane; any doubt -> verbatim block fallback). Verbatim Q4
once as bit-exact reference (maxdiff==0.0 required). Gate Q1/Q2/Q3/Q5 run
the verbatim production chain (untouched) best-of-3 vs
scratch/bench_p2_accept_100M.json baselines (no regressions).
Boundary: NFS blocks resident once (IO outside timers); best-of-3 E2E.
Usage (Git Bash, sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_q4_direct_100M.py
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
_CPU_LIB = str(FORK / "src" / "Drivers" / "CPU")
if _CPU_LIB not in sys.path:
    sys.path.insert(0, _CPU_LIB)
import numpy as np


def main():
    global _C
    t = max(1, min(64, int(os.environ.get("NUMFAST_THREADS", "1"))))
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    import importlib as _il
    for _m in [m for m in sys.modules if m == "_lib" or m.startswith("_lib.")]:
        del sys.modules[_m]
    import _lib.cpu as _Cf
    _C = _Cf
    a = kernel.alias
    plan_groupby = a["plan_groupby"]
    from _lib.native_cpu import available as _nok
    from _lib.native_cpu import mixed_sum_only_count as _mso
    from _lib.native_cpu import pack_i32_direct as _nfpack
    assert _nok(), "native DLL missing"
    base = json.loads((FORK / "scratch" / "bench_p2_accept_100M.json").read_text())
    nfs = str(FORK / "scratch" / "nfs_stream" / "G1_1e8.stream.nfs")
    h = a["nfs_stream_open"](nfs, force_lazy=True)
    idx = h["index"]
    refs = h["service"]["refs"]
    N = int(h["header"]["N"])
    M1 = int(max(m["maxs"]["k1"] for m in idx)) + 1
    M2k2 = int(max(m["maxs"]["k2"] for m in idx)) + 1
    M2 = M1 * M2k2
    M3 = int(max(m["maxs"]["k3"] for m in idx)) + 1
    M4 = int(max(m["maxs"]["id4"] for m in idx)) + 1
    M5 = int(max(m["maxs"]["id6"] for m in idx)) + 1
    V1MAX = int(max(m["maxs"]["v1"] for m in idx))
    V2MAX = int(max(m["maxs"]["v2"] for m in idx))
    t0 = time.perf_counter()
    blocks = []
    with open(nfs, "rb") as fh:
        for m in idx:
            fh.seek(int(m["file_offset"]))
            raw = fh.read(int(m["blob_len"]))
            z = np.load(io.BytesIO(raw))
            blocks.append({k: np.asarray(z[k]) for k in
                           ("k1", "k2", "k3", "id4", "id6", "v1", "v2", "v3")})
            z.close()
            del raw
    print(f"resident rows={sum(len(b['k1']) for b in blocks)} "
          f"load={(time.perf_counter()-t0)*1000:.0f}ms T={t} N={N} "
          f"M1={M1} M2={M2} M3={M3} M4={M4}", flush=True)
    _wk = np.ascontiguousarray(np.array([0, 1, 0], dtype=np.int32))
    _wv = np.ascontiguousarray(np.array([1, 2, 3], dtype=np.int32))
    _wf = np.ascontiguousarray(np.array([1.0, 2.0, 3.0]))
    assert _mso(_wk, [_wv, _wf], 2)[0][0].tolist() == [4, 2]
    print("warmup: ST incl. P2 path OK", flush=True)
    out = {"N": N, "T": t, "rows": {}}

    def verbatim_block(k, vs):
        kb, vb = _C._probe_all(k, vs, t)
        dec, mm, _ = _C._dense_gate(k, plan_groupby, naggs=len(vs), _kb=kb)
        assert dec is not None
        r = _C._native_mt(k, vs, int(mm), t, _vb=vb)
        assert r is not None
        return r

    def merge_global(M, sums_list, uk, cnt):
        accs = [np.zeros(M, dtype=np.asarray(x).dtype) for x in sums_list]
        cc = np.zeros(M, dtype=np.int64)
        ui = np.asarray(uk, dtype=np.int64)
        for d, sl in zip(accs, sums_list):
            d[ui] += np.asarray(sl, dtype=d.dtype)
        cc[ui] += np.asarray(cnt, dtype=np.int64)
        return accs, cc

    # ---- Q4 verbatim reference (once) ----
    gc.collect()
    raccs, rcc = None, None
    s = time.perf_counter()
    for b in blocks:
        k = np.asarray(b["id4"])
        vs = [np.asarray(b[c]) for c in ("v1", "v2", "v3")]
        uk, cnt, sums, _, _, _ = verbatim_block(k, vs)
        a4, c4 = merge_global(M4, sums, uk, cnt) if raccs is None else (None, None)
        if raccs is None:
            raccs, rcc = a4, c4
        else:
            ui = np.asarray(uk, dtype=np.int64)
            for d, sl in zip(raccs, sums):
                d[ui] += np.asarray(sl, dtype=d.dtype)
            rcc[ui] += np.asarray(cnt, dtype=np.int64)
    verb_ms = (time.perf_counter() - s) * 1000
    assert int(raccs[0].sum()) == refs["v1"] and int(raccs[1].sum()) == refs["v2"]
    print(f"Q4-verbatim {verb_ms:.1f}ms ref-ok", flush=True)

    # ---- Q4 direct best-of-3 ----
    def q4direct_once():
        accs, cc, fb = None, None, 0
        s = time.perf_counter()
        kb = (0, M4 - 1)
        vb = [(0, V1MAX), (0, V2MAX), None]
        for b in blocks:
            k = np.asarray(b["id4"])
            vs = [np.asarray(b[c]) for c in ("v1", "v2", "v3")]
            dec, mm, _ = _C._dense_gate(k, plan_groupby, naggs=3, _kb=kb)
            r = None
            if dec is not None and int(mm) == M4:
                r = _C._native_mt_direct(k, vs, int(mm), t, vb)
            if r is None:
                fb += 1
                r = verbatim_block(k, vs)
            uk, cnt, sums, _, _, _ = r
            if accs is None:
                accs, cc = merge_global(M4, sums, uk, cnt)
            else:
                ui = np.asarray(uk, dtype=np.int64)
                for d, sl in zip(accs, sums):
                    d[ui] += np.asarray(sl, dtype=d.dtype)
                cc[ui] += np.asarray(cnt, dtype=np.int64)
            del r, uk, cnt, sums
        return (time.perf_counter() - s) * 1000, accs, cc, fb

    gc.collect()
    dt0, a0, c0, fb0 = q4direct_once()
    md0 = max(float(np.max(np.abs(np.asarray(a0[2], dtype=np.float64)
                                  - np.asarray(raccs[2], dtype=np.float64)))),
              float(np.max(np.abs((np.asarray(a0[0], dtype=np.float64))
                                  - np.asarray(raccs[0], dtype=np.float64)))),
              float(np.max(np.abs((np.asarray(a0[1], dtype=np.float64))
                                  - np.asarray(raccs[1], dtype=np.float64)))))
    ex0 = bool((np.asarray(a0[0]) == np.asarray(raccs[0])).all()
               and (np.asarray(a0[1]) == np.asarray(raccs[1])).all()
               and (np.asarray(c0) == np.asarray(rcc)).all())
    print(f"Q4-direct cold={dt0:.1f}ms fb={fb0} maxdiff_vs_verbatim={md0:.3g} "
          f"int_exact={ex0} chk=({int(a0[0].sum())},{int(a0[1].sum())})", flush=True)
    assert fb0 == 0 and md0 == 0.0 and ex0
    best, runs = 1e18, []
    for _ in range(3):
        gc.collect()
        dt, accs, cc, fb = q4direct_once()
        assert fb == 0
        assert int(accs[0].sum()) == refs["v1"] and int(accs[1].sum()) == refs["v2"] \
            and int(cc.sum()) == N
        md = float(np.max(np.abs(np.asarray(accs[2], dtype=np.float64)
                                 - np.asarray(raccs[2], dtype=np.float64))))
        assert md == 0.0, md
        runs.append(dt)
        best = min(best, dt)
    means = [float((np.asarray(a0[i], dtype=np.float64)
                    / np.maximum(np.asarray(c0, dtype=np.float64), 1.0)).sum())
             for i in range(3)]
    print(f"Q4-direct best3={best:.1f}ms runs={[round(x,1) for x in runs]} "
          f"means={[round(x,6) for x in means]}", flush=True)
    out["rows"]["Q4-direct"] = {"best3_ms": round(best, 1), "verbatim_ms": round(verb_ms, 1),
                                "maxdiff": 0.0, "means": means,
                                "chk": {"s1": int(a0[0].sum()), "s2": int(a0[1].sum())}}
    out["rows"]["Q4-baseline"] = base["rows"]["Q4"]
    del raccs, rcc, a0, c0, accs, cc

    # ---- gate Q1/Q2/Q3/Q5 (verbatim production chain, untouched) ----
    def run_query(tag, keyname, colnames, M, check, pack=None):
        best, chk = None, None
        for _ in range(3):
            gc.collect()
            accs, cc = None, None
            s = time.perf_counter()
            for b in blocks:
                k = np.asarray(b[keyname])
                if pack == "new":
                    k = _nfpack(np.ascontiguousarray(b["k1"], dtype=np.int32),
                                np.ascontiguousarray(b["k2"], dtype=np.int32), M2k2)
                vs = [np.asarray(b[c]) for c in colnames]
                uk, cnt, sums, _, _, _ = verbatim_block(k, vs)
                if accs is None:
                    accs = [np.zeros(M, dtype=np.asarray(x).dtype) for x in sums]
                    cc = np.zeros(M, dtype=np.int64)
                ui = np.asarray(uk, dtype=np.int64)
                for d, sl in zip(accs, sums):
                    d[ui] += np.asarray(sl, dtype=d.dtype)
                cc[ui] += np.asarray(cnt, dtype=np.int64)
                del uk, cnt, sums
            dt = (time.perf_counter() - s) * 1000
            chk, ok, _ = check(accs, cc)
            assert ok, (tag, chk)
            best = dt if best is None else min(best, dt)
            del accs, cc
        print(f"{tag} best3={best:.1f}ms chk={chk} base={base['rows'][tag]['best3_ms']}",
              flush=True)
        assert best <= float(base["rows"][tag]["best3_ms"]) * 1.05, (tag, best)
        out["rows"][tag] = {"best3_ms": round(best, 1), "chk": chk,
                            "base_ms": base["rows"][tag]["best3_ms"]}

    def _f64ok(x, ref):
        return abs(float(x) - ref) <= 1e-6 * abs(ref) + 1.0

    run_query("Q1", "k1", ["v1"], M1,
              lambda a, c: ({"total": int(a[0].sum())},
                            int(a[0].sum()) == refs["v1"], None))
    run_query("Q2-npack", "k1", ["v1"], M2,
              lambda a, c: ({"total": int(a[0].sum()), "ng": int((a[0] != 0).sum())},
                            int(a[0].sum()) == refs["v1"]
                            and int((a[0] != 0).sum()) == 10000, None), pack="new")
    run_query("Q3", "k3", ["v1", "v3"], M3,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "v3": float(np.asarray(a[1], dtype=np.float64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum()) == refs["v1"]
                            and _f64ok(float(np.asarray(a[1], dtype=np.float64).sum()),
                                       refs["v3"]), None))
    run_query("Q5", "id6", ["v1", "v2", "v3"], M5,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "s2": int(np.asarray(a[1], dtype=np.int64).sum()),
                             "v3": float(np.asarray(a[2], dtype=np.float64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum()) == refs["v1"]
                            and int(np.asarray(a[1], dtype=np.int64).sum()) == refs["v2"]
                            and _f64ok(float(np.asarray(a[2], dtype=np.float64).sum()),
                                       refs["v3"]), None))
    o = FORK / "tests" / "heavy" / "bench_q4_direct_100M.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o}", flush=True)


_C = None

if __name__ == "__main__":
    main()
