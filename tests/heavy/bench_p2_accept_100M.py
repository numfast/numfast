# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P2 acceptance: production Q1-Q5, 100M/16T, NFS load-once, best-of-3.

Boundary mirrors bench_opt_phase3_100M (verbatim): NFS stream blocks
loaded ONCE (file IO outside all timers), per-block MT dense aggregate
(production _native_mt: P2 sum-only path where gated, proven path
elsewhere) + global dense merge, best-of-3, exact chk vs service refs.
Compares against the fb8989d baseline (phase3 NEW + research BASE).

Two harness lessons baked in (bench-only, production untouched):
- Builder exec() purges _lib.* from sys.modules: rebind _lib.cpu fresh
  post-build (top-level import would be a stale pre-build instance).
- Single-threaded native warmup outside timers (lazy ctypes config).

Usage (Git Bash, strictly sequential, timeout 550):
  NUMFAST_THREADS=16 PYTHONPATH=/c/App/numfast/numfast-ponytail:/c/App/numfast/app-builder-ponytail \
    /c/App/numfast/.venv/Scripts/python.exe tests/heavy/bench_p2_accept_100M.py
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
    from _lib.native_cpu import fused_sum_count as _nf1  # noqa: F401
    from _lib.native_cpu import mixed_available as _mok
    from _lib.native_cpu import pack_i32_direct as _nfpack
    from _lib.native_cpu import sum_count_i32 as _ni32  # noqa: F401
    assert _nok() and _mok(), "P2 kernels missing in working DLL"
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
          f"M1={M1} M2={M2} M3={M3} M4={M4} M5={M5} P2={_mok()}", flush=True)
    out = {"N": N, "T": t, "rows": {}}

    # ST warmup outside timers (see header).
    _wk = np.ascontiguousarray(np.array([0, 1, 0], dtype=np.int32))
    _wv = np.ascontiguousarray(np.array([1, 2, 3], dtype=np.int32))
    _wf = np.ascontiguousarray(np.array([1.0, 2.0, 3.0]))
    from _lib.native_cpu import mixed_sum_only_count as _mso
    assert _mso(_wk, [_wv, _wf], 2)[0][0].tolist() == [4, 2]
    print("warmup: ST incl. P2 path OK", flush=True)

    def run_query(tag, keyname, colnames, M, check, pack=None):
        best, chk, f64runs = None, None, []
        for _ in range(3):
            gc.collect()
            accs, cc = None, None
            s = time.perf_counter()
            for b in blocks:
                k = np.asarray(b[keyname])
                if pack == "old":
                    k = (np.asarray(b["k1"]).astype(np.int64)
                         * np.int64(M2k2)
                         + np.asarray(b["k2"]).astype(np.int64)).astype(np.int32)
                elif pack == "new":
                    k = _nfpack(np.ascontiguousarray(b["k1"], dtype=np.int32),
                                np.ascontiguousarray(b["k2"], dtype=np.int32),
                                M2k2)
                vs = [np.asarray(b[c]) for c in colnames]
                kb, vb = _C._probe_all(k, vs, t)
                dec, mm, _ = _C._dense_gate(k, plan_groupby,
                                            naggs=len(vs), _kb=kb)
                assert dec is not None, (tag, "gate")
                r = _C._native_mt(k, vs, int(mm), t, _vb=vb)
                assert r is not None, (tag, "agg")
                uk, cnt, sums, _, _, _ = r
                if accs is None:
                    accs = [np.zeros(M, dtype=np.asarray(x).dtype)
                            for x in sums]
                    cc = np.zeros(M, dtype=np.int64)
                ui = np.asarray(uk, dtype=np.int64)
                for d, sl in zip(accs, sums):
                    d[ui] += np.asarray(sl, dtype=d.dtype)
                cc[ui] += np.asarray(cnt, dtype=np.int64)
                del r, uk, cnt, sums
            dt = (time.perf_counter() - s) * 1000
            chk, ok, f64 = check(accs, cc)
            assert ok, (tag, chk)
            f64runs.append(f64)
            best = dt if best is None else min(best, dt)
            del accs, cc
        # f64 determinism: bit-identical across reps (same accumulation order)
        if f64runs[0] is not None:
            assert f64runs[1] == f64runs[0] and f64runs[2] == f64runs[0], \
                (tag, f64runs)
        print(f"{tag} best3={best:.1f}ms chk={chk} f64det=OK", flush=True)
        out["rows"][tag] = {"best3_ms": round(best, 1), "chk": chk}

    def _f64ok(x, ref):
        return abs(float(x) - ref) <= 1e-6 * abs(ref) + 1.0

    run_query("Q1", "k1", ["v1"], M1,
              lambda a, c: ({"total": int(a[0].sum())},
                            int(a[0].sum()) == refs["v1"], None))
    run_query("Q2-chain", "k1", ["v1"], M2,
              lambda a, c: ({"total": int(a[0].sum()),
                             "ng": int((a[0] != 0).sum())},
                            int(a[0].sum()) == refs["v1"]
                            and int((a[0] != 0).sum()) == 10000, None),
              pack="old")
    run_query("Q2-npack", "k1", ["v1"], M2,
              lambda a, c: ({"total": int(a[0].sum()),
                             "ng": int((a[0] != 0).sum())},
                            int(a[0].sum()) == refs["v1"]
                            and int((a[0] != 0).sum()) == 10000, None),
              pack="new")
    run_query("Q3", "k3", ["v1", "v3"], M3,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "v3": float(np.asarray(a[1],
                                                    dtype=np.float64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum())
                            == refs["v1"] and _f64ok(
                                float(np.asarray(a[1],
                                                 dtype=np.float64).sum()),
                                refs["v3"]),
                            float(np.asarray(a[1], dtype=np.float64).sum())))
    run_query("Q4", "id4", ["v1", "v2", "v3"], M4,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "s2": int(np.asarray(a[1], dtype=np.int64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum())
                            == refs["v1"] and int(np.asarray(a[1],
                                                             dtype=np.int64).sum())
                            == refs["v2"] and int(c.sum()) == N, None))
    run_query("Q5", "id6", ["v1", "v2", "v3"], M5,
              lambda a, c: ({"s1": int(np.asarray(a[0], dtype=np.int64).sum()),
                             "s2": int(np.asarray(a[1], dtype=np.int64).sum()),
                             "v3": float(np.asarray(a[2],
                                                    dtype=np.float64).sum())},
                            int(np.asarray(a[0], dtype=np.int64).sum())
                            == refs["v1"] and int(np.asarray(a[1],
                                                             dtype=np.int64).sum())
                            == refs["v2"] and _f64ok(
                                float(np.asarray(a[2],
                                                 dtype=np.float64).sum()),
                                refs["v3"]),
                            float(np.asarray(a[2], dtype=np.float64).sum())))

    o = FORK / "scratch" / "bench_p2_accept_100M.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    print(f"JSON {o}", flush=True)


_C = None

if __name__ == "__main__":
    main()
