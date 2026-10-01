# S6 NATIVE SHOWDOWN — GroupedHashMT Numba-warm vs native-warm vs native-cold.
# ClickBench 10M Q9 (RegionID x DISTINCT UserID) kernel-level + exact gates
# (DuckDB full map, 3586 groups) + 300-case fuzz (seed 42, lexsort oracle)
# + MT ladder. Old Numba lane intact (fallback/reference, never modified);
# native lane = new `_lib/grouped_native_hash.py` (standalone, no numba
# import). Generic only: no hardcoded cardinalities, dtypes preserved.
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

P10 = "C:/App/competitions/ClickBench/data/hits_10m.parquet"
NAT_PATH = "C:/App/numfast/numfast/src/Drivers/GroupedHashMT/_lib/grouped_native_hash.py"
NUMBA_PATH = "C:/App/numfast/numfast/src/Drivers/GroupedHashMT/_lib/grouped_mt_hash.py"
TMP = "C:/Users/Mikech/AppData/Local/Temp/nfzip"
OUT = "C:/App/numfast/numfast/numfast-native/results/bench_ghash_mt.json"


def t():
    return time.perf_counter()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def med(v):
    return sorted(v)[len(v) // 2]


def main():
    nat = load(NAT_PATH, "gnh_s6")
    nb = load(NUMBA_PATH, "gmh_s6")
    assert nat.available(), nat.why()
    print("native backend: %s" % nat.why(), flush=True)

    import pyarrow.parquet as pq
    tbl = pq.read_table(P10, columns=["RegionID", "UserID"])
    RID = np.ascontiguousarray(tbl.column("RegionID").to_numpy(zero_copy_only=False))
    UID = np.ascontiguousarray(tbl.column("UserID").to_numpy(zero_copy_only=False))
    n = int(RID.size)
    print("10M N=%d kdt=%s vdt=%s cpus=%s" % (n, RID.dtype, UID.dtype, os.cpu_count()),
          flush=True)
    os.makedirs(TMP, exist_ok=True)
    np.save(os.path.join(TMP, "K.npy"), RID)
    np.save(os.path.join(TMP, "V.npy"), UID)

    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    s = t()
    ref = {int(k): int(u) for k, u in con.execute(
        "SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM read_parquet('%s') "
        "GROUP BY RegionID" % P10).fetchall()}
    print("duck ref: %.0fms groups=%d" % ((t() - s) * 1000, len(ref)), flush=True)

    def asmap(uk, cn):
        return {int(k): int(v) for k, v in zip(uk.tolist(), cn.tolist())}

    # Numba warm: first call compiles (discarded), then timed reps.
    s = t()
    _ = nb.grouped_distinct_hash(RID, UID)
    print("numba first-call (compile+run): %.0fms" % ((t() - s) * 1000), flush=True)
    nb_reps, nb_maps = [], None
    for i in range(5):
        s = t()
        uk, cn, hms, sms = nb.grouped_distinct_hash(RID, UID)
        ms = (t() - s) * 1000
        nb_reps.append(ms)
        nb_maps = asmap(uk, cn)
        print("numba-warm rep%d: %.0fms (hash %.0f+scan %.0f) exact=%s" % (
            i, ms, hms, sms, nb_maps == ref), flush=True)

    # Native warm: first call (pool+thread warm, no JIT), then reps.
    s = t()
    _ = nat.grouped_distinct_hash_native(RID, UID)
    print("native first-call: %.0fms" % ((t() - s) * 1000), flush=True)
    nat_reps, nat_maps = [], None
    for i in range(5):
        s = t()
        uk, cn, hms, sms = nat.grouped_distinct_hash_native(RID, UID)
        ms = (t() - s) * 1000
        nat_reps.append(ms)
        nat_maps = asmap(uk, cn)
        print("native-warm rep%d: %.0fms (hash %.0f+scan %.0f) exact=%s" % (
            i, ms, hms, sms, nat_maps == ref), flush=True)

    # Native cold: fresh interpreter per rep, native module only.
    # Proves "cold = without JIT at all" (subprocess asserts numba
    # never imported). Kernel ms is timed inside; wall covers
    # interpreter+import+mmap-load for context.
    cold_code = (
        "import importlib.util,sys,time,numpy as np,json;"
        "s=importlib.util.spec_from_file_location('g',r'%s');"
        "g=importlib.util.module_from_spec(s);s.loader.exec_module(g);"
        "K=np.load(r'%s',mmap_mode='r');V=np.load(r'%s',mmap_mode='r');"
        "a=time.perf_counter();uk,cn,h,sc=g.grouped_distinct_hash_native(K,V);"
        "k=(time.perf_counter()-a)*1000;"
        "print(json.dumps({'kernel_ms':k,'hash_ms':h,'scan_ms':sc,"
        "'groups':int(len(uk)),'sum_nunique':int(cn.sum()),"
        "'numba_absent':'numba' not in sys.modules}))"
        % (NAT_PATH, os.path.join(TMP, "K.npy"), os.path.join(TMP, "V.npy")))
    cold = []
    for i in range(3):
        s = t()
        r = subprocess.run([sys.executable, "-B", "-c", cold_code],
                           capture_output=True, text=True, timeout=600)
        wall = (t() - s) * 1000
        assert r.returncode == 0, r.stderr[-500:]
        d = json.loads(r.stdout.strip().splitlines()[-1])
        d["wall_ms"] = wall
        cold.append(d)
        print("native-cold rep%d: kernel=%.0fms (hash %.0f+scan %.0f) wall=%.0fms "
              "groups=%d numba_absent=%s" % (
                  i, d["kernel_ms"], d["hash_ms"], d["scan_ms"], wall,
                  d["groups"], d["numba_absent"]), flush=True)
        assert d["numba_absent"] is True
        assert d["groups"] == len(ref) and d["sum_nunique"] == sum(ref.values())

    # MT ladder (native T override; exact at every rung vs DuckDB map).
    ladder = {}
    for T in (1, 2, 4, 8, 12, 16):
        _ = nat.grouped_distinct_hash_native(RID, UID, nthreads=T)
        reps = []
        ok = True
        for i in range(3):
            s = t()
            uk, cn, hms, sms = nat.grouped_distinct_hash_native(RID, UID, nthreads=T)
            reps.append((t() - s) * 1000)
            ok = ok and (asmap(uk, cn) == ref)
        ladder[T] = {"median_ms": med(reps), "all_ms": [round(x) for x in reps],
                     "hash_ms": round(hms), "scan_ms": round(sms), "exact": ok}
        print("ladder T=%2d: median=%.0fms %s exact=%s" % (
            T, ladder[T]["median_ms"], reps, ok), flush=True)

    # Fuzz: 300 randomized cases, seed 42, lexsort oracle (s5 shapes).
    # Native-covered domains must match exactly; uint64-huge lanes must
    # raise the miss (fallback owns them) with the Numba lane exact.
    def oracle(K, V):
        if K.size == 0:
            return {}
        px = np.lexsort((V, K))
        sk, sv = K[px], V[px]
        out = {}
        ck, lastv, c = sk[0], sv[0], 1
        for i in range(1, sk.size):
            if sk[i] != ck:
                out[int(ck)] = c
                ck, lastv, c = sk[i], sv[i], 1
            elif sv[i] != lastv:
                lastv, c = sv[i], c + 1
        out[int(ck)] = c
        return out

    rng = np.random.default_rng(42)
    fails = miss = exact = 0
    t0 = t()
    for trial in range(300):
        shape = trial % 10
        nn = int(rng.integers(0, 60000)) if shape < 7 else int(rng.integers(0, 8))
        if shape == 0:
            K = rng.integers(-1000, 1000, size=nn).astype(np.int32)
            V = rng.integers(-10**6, 10**6, size=nn).astype(np.int64)
            native_case = True
        elif shape == 1:
            K = rng.integers(0, 2**31 - 1, size=nn).astype(np.int64)
            V = rng.integers(0, 2**40, size=nn).astype(np.int64)
            native_case = True
        elif shape == 2:
            K = rng.integers(0, 50, size=nn).astype(np.int64)
            V = rng.integers(-5, 5, size=nn).astype(np.int64)
            native_case = True
        elif shape == 3:
            K = rng.integers(-2**31, 2**31 - 1, size=nn).astype(np.int64)
            V = rng.integers(0, 2**31 - 1, size=nn).astype(np.int64)
            native_case = True
        elif shape == 4:
            K = (rng.integers(0, 2**40, size=nn).astype(np.int64) * (2**24 + 3))
            K = np.where(K > 2**31 - 1, (K % 200000).astype(np.int64), K).astype(np.int64)
            V = rng.integers(0, 10**6, size=nn).astype(np.int64)
            native_case = True
        elif shape == 5:
            K = rng.integers(0, 2**32 - 1, size=nn).astype(np.uint32)
            V = rng.integers(-10**6, 10**6, size=nn).astype(np.int64)
            native_case = True
        elif shape == 6:
            K = rng.integers(0, 1000, size=nn).astype(np.int64)
            V = rng.integers(0, 2**63 - 1, size=nn).astype(np.int64)
            native_case = True
        elif shape == 7:
            K = rng.integers(-3, 3, size=nn).astype(np.int64)
            V = rng.integers(-3, 3, size=nn).astype(np.int64)
            native_case = True
        elif shape == 8:
            K = np.full(nn, 7, dtype=np.int64)
            V = rng.integers(0, max(nn * 2, 2), size=nn).astype(np.int64)
            native_case = True
        else:
            K = rng.integers(0, 100, size=nn).astype(np.int64)
            V = rng.choice(np.array([-2**63, 2**63 - 1, -1, 0, 1], dtype=np.int64),
                           size=nn).astype(np.int64)
            native_case = True
        exp = oracle(np.ascontiguousarray(K).ravel(), np.ascontiguousarray(V).ravel())
        try:
            uk, cn, _, _ = nat.grouped_distinct_hash_native(K, V)
            got = {int(k): int(v) for k, v in zip(uk.tolist(), cn.tolist())}
            if got == exp:
                exact += 1
            else:
                print("FUZZ MISMATCH t%d shape=%d n=%d" % (trial, shape, nn), flush=True)
                fails += 1
                if fails > 3:
                    break
        except Exception as e:
            print("FUZZ NATIVE-RAISED t%d shape=%d n=%d %r" % (trial, shape, nn, e),
                  flush=True)
            fails += 1
            if fails > 3:
                break
        if trial % 50 == 49:  # determinism repeat
            uk2, cn2, _, _ = nat.grouped_distinct_hash_native(K, V)
            if not (np.array_equal(uk, uk2) and np.array_equal(cn, cn2)):
                print("FUZZ NONDETERMINISM t%d" % trial, flush=True)
                fails += 1
                break
    # Miss gates: uint64-huge values + non-integer keys must miss cleanly.
    for K, V, tag in [
            (np.array([1, 1, 2], dtype=np.int64),
             np.array([0, 2**64 - 1, 0], dtype=np.uint64), "u64-huge-val"),
            (np.array([1.5, 2.5], dtype=np.float64),
             np.array([1, 1], dtype=np.int64), "float-key")]:
        try:
            nat.grouped_distinct_hash_native(K, V)
            print("MISS-GATE %s: NO-MISS (unexpected)" % tag, flush=True)
            fails += 1
        except Exception:
            miss += 1
            print("MISS-GATE %s: clean miss" % tag, flush=True)
    print("fuzz: %d/300 exact, miss-gates %d/2 clean, %.0fs" % (exact, miss, t() - t0),
          flush=True)

    out = {
        "n": n, "kdt": str(RID.dtype), "vdt": str(UID.dtype),
        "groups": len(ref), "cpus": os.cpu_count(),
        "numba_warm_ms": {"median": med(nb_reps), "all": [round(x) for x in nb_reps],
                          "exact": nb_maps == ref},
        "native_warm_ms": {"median": med(nat_reps), "all": [round(x) for x in nat_reps],
                           "exact": nat_maps == ref},
        "native_cold": cold,
        "ladder": ladder,
        "fuzz": {"exact": exact, "of": 300, "miss_gates": miss, "fails": fails},
    }
    Q = out["numba_warm_ms"]["median"] / out["native_warm_ms"]["median"]
    out["native_vs_numba_warm"] = Q
    json.dump(out, open(OUT, "w"), indent=1)
    print("S6 DONE groups=%d numba=%.0f native=%.0f speedup=%.2f fuzz=%d/300 fails=%d" % (
        len(ref), out["numba_warm_ms"]["median"], out["native_warm_ms"]["median"],
        Q, exact, fails), flush=True)


if __name__ == "__main__":
    main()
