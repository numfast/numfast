#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SESSION ARTIFACT (not src, NOT committed): group_index integration gate.

Exact correctness of the INTEGRATED path (planner gate + physical kernels)
vs np.unique on all modes: M=100/10K/100K/1M/10M x uniform/skewed/sorted/
sparse-range, plus edges (empty, n=1, all-equal, negatives, [0,1e9] trap,
packed composite). Every physical strategy is also compared vs np.unique
directly. Contracts: ukeys exact + sorted, counts/sums int exact, float sums
rtol<=1e-9 (reduceat order may differ in last ulp), inverse contract
ukeys[inverse]==keys. Seed 42.

Usage (Git Bash):
  .../python.exe .session-groupindex/check_groupindex_correct.py
Writes .session-groupindex/results_correct.json
"""
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORK = HERE.parent
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


from builder import MAIN

_kernel = MAIN["build"](str(FORK))
_plan = _kernel.alias["plan_groupby"]
# session-only direct import: resolve THIS extension's _lib (builder already
# loaded all extensions; CPU dir first so _lib == CPU/_lib).
sys.path.insert(0, str(FORK / "src" / "Drivers" / "CPU"))
from _lib.groupindex import (dense_direct_index, dense_estimate_bytes,
                              group_index, hash_dynamic_index, is_sorted_early,
                              range_probe, sorted_run_index,
                              unique_fallback_index)

SEED = 42
BUDGET = _kernel.alias["cpu_capability"]()["max_buffer_bytes"]
print(f"budget(max_buffer_bytes)={BUDGET/1e6:.1f}MB", flush=True)


def gen(M, dist, n, rng):
    if dist == "uniform":
        return rng.integers(0, M, size=n, dtype=np.int64)
    if dist == "skewed":
        z = rng.zipf(1.5, size=n)
        z = np.minimum(z, np.int64(10 * M + 7))
        return (z % np.int64(M)).astype(np.int64)
    if dist == "sorted":
        return np.sort(rng.integers(0, M, size=n, dtype=np.int64))
    if dist == "sparse":
        # half 0, half 1e9-class far key: trap for max-min-only gates
        k = np.zeros(n, dtype=np.int64)
        k[n // 2:] = np.int64(1_000_000_000)
        rng.shuffle(k)
        return k
    if dist == "neg":
        return rng.integers(-M, M, size=n, dtype=np.int64)
    if dist == "packed":
        hi = rng.integers(0, 100, size=n, dtype=np.int64)
        lo = rng.integers(0, 100, size=n, dtype=np.int64)
        return (hi << np.int64(32)) | lo
    raise ValueError(dist)


def ref(keys, vals_i, vals_f):
    uk, inv = np.unique(keys, return_inverse=True)
    ng = uk.size
    cnt = np.bincount(inv, minlength=ng).astype(np.int64)
    si = np.bincount(inv, weights=vals_i.astype(np.int64), minlength=ng).astype(np.int64)
    sf = np.bincount(inv, weights=vals_f.astype(np.float64), minlength=ng)
    return uk, inv, cnt, si, sf


def check_case(gi, keys, ruk, rcnt, rsi, rsf, label, strat_fn_name):
    errs = []
    uk = np.asarray(gi["ukeys"])
    if uk.size != ruk.size:
        return [f"ngroups {uk.size}!={ruk.size}"]
    if not np.array_equal(uk, ruk):
        errs.append("ukeys mismatch")
    if not bool(np.all(uk[1:] >= uk[:-1])) if uk.size > 1 else False:
        errs.append("ukeys not sorted")
    inv = gi["inverse"]
    if inv is not None:
        inv = np.asarray(inv)
        if not np.array_equal(uk[inv], np.asarray(keys)):
            errs.append("inverse contract ukeys[inverse]==keys FAILED")
        cnt = np.bincount(inv, minlength=uk.size).astype(np.int64)
        if not np.array_equal(cnt, rcnt):
            errs.append("counts mismatch")
    else:
        st = np.asarray(gi["starts"])
        ends = np.empty(st.size, dtype=np.int64)
        ends[:-1] = st[1:]
        ends[-1] = gi["n"]
        if not np.array_equal(ends - st, rcnt):
            errs.append("run-length counts mismatch")
    return errs


def run_case(label, keys):
    keys = np.asarray(keys)
    n = keys.size
    rng = np.random.default_rng(SEED + n + len(label))
    vals_i = rng.integers(1, 100, size=n).astype(np.int64)
    vals_f = (rng.random(size=n) * 100).astype(np.float64)
    out = {"label": label, "n": n, "ok": True, "errors": []}
    if n == 0:
        gi, dec = group_index(keys, _plan)
        assert gi["strategy"] == "empty" and gi["ukeys"].size == 0
        out.update({"strategy": "empty", "reason": dec.get("reason")})
        print(f"{label}: empty OK", flush=True)
        return out
    t0 = time.perf_counter()
    ruk, rinv, rcnt, rsi, rsf = ref(keys, vals_i, vals_f)
    out["ref_ms"] = (time.perf_counter() - t0) * 1000
    # integrated decision
    r0 = rss()
    t0 = time.perf_counter()
    gi, dec = group_index(keys, _plan)
    out["gi_ms"] = (time.perf_counter() - t0) * 1000
    out["rss_gb"] = rss() - r0
    out["strategy"] = gi["strategy"]
    out["reason"] = dec.get("reason")
    kmn, kmx, R = (range_probe(keys) if n else (0, 0, 0))
    out["R"] = R
    out["est_dense_MB"] = dense_estimate_bytes(R, n) / 1e6
    errs = check_case(gi, keys, ruk, rcnt, rsi, rsf, label, "integrated")
    # sums check on integrated inverse/starts path (same arithmetic as driver)
    if gi["inverse"] is not None:
        inv = np.asarray(gi["inverse"])
        si = np.bincount(inv, weights=vals_i.astype(np.int64),
                         minlength=ruk.size).astype(np.int64)
        sf = np.bincount(inv, weights=vals_f, minlength=ruk.size)
    else:
        si = np.add.reduceat(vals_i.astype(np.int64), np.asarray(gi["starts"]))
        sf = np.add.reduceat(vals_f.astype(np.float64), np.asarray(gi["starts"]))
    if not np.array_equal(si, rsi):
        errs.append("int sums mismatch")
    den = np.maximum(np.abs(rsf), 1e-300)
    if not bool(np.all(np.abs(sf - rsf) / den <= 1e-9)):
        errs.append(f"float sums rel-err {np.max(np.abs(sf-rsf)/den):.2e} > 1e-9")
    # every physical strategy directly vs ref
    strat_tests = [("unique", lambda: unique_fallback_index(keys))]
    if bool(np.all(keys[1:] >= keys[:-1])) if n > 1 else True:
        strat_tests.append(("sorted", lambda: sorted_run_index(keys)))
    if R <= min(64_000_000, 4 * n) and dense_estimate_bytes(R, n) <= BUDGET:
        strat_tests.append(("dense", lambda: dense_direct_index(keys, kmn, R)))
    else:
        out["dense_skipped_gate"] = True
    strat_tests.append(("hash", lambda: hash_dynamic_index(keys)))
    for sname, fn in strat_tests:
        r0 = rss()
        t0 = time.perf_counter()
        sgi = fn()
        ms = (time.perf_counter() - t0) * 1000
        mem = rss() - r0
        se = check_case(sgi, keys, ruk, rcnt, rsi, rsf, label, sname)
        if sname in ("dense", "hash", "unique") and sgi["inverse"] is not None:
            si2 = np.bincount(np.asarray(sgi["inverse"]),
                              weights=vals_i.astype(np.int64),
                              minlength=ruk.size).astype(np.int64)
            if not np.array_equal(si2, rsi):
                se.append("int sums mismatch")
        out.setdefault("strategies", {})[sname] = {
            "ms": ms, "mem_gb": mem, "ok": not se, "errors": se}
        if se:
            errs.append(f"{sname}: {se}")
        del sgi
        gc.collect()
    out["errors"] = errs
    out["ok"] = not errs
    status = "OK " if out["ok"] else "FAIL"
    print(f"{status} {label}: n={n} R={R} -> {out['strategy']} "
          f"gi={out['gi_ms']:.1f}ms (ref {out['ref_ms']:.1f}ms) "
          f"mem+{out['rss_gb']:.2f}GB reason: {out['reason']} "
          f"{errs if errs else ''}", flush=True)
    del keys, vals_i, vals_f, ruk, rinv, rcnt, rsi, rsf, gi
    gc.collect()
    return out


def main():
    # numba warmup (compile excluded from timings)
    group_index(np.array([2, 1, 2, 1, 0, 3], dtype=np.int64), _plan)
    hash_dynamic_index(np.array([5, 4, 5], dtype=np.int64))
    print("warmup done", flush=True)
    results = []
    # sweep A: N=1M, full M x dist (gates scale identically to 10M)
    N1 = 1_000_000
    for M in (100, 10_000, 100_000, 1_000_000, 10_000_000):
        for dist in ("uniform", "skewed", "sorted"):
            rng = np.random.default_rng(SEED)
            results.append(run_case(f"A/M={M}/{dist}", gen(M, dist, N1, rng)))
    # edges
    rng = np.random.default_rng(SEED)
    results.append(run_case("edge/empty", np.zeros(0, dtype=np.int64)))
    results.append(run_case("edge/n=1", np.array([7], dtype=np.int64)))
    results.append(run_case("edge/all-equal-1K", np.full(1000, 5, dtype=np.int64)))
    results.append(run_case("edge/trap-0-vs-1e9",
                            np.array([0, 1_000_000_000], dtype=np.int64)))
    results.append(run_case("edge/neg-100K",
                            gen(50_000, "neg", 100_000, np.random.default_rng(SEED))))
    # sweep B: N=10M production scale spot checks
    N2 = 10_000_000
    results.append(run_case("B/M=100/uniform",
                            gen(100, "uniform", N2, np.random.default_rng(SEED))))
    results.append(run_case("B/M=100/sorted",
                            gen(100, "sorted", N2, np.random.default_rng(SEED))))
    results.append(run_case("B/M=10M/uniform",
                            gen(10_000_000, "uniform", N2, np.random.default_rng(SEED))))
    results.append(run_case("B/sparse-0-vs-1e9",
                            gen(0, "sparse", N2, np.random.default_rng(SEED))))
    results.append(run_case("B/packed-Q2-like",
                            gen(0, "packed", N2, np.random.default_rng(SEED))))
    results.append(run_case("B/neg-10M",
                            gen(10_000_000, "neg", N2, np.random.default_rng(SEED))))
    fails = [r for r in results if not r["ok"]]
    print(f"\n=== {len(results)-len(fails)}/{len(results)} OK; "
          f"FAIL={len(fails)} ===", flush=True)
    for r in fails:
        print(f"FAIL {r['label']}: {r['errors']}", flush=True)
    (HERE / "results_correct.json").write_text(
        json.dumps(json.loads(json.dumps(results, default=float)), indent=1))
    print("DONE", flush=True)
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
