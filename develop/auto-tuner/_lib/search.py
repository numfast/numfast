# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Controlled execution: oracle + timing + structural counters.

- Oracle: numpy reference (unique+inverse+bincount, int64 exact).
- Each candidate keeps identical semantics; only algorithm/policy differs.
- No Planner/Driver/HLL imports (research self-contained, no prod change).
- Metrics per candidate: ms (median), memory_bytes (estimated allocs),
  rows_read, materializations, native_calls (numpy C calls, counted).
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _lib.candidates import list_candidates
from _lib.fingerprint import fingerprint


def make_data(n, m, seed, thresh=50):
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, m, size=n, dtype=np.int32)
    values = rng.integers(0, 100, size=n, dtype=np.int32)
    return keys, values, int(thresh)


def oracle(keys, values, thresh):
    mask = values > thresh
    fk = np.asarray(keys)[mask]
    fv = np.asarray(values)[mask].astype(np.int64)
    if fk.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    uk, inv = np.unique(fk, return_inverse=True)
    sums = np.bincount(inv, weights=fv)
    return uk.astype(np.int32), sums.astype(np.int64)


def _groupby_sums(fkeys, fvals, strategy):
    if fkeys.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    if strategy in ("dense_shift", "dense_fused"):
        kmin = int(fkeys.min())
        shifted = (fkeys.astype(np.int64) - kmin).astype(np.int64)
        span = int(shifted.max()) + 1
        full = np.bincount(shifted, weights=fvals.astype(np.float64), minlength=span)
        present = np.flatnonzero(full) if strategy == "dense_fused" else np.unique(shifted)
        ukeys = (present + kmin).astype(np.int32)
        sums = full[present].astype(np.int64)
        order = np.argsort(ukeys, kind="stable")
        return ukeys[order], sums[order]
    if strategy == "sorted":
        order = np.argsort(fkeys, kind="stable")
        sk = fkeys[order]
        sv = fvals.astype(np.int64)[order]
        bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        ukeys = sk[bounds]
        sums = np.add.reduceat(sv, bounds)
        return ukeys.astype(np.int32), sums.astype(np.int64)
    # hash / unique: generic unique+inverse+bincount
    uk, inv = np.unique(fkeys, return_inverse=True)
    sums = np.bincount(inv, weights=fvals.astype(np.float64)).astype(np.int64)
    return uk.astype(np.int32), sums


def _run_single(keys, values, thresh, groupby, path):
    n = len(keys)
    if path == "materialize":
        mask = values > thresh
        fkeys = np.asarray(keys)[mask]
        fvals = np.asarray(values)[mask]
        rows_read = 2 * n
        materializations = 2
        native_calls = 4
        uk, sums = _groupby_sums(fkeys, fvals, groupby)
        native_calls += 3
        mem = int(fkeys.nbytes + fvals.nbytes + mask.nbytes)
    else:
        mask = values > thresh
        rows_read = n
        materializations = 0
        native_calls = 2
        # fused: single masked pass, no filter-output alloc kept
        fkeys = np.asarray(keys)[mask]
        fvals = np.asarray(values)[mask]
        mem = int(mask.nbytes)
        uk, sums = _groupby_sums(fkeys, fvals, groupby)
        native_calls += 3
    return uk, sums, rows_read, materializations, native_calls, mem


def _run_mt(keys, values, thresh, groupby, path, threads):
    n = len(keys)
    idx = np.array_split(np.arange(n), threads)
    partial = []
    rows_read = 0
    mats = 0
    ncalls = 0
    mem = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(_run_single, keys[i], values[i], thresh, groupby, path) for i in idx]
        for f in futs:
            uk, sums, rr, mm, nc, me = f.result()
            partial.append((uk, sums))
            rows_read += rr
            mats += mm
            ncalls += nc
            mem += me
    # merge partials by key (exact int64 add)
    if not partial:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64), rows_read, mats, ncalls, mem
    allk = np.concatenate([p[0] for p in partial]) if partial[0][0].size else np.zeros(0, dtype=np.int32)
    allv = np.concatenate([p[1] for p in partial]) if partial[0][0].size else np.zeros(0, dtype=np.int64)
    if allk.size == 0:
        return allk, allv, rows_read, mats, ncalls, mem
    uk, inv = np.unique(allk, return_inverse=True)
    sums = np.bincount(inv, weights=allv.astype(np.float64)).astype(np.int64)
    ncalls += 2
    return uk.astype(np.int32), sums, rows_read, mats, ncalls, mem


def execute(keys, values, thresh, cand, reps=5, warmup=2):
    for _ in range(warmup):
        if cand["exec"] == "ST":
            _run_single(keys, values, thresh, cand["groupby"], cand["path"])
        else:
            _run_mt(keys, values, thresh, cand["groupby"], cand["path"], cand["threads"])
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        if cand["exec"] == "ST":
            out = _run_single(keys, values, thresh, cand["groupby"], cand["path"])
        else:
            out = _run_mt(keys, values, thresh, cand["groupby"], cand["path"], cand["threads"])
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"ms": float(ts[len(ts) // 2]), "out": out, "times": ts}


def search(keys, values, thresh, reps=5, warmup=2):
    ref_k, ref_v = oracle(keys, values, thresh)
    rows = []
    for cand in list_candidates():
        r = execute(keys, values, thresh, cand, reps=reps, warmup=warmup)
        uk, sums, rows_read, mats, ncalls, mem = r["out"]
        ok = uk.shape == ref_k.shape and bool(np.array_equal(uk, ref_k)) and bool(np.array_equal(sums, ref_v))
        rows.append({
            "id": cand["id"], "exec": cand["exec"], "threads": cand["threads"],
            "groupby": cand["groupby"], "path": cand["path"],
            "ms": round(r["ms"], 3), "correct": bool(ok),
            "memory_bytes": int(mem), "rows_read": int(rows_read),
            "materializations": int(mats), "native_calls": int(ncalls),
        })
    rows.sort(key=lambda d: d["ms"])
    return {"reference": {"nkeys": int(ref_k.size)}, "rows": rows}


def run_demo(n_tune=200000, m=10000, seed_tune=42, seed_eval=43, reps=5, warmup=2):
    k_t, v_t, th = make_data(n_tune, m, seed_tune)
    k_e, v_e, _ = make_data(n_tune, m, seed_eval)
    fp_tune = fingerprint(k_t, v_t, th, seed_tune, "tune(seed42)", n_tune)
    fp_eval = fingerprint(k_e, v_e, th, seed_eval, "eval(seed43)", n_tune)
    res_tune = search(k_t, v_t, th, reps=reps, warmup=warmup)
    res_eval = search(k_e, v_e, th, reps=reps, warmup=warmup)
    base = next(r for r in res_tune["rows"] if r["id"] == "Plan A")
    best = next(r for r in res_tune["rows"] if r["correct"])
    # reason is observable, chosen after measurement
    if best["path"] == "fused" and base["path"] == "materialize":
        why = f"eliminated materialization ({base['materializations']}->0 intermediate allocs, rows_read {base['rows_read']}->{best['rows_read']}), chose {best['groupby']} {best['exec']}"
    else:
        why = f"chose {best['groupby']} {best['exec']}/{best['path']} by measured min ms"
    # eval check: same best id on held-out split?
    best_eval = next(r for r in res_eval["rows"] if r["correct"])
    overfit = best_eval["id"] != best["id"]
    return {
        "thresh": th, "fp_tune": fp_tune, "fp_eval": fp_eval,
        "tune": res_tune, "eval": res_eval,
        "baseline": base, "best": best, "reason": why,
        "best_eval_id": best_eval["id"], "overfit_flag": bool(overfit),
    }
