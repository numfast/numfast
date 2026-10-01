# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only: generic GROUP OWNERSHIP / WEIGHTED PARTITION planners.

No prod change. No NFS change. All functions pure numpy (planner side);
execution kernels are reused from mtgroup/kernels_mt (no duplication).

Contract: input = unique group codes (int64, ascending) + optional weights
(row counts per group, int64). Output = per-worker group lists (disjoint,
covering all groups) + loads. Workers compute their own tuples exclusively
from a formula (no ID shipping needed for contiguous/round-robin/hash).
"""
import time

import numpy as np

SEED = 42


# ---------- BASELINE (K-only, no weights) ----------
def baseline_contiguous(ukeys, P):
    """Worker i ~= K/P groups: contiguous ranges over sorted ukeys."""
    idx = np.array_split(np.arange(ukeys.size), P)
    return [np.ascontiguousarray(ukeys[s]) for s in idx]


def baseline_roundrobin(ukeys, P):
    """Worker i gets every P-th group (interleaved)."""
    n = ukeys.size
    return [np.ascontiguousarray(ukeys[w::P]) for w in range(P)]


def formula_contiguous(K, P, worker):
    """Formula assignment: worker computes its own [start,end) in code space.

    No ID lists shipped. Exclusive by construction. Only valid when group
    codes are dense 0..K-1 (true for NumFast resident codes).
    """
    base, rem = divmod(K, P)
    start = worker * base + min(worker, rem)
    end = start + base + (1 if worker < rem else 0)
    return (int(start), int(end))


def formula_roundrobin(K, P, worker):
    """Formula assignment for round-robin: worker owns codes w, w+P, ..."""
    return np.arange(worker, K, P, dtype=np.int64)


def formula_hash_owner(code, P, mult=2654435761):
    """Stateless hash ownership: owner = (code*mult mod 2^32) % P."""
    return int((int(code) * mult & 0xFFFFFFFF) % P)


# ---------- GLOBAL GROUP INDEX (metadata, not answer) ----------
def build_gindex_exact(codes_sorted, counts):
    """Exact index: (group_code int32, weight int64). No SUM/MEAN result."""
    t0 = time.perf_counter()
    gc = np.ascontiguousarray(codes_sorted, dtype=np.int32)
    w = np.ascontiguousarray(counts, dtype=np.int64)
    ms = (time.perf_counter() - t0) * 1000.0
    raw = int(gc.nbytes + w.nbytes)
    return {"codes": gc, "weight": w, "build_ms": ms, "raw_bytes": raw,
            "bytes_per_group": raw / max(1, gc.size)}


def build_gindex_quantized(counts, bits=8):
    """Quantized weights: log2 bucket into u8 (or u16). Returns max rel err."""
    t0 = time.perf_counter()
    c = np.ascontiguousarray(counts, dtype=np.int64)
    mx = int(c.max()) if c.size else 1
    # log2 scale: q = ceil(255 * log(1+c)/log(1+mx))
    q = np.ceil(255.0 * np.log1p(c.astype(np.float64)) / np.log1p(float(mx)))
    q = q.astype(np.uint8)
    # dequant for error measurement: invert
    rec = np.expm1(q.astype(np.float64) / 255.0 * np.log1p(float(mx)))
    denom = np.maximum(c.astype(np.float64), 1.0)
    rel = np.abs(rec - c.astype(np.float64)) / denom
    ms = (time.perf_counter() - t0) * 1000.0
    raw = int(c.size * 4 + c.size * 1)  # int32 code + u8 weight
    return {"q": q, "build_ms": ms, "raw_bytes": raw,
            "bytes_per_group": raw / max(1, c.size),
            "max_rel_err": float(rel.max()) if rel.size else 0.0,
            "mean_rel_err": float(rel.mean()) if rel.size else 0.0}


def build_gindex_approx(keys_sample, m, frac=0.01, seed=SEED):
    """Approximate weights from frac subsample, scaled by 1/frac.

    Reports max/mean relative error vs exact bincount (research only).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = keys_sample.shape[0]
    take = max(1, int(n * frac))
    idx = rng.choice(n, size=take, replace=False)
    sub = np.bincount(keys_sample[idx].astype(np.int64), minlength=m)
    est = (sub.astype(np.float64) / frac).astype(np.int64)
    ms = (time.perf_counter() - t0) * 1000.0
    return {"weight_est": est, "build_ms": ms, "frac": frac, "take": take}


def gindex_storage_report(codes, counts):
    """Size analysis: raw, delta-compressed estimate, quantized."""
    raw = int(codes.nbytes + counts.nbytes)
    # delta of sorted codes: codes are dense 0..K-1 -> deltas ~1 byte each
    d = np.diff(codes.astype(np.int64))
    uniq_d, cntd = np.unique(d, return_counts=True)
    # entropy estimate of delta stream
    p = cntd.astype(np.float64) / cntd.sum()
    h_bits = float(-(p * np.log2(np.maximum(p, 1e-12))).sum())
    est_delta_bytes = int(np.ceil(codes.size * h_bits / 8.0)) + 8
    q = build_gindex_quantized(counts)
    return {"raw_bytes": raw, "bytes_per_group_raw": raw / max(1, codes.size),
            "delta_entropy_bits": h_bits,
            "est_delta_codes_bytes": est_delta_bytes,
            "est_total_delta_plus_i64": est_delta_bytes + int(counts.nbytes),
            "est_total_delta_plus_u8q": est_delta_bytes + int(codes.size),
            "quant_max_rel_err": q["max_rel_err"],
            "quant_mean_rel_err": q["mean_rel_err"]}


# ---------- WEIGHTED OWNERSHIP A-E ----------
def plan_A_equal_number(ukeys, P):
    return baseline_contiguous(ukeys, P)


def plan_B_contig_weighted(ukeys, weights, P):
    """Contiguous ranges cut by cumulative weight ~= total/P."""
    t0 = time.perf_counter()
    w = weights.astype(np.int64)
    total = int(w.sum())
    target = total / P
    cum = np.cumsum(w)
    bounds = []
    lo = 0
    for p in range(1, P):
        hi = int(np.searchsorted(cum, p * target, side="left")) + 1
        hi = min(hi, ukeys.size)
        bounds.append((lo, hi))
        lo = hi
    bounds.append((lo, ukeys.size))
    out = [np.ascontiguousarray(ukeys[a:b]) for a, b in bounds]
    ms = (time.perf_counter() - t0) * 1000.0
    return out, {"sched_ms": ms}


def plan_C_greedy(ukeys, weights, P):
    """Greedy in id order: next group -> currently least-loaded worker."""
    t0 = time.perf_counter()
    loads = np.zeros(P, dtype=np.int64)
    groups = [[] for _ in range(P)]
    for g, w in zip(ukeys.tolist(), weights.tolist()):
        b = int(np.argmin(loads))
        groups[b].append(int(g))
        loads[b] += int(w)
    ms = (time.perf_counter() - t0) * 1000.0
    return [np.array(x, dtype=np.int64) for x in groups], {"sched_ms": ms}


def plan_D_sorted_lpt(ukeys, weights, P):
    """Sorted-by-weight desc + least-loaded (LPT)."""
    t0 = time.perf_counter()
    order = np.argsort(-weights.astype(np.int64), kind="stable")
    loads = np.zeros(P, dtype=np.int64)
    groups = [[] for _ in range(P)]
    for j in order.tolist():
        b = int(np.argmin(loads))
        groups[b].append(int(ukeys[j]))
        loads[b] += int(weights[j])
    ms = (time.perf_counter() - t0) * 1000.0
    return [np.array(x, dtype=np.int64) for x in groups], {"sched_ms": ms}


def plan_E_firstlast(ukeys, weights, P):
    """Heuristic only: sort desc, pair largest+smallest, pair -> least-loaded."""
    t0 = time.perf_counter()
    order = np.argsort(-weights.astype(np.int64), kind="stable")
    pairs = []
    i, j = 0, len(order) - 1
    while i < j:
        pairs.append((int(order[i]), int(order[j])))
        i += 1
        j -= 1
    if i == j:
        pairs.append((int(order[i]), None))
    loads = np.zeros(P, dtype=np.int64)
    groups = [[] for _ in range(P)]
    for a, b in pairs:
        w = int(np.argmin(loads))
        groups[w].append(int(ukeys[a]))
        loads[w] += int(weights[a])
        if b is not None:
            groups[w].append(int(ukeys[b]))
            loads[w] += int(weights[b])
    ms = (time.perf_counter() - t0) * 1000.0
    return [np.array(x, dtype=np.int64) for x in groups], {"sched_ms": ms}


def plan_hash_full(m, P):
    g = np.arange(m, dtype=np.int64)
    owner = ((g * np.int64(2654435761)) & np.int64(0xFFFFFFFF)) % np.int64(P)
    return [np.flatnonzero(owner == w) for w in range(P)], owner.astype(np.int32)


def owner_array(m, groups, P):
    go = np.empty(m, dtype=np.int32)
    for w, seg in enumerate(groups):
        go[np.asarray(seg)] = np.int32(w)
    return go


# ---------- METRICS ----------
def load_stats(counts_m, groups):
    loads = np.array([int(counts_m[np.asarray(s)].sum()) if len(s) else 0
                      for s in groups], dtype=np.int64)
    mean = float(loads.mean()) if loads.size else 0.0
    mx, mn = (int(loads.max()), int(loads.min())) if loads.size else (0, 0)
    imb = (mx - mean) / mean * 100.0 if mean else 0.0
    return {"loads": loads.tolist(), "max": mx, "min": mn, "mean": mean,
            "imb_pct": float(imb),
            "groups_per_worker": [int(len(s)) for s in groups]}


def check_disjoint(groups, K=None):
    seen = set()
    for s in groups:
        for g in np.asarray(s).tolist():
            if g in seen:
                return False, int(g)
            seen.add(int(g))
    if K is not None and len(seen) != int(K):
        return False, -1
    return True, -1


def lower_bound_unsplittable(weights, P):
    total = int(weights.sum())
    return max(int(weights.max()), int((total + P - 1) // P))
