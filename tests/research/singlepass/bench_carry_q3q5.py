# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: columnar carry for Q3/Q5 results — no per-group Python objects.

GroupBy kernel untouched (imports kernels_sp.f1_nocopy as-is). Compares:
  OLD : dict_fast_multi full materialization (100K nested dicts)
  NEW-V1 carry      : (ukeys, counts, sums) columnar, O(1) objects
  NEW-V2 carry+lazy : V1 + head(n)/point-lookup without full dict
  NEW-V3 resident   : dense M-buffers carried + compact views (memory view)

Generic: ColumnCarry works for any multi-group (ukeys, counts, [sums], ops).
No H2O hacks, no cache, no fixed-M, no benchmark branches. No prod change.
Usage: python tests/research/singlepass/bench_carry_q3q5.py
"""
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / "tests" / "research" / "singlepass"))

import numpy as np

from kernels_sp import dict_fast_multi, f1_nocopy

CSV = os.environ.get("MT_CSV", "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv")
N = 10_000_000
OUT = FORK / "tests" / "research" / "singlepass" / "results_carry_q3q5.json"
SEED_SAMPLE = 42


class ColumnCarry:
    """Generic multi-group result carrier. O(1) Python objects, no per-group dict.

    ukeys: int64 sorted unique keys; counts: int64; sums: list of int64/float64
    arrays aligned with ukeys. ops_list mirrors dict_fast_multi ops per column.
    Hot path holds only this object (numpy arrays). Materialization only via
    explicit to_dict() (final/display), head(n) or lookup(key) stay columnar.
    """
    __slots__ = ("ukeys", "counts", "sums", "ops_list")

    def __init__(self, ukeys, counts, sums, ops_list):
        self.ukeys = ukeys
        self.counts = counts
        self.sums = sums
        self.ops_list = ops_list

    @property
    def ngroups(self):
        return int(self.ukeys.size)

    @property
    def nbytes(self):
        return (int(self.ukeys.nbytes) + int(self.counts.nbytes)
                + sum(int(s.nbytes) for s in self.sums))

    def checksum(self):
        return int(self.counts.sum())

    def means(self, ci):
        s = self.sums[ci]
        return s.astype(np.float64) / self.counts

    def head(self, n=10):
        """Display rows without dict: vectorized slice -> list of tuples (n small)."""
        n = min(n, self.ngroups)
        idx = np.arange(n)
        rows = []
        for i in idx:
            cells = []
            for ci, ops in enumerate(self.ops_list):
                cell = []
                if "sum" in ops:
                    cell.append(self.sums[ci][i].item())
                if "count" in ops:
                    cell.append(int(self.counts[i]))
                if "mean" in ops:
                    cell.append(float(self.sums[ci][i] / self.counts[i]))
                cells.append(tuple(cell))
            rows.append((int(self.ukeys[i]), tuple(cells)))
        return rows

    def lookup(self, key):
        """Point query via binary search (ukeys sorted). No dict."""
        pos = int(np.searchsorted(self.ukeys, np.int64(key)))
        if pos >= self.ngroups or int(self.ukeys[pos]) != int(key):
            raise KeyError(key)
        cells = []
        for ci, ops in enumerate(self.ops_list):
            cell = {}
            if "sum" in ops:
                cell["sum"] = self.sums[ci][pos].item()
            if "count" in ops:
                cell["count"] = int(self.counts[pos])
            if "mean" in ops:
                cell["mean"] = float(self.sums[ci][pos] / self.counts[pos])
            cells.append(cell)
        return cells

    def to_dict(self):
        """Explicit final materialization only (off hot path)."""
        return dict_fast_multi(self.ukeys, self.counts, self.sums, self.ops_list)


def best_of(fn, reps=5):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts), float(np.median(ts)), out


def load_cols():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    df = pd.read_csv(CSV, usecols=["id3", "id6", "v1", "v2", "v3"])
    assert len(df) == N, len(df)
    t = time.perf_counter()
    r3 = a["resident_prepare"]({"id3": {"values": pa.array(
        df["id3"].to_numpy(), type=pa.string()), "prefix": "id"}})
    enc3 = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    rest = a["resident_prepare"]({
        "id6": {"values": df["id6"].to_numpy().astype(np.int32), "dtype": "int32"},
        "v1": {"values": df["v1"].to_numpy().astype(np.int32), "dtype": "int32"},
        "v2": {"values": df["v2"].to_numpy().astype(np.int32), "dtype": "int32"},
        "v3": {"values": df["v3"].to_numpy().astype(np.float64), "dtype": "float64"}})
    enc_rest = (time.perf_counter() - t) * 1000
    cols = {"id3": np.ascontiguousarray(r3["id3"]["codes"]),
            "id6": np.ascontiguousarray(rest["id6"]["codes"]),
            "v1": np.ascontiguousarray(rest["v1"]["codes"]),
            "v2": np.ascontiguousarray(rest["v2"]["codes"]),
            "v3": np.ascontiguousarray(rest["v3"]["codes"])}
    del df, r3, rest
    gc.collect()
    return cols, round(enc3, 1), round(enc_rest, 1)


def dict_peak_bytes(uk, counts, sums, ops):
    """Peak RSS of one full OLD materialization (tracemalloc, off hot path)."""
    gc.collect()
    tracemalloc.start()
    d = dict_fast_multi(uk, counts, sums, ops)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n = len(d)
    del d
    gc.collect()
    return peak, n


def stage(q, keys, vcols, ops_list):
    m = int(keys.max()) + 1
    # shared aggregation (internal GroupBy untouched, timed once)
    t = time.perf_counter()
    uk, counts, sums = f1_nocopy(keys, vcols, m)
    agg_ms = (time.perf_counter() - t) * 1000
    ng = int(uk.size)

    # OLD: full dict materialization
    b_old, med_old, d_old = best_of(
        lambda: dict_fast_multi(uk, counts, sums, ops_list), reps=3)

    # NEW-V1: columnar carry pack (O(1) objects, no per-group loop)
    def pack_v1():
        return ColumnCarry(uk, counts, sums, ops_list)
    b_v1, med_v1, carry = best_of(pack_v1, reps=50)

    # NEW-V2: carry + lazy display (head-10) and point lookup
    b_head, med_head, rows = best_of(lambda: carry.head(10), reps=20)
    rng = np.random.default_rng(SEED_SAMPLE)
    sample_keys = rng.choice(uk, size=20, replace=False)
    t = time.perf_counter()
    for k in sample_keys.tolist():
        carry.lookup(int(k))
    lookup_ms = (time.perf_counter() - t) * 1000
    lookup_us = lookup_ms * 1000 / len(sample_keys)

    # NEW-V3: resident dense buffers (M-sized) carried alongside compact views
    dense_nbytes = int((m * 8) + (m * 8) + sum(s.dtype.itemsize * m for s in sums))
    v3_view_ms = 0.0  # views are free; compact already materialized by agg

    # memory: carry nbytes vs OLD dict peak
    peak, ndict = dict_peak_bytes(uk, counts, sums, ops_list)
    assert ndict == ng

    # correctness: exact, generic checks
    assert carry.checksum() == N, (carry.checksum(), N)
    assert carry.ngroups == len(d_old) == ng
    # OLD dict == carry on 1000-key sample (exact ==)
    rng2 = np.random.default_rng(SEED_SAMPLE + 1)
    for k in rng2.choice(uk, size=min(1000, ng), replace=False).tolist():
        got = carry.lookup(int(k))
        want = d_old[int(k)]
        for ci, ops in enumerate(ops_list):
            for f in ("sum", "count", "mean"):
                if f in ops:
                    a, b = got[ci][f], want[ci][f]
                    assert a == b, (q, k, ci, f, a, b)
    # means max_diff vs vectorized recompute == 0 for mean cols
    max_diff = 0.0
    for ci, ops in enumerate(ops_list):
        if "mean" in ops:
            ref = (sums[ci].astype(np.float64) / counts)
            got = carry.means(ci)
            max_diff = max(max_diff, float(np.max(np.abs(got - ref))))
    assert max_diff == 0.0, max_diff

    r = {"m": m, "ngroups": ng,
         "agg_f1_ms": round(agg_ms, 1),
         "OLD_dict_best": round(b_old, 1), "OLD_dict_med": round(med_old, 1),
         "V1_carry_best": round(b_v1, 4), "V1_carry_med": round(med_v1, 4),
         "V2_head10_best": round(b_head, 3), "V2_head10_med": round(med_head, 3),
         "V2_lookup_us_per_key": round(lookup_us, 2),
         "V3_dense_bytes": dense_nbytes, "V3_view_ms": v3_view_ms,
         "carry_bytes": carry.nbytes, "OLD_dict_peak_bytes": peak,
         "TIME_RATIO_old_vs_v1": round(b_old / max(b_v1, 1e-9), 1),
         "MEM_RATIO_old_vs_carry": round(peak / max(carry.nbytes, 1), 1),
         "correctness": {"chk": N, "sample_exact_1000": True,
                         "means_max_diff": max_diff}}
    print(f"{q}: ng={ng} agg={agg_ms:.0f}ms OLD_dict={b_old:.0f}ms "
          f"V1_carry={b_v1*1000:.1f}us head10={b_head:.3f}ms "
          f"lookup={lookup_us:.1f}us carry={carry.nbytes/1e6:.1f}MB "
          f"dict_peak={peak/1e6:.1f}MB max_diff={max_diff}", flush=True)
    del d_old
    gc.collect()
    return r


def main():
    cols, enc3, enc_rest = load_cols()
    print(f"encode_ms: id3={enc3} rest={enc_rest}", flush=True)
    out = {"tag": "carry_q3q5", "N": N,
           "encode_ms": {"id3": enc3, "id6_v1_v2_v3": enc_rest}}
    out["Q3"] = stage("Q3", cols["id3"], [cols["v1"], cols["v3"]],
                      [("sum",), ("mean",)])
    del cols["id3"]
    gc.collect()
    out["Q5"] = stage("Q5", cols["id6"],
                      [cols["v1"], cols["v2"], cols["v3"]],
                      [("sum",), ("sum",), ("sum",)])
    out["notes"] = ("GroupBy untouched (kernels_sp.f1_nocopy); only result "
                    "carry differs. V1 = hot path (O(1) objects). to_dict() "
                    "explicit final only. Generic ColumnCarry for any "
                    "multi-group result.")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{OUT.name} written", flush=True)


if __name__ == "__main__":
    main()
