# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Result research prototype (NEW file, prod/Rust untouched).

NativeGroupResult as a separate entity over the engine ColumnCarry:
  {keys (int64 view), columns {name: sums view}, counts (u64/i64 view),
   validity, dtype/schema, group_count, metadata} — zero Python objects
per group. Rust agg -> typed contiguous arrays -> NumPy views (no copy);
dict ONLY via explicit .to_dict() (compat). Chained operator
(.groupby().sum().mean() second stage) consumes the native result
directly, never touching dict.

Proves: zero-copy (shares_memory), dict_ms -> native_ms, memory,
correctness exact, Q1/Q4 then Q3/Q5 vs Polars same-T.

Usage (Git Bash, strictly sequential, one process per run):
  timeout 1200 /c/Users/Mikech/AppData/Local/Programs/Python/Python314/python \
    tests/research/native_result/bench_native_result.py 16 Q1Q4
  Env: NF_SOURCE=csv10M NF_OUT=tests/research/native_result/res.jsonl
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

T = int(sys.argv[1]) if len(sys.argv) > 1 else 16
WHICH = sys.argv[2] if len(sys.argv) > 2 else "Q1Q4"
os.environ["NUMFAST_THREADS"] = str(T)
os.environ["POLARS_MAX_THREADS"] = str(T)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

SOURCE = os.environ.get("NF_SOURCE", "csv10M")
OUT = FORK / os.environ.get("NF_OUT", "tests/research/native_result/res.jsonl")

DATA = "C:/App/competitions/H2O/data"
SRC = {"csv10M": (DATA + "/G1_1e7_1e2_0_0.csv", 10_000_000)}
GOLD = {
    "Q1": {"ngroups": 100, "total": 29998789},
    "Q3": {"ngroups": 100000, "v1": 29998789, "mean3_sum": 4999719.622344427},
    "Q4": {"ngroups": 100, "m1": 299.98798187506526,
           "m2": 799.894179409978, "m3": 4999.766872833688},
    "Q5": {"ngroups": 100000, "s1": 29998789, "s2": 79989360,
           "s3_scaled": 499976651408061},
}

import numpy as np
import psutil

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 1e9


def best_of(fn, reps=3):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return out, best


class NativeGroupResult:
    """Zero-copy view over ColumnCarry: no new buffers, no per-group objects.

    keys/columns/counts are NumPy views (shares_memory with carry).
    validity derived (counts>0, all-True by compact construction).
    to_dict() = explicit compat only. Second-stage ops (group_mean,
    global_sum, mean_of_sums = .groupby().sum().mean() chain) run
    vectorized straight on the native columns.
    """

    __slots__ = ("_c", "cols", "schema", "metadata", "_valid")

    def __init__(self, carry, cols, metadata):
        self._c = carry
        self.cols = list(cols)
        self.schema = {"key": "int64", "count": "int64",
                       "sums": {c: str(carry.sums[c].dtype) for c in cols}}
        self.metadata = dict(metadata)
        self._valid = None

    @property
    def keys(self):
        return self._c.ukeys

    @property
    def counts(self):
        return self._c.counts

    def column(self, name):
        return self._c.sums[name]

    @property
    def group_count(self):
        return self._c.ngroups

    @property
    def validity(self):
        if self._valid is None:
            self._valid = np.ascontiguousarray(self._c.counts > 0)
        return self._valid

    def assert_zero_copy(self):
        assert self.keys.base is not None or np.shares_memory(
            self.keys, self._c.ukeys)
        for c in self.cols:
            assert np.shares_memory(self.column(c), self._c.sums[c])
            assert self.column(c).flags["C_CONTIGUOUS"]
        assert self.keys.dtype == np.int64
        assert self.counts.dtype == np.int64
        assert self.keys.flags["C_CONTIGUOUS"]

    # ---- explicit compat (dict ONLY on request) ----
    def to_dict(self, ops_map, threads=1):
        """ops_map: None -> single-op flat; dict col->ops otherwise."""
        c = self._c
        if ops_map is None:
            return c.to_dict_flat("sum", threads=threads)
        if len(self.cols) == 1:
            col = self.cols[0]
            ops = set(ops_map[col])
            if len(ops) == 1:
                return c.to_dict_flat(next(iter(ops)), threads=threads)
            return c.to_dict_single(col, ops, threads=threads)
        return c.to_dict_multi(self.cols, ops_map, threads=threads)

    # ---- chained next-operator on native result (no dict) ----
    def group_mean(self, col):
        return self._c.means(col)

    def global_sum(self, col):
        return self.column(col).sum()

    def mean_of_sums(self, col):
        """Second stage of .groupby().sum().mean(): mean over group sums."""
        return float(self.column(col).mean())


def main():
    from builder import MAIN

    path, N = SRC[SOURCE]
    qs = ["Q1", "Q4"] if WHICH == "Q1Q4" else (
        ["Q3", "Q5"] if WHICH == "Q3Q5" else ["Q1", "Q3", "Q4", "Q5"])
    print(f"=== native-result {SOURCE} N={N} T={T} qs={qs} ===", flush=True)
    import pandas as pd

    t0 = time.perf_counter()
    df = pd.read_csv(path, usecols=["id1", "id3", "id4", "id6", "v1", "v2", "v3"])
    assert len(df) == N
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"load {load_ms:.0f}ms RSS {rss():.2f}GB", flush=True)

    kernel = MAIN["build"](str(FORK))
    a = kernel.alias

    def enc(col, prefix="id"):
        g = a["compile"]([a["ir_encode_pattern"]("c", col, prefix)])
        return np.ascontiguousarray(
            a["evaluate"](a["optimize"](g), "cpu", len(col))["result"])

    t0 = time.perf_counter()
    K1 = enc(df["id1"].to_numpy(), "id")
    K3 = enc(df["id3"].to_numpy(), "id")
    enc_ms = (time.perf_counter() - t0) * 1000
    K4 = np.ascontiguousarray(
        (df["id4"].to_numpy().astype(np.int32) -
         int(df["id4"].to_numpy().min())).astype(np.int32))
    K6 = np.ascontiguousarray(
        (df["id6"].to_numpy().astype(np.int32) -
         int(df["id6"].to_numpy().min())).astype(np.int32))
    V1 = np.ascontiguousarray(df["v1"].to_numpy().astype(np.int32))
    V2 = np.ascontiguousarray(df["v2"].to_numpy().astype(np.int32))
    V3 = np.ascontiguousarray(df["v3"].to_numpy().astype(np.float64))
    print(f"encode {enc_ms:.0f}ms RSS {rss():.2f}GB", flush=True)

    def _nodes(q, result):
        if q == "Q1":
            return [a["ir_series"]("k", K1), a["ir_series"]("v", V1),
                    a["ir_groupby"]("g", "v", "k", "sum", result=result)]
        if q == "Q4":
            return [a["ir_series"]("k", K4), a["ir_series"]("v1", V1),
                    a["ir_series"]("v2", V2),
                    a["ir_series"]("v3", V3, "float64"),
                    a["ir_groupby_multi"](
                        "g", ["v1", "v2", "v3"], "k",
                        {"v1": ("mean",), "v2": ("mean",), "v3": ("mean",)},
                        result=result)]
        if q == "Q3":
            return [a["ir_series"]("k", K3), a["ir_series"]("v1", V1),
                    a["ir_series"]("v3", V3, "float64"),
                    a["ir_groupby_multi"](
                        "g", ["v1", "v3"], "k",
                        {"v1": ("sum",), "v3": ("mean",)}, result=result)]
        return [a["ir_series"]("k", K6), a["ir_series"]("v1", V1),
                a["ir_series"]("v2", V2),
                a["ir_series"]("v3", V3, "float64"),
                a["ir_groupby_multi"](
                    "g", ["v1", "v2", "v3"], "k",
                    {"v1": ("sum",), "v2": ("sum",), "v3": ("sum",)},
                    result=result)]

    COLS = {"Q1": ["v"], "Q4": ["v1", "v2", "v3"],
            "Q3": ["v1", "v3"], "Q5": ["v1", "v2", "v3"]}
    OPS = {"Q1": None,
           "Q4": {"v1": ("mean",), "v2": ("mean",), "v3": ("mean",)},
           "Q3": {"v1": ("sum",), "v3": ("mean",)},
           "Q5": {"v1": ("sum",), "v2": ("sum",), "v3": ("sum",)}}

    res = {}
    for q in qs:
        cols, ops = COLS[q], OPS[q]
        g = a["optimize"](a["compile"](_nodes(q, "dict")))
        gc_c = a["optimize"](a["compile"](_nodes(q, "carry")))

        def _run_carry():
            return a["cpu_execute"](gc_c["nodes"])

        bufs, native_ms = best_of(_run_carry, 3)
        carry = bufs["g#carry"]
        gi = bufs.get("g#groupindex", {})
        stages = {k: gi.get(k) for k in
                  ("strategy", "backend", "threads", "gi_ms", "agg_ms",
                   "carry_ms", "merge_ms")}
        nr = NativeGroupResult(carry, cols,
                               {"backend": stages.get("backend"),
                                "threads": stages.get("threads"),
                                "strategy": stages.get("strategy")})
        nr.assert_zero_copy()
        assert bool(nr.validity.all())
        assert nr.group_count == int(carry.ukeys.size)
        nbytes = (int(carry.ukeys.nbytes) + int(carry.counts.nbytes)
                  + sum(int(carry.sums[c].nbytes) for c in cols))

        _, chain_ms = best_of(
            lambda: (nr.group_mean(cols[-1]), nr.mean_of_sums(cols[0])), 3)
        chain = {"mean_sum": float(nr.group_mean(cols[-1]).sum()),
                 "mean_of_sums": nr.mean_of_sums(cols[0])}

        r0 = rss()
        d, compat_ms = best_of(lambda: nr.to_dict(ops), 3)
        compat_rss = rss() - r0
        del d
        gc.collect()

        _, dict_ms = best_of(lambda: a["cpu_execute"](g["nodes"])["g"], 3)
        ref = a["cpu_execute"](g["nodes"])["g"]
        assert nr.to_dict(ops) == ref, f"{q}: native->dict != dict-default"

        # correctness: native checks vs gold (exact int, 1e-9 rel float)
        if q == "Q1":
            chk = {"total": int(nr.global_sum("v")),
                   "ngroups": nr.group_count}
            assert chk["total"] == GOLD[q]["total"]
            assert chk["ngroups"] == GOLD[q]["ngroups"]
            assert int(sum(int(v) for v in ref.values())) == chk["total"]
        elif q == "Q4":
            ms = [float(nr.group_mean(c).sum()) for c in cols]
            md = max(abs(ms[0] - GOLD[q]["m1"]), abs(ms[1] - GOLD[q]["m2"]),
                     abs(ms[2] - GOLD[q]["m3"]))
            assert md < 1e-9, (md, ms)
            chk = {"means": ms, "ngroups": nr.group_count, "maxdiff": md}
            pr = [float(sum(cell[c]["mean"] for cell in ref.values()))
                  for c in cols]
            assert max(abs(x - y) for x, y in zip(ms, pr)) <= 1e-9, (ms, pr)
            assert abs(chain["mean_of_sums"] * nr.group_count - float(
                nr.global_sum(cols[0]))) <= 1e-6
        elif q == "Q3":
            m3 = float(nr.group_mean("v3").sum())
            assert int(nr.global_sum("v1")) == GOLD[q]["v1"]
            assert abs(m3 - GOLD[q]["mean3_sum"]) <= 1e-6 * abs(
                GOLD[q]["mean3_sum"]) + 1e-6
            assert nr.group_count == GOLD[q]["ngroups"]
            chk = {"v1": int(nr.global_sum("v1")), "mean3_sum": m3,
                   "ngroups": nr.group_count}
            assert abs(m3 - float(sum(c["v3"]["mean"]
                                      for c in ref.values()))) <= 1e-9 * max(
                                          1.0, abs(m3))
        else:
            s = [int(nr.global_sum(c)) for c in ("v1", "v2")]
            sc = int(round(float(nr.global_sum("v3")) * 1e6))
            assert s[0] == GOLD[q]["s1"] and s[1] == GOLD[q]["s2"]
            assert sc == GOLD[q]["s3_scaled"]
            assert nr.group_count == GOLD[q]["ngroups"]
            chk = {"s1": s[0], "s2": s[1], "s3_scaled": sc,
                   "ngroups": nr.group_count}
        win = (dict_ms - native_ms) / dict_ms * 100
        res[q] = {"stages": stages, "native_ms": native_ms,
                  "chain_ms": chain_ms, "compat_ms": compat_ms,
                  "dict_ms": dict_ms, "win_pct": win,
                  "carry_kb": nbytes / 1024, "compat_rss_gb": compat_rss,
                  "chk": chk, "chain": chain}
        print(f"{q}: native={native_ms:.1f}ms chain={chain_ms:.3f}ms "
              f"compat(dict-on-request)={compat_ms:.1f}ms "
              f"dict(default)={dict_ms:.1f}ms win={win:+.1f}% "
              f"carry={nbytes/1024:.0f}KB dRSS={compat_rss:.3f}GB "
              f"chk_ng={chk['ngroups']}", flush=True)
        del bufs, ref, nr
        gc.collect()

    import polars as pl

    pf = pl.from_pandas(df)
    del df
    gc.collect()
    PQ = {"Q1": lambda: pf.group_by("id1").agg(pl.col("v1").sum()),
          "Q4": lambda: pf.group_by("id4").agg(pl.col("v1").mean(),
                                               pl.col("v2").mean(),
                                               pl.col("v3").mean()),
          "Q3": lambda: pf.group_by("id3").agg(pl.col("v1").sum(),
                                               pl.col("v3").mean()),
          "Q5": lambda: pf.group_by("id6").agg(pl.col("v1").sum(),
                                               pl.col("v2").sum(),
                                               pl.col("v3").sum())}
    pol = {}
    for q in qs:
        _, ms = best_of(lambda q=q: PQ[q](), 3)
        pol[q] = ms
    assert int(pf.group_by("id1").agg(
        pl.col("v1").sum()).to_series(-1).sum()) == GOLD["Q1"]["total"]
    print("polars T=%d: " % T + " ".join(f"{q}={pol[q]:.1f}ms" for q in qs)
          + f" RSS {rss():.2f}GB", flush=True)
    for q in qs:
        print(f"{q}: native {res[q]['native_ms']:.1f}ms vs polars "
              f"{pol[q]:.1f}ms ratio "
              f"{res[q]['native_ms']/pol[q]:.2f}x chk={res[q]['chk']}",
              flush=True)

    row = {"T": T, "which": WHICH, "source": SOURCE, "n": N,
           "load_ms": load_ms, "enc_ms": enc_ms, "res": res,
           "polars_ms": pol, "rss_gb": rss()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"appended {OUT}", flush=True)


if __name__ == "__main__":
    main()
