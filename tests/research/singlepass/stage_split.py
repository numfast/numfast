# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research: Q1-Q5 stage split via direct cpu_execute_impl (prod code, read-only).

Loads prod groupindex/planner/cpu modules by file path with a stub `_lib`
package (same source files prod uses — no copies). Reports per Q:
series/pack ms, gi_ms, agg_ms, dict-materialize ms, total. Single-thread.
"""
import gc
import importlib.util
import sys
import time
import types
from pathlib import Path

FORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORK.parent / "app-builder-ponytail"))
sys.path.insert(0, str(FORK))

import numpy as np

CPU_LIB = FORK / "src" / "Drivers" / "CPU" / "_lib"
PLAN_LIB = FORK / "src" / "Runtime" / "Planner" / "_lib"


def load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


gi_mod = cpu_mod = plan_mod = None


def load_prod_modules():
    global gi_mod, cpu_mod, plan_mod
    # stub `_lib` package resolving to the real prod CPU _lib dir (same files).
    # Installed AFTER kernel build: the Builder loader purges `_lib*` from
    # sys.modules, so installing earlier would shadow its isolated loads.
    _lib = types.ModuleType("_lib")
    _lib.__path__ = [str(CPU_LIB)]
    sys.modules["_lib"] = _lib
    gi_mod = load("_lib.groupindex", CPU_LIB / "groupindex.py")
    cpu_mod = load("prod_cpu", CPU_LIB / "cpu.py")
    plan_mod = load("prod_planner", PLAN_LIB / "planner.py")

CSV = "C:/App/competitions/H2O/data/G1_1e7_1e2_0_0.csv"
N = 10_000_000


def main():
    from builder import MAIN
    kernel = MAIN["build"](str(FORK))
    a = kernel.alias
    import pandas as pd
    import pyarrow as pa
    df = pd.read_csv(CSV, usecols=["id1", "id2", "id3", "id4", "id6", "v1", "v2", "v3"])
    assert len(df) == N
    s_id1 = pa.array(df["id1"].to_numpy(), type=pa.string())
    s_id2 = pa.array(df["id2"].to_numpy(), type=pa.string())
    s_id3 = pa.array(df["id3"].to_numpy(), type=pa.string())
    id4 = df["id4"].to_numpy().astype(np.int32)
    id6 = df["id6"].to_numpy().astype(np.int32)
    v1 = df["v1"].to_numpy().astype(np.int32)
    v2 = df["v2"].to_numpy().astype(np.int32)
    v3 = df["v3"].to_numpy().astype(np.float64)
    del df
    gc.collect()

    res = {}
    for name, col in (("id1", {"values": s_id1, "prefix": "id"}),
                      ("id2", {"values": s_id2, "prefix": "id"}),
                      ("id3", {"values": s_id3, "prefix": "id"})):
        res[name] = a["resident_prepare"]({name: col})[name]
    c = a["resident_prepare"]({
        "id4": {"values": id4, "dtype": "int32"}, "id6": {"values": id6, "dtype": "int32"},
        "v1": {"values": v1, "dtype": "int32"}, "v2": {"values": v2, "dtype": "int32"},
        "v3": {"values": v3, "dtype": "float64"}})
    for k, vv in c.items():
        res[k] = vv
    load_prod_modules()
    K1, K2 = res["id1"]["codes"], res["id2"]["codes"]
    K3, K4 = res["id3"]["codes"], res["id4"]["codes"]
    K6 = res["id6"]["codes"]
    V1, V2, V3 = res["v1"]["codes"], res["v2"]["codes"], res["v3"]["codes"]

    # canonical_dtype / format_error / accum_dtype from built kernel internals:
    # replicate minimal: logical dtypes int32/float32/float64, accum int64/float64.
    def canonical_dtype(name):
        t = {"int32": "int32", "float32": "float32", "f32": "float32",
             "float64": "float64", "f64": "float64"}
        if name not in t:
            raise ValueError(name)
        return {"logical": t[name]}

    def accum_dtype(name):
        return "int64" if name in ("int32", "int64") else "float64"

    def fmt(what, fix="", doc=""):
        return ValueError(str(what))

    _impl = plan_mod.plan_groupby_impl
    _budget = cpu_mod.cpu_capability_impl()["max_buffer_bytes"]

    def plan_groupby(n, is_sorted, nunique_hint=None, **kw):
        if kw.get("budget_bytes") is None and kw.get("span") is not None:
            kw["budget_bytes"] = _budget
        return _impl(n, is_sorted, nunique_hint, **kw)

    queries = {
        "Q1": [a["ir_series"]("k", K1), a["ir_series"]("v", V1),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q2": [a["ir_series"]("c1", K1), a["ir_series"]("c2", K2),
               a["ir_series"]("v", V1), a["ir_pack_keys"]("k", "c1", "c2"),
               a["ir_groupby"]("g", "v", "k", "sum")],
        "Q3": [a["ir_series"]("k", K3), a["ir_series"]("v1", V1),
               a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v3"], "k",
                                     {"v1": ("sum",), "v3": ("mean",)})],
        "Q4": [a["ir_series"]("k", K4), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("mean",), "v2": ("mean",),
                                      "v3": ("mean",)})],
        "Q5": [a["ir_series"]("k", K6), a["ir_series"]("v1", V1),
               a["ir_series"]("v2", V2), a["ir_series"]("v3", V3, "float64"),
               a["ir_groupby_multi"]("g", ["v1", "v2", "v3"], "k",
                                     {"v1": ("sum",), "v2": ("sum",),
                                      "v3": ("sum",)})],
    }
    g = a["optimize"](a["compile"](queries["Q1"]))  # warmup numba paths in prod modules
    cpu_mod.cpu_execute_impl(
        [{"kernel_id": n["kernel_id"], "inputs": n["inputs"],
          "params": n["params"], "out": n["out"]} for n in g["nodes"]],
        accum_dtype, canonical_dtype, fmt, plan_groupby)

    for q, jobs in queries.items():
        g = a["optimize"](a["compile"](jobs))
        nodes = [{"kernel_id": n["kernel_id"], "inputs": n["inputs"],
                  "params": n["params"], "out": n["out"]} for n in g["nodes"]]
        for rep in range(3):
            t = time.perf_counter()
            bufs = cpu_mod.cpu_execute_impl(
                nodes, accum_dtype, canonical_dtype, fmt,
                plan_groupby)
            tot = (time.perf_counter() - t) * 1000
            gi_info = bufs.get("g#groupindex", {})
            strat = bufs.get("g#strategy", "?")
            # dict materialization is inside cpu_execute; re-time it alone:
            t = time.perf_counter()
            r = bufs["g"]
            nkeys = len(r)
            _ = time.perf_counter() - t
            print(f"{q} rep{rep}: total={tot:.0f}ms strategy={strat} "
                  f"gi={gi_info.get('gi_ms', -1):.0f} "
                  f"agg={gi_info.get('agg_ms', -1):.0f} "
                  f"other={(tot - gi_info.get('gi_ms', 0) - gi_info.get('agg_ms', 0)):.0f} "
                  f"ngroups={nkeys} reason={gi_info.get('reason', '')[:80]}",
                  flush=True)


if __name__ == "__main__":
    main()
