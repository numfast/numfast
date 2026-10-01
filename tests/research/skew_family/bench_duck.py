# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only: DuckDB thread ladder 1/4/8/16T on skew family 10M.

Per dataset: load parquet once (load_ms separate), then per T: PRAGMA
threads=T + proof SELECT current_setting('threads'), Q1-Q5 query-only
run1/run2 (2 reps, same t_start->t boundary), chk AFTER timing vs own
stats_chk.json reference (official chk N/A on synthetic). No prod change.
Usage (Git Bash):
  C:/App/competitions/H2O/python310/python.exe tests/research/skew_family/bench_duck.py [level|all]
"""
import gc
import json
import sys
import time
from pathlib import Path

SESS = Path("C:/App/competitions/H2O/session/skew_family")
LEVELS_ALL = ["uniform", "mild-skew", "medium-skew", "heavy-skew"]
LADDER = [1, 4, 8, 16]
QS = {
    "Q1": "SELECT id1, SUM(v1) s FROM t GROUP BY id1",
    "Q2": "SELECT id1, id2, SUM(v1) s FROM t GROUP BY id1, id2",
    "Q3": "SELECT id3, SUM(v1) s, AVG(v3) m FROM t GROUP BY id3",
    "Q4": "SELECT id4, AVG(v1) a, AVG(v2) b, AVG(v3) c FROM t GROUP BY id4",
    "Q5": "SELECT id6, SUM(v1) a, SUM(v2) b, SUM(v3) c FROM t GROUP BY id6",
}


def main():
    import duckdb
    print(f"duckdb {duckdb.__version__}", flush=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    levels = LEVELS_ALL if arg == "all" else [arg]
    for level in levels:
        ref = json.load(open(SESS / f"skew_{level}_10M_stats_chk.json"))
        out = {"level": level, "engine": f"duckdb {duckdb.__version__}",
               "N": ref["N"], "cells": {}}
        con = duckdb.connect()
        t = time.perf_counter()
        con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{SESS}/skew_{level}_10M.parquet')")
        out["load_ms"] = (time.perf_counter() - t) * 1000
        n = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert n == ref["N"], n
        print(f"=== {level} load={out['load_ms']:.0f}ms n={n} ===", flush=True)
        for T in LADDER:
            con.execute(f"PRAGMA threads={T}")
            proof = con.execute("SELECT current_setting('threads')").fetchone()[0]
            assert int(proof) == T, (T, proof)
            for q, sql in QS.items():
                reps = []
                res = None
                for _ in range(2):
                    s = time.perf_counter()
                    res = con.execute(sql).fetchall()
                    reps.append((time.perf_counter() - s) * 1000)
                # chk separate vs own reference
                assert len(res) == ref["chk"][q]["ngroups"], (level, q, T, len(res))
                if q == "Q1":
                    tot = int(sum(r[1] for r in res))
                    assert tot == ref["chk"][q]["total"], (level, q, T, tot)
                if q == "Q2":
                    tot = int(sum(r[2] for r in res))
                    assert tot == ref["chk"][q]["total"], (level, q, T, tot)
                out["cells"][f"{q}@{T}"] = {"run1_ms": reps[0], "run2_ms": reps[1],
                                            "best_ms": min(reps), "threads_proof": int(proof)}
                print(f"{q}@{T}(thr={proof}): run1={reps[0]:.0f} run2={reps[1]:.0f}",
                      flush=True)
            gc.collect()
        con.close()
        with open(SESS / f"results_duck_{level}.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"results_duck_{level}.json written", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
