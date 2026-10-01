# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""ClickBench ORDER BY/LIMIT gate: NumFast sort/slice/gather vs DuckDB on hits_1m.

Bulk-only (999978 rows, parquet). No 1B. Heavy: run explicitly with the
system python (see Stage 2 report for the exact command).

Probes (all bulk, all exact-gated):
  P1 Q26-shape: WHERE SearchPhrase<>'' ORDER BY EventTime LIMIT 10
  P2 Q27-shape: WHERE SearchPhrase<>'' ORDER BY SearchPhrase LIMIT 10
  P3 Q8-shape:  WHERE AdvEngineID<>0 GROUP BY AdvEngineID ORDER BY count DESC
  P4 Q13-shape: WHERE SearchPhrase<>'' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10

Gate per probe (honest about ties -- SQL tie order is unspecified):
  1. multiset of full ordered keys NumFast == DuckDB (Counter equality);
  2. NumFast key sequence sorted in the requested direction;
  3. NumFast ties keep CSV input order (stable sort, verified by positions);
  4. LIMIT rows exact vs DuckDB when the LIMIT boundary is tie-free.
Stage breakdown (ms per stage) + throughput printed per probe. Seed: no RNG.
"""

import sys
import time
from collections import Counter

PARQUET = "C:/App/competitions/ClickBench/data/hits_1m.csv".replace("hits_1m.csv",
                                                                    "hits_1m.parquet")
APP_DIR = "C:/App/numfast/numfast-ponytail"
sys.path.insert(0, "C:/App/numfast/numfast-ponytail")
sys.path.insert(0, "C:/App/numfast/app-builder-ponytail")

import numpy as np  # noqa: E402

from builder import MAIN  # noqa: E402

import duckdb  # noqa: E402


def _load_columns():
    import pyarrow.parquet as pq

    t = {}
    s = time.perf_counter()
    tbl = pq.read_table(PARQUET, columns=["SearchPhrase", "EventTime", "AdvEngineID"])
    t["parquet_read"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    phrases = np.asarray(tbl.column("SearchPhrase").to_pylist(), dtype=object)
    et = tbl.column("EventTime").to_numpy(zero_copy_only=False)
    adv = tbl.column("AdvEngineID").to_numpy(zero_copy_only=False)
    t["to_numpy"] = (time.perf_counter() - s) * 1000
    return phrases, et, adv, t


def _resident(a, phrases, et):
    t = {}
    s = time.perf_counter()
    resident = a["resident_prepare"]({"sp": {"values": phrases}})
    t["resident_dictionary"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    enc = a["date_encode"](et)
    t["date_encode"] = (time.perf_counter() - s) * 1000
    return resident, enc, t


def _nonempty_filter_jobs(a, codes, dictionary, validity, etcodes, tag):
    allowed = a["dict_not_empty_codes"](dictionary)
    mask = a["codes_member_mask"](codes, allowed, validity)
    return mask, [
        a["ir_series"](f"{tag}_spc", np.ascontiguousarray(codes)),
        a["ir_series"](f"{tag}_et", np.ascontiguousarray(etcodes)),
        a["ir_series"](f"{tag}_m", np.ascontiguousarray(mask), dtype="bool"),
        a["ir_filter"](f"{tag}_spf", f"{tag}_spc", f"{tag}_m"),
        a["ir_filter"](f"{tag}_etf", f"{tag}_et", f"{tag}_m"),
    ]


def _check_order(name, nf_keys, dd_keys, nf_pos, descending, limit, nf_rows, dd_rows):
    """Four sub-gates; returns (ok, detail). nf_pos = input positions of nf order."""
    ok = True
    det = {}
    det["multiset_equal"] = Counter(list(nf_keys)) == Counter(list(dd_keys))
    ok &= det["multiset_equal"]
    k = np.asarray(nf_keys)
    det["sorted"] = bool((k[:-1] >= k[1:]).all() if descending
                         else (k[:-1] <= k[1:]).all()) if k.size > 1 else True
    ok &= det["sorted"]
    # stability: positions increase inside every equal-key run
    p = np.asarray(nf_pos)
    det["stable"] = True
    if k.size:
        ch = np.empty(k.shape, dtype=bool)
        ch[0] = True
        ch[1:] = k[1:] != k[:-1]
        starts = np.flatnonzero(ch)
        ends = np.append(starts[1:], k.size)
        for lo, hi in zip(starts.tolist(), ends.tolist()):
            if not bool((np.diff(p[lo:hi]) > 0).all()):
                det["stable"] = False
                break
    ok &= det["stable"]
    det["boundary"] = (limit is None or len(nf_keys) <= limit
                       or nf_keys[limit - 1] != nf_keys[limit])
    if limit is not None and det["boundary"]:
        det["limit_exact"] = [tuple(r) for r in nf_rows] == [tuple(r) for r in dd_rows]
        ok &= det["limit_exact"]
    else:
        det["limit_exact"] = "tie-at-boundary-or-no-limit: multiset+sorted+stable only"
    return ok, det


def main():
    t_all = time.perf_counter()
    kernel = MAIN["build"](APP_DIR)
    a = kernel.alias
    phrases, et, adv, t_load = _load_columns()
    n = len(phrases)
    assert n == 999978, f"hits_1m rows {n} != 999978"
    resident, etenc, t_res = _resident(a, phrases, et)
    spc, spv, spdict = resident["sp"]["codes"], resident["sp"]["validity"], \
        resident["sp"]["dictionary"]
    etc = etenc["codes"]
    results = {}

    # ---- P1: Q26-shape ----
    s = time.perf_counter()
    mask, jobs = _nonempty_filter_jobs(a, spc, spdict, spv, etc, "p1")
    jobs += [a["ir_sort"]("p1_p", "p1_etf"),
             a["ir_slice"]("p1_t", "p1_p", limit=10),
             a["ir_gather"]("p1_g", "p1_spf", "p1_t")]
    graph = a["optimize"](a["compile"](jobs))
    t_comp = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    # full NumFast order (pre-slice) for the gate: re-sort positions on all rows
    s = time.perf_counter()
    g2 = a["optimize"](a["compile"]([
        a["ir_series"]("e", bufs["p1_etf"]), a["ir_sort"]("p", "e")]))
    perm_full = a["cpu_execute"](g2["nodes"])["p"]
    t_full = (time.perf_counter() - s) * 1000
    nf_rows = a["dictionary_decode"](bufs["p1_g"], spdict)
    s = time.perf_counter()
    dd_all = duckdb.query(
        f"SELECT EXTRACT(EPOCH FROM EventTime)::BIGINT, SearchPhrase FROM read_parquet('{PARQUET}') "
        "WHERE SearchPhrase <> '' ORDER BY EventTime").fetchall()
    dd_top = duckdb.query(
        f"SELECT SearchPhrase FROM read_parquet('{PARQUET}') "
        "WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10").fetchall()
    t_dd = (time.perf_counter() - s) * 1000
    etf = bufs["p1_etf"]
    nf_keys = [int(etf[int(i)]) for i in perm_full]
    nf_pos = [int(i) for i in perm_full]
    dd_keys = [int(v[0]) for v in dd_all]
    ok, det = _check_order("P1", nf_keys, dd_keys, nf_pos, False, 10,
                           [(r,) for r in nf_rows], dd_top)
    results["P1_Q26"] = {"ok": ok, "detail": det,
                         "stages_ms": {**t_load, **t_res, "compile": t_comp,
                                       "execute": t_exec, "full_perm": t_full,
                                       "duckdb": t_dd},
                         "rows": n, "nf_top": list(nf_rows),
                         "dd_top": [r[0] for r in dd_top]}

    # ---- P2: Q27-shape (ORDER BY SearchPhrase == rank order) ----
    s = time.perf_counter()
    mask, jobs = _nonempty_filter_jobs(a, spc, spdict, spv, etc, "p2")
    jobs += [a["ir_sort"]("p2_p", "p2_spf"),
             a["ir_slice"]("p2_t", "p2_p", limit=10),
             a["ir_gather"]("p2_g", "p2_spf", "p2_t")]
    graph = a["optimize"](a["compile"](jobs))
    t_comp = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    g2 = a["optimize"](a["compile"]([
        a["ir_series"]("c", bufs["p2_spf"]), a["ir_sort"]("p", "c")]))
    perm_full = a["cpu_execute"](g2["nodes"])["p"]
    t_full = (time.perf_counter() - s) * 1000
    nf_rows = a["dictionary_decode"](bufs["p2_g"], spdict)
    s = time.perf_counter()
    dd_all = duckdb.query(
        f"SELECT SearchPhrase FROM read_parquet('{PARQUET}') "
        "WHERE SearchPhrase <> '' ORDER BY SearchPhrase").fetchall()
    dd_top = duckdb.query(
        f"SELECT SearchPhrase FROM read_parquet('{PARQUET}') "
        "WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10").fetchall()
    t_dd = (time.perf_counter() - s) * 1000
    spf = bufs["p2_spf"]
    nf_keys = [spdict[int(spf[int(i)])] for i in perm_full]
    nf_pos = [int(i) for i in perm_full]
    dd_keys = [r[0] for r in dd_all]
    ok, det = _check_order("P2", nf_keys, dd_keys, nf_pos, False, 10,
                           [(r,) for r in nf_rows], dd_top)
    results["P2_Q27"] = {"ok": ok, "detail": det,
                         "stages_ms": {"compile": t_comp, "execute": t_exec,
                                       "full_perm": t_full, "duckdb": t_dd},
                         "rows": int(spf.size), "nf_top": list(nf_rows),
                         "dd_top": [r[0] for r in dd_top]}

    # ---- P3: Q8-shape (GROUP BY AdvEngineID ORDER BY count DESC) ----
    s = time.perf_counter()
    adv32 = np.ascontiguousarray(adv.astype(np.int64))
    jobs = [a["ir_series"]("v", adv32), a["ir_compare"]("m", "v", 0, op="!="),
            a["ir_filter"]("f", "v", "m"),
            a["ir_groupby"]("g", "f", "f", "count", result="carry")]
    graph = a["optimize"](a["compile"](jobs))
    t_comp = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    carry = bufs["g"]
    s = time.perf_counter()
    g2 = a["optimize"](a["compile"]([
        a["ir_series"]("c", np.ascontiguousarray(carry.counts)),
        a["ir_series"]("k", np.ascontiguousarray(carry.ukeys)),
        a["ir_sort"]("p", "c", descending=True),
        a["ir_gather"]("gc", "c", "p"),
        a["ir_gather"]("gk", "k", "p")]))
    b2 = a["cpu_execute"](g2["nodes"])
    t_sort = (time.perf_counter() - s) * 1000
    nf_pairs = [(int(k), int(c)) for k, c in zip(b2["gk"], b2["gc"])]
    s = time.perf_counter()
    dd_pairs = duckdb.query(
        f"SELECT AdvEngineID, COUNT(*) FROM read_parquet('{PARQUET}') "
        "WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC").fetchall()
    t_dd = (time.perf_counter() - s) * 1000
    dd_pairs = [(int(k), int(c)) for k, c in dd_pairs]
    nf_counts = [c for _, c in nf_pairs]
    pos = list(range(len(nf_pairs)))
    ok, det = _check_order("P3", nf_counts, [c for _, c in dd_pairs], pos,
                           True, None, nf_pairs, dd_pairs)
    det["pairs_exact"] = nf_pairs == dd_pairs
    ok &= det["pairs_exact"] if det["boundary"] else True
    results["P3_Q8"] = {"ok": ok, "detail": det,
                        "stages_ms": {"compile": t_comp, "execute": t_exec,
                                      "sort_slice": t_sort, "duckdb": t_dd},
                        "ngroups": len(nf_pairs), "nf_top5": nf_pairs[:5],
                        "dd_top5": dd_pairs[:5]}

    # ---- P4: Q13-shape (GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10) ----
    s = time.perf_counter()
    allowed = a["dict_not_empty_codes"](spdict)
    mask = a["codes_member_mask"](spc, allowed, spv)
    jobs = [a["ir_series"]("c", np.ascontiguousarray(spc)),
            a["ir_series"]("m", np.ascontiguousarray(mask), dtype="bool"),
            a["ir_filter"]("f", "c", "m"),
            a["ir_groupby"]("g", "f", "f", "count", result="carry")]
    graph = a["optimize"](a["compile"](jobs))
    t_comp = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    bufs = a["cpu_execute"](graph["nodes"])
    t_exec = (time.perf_counter() - s) * 1000
    carry = bufs["g"]
    s = time.perf_counter()
    g2 = a["optimize"](a["compile"]([
        a["ir_series"]("cc", np.ascontiguousarray(carry.counts)),
        a["ir_series"]("kk", np.ascontiguousarray(carry.ukeys)),
        a["ir_sort"]("pp", "cc", descending=True),
        a["ir_slice"]("tt", "pp", limit=10),
        a["ir_gather"]("gc", "cc", "tt"),
        a["ir_gather"]("gk", "kk", "tt")]))
    b2 = a["cpu_execute"](g2["nodes"])
    t_sort = (time.perf_counter() - s) * 1000
    nf_top = [(spdict[int(k)], int(c)) for k, c in zip(b2["gk"], b2["gc"])]
    # full NumFast count order for multiset gate
    g3 = a["optimize"](a["compile"]([
        a["ir_series"]("cc", np.ascontiguousarray(carry.counts)),
        a["ir_sort"]("pp", "cc", descending=True)]))
    perm_full = a["cpu_execute"](g3["nodes"])["pp"]
    nf_counts = [int(carry.counts[int(i)]) for i in perm_full]
    s = time.perf_counter()
    dd_all = duckdb.query(
        f"SELECT COUNT(*) AS c FROM read_parquet('{PARQUET}') "
        "WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC").fetchall()
    dd_top = duckdb.query(
        f"SELECT SearchPhrase, COUNT(*) AS c FROM read_parquet('{PARQUET}') "
        "WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"
    ).fetchall()
    t_dd = (time.perf_counter() - s) * 1000
    dd_top = [(r[0], int(r[1])) for r in dd_top]
    dd_counts = [int(r[0]) for r in dd_all]
    ok, det = _check_order("P4", nf_counts, dd_counts,
                           [int(i) for i in perm_full], True, 10,
                           nf_top, dd_top)
    results["P4_Q13"] = {"ok": ok, "detail": det,
                         "stages_ms": {"compile": t_comp, "execute": t_exec,
                                       "sort_slice": t_sort, "duckdb": t_dd},
                         "ngroups": int(carry.ukeys.size), "nf_top": nf_top,
                         "dd_top": dd_top}

    results["_total_s"] = time.perf_counter() - t_all
    print("=== ClickBench sort/slice gate (hits_1m, 999978 rows) ===")
    allok = True
    for name, r in results.items():
        if name.startswith("_"):
            continue
        allok &= r["ok"]
        st = " ".join(f"{k}={v:.1f}" for k, v in r["stages_ms"].items())
        print(f"{name}: {'PASS' if r['ok'] else 'FAIL'} | {r['detail']} | {st}")
        if name in ("P1_Q26", "P2_Q27"):
            print(f"  nf_top={r['nf_top']}")
            print(f"  dd_top={r['dd_top']}")
        else:
            print(f"  nf_top5={r.get('nf_top5', r.get('nf_top'))}")
            print(f"  dd_top5={r.get('dd_top5', r.get('dd_top'))}")
    print(f"TOTAL s: {results['_total_s']:.1f} ALL_OK={allok}")
    if not allok:
        sys.exit(1)


if __name__ == "__main__":
    main()
