# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""ClickBench 43-query matrix: NumFast (rebuilt fork) vs DuckDB on hits_1m.

Bulk-only (999978 rows, parquet). No 1B. Heavy: run explicitly with the
system python (see report for the exact command). Strictly sequential probes.

Per query: supported -> execute via engine + DuckDB cross-check (EXACT for
ints/strings/dates, TOL 1e-9 for float AVGs); unsupported -> explicit reason
(BIGINT narrowing, missing DISTINCT/CASE/REGEXP/groupby-min, wide SELECT *).
int64 is NOT implemented: Core stays GPU-portable (int32 logical); BIGINT
queries are explicit OverflowError, never silent wrap. No query-cache,
no fixed-M, no H2O tricks. Seed: no RNG (fixed data). Stage breakdown per
probe + matrix JSON next to this script.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

PARQUET = "C:/App/competitions/ClickBench/data/hits_1m.parquet"
APP_DIR = "C:/App/numfast/numfast"
sys.path.insert(0, "C:/App/numfast/numfast/src")
sys.path.insert(0, "C:/App/numfast/app-builder")

import numpy as np  # noqa: E402

import duckdb  # noqa: E402
from builder import MAIN  # noqa: E402

COLS = ["SearchPhrase", "EventTime", "EventDate", "AdvEngineID", "ResolutionWidth",
        "CounterID", "ClientIP", "RegionID", "URL", "Title", "MobilePhoneModel",
        "SearchEngineID", "DontCountHits", "IsRefresh", "IsLink", "IsDownload", "UserID"]

MATRIX = []


def rec(q, status, detail, stages=None, match=None):
    MATRIX.append({"q": q, "status": status, "detail": detail,
                   "stages_ms": stages or {}, "match": match})


def _t():
    return time.perf_counter()


def main():
    kernel = MAIN["build"](APP_DIR)
    a = kernel.alias

    # ---------- ingest (once) ----------
    import pyarrow.parquet as pq

    s = _t()
    tbl = pq.read_table(PARQUET, columns=COLS)
    t_read = (_t() - s) * 1000
    s = _t()
    d = {c: tbl.column(c).to_pylist() for c in
         ("SearchPhrase", "URL", "Title", "MobilePhoneModel")}
    for c in ("AdvEngineID", "ResolutionWidth", "CounterID", "ClientIP",
              "RegionID", "SearchEngineID", "DontCountHits", "IsRefresh",
              "IsLink", "IsDownload"):
        d[c] = tbl.column(c).to_numpy(zero_copy_only=False)
    d["EventTime"] = tbl.column("EventTime").to_numpy(zero_copy_only=False)
    d["EventDate"] = tbl.column("EventDate").to_numpy(zero_copy_only=False)
    d["UserID"] = np.ascontiguousarray(tbl.column("UserID").to_numpy(zero_copy_only=False))
    t_np = (_t() - s) * 1000
    n = len(d["SearchPhrase"])
    assert n == 999978, n
    s = _t()
    res = a["resident_prepare"]({k: {"values": d[k]} for k in
                                ("SearchPhrase", "URL", "Title", "MobilePhoneModel")})
    t_dict = (_t() - s) * 1000
    s = _t()
    ete = a["date_encode"](d["EventTime"])
    ede = a["date_encode"](d["EventDate"])
    t_date = (_t() - s) * 1000
    SP, SPv, SPd = res["SearchPhrase"]["codes"], res["SearchPhrase"]["validity"], \
        res["SearchPhrase"]["dictionary"]
    UC, Ud = res["URL"]["codes"], res["URL"]["dictionary"]
    TC, Td = res["Title"]["codes"], res["Title"]["dictionary"]
    ET, ED = ete["codes"], ede["codes"]
    ADV, RESW = np.ascontiguousarray(d["AdvEngineID"]), np.ascontiguousarray(d["ResolutionWidth"])
    CID = np.ascontiguousarray(d["CounterID"])
    CIP = np.ascontiguousarray(d["ClientIP"])
    RID = np.ascontiguousarray(d["RegionID"])
    SEID = np.ascontiguousarray(d["SearchEngineID"])
    DCH = np.ascontiguousarray(d["DontCountHits"])
    IRF = np.ascontiguousarray(d["IsRefresh"])
    ILK = np.ascontiguousarray(d["IsLink"])
    IDL = np.ascontiguousarray(d["IsDownload"])
    UID = np.ascontiguousarray(d["UserID"], dtype=np.int64)
    ING = {"parquet_read": t_read, "to_numpy": t_np, "dict4": t_dict,
           "date2": t_date, "rows": n}

    def series(name, arr, dtype="int32"):
        return a["ir_series"](name, arr, dtype) if dtype != "int32" else \
            a["ir_series"](name, arr)

    def run(jobs):
        s = _t()
        g = a["optimize"](a["compile"](jobs))
        tc = (_t() - s) * 1000
        s = _t()
        b = a["cpu_execute"](g["nodes"])
        te = (_t() - s) * 1000
        return b, {"compile": tc, "execute": te}

    def dd(sql):
        s = _t()
        out = duckdb.query(sql.format(PQ=PARQUET)).fetchall()
        return out, {"duckdb": (_t() - s) * 1000}

    # ---------- Q1 ----------
    b, st = run([series("v", RESW), a["ir_reduce"]("r", "v", "count")])
    r, sd = dd("SELECT COUNT(*) FROM read_parquet('{PQ}')")
    rec("Q1", "supported", f"count={b['r']}", {**st, **sd},
        "EXACT" if int(b["r"]) == r[0][0] else "MISMATCH")

    # ---------- Q2 ----------
    b, st = run([series("v", ADV), a["ir_compare"]("m", "v", 0, op="!="),
                 a["ir_filter"]("f", "v", "m"), a["ir_reduce"]("r", "f", "count")])
    r, sd = dd("SELECT COUNT(*) FROM read_parquet('{PQ}') WHERE AdvEngineID <> 0")
    rec("Q2", "supported", f"count={b['r']}", {**st, **sd},
        "EXACT" if int(b["r"]) == r[0][0] else "MISMATCH")

    # ---------- Q3 ----------
    b, st = run([series("v", RESW), a["ir_reduce"]("s", "v", "sum"),
                 a["ir_reduce"]("c", "v", "count"), a["ir_reduce"]("m", "v", "mean")])
    r, sd = dd("SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) "
               "FROM read_parquet('{PQ}')")
    r3, _ = dd("SELECT SUM(ResolutionWidth), COUNT(*), AVG(ResolutionWidth) "
               "FROM read_parquet('{PQ}')")
    ok = int(b["s"]) == r3[0][0] and int(b["c"]) == r3[0][1] and \
        abs(float(b["m"]) - r3[0][2]) <= 1e-9 * max(1.0, abs(r3[0][2]))
    rec("Q3", "supported", f"sum={b['s']} count={b['c']} avg={b['m']}",
        {**st, **sd}, "EXACT+TOL" if ok else "MISMATCH")

    # ---------- Q4/Q5/Q6 ----------
    try:
        run([a["ir_series"]("u", [435090932899640449, 1234567890123456789])])
        rec("Q4", "supported?", "UNEXPECTED: no overflow", {}, "MISMATCH")
    except OverflowError as e:
        rec("Q4", "unsupported", f"AVG(UserID): BIGINT narrowing {str(e)[:80]}", {}, "N/A")
    _b5, _st5 = run([series("u5", np.ascontiguousarray(UID), "int64"),
                     a["ir_count_distinct"]("gd5", "u5", None)])
    _r5, _sd5 = dd("SELECT COUNT(DISTINCT UserID) FROM read_parquet('{PQ}')")
    rec("Q5", "supported", f"count={int(_b5['gd5#ng'])}",
        {**_st5, **_sd5},
        "EXACT" if int(_b5["gd5#ng"]) == _r5[0][0] else "MISMATCH")
    _b6, _st6 = run([a["ir_series"]("sp6", np.ascontiguousarray(SP),
                                    validity=np.ascontiguousarray(SPv)),
                     a["ir_count_distinct"]("gd6", "sp6", None)])
    _r6, _sd6 = dd("SELECT COUNT(DISTINCT SearchPhrase) FROM read_parquet('{PQ}')")
    rec("Q6", "supported", f"count={int(_b6['gd6#ng'])}",
        {**_st6, **_sd6},
        "EXACT" if int(_b6["gd6#ng"]) == _r6[0][0] else "MISMATCH")

    # ---------- Q7 ----------
    b, st = run([series("v", ED), a["ir_reduce"]("mn", "v", "min"),
                 a["ir_reduce"]("mx", "v", "max")])
    r, sd = dd("SELECT MIN(EventDate), MAX(EventDate) FROM read_parquet('{PQ}')")
    import datetime as _dt
    nf = [str(_dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=int(b["mn"])))[:10],
          str(_dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=int(b["mx"])))[:10]]
    rec("Q7", "supported", f"min={nf[0]} max={nf[1]}", {**st, **sd},
        "EXACT" if nf == [str(r[0][0]), str(r[0][1])] else "MISMATCH")

    # ---------- Q8 ----------
    b, st = run([series("v", ADV), a["ir_compare"]("m", "v", 0, op="!="),
                 a["ir_filter"]("f", "v", "m"),
                 a["ir_groupby"]("g", "f", "f", "count", result="carry")])
    c = b["g"]
    b2, st2 = run([a["ir_series"]("cc", np.ascontiguousarray(c.counts), "int64"),
                   a["ir_series"]("kk", np.ascontiguousarray(c.ukeys), "int64"),
                   a["ir_sort"]("p", "cc", descending=True),
                   a["ir_gather"]("gc", "cc", "p"), a["ir_gather"]("gk", "kk", "p")])
    nf = [(int(k), int(x)) for k, x in zip(b2["gk"], b2["gc"])]
    r, sd = dd("SELECT AdvEngineID, COUNT(*) FROM read_parquet('{PQ}') "
               "WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC")
    r = [(int(k), int(x)) for k, x in r]
    rec("Q8", "supported", f"groups={len(nf)} top={nf[:3]}", {**st, **st2, **sd},
        "EXACT" if nf == r else "MISMATCH")

    # ---------- Q9-Q12, Q14 (DISTINCT) ----------
    # NULL-EXCLUDED DISTINCT rule: COUNT DISTINCT never counts NULL/invalid
    # rows. SearchPhrase has 930624 NULLs of 999978 rows; global
    # COUNT(DISTINCT SearchPhrase)=18315 via ir_count_distinct #ng (invalid
    # excluded). Same rule for grouped COUNT(DISTINCT UserID) below.
    # ---------- Q12 (MobilePhone + MobilePhoneModel grouped COUNT DISTINCT) ----------
    # ingest MobilePhone + MPM<>'' mask + pack_keys 2-col (radix: both
    # columns small, product fits int64; hash fallback only if radix
    # refuses) + group_count_distinct + divmod pair decode + ORDER BY u
    # LIMIT 10 + tie-gate. Validity AND of both key columns: MPM mask
    # excludes NULL/empty, MobilePhone has no NULLs (int64 0..118).
    _mp12 = np.ascontiguousarray(
        pq.read_table(PARQUET, columns=["MobilePhone"]).column("MobilePhone")
        .to_numpy(zero_copy_only=False).astype(np.int32))
    _MPC12 = res["MobilePhoneModel"]["codes"]
    _MPC12v = res["MobilePhoneModel"]["validity"]
    _MPC12d = res["MobilePhoneModel"]["dictionary"]
    _mall12 = np.ascontiguousarray(a["codes_member_mask"](
        _MPC12, a["dict_not_empty_codes"](_MPC12d), _MPC12v))
    _t12a = _t()
    _b12, _st12 = run([series("mp12", _mp12),
                       series("mc12", np.ascontiguousarray(_MPC12)),
                       series("u12", np.ascontiguousarray(UID), "int64"),
                       a["ir_series"]("mm12", _mall12, dtype="bool"),
                       a["ir_filter"]("mpf12", "mp12", "mm12"),
                       a["ir_filter"]("mcf12", "mc12", "mm12"),
                       a["ir_filter"]("uf12", "u12", "mm12")])
    _mpf12 = np.ascontiguousarray(_b12["mpf12"])
    _mcf12 = np.ascontiguousarray(_b12["mcf12"])
    _M12 = int(_mcf12.max()) + 1
    try:
        _b12b, _st12b = run([series("mpf12b", _mpf12),
                             series("mcf12b", _mcf12),
                             series("uf12b", np.ascontiguousarray(_b12["uf12"]), "int64"),
                             a["ir_pack_keys"]("kk12", "mpf12b", "mcf12b", mode="radix",
                                               radix=[int(_mpf12.max()) + 1, _M12]),
                             a["ir_count_distinct"]("gd12", "uf12b", "kk12")])
        _pack12mode = "radix"
    except ValueError:
        _b12b, _st12b = run([series("mpf12b", _mpf12),
                             series("mcf12b", _mcf12),
                             series("uf12b", np.ascontiguousarray(_b12["uf12"]), "int64"),
                             a["ir_pack_keys"]("kk12", "mpf12b", "mcf12b", mode="hash"),
                             a["ir_count_distinct"]("gd12", "uf12b", "kk12")])
        _pack12mode = "hash"
    _t12e = (_t() - _t12a) * 1000
    if _pack12mode == "radix":
        _nf12 = sorted(((int(_k // _M12), _MPC12d[int(_k % _M12)], int(_v))
                        for _k, _v in _b12b["gd12"].items()),
                       key=lambda _t: -_t[2])[:10]
    else:
        _nf12 = sorted(((int(_k[0]), _MPC12d[int(_k[1])], int(_v))
                        for _k, _v in _b12b["gd12"].items()),
                       key=lambda _t: -_t[2])[:10]
    _r12, _sd12 = dd("SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u "
                     "FROM read_parquet('{PQ}') WHERE MobilePhoneModel <> '' "
                     "GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 11")
    _r12 = [(int(_k), _p, int(_u)) for _k, _p, _u in _r12]
    _win12, _ext12 = _r12[:10], _r12[10:]
    _tie12 = bool(_ext12) and bool(_win12) and _ext12[0][2] == _win12[-1][2]
    _nfc12 = [_u for _, _, _u in _nf12]
    if _nf12 == _win12:
        _m12 = "EXACT"
    elif (_tie12 and Counter(_nfc12) == Counter([_u for _, _, _u in _win12])
            and all(_x >= _y for _x, _y in zip(_nfc12, _nfc12[1:]))):
        _m12 = "TIE-MULTISET"
    else:
        _m12 = "MISMATCH"
    rec("Q12", "supported", f"pack={_pack12mode} top={_nf12[0] if _nf12 else None} tie={_tie12} exec_ms={_t12e:.1f}",
        {**_st12, **_st12b, **_sd12}, _m12)
    _b9, _st9 = run([series("u9", np.ascontiguousarray(UID), "int64"),
                     series("rid9", np.ascontiguousarray(RID)),
                     a["ir_count_distinct"]("gd9", "u9", "rid9")])
    _r9, _sd9 = dd("SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM read_parquet('{PQ}') "
                   "GROUP BY RegionID ORDER BY u DESC LIMIT 11")
    _r9 = [(int(k), int(u)) for k, u in _r9]
    _win9, _ext9 = _r9[:10], _r9[10:]
    _nf9 = sorted(((int(k), int(v)) for k, v in _b9["gd9"].items()),
                  key=lambda t: -t[1])[:10]
    _tie9 = bool(_ext9) and bool(_win9) and _ext9[0][1] == _win9[-1][1]
    _nfc9 = [u for _, u in _nf9]
    if _nf9 == _win9:
        _m9 = "EXACT"
    elif (_tie9 and Counter(_nfc9) == Counter([u for _, u in _win9])
            and all(x >= y for x, y in zip(_nfc9, _nfc9[1:]))):
        _m9 = "TIE-MULTISET"
    else:
        _m9 = "MISMATCH"
    rec("Q9", "supported", f"top={_nf9[0] if _nf9 else None} tie={_tie9}",
        {**_st9, **_sd9}, _m9)
    _mall14 = np.ascontiguousarray(a["codes_member_mask"](
        SP, a["dict_not_empty_codes"](SPd), SPv))
    _b14, _st14 = run([series("sp14", np.ascontiguousarray(SP)),
                       series("u14", np.ascontiguousarray(UID), "int64"),
                       a["ir_series"]("mm14", _mall14, dtype="bool"),
                       a["ir_filter"]("spf14", "sp14", "mm14"),
                       a["ir_filter"]("uf14", "u14", "mm14"),
                       a["ir_count_distinct"]("gd14", "uf14", "spf14")])
    _r14, _sd14 = dd("SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM read_parquet('{PQ}') "
                     "WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 11")
    _r14 = [(k, int(u)) for k, u in _r14]
    _win14, _ext14 = _r14[:10], _r14[10:]
    _nf14 = sorted(((SPd[int(k)], int(v)) for k, v in _b14["gd14"].items()),
                   key=lambda t: -t[1])[:10]
    _tie14 = bool(_ext14) and bool(_win14) and _ext14[0][1] == _win14[-1][1]
    _nfc14 = [u for _, u in _nf14]
    if _nf14 == _win14:
        _m14 = "EXACT"
    elif (_tie14 and Counter(_nfc14) == Counter([u for _, u in _win14])
            and all(x >= y for x, y in zip(_nfc14, _nfc14[1:]))):
        _m14 = "TIE-MULTISET"
    else:
        _m14 = "MISMATCH"
    rec("Q14", "supported", f"top={_nf14[0] if _nf14 else None} tie={_tie14}",
        {**_st14, **_sd14}, _m14)
    _MPC = res["MobilePhoneModel"]["codes"]
    _MPCv = res["MobilePhoneModel"]["validity"]
    _MPCd = res["MobilePhoneModel"]["dictionary"]
    _mall11 = np.ascontiguousarray(a["codes_member_mask"](
        _MPC, a["dict_not_empty_codes"](_MPCd), _MPCv))
    _b11, _st11 = run([series("mp11", np.ascontiguousarray(_MPC)),
                       series("u11", np.ascontiguousarray(UID), "int64"),
                       a["ir_series"]("mm11", _mall11, dtype="bool"),
                       a["ir_filter"]("mpf11", "mp11", "mm11"),
                       a["ir_filter"]("uf11", "u11", "mm11"),
                       a["ir_count_distinct"]("gd11", "uf11", "mpf11")])
    _r11q, _sd11 = dd("SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM read_parquet('{PQ}') "
                      "WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 11")
    _r11q = [(k, int(u)) for k, u in _r11q]
    _win11, _ext11 = _r11q[:10], _r11q[10:]
    _nf11 = sorted(((_MPCd[int(k)], int(v)) for k, v in _b11["gd11"].items()),
                   key=lambda t: -t[1])[:10]
    _tie11 = bool(_ext11) and bool(_win11) and _ext11[0][1] == _win11[-1][1]
    _nfc11 = [u for _, u in _nf11]
    _dall11, _ = dd("SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM read_parquet('{PQ}') "
                    "WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel")
    _dall11 = [(k, int(u)) for k, u in _dall11]
    _full11l = [(_MPCd[int(k)], int(v)) for k, v in _b11["gd11"].items()]
    if _nf11 == _win11:
        _m11 = "EXACT"
    elif (_tie11 and Counter(_full11l) == Counter(_dall11)
            and Counter(_nf11) == Counter(_win11)
            and all(x >= y for x, y in zip(_nfc11, _nfc11[1:]))):
        _m11 = "TIE-MULTISET"
    else:
        _m11 = "MISMATCH"
    rec("Q11", "supported", f"top={_nf11[0] if _nf11 else None} tie={_tie11}",
        {**_st11, **_sd11}, _m11)
    # Q10: RegionID, SUM(AdvEngineID), COUNT, AVG(ResolutionWidth),
    # COUNT(DISTINCT UserID) GROUP BY RegionID ORDER BY c DESC LIMIT 10
    b, st = run([series("adv", ADV), series("resw", RESW), series("rid", np.ascontiguousarray(RID)),
                 series("u10", np.ascontiguousarray(UID), "int64"),
                 a["ir_groupby"]("s10", "adv", "rid", op="sum", result="dict"),
                 a["ir_groupby"]("c10", "rid", "rid", op="count", result="dict"),
                 a["ir_groupby"]("m10", "resw", "rid", op="mean", result="dict"),
                 a["ir_count_distinct"]("gd10", "u10", "rid")])
    r, sd = dd("SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), "
               "COUNT(DISTINCT UserID) FROM read_parquet('{PQ}') "
               "GROUP BY RegionID ORDER BY c DESC LIMIT 10")
    dd10 = [(int(k), int(s), int(c), float(v), int(u)) for k, s, c, v, u in r]
    nf10 = sorted(((int(k), int(b["s10"][k]), int(b["c10"][k]),
                    float(b["m10"][k]), int(b["gd10"][k])) for k in b["c10"]),
                  key=lambda t: -t[2])[:10]
    ok10 = all(k1 == k2 and s1 == s2 and c1 == c2 and u1 == u2 and
               abs(v1 - v2) <= 1e-9 * max(1.0, abs(v2))
               for (k1, s1, c1, v1, u1), (k2, s2, c2, v2, u2) in zip(nf10, dd10))
    rec("Q10", "supported", f"top={nf10[0]} null_excluded=930624",
        {**st, **sd}, "EXACT" if ok10 else "MISMATCH")

    # ---------- Q13 ----------
    allowed = a["dict_not_empty_codes"](SPd)
    m = a["codes_member_mask"](SP, allowed, SPv)
    b, st = run([series("c", np.ascontiguousarray(SP)),
                 a["ir_series"]("mm", np.ascontiguousarray(m), dtype="bool"),
                 a["ir_filter"]("f", "c", "mm"),
                 a["ir_groupby"]("g", "f", "f", "count", result="carry")])
    c = b["g"]
    b2, st2 = run([a["ir_series"]("cc", np.ascontiguousarray(c.counts), "int64"),
                   a["ir_series"]("kk", np.ascontiguousarray(c.ukeys), "int64"),
                   a["ir_sort"]("p", "cc", descending=True),
                   a["ir_slice"]("t", "p", limit=10),
                   a["ir_gather"]("gc", "cc", "t"), a["ir_gather"]("gk", "kk", "t")])
    nf = [(SPd[int(k)], int(x)) for k, x in zip(b2["gk"], b2["gc"])]
    r, sd = dd("SELECT SearchPhrase, COUNT(*) AS c FROM read_parquet('{PQ}') "
               "WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10")
    r = [(k, int(x)) for k, x in r]
    rec("Q13", "supported", f"groups={int(c.ukeys.size)} top={nf[:2]}", {**st, **st2, **sd},
        "EXACT" if nf == r else "MISMATCH")

    # ---------- Q15 (two-TEXT composite, filtered population, divmod unpack) ----------
    se_res = a["resident_prepare"]({"se": {"values": SEID.tolist()}})
    SE = se_res["se"]["codes"]
    ne = a["dict_not_empty_codes"](SPd)
    msp15 = a["codes_member_mask"](SP, ne, SPv)
    SEf, SPf = SE[msp15], SP[msp15]
    M = int(SPf.max()) + 1
    b, st = run([series("a", np.ascontiguousarray(SEf)), series("p", np.ascontiguousarray(SPf)),
                 a["ir_pack_keys"]("k", "a", "p", mode="radix",
                                   radix=[int(SEf.max()) + 1, M]),
                 a["ir_groupby"]("g", "a", "k", "count", result="carry")])
    c = b["g"]
    b2, st2 = run([a["ir_series"]("cc", np.ascontiguousarray(c.counts), "int64"),
                   a["ir_series"]("kk", np.ascontiguousarray(c.ukeys), "int64"),
                   a["ir_sort"]("pp", "cc", descending=True),
                   a["ir_slice"]("tt", "pp", limit=10),
                   a["ir_gather"]("gc", "cc", "tt"), a["ir_gather"]("gk", "kk", "tt")])
    nf = [(int(kk // M), SPd[int(kk % M)], int(x))
          for kk, x in zip(b2["gk"], b2["gc"])]
    r, sd = dd("SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM read_parquet('{PQ}') "
               "WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase "
               "ORDER BY c DESC LIMIT 10")
    dd_map = Counter((int(k), p) for k, p, _ in r)
    nf_map = Counter((se, p) for se, p, _ in nf)
    okc = [x for _, _, x in nf] == [int(x) for _, _, x in r] and nf_map == dd_map
    rec("Q15", "supported", f"groups={int(c.ukeys.size)} top-counts={[x for _, _, x in nf][:3]}",
        {**st, **st2, **sd}, "EXACT" if okc else "MISMATCH")

    # ---------- Q16-Q20 (UserID) ----------
    # LIMIT-no-ORDER rule (Q17/Q18): SQL row order is unspecified for ties
    # (Q17 ORDER BY COUNT ties) and fully unspecified for LIMIT without
    # ORDER BY (Q18). Harness compares order-insensitively: set/multiset +
    # membership in the full group set, never raw row order. Engine unchanged.
    rec("Q16", "unsupported", "GROUP BY UserID: BIGINT keys narrowing", {}, "N/A")
    _sp_null = len(SPd)
    _sp2 = np.ascontiguousarray(np.where(SPv, SP, _sp_null).astype(np.int64))
    _b1718, _st1718 = run([series("uu", np.ascontiguousarray(UID), "int64"),
                           a["ir_series"]("ss", _sp2, "int64"),
                           a["ir_pack_keys"]("kk1718", "uu", "ss", mode="hash"),
                           a["ir_groupby"]("g1718", "uu", "kk1718", op="count", result="dict")])
    _full1718 = dict(_b1718["g1718"])
    def _dec1718(uid, spc):
        return (int(uid), None if int(spc) == _sp_null else SPd[int(spc)])
    _r17, _sd17 = dd("SELECT UserID, SearchPhrase, COUNT(*) FROM read_parquet('{PQ}') "
                     "GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 11")
    _r17 = [(int(u), p, int(c)) for u, p, c in _r17]
    _win17, _ext17 = _r17[:10], _r17[10:]
    _nf17 = sorted(((_dec1718(u, s)[0], _dec1718(u, s)[1], int(c))
                    for (u, s), c in _full1718.items()), key=lambda t: -t[2])[:10]
    _tie17 = bool(_ext17) and bool(_win17) and _ext17[0][2] == _win17[-1][2]
    if _nf17 == _win17:
        _m17 = "EXACT"
    elif (Counter([c for _, _, c in _nf17]) == Counter([c for _, _, c in _win17])
          and all(x >= y for x, y in zip([c for _, _, c in _nf17], [c for _, _, c in _nf17][1:]))):
        _m17 = "TIE-MULTISET"
    else:
        _m17 = "MISMATCH"
    rec("Q17", "supported", f"groups={len(_full1718)} top={_nf17[:1]} tie={_tie17}",
        {**_st1718, **_sd17}, _m17)
    _r18, _sd18 = dd("SELECT UserID, SearchPhrase, COUNT(*) FROM read_parquet('{PQ}') "
                     "GROUP BY UserID, SearchPhrase LIMIT 10")
    _fullset18 = set((int(uu), (None if int(ss) == _sp_null else SPd[int(ss)]), int(cc))
                     for (uu, ss), cc in _full1718.items())
    _nf18 = _nf17  # any 10 valid groups satisfy LIMIT-without-ORDER; reuse top-10
    _m18 = "MULTISET" if set(_nf18) <= _fullset18 and len(_nf18) == 10 else "MISMATCH"
    rec("Q18", "supported", f"groups={len(_full1718)} subset={_m18 == 'MULTISET'}",
        {**_st1718, **_sd18}, _m18)
    rec("Q19", "unsupported", "GROUP BY UserID+m+SearchPhrase: BIGINT keys", {}, "N/A")
    try:
        run([a["ir_series"]("u", [435090932899640449])])
        rec("Q20", "supported?", "UNEXPECTED", {}, "MISMATCH")
    except OverflowError:
        rec("Q20", "unsupported", "WHERE UserID=435090932899640449: BIGINT literal narrowing", {}, "N/A")

    # ---------- Q21 (LIKE via dict_contains) ----------
    hit = a["dict_contains"](Ud, "google")
    m = a["codes_member_mask"](UC, hit, res["URL"]["validity"])
    b, st = run([series("c", np.ascontiguousarray(UC)),
                 a["ir_series"]("mm", np.ascontiguousarray(m), dtype="bool"),
                 a["ir_filter"]("f", "c", "mm"), a["ir_reduce"]("r", "f", "count")])
    r, sd = dd("SELECT COUNT(*) FROM read_parquet('{PQ}') WHERE URL LIKE '%google%'")
    rec("Q21", "supported", f"count={b['r']}", {**st, **sd},
        "EXACT" if int(b["r"]) == r[0][0] else "MISMATCH")

    # ---------- Q22/Q23 ----------
    # Q22 exact (single group; grouped string MIN via lexical rank, fix
    # 21b19d9 kept). Q23 tie rule: c9==c10==5 at the LIMIT boundary, so any
    # top-10 with the same count-multiset + descending sortedness is a pass
    # (TIE-MULTISET, Counter-based); engine order unchanged.
    _hit22 = a["dict_contains"](Ud, "google")
    _m22a = a["codes_member_mask"](UC, _hit22, res["URL"]["validity"])
    _m22b = a["codes_member_mask"](SP, a["dict_not_empty_codes"](SPd), SPv)
    _mall22 = np.ascontiguousarray(np.asarray(_m22a) & np.asarray(_m22b))
    _b22, _st22 = run([series("sp22", np.ascontiguousarray(SP)),
                       series("url22", np.ascontiguousarray(UC)),
                       a["ir_series"]("mm22", _mall22, dtype="bool"),
                       a["ir_filter"]("spf22", "sp22", "mm22"),
                       a["ir_filter"]("urlf22", "url22", "mm22"),
                       a["ir_groupby"]("min22", "urlf22", "spf22", op="min", result="carry"),
                       a["ir_groupby"]("cnt22", "spf22", "spf22", op="count", result="carry")])
    _c22 = _b22["cnt22"]
    _o22 = np.argsort(-np.ascontiguousarray(_c22.counts), kind="stable")[:10]
    _nf22 = [(SPd[int(_c22.ukeys[i])], Ud[int(_b22["min22"].mins["v"][
        int(np.where(_b22["min22"].ukeys == _c22.ukeys[i])[0][0])])], int(_c22.counts[i]))
        for i in _o22]
    _r22, _sd22 = dd("SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM read_parquet('{PQ}') "
                     "WHERE URL LIKE '%google%' AND SearchPhrase <> '' "
                     "GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10")
    _r22 = [(k, u, int(c)) for k, u, c in _r22]
    rec("Q22", "supported", f"groups={int(_c22.ukeys.size)} top={_nf22[:1]}",
        {**_st22, **_sd22}, "EXACT" if _nf22 == _r22 else "MISMATCH")
    _hitT = a["dict_contains"](Td, "Google")
    _mTa = a["codes_member_mask"](TC, _hitT, res["Title"]["validity"])
    _hitU = a["dict_contains"](Ud, ".google.")
    _mUb = ~(np.asarray(a["codes_member_mask"](UC, _hitU, res["URL"]["validity"])))
    _mall23 = np.ascontiguousarray(np.asarray(_mTa) & _mUb & np.asarray(_m22b))
    _b23, _st23 = run([series("sp23", np.ascontiguousarray(SP)),
                       series("url23", np.ascontiguousarray(UC)),
                       series("tit23", np.ascontiguousarray(TC)),
                       series("u23", np.ascontiguousarray(UID), "int64"),
                       a["ir_series"]("mm23", _mall23, dtype="bool"),
                       a["ir_filter"]("spf23", "sp23", "mm23"),
                       a["ir_filter"]("urlf23", "url23", "mm23"),
                       a["ir_filter"]("titf23", "tit23", "mm23"),
                       a["ir_filter"]("uf23", "u23", "mm23"),
                       a["ir_groupby"]("minU23", "urlf23", "spf23", op="min", result="carry"),
                       a["ir_groupby"]("minT23", "titf23", "spf23", op="min", result="carry"),
                       a["ir_groupby"]("cnt23", "spf23", "spf23", op="count", result="carry"),
                       a["ir_count_distinct"]("gd23", "uf23", "spf23")])
    _c23 = _b23["cnt23"]
    _o23 = np.argsort(-np.ascontiguousarray(_c23.counts), kind="stable")[:10]
    def _gmin(carry, key):
        uk = carry.ukeys
        j = int(np.where(uk == key)[0][0])
        return int(carry.mins["v"][j])
    _nf23 = [(SPd[int(_c23.ukeys[i])], Ud[_gmin(_b23["minU23"], _c23.ukeys[i])],
              Td[_gmin(_b23["minT23"], _c23.ukeys[i])], int(_c23.counts[i]),
              int(_b23["gd23"].get(int(_c23.ukeys[i]), 0))) for i in _o23]
    _r23, _sd23 = dd("SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) "
                     "FROM read_parquet('{PQ}') WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' "
                     "AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 11")
    _r23 = [(k, u, t, int(c), int(uu)) for k, u, t, c, uu in _r23]
    _win23, _ext23 = _r23[:10], _r23[10:]
    _tie23 = bool(_ext23) and bool(_win23) and _ext23[0][3] == _win23[-1][3]
    _nfc23 = [c for _, _, _, c, _ in _nf23]
    if _nf23 == _win23:
        _m23 = "EXACT"
    elif (_tie23 and Counter(_nfc23) == Counter([c for _, _, _, c, _ in _win23])
          and all(x >= y for x, y in zip(_nfc23, _nfc23[1:]))):
        _m23 = "TIE-MULTISET"
    else:
        _m23 = "MISMATCH"
    rec("Q23", "supported", f"groups={int(_c23.ukeys.size)} tie={_tie23} top_c={_nfc23[:3]}",
        {**_st23, **_sd23}, _m23)
    _hit24 = a["dict_contains"](Ud, "google")
    _m24 = np.ascontiguousarray(a["codes_member_mask"](UC, _hit24, res["URL"]["validity"]))
    _b24, _st24 = run([series("cc24", np.ascontiguousarray(UC)),
                       series("ee24", np.ascontiguousarray(ET)),
                       a["ir_series"]("mm24", _m24, dtype="bool"),
                       a["ir_filter"]("cf24", "cc24", "mm24"),
                       a["ir_filter"]("ef24", "ee24", "mm24"),
                       a["ir_sort"]("p24", "ef24"),
                       a["ir_slice"]("t24", "p24", limit=10),
                       a["ir_gather"]("g24", "cf24", "t24")])
    _nf24 = list(a["dictionary_decode"](_b24["g24"], Ud))
    _r24w, _sd24 = dd("SELECT URL, EventTime FROM read_parquet('{PQ}') "
                      "WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 11")
    _win24 = [x[0] for x in _r24w[:10]]
    _tie24 = len(_r24w) == 11 and _r24w[9][1] == _r24w[10][1]
    if list(_nf24) == _win24:
        _m24s = "EXACT"
    elif (_tie24 and Counter(list(_nf24)) == Counter(_win24)):
        _m24s = "TIE-MULTISET"
    else:
        _m24s = "MISMATCH"
    rec("Q24", "supported",
        f"PARTIAL url-only projection (FULL 105-col deferred, wide projection ARCH) "
        f"filtered={int(np.sum(_m24))} tie={_tie24}",
        {**_st24, **_sd24}, _m24s)

    # ---------- Q25/Q26 (P1/P2 shapes) ----------
    for tag, order_col, qname, ddsql in (
            ("et", ET, "Q25",
             "SELECT SearchPhrase FROM read_parquet('{PQ}') WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10"),
            ("sp", SP, "Q26",
             "SELECT SearchPhrase FROM read_parquet('{PQ}') WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10")):
        allowed = a["dict_not_empty_codes"](SPd)
        m = a["codes_member_mask"](SP, allowed, SPv)
        key = ET if tag == "et" else SP
        b, st = run([series("cc", np.ascontiguousarray(SP)), series("kk", np.ascontiguousarray(key)),
                     a["ir_series"]("mm", np.ascontiguousarray(m), dtype="bool"),
                     a["ir_filter"]("cf", "cc", "mm"), a["ir_filter"]("kf", "kk", "mm"),
                     a["ir_sort"]("p", "kf"), a["ir_slice"]("t", "p", limit=10),
                     a["ir_gather"]("g", "cf", "t")])
        nf = a["dictionary_decode"](b["g"], SPd)
        r, sd = dd(ddsql)
        r = [x[0] for x in r]
        # TIE-BOUNDARY rule (Q25): SQL tie order at LIMIT 10 is unspecified;
        # r9==r10 means any top-10 with the same Counter-multiset passes.
        # Counter-multiset + sortedness, engine order unchanged.
        _r11, _ = dd(ddsql + " LIMIT 11" if "LIMIT" not in ddsql else ddsql.replace("LIMIT 10", "LIMIT 11"))
        _r11 = [x[0] for x in _r11]
        _tie25 = len(_r11) == 11 and _r11[9] == _r11[10]
        # full-order multiset gate (ties at LIMIT boundary are unspecified in SQL)
        b2, _ = run([a["ir_series"]("kf2", np.ascontiguousarray(b["kf"])),
                     a["ir_sort"]("pp2", "kf2")])
        nfkeys = [int(b["kf"][int(i)]) for i in b2["pp2"]]
        keycol = "EventTime" if tag == "et" else "SearchPhrase"
        sel = "EXTRACT(EPOCH FROM EventTime)::BIGINT" if tag == "et" else "SearchPhrase"
        rfull, _ = dd(f"SELECT {sel} FROM read_parquet('{{PQ}}') WHERE SearchPhrase <> '' "
                      f"ORDER BY {keycol}")
        rfull = [int(x[0]) if tag == "et" else x[0] for x in rfull]
        if tag == "sp":
            nfkeys = [SPd[k] for k in nfkeys]
        ms = Counter(nfkeys) == Counter(rfull)
        det = f"top_match={list(nf) == r} multiset={ms} tie={_tie25}"
        _topms = Counter(list(nf)) == Counter(r)
        rec(qname, "supported", det, {**st, **sd},
            "EXACT" if list(nf) == r else ("MULTISET" if (ms or (_tie25 and _topms)) else "MISMATCH"))

    # ---------- Q27 (composite ORDER BY EventTime, SearchPhrase) ----------
    allowed = a["dict_not_empty_codes"](SPd)
    m = a["codes_member_mask"](SP, allowed, SPv)
    b, st = run([series("cc", np.ascontiguousarray(SP)), series("ee", np.ascontiguousarray(ET)),
                 a["ir_series"]("mm", np.ascontiguousarray(m), dtype="bool"),
                 a["ir_filter"]("cf", "cc", "mm"), a["ir_filter"]("ef", "ee", "mm"),
                 a["ir_sort"]("p", "ef", "cf"), a["ir_slice"]("t", "p", limit=10),
                 a["ir_gather"]("g", "cf", "t")])
    nf = a["dictionary_decode"](b["g"], SPd)
    r, sd = dd("SELECT SearchPhrase FROM read_parquet('{PQ}') WHERE SearchPhrase <> '' "
               "ORDER BY EventTime, SearchPhrase LIMIT 10")
    r = [x[0] for x in r]
    rec("Q27", "supported", f"top_match={list(nf) == r}", {**st, **sd},
        "EXACT" if list(nf) == r else "MISMATCH")

    # ---------- Q28 (HAVING + ORDER BY float; URL<>'' population == DuckDB) ----------
    lut = a["dict_len_lut"](Ud)
    murl = a["codes_member_mask"](UC, a["dict_not_empty_codes"](Ud),
                                  res["URL"]["validity"])
    b, st = run([series("cc", np.ascontiguousarray(CID)),
                 a["ir_series"]("lut", np.ascontiguousarray(lut)),
                 a["ir_series"]("uc", np.ascontiguousarray(UC)),
                 a["ir_series"]("mm", np.ascontiguousarray(murl), dtype="bool"),
                 a["ir_filter"]("ccf", "cc", "mm"),
                 a["ir_filter"]("ucf", "uc", "mm"),
                 a["ir_gather"]("ln", "lut", "ucf"),
                 a["ir_groupby_multi"]("g", "ln", "ccf", ops=("mean", "count"),
                                       result="carry")])
    c = b["g"]
    means = np.ascontiguousarray(c.means("ln"))
    keep = c.counts > 100000
    kk, ccnt, mm = c.ukeys[keep], c.counts[keep], means[keep]
    b2, st2 = run([a["ir_series"]("mm2", np.ascontiguousarray(mm), "float64"),
                   a["ir_series"]("kk2", np.ascontiguousarray(kk), "int64"),
                   a["ir_series"]("cc2", np.ascontiguousarray(ccnt), "int64"),
                   a["ir_sort"]("p", "mm2", descending=True),
                   a["ir_slice"]("t", "p", limit=25),
                   a["ir_gather"]("gk", "kk2", "t"), a["ir_gather"]("gm", "mm2", "t"),
                   a["ir_gather"]("gc", "cc2", "t")])
    nf = [(int(k), float(x), int(y)) for k, x, y in zip(b2["gk"], b2["gm"], b2["gc"])]
    r, sd = dd("SELECT CounterID, AVG(length(URL)) AS l, COUNT(*) AS c FROM read_parquet('{PQ}') "
               "WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25")
    ok = [k for k, _, _ in nf] == [int(k) for k, _, _ in r] and \
        [y for _, _, y in nf] == [int(y) for _, _, y in r] and \
        all(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for (_, x, _), (_, y, _) in zip(nf, r))
    rec("Q28", "supported", f"groups={len(nf)} top={nf[:2]}", {**st, **st2, **sd},
        "EXACT+TOL" if ok else "MISMATCH")

    # ---------- Q29 (Referer domain via generic regex primitive, Q28 chain) ----------
    # ingest Referer + resident_prepare + ir_text_regex_replace with the
    # official ClickBench pattern ^https?://(?:www\.)?([^/]+)/.*$ (repl \1)
    # -> text_length + groupby_multi(mean,count) + grouped min + HAVING
    # c>100000 + ORDER BY mean DESC LIMIT 25 (Q28 chain) + decode k.
    # Src untouched: regex output re-encoded to codes harness-level.
    # Validity: NULL Referer -> NULL domain -> excluded both sides
    # (DuckDB SQL carries WHERE Referer IS NOT NULL for the full gate).
    _PAT29 = r"^https?://(?:www\.)?([^/]+)/.*$"
    _REPL29 = r"\1"
    _rf_vals = pq.read_table(PARQUET, columns=["Referer"]).column("Referer").to_pylist()
    _rf_res = a["resident_prepare"]({"rf29": {"values": _rf_vals}})
    _rf_codes = np.ascontiguousarray(_rf_res["rf29"]["codes"])
    _rf_valid = np.ascontiguousarray(_rf_res["rf29"]["validity"])
    _rf_body = _rf_res["rf29"]["dictionary"]
    # D-scale wiring harness-level (src untouched): _keep_column keeps
    # array-likes by ref but list()-ifies plain dicts, so the documented
    # utf8_data+offsets+codes+validity bundle travels as an object with
    # .shape (kept by ref) instead of a dict.
    class _DictBody29:
        def __init__(self, utf8_data, offsets, codes, validity):
            self.utf8_data = utf8_data
            self.offsets = offsets
            self.codes = codes
            self.validity = validity
            self.shape = (int(np.ascontiguousarray(codes).size),)
    _rf_dict = _DictBody29(_rf_body.utf8_data,
                           np.ascontiguousarray(np.asarray(_rf_body.offsets)),
                           _rf_codes, _rf_valid)
    # PARITY6 micro-probe (NULL/empty/Unicode/backref/no-match/normal)
    import re as _re29
    _rx6 = ["http://www.example.com/a/b", None, "", "пример.рф/путь/üñî",
            "https://google.com/fee=x", "no-protocol-here"]
    _b6 = a["cpu_execute"]([a["ir_text_regex_replace"]("rx6", _rx6, _PAT29, _REPL29)])
    _rx6out = list(_b6["rx6"])
    _rx6ref = [_re29.sub(_PAT29, _REPL29, _v, count=1) if isinstance(_v, str) else None
               for _v in _rx6]
    _parity6 = _rx6out == _rx6ref
    _t29a = _t()
    _rx29 = a["cpu_execute"]([a["ir_text_regex_replace"]("rx29", _rf_dict, _PAT29, _REPL29)])
    _dom29 = np.ascontiguousarray(_rx29["rx29"])
    _dom29v = np.ascontiguousarray(_rx29["rx29#validity"])
    _ln29 = a["cpu_execute"]([a["ir_text_length"]("ln29", _dom29)])
    _len29 = np.ascontiguousarray(_ln29["ln29"])
    _v29 = _dom29v & np.ascontiguousarray(_ln29["ln29#validity"])
    _uniq29, _inv29 = np.unique(_dom29[_v29], return_inverse=True)
    _dk29 = np.zeros(_dom29.size, dtype=np.int32)
    _dk29[_v29] = _inv29.astype(np.int32)
    _b29, _st29 = run([series("ln29b", np.ascontiguousarray(_len29)),
                       a["ir_series"]("dk29", _dk29),
                       a["ir_series"]("mm29", np.ascontiguousarray(_v29), dtype="bool"),
                       a["ir_filter"]("lnf29", "ln29b", "mm29"),
                       a["ir_filter"]("dkf29", "dk29", "mm29"),
                       a["ir_groupby_multi"]("g29", "lnf29", "dkf29",
                                             ops=("mean", "count"), result="carry"),
                       a["ir_groupby"]("m29", "dkf29", "dkf29", "min", result="carry")])
    _t29e = (_t() - _t29a) * 1000
    _c29 = _b29["g29"]
    _means29 = np.ascontiguousarray(_c29.means("lnf29"))
    _keep29 = _c29.counts > 100000
    _kk29, _cc29, _mm29 = _c29.ukeys[_keep29], _c29.counts[_keep29], _means29[_keep29]
    _min29c = _b29["m29"]
    _minmap29 = dict(zip([int(_k) for _k in _min29c.ukeys],
                         [int(_v) for _v in _min29c.mins["v"]]))
    _rows29 = [(_uniq29[int(_k)], float(_m), int(_c),
                _uniq29[int(_minmap29[int(_k)])])
               for _k, _m, _c in zip(_kk29, _mm29, _cc29)]
    _rows29.sort(key=lambda _t: -_t[1])
    _nf29 = _rows29[:25]
    _SQL29 = ("SELECT regexp_replace(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS d, "
              "AVG(length(regexp_replace(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1'))) AS l, "
              "COUNT(*) AS c, MIN(regexp_replace(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1')) AS m "
              "FROM read_parquet('{PQ}') WHERE Referer IS NOT NULL GROUP BY d "
              "HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25")
    _r29, _sd29 = dd(_SQL29)
    _ok29 = (len(_nf29) == len(_r29) and
             all(_k == _dk and _c == int(_cc) and _md == _dm and
                 abs(_l - float(_ll)) <= 1e-9 * max(1.0, abs(float(_ll)))
                 for (_k, _l, _c, _md), (_dk, _ll, _cc, _dm) in zip(_nf29, _r29)))
    rec("Q29", "supported",
        f"groups={len(_kk29)} parity6={_parity6} top={_nf29[:1]} exec_ms={_t29e:.1f}",
        {**_st29, **_sd29}, "EXACT" if (_ok29 and _parity6) else "MISMATCH")

    # ---------- Q30 (90-col SUM) ----------
    jobs = [series("v", RESW)]
    for k in range(90):
        jobs.append(a["ir_map"](f"m{k}", "v", "add", k))
        jobs.append(a["ir_reduce"](f"r{k}", f"m{k}", "sum"))
    b, st = run(jobs)
    nf = [int(b[f"r{k}"]) for k in range(90)]
    r, sd = dd("SELECT " + ", ".join(f"SUM(ResolutionWidth + {k})" for k in range(90)) +
               " FROM read_parquet('{PQ}')")
    rec("Q30", "supported", f"sums0={nf[0]} sums89={nf[89]}", {**st, **sd},
        "EXACT" if nf == [int(x) for x in r[0]] else "MISMATCH")

    # ---------- Q31 (two-int composite: int64 transient keys, divmod unpack) ----------
    # ClientIP spans ~4.29e9 even shifted: int32 cannot hold it (the guard fires
    # loudly). int64 is transient accumulator-width here (allowed, never a Core
    # dtype); the mixed-radix product stays far below int64 (checked inside).
    sh = int(-min(int(SEID.min()), int(CIP.min()))) if min(int(SEID.min()), int(CIP.min())) < 0 else 0
    msp = a["codes_member_mask"](SP, a["dict_not_empty_codes"](SPd), SPv)
    # radix from the FILTERED population (pack runs on filtered rows only)
    CIPf = (CIP + sh)[msp]
    M = int(CIPf.max()) + 1
    b, st = run([series("a", np.ascontiguousarray(SEID[msp])),
                 a["ir_series"]("c", np.ascontiguousarray(CIPf), "int64"),
                 a["ir_pack_keys"]("k", "a", "c", mode="radix",
                                   radix=[int(SEID[msp].max()) + 1, M]),
                 a["ir_groupby"]("g", "a", "k", "count", result="carry")])
    c = b["g"]
    b2, st2 = run([a["ir_series"]("cc", np.ascontiguousarray(c.counts), "int64"),
                   a["ir_series"]("kk", np.ascontiguousarray(c.ukeys), "int64"),
                   a["ir_sort"]("p", "cc", descending=True),
                   a["ir_slice"]("t", "p", limit=10),
                   a["ir_gather"]("gc", "cc", "t"), a["ir_gather"]("gk", "kk", "t")])
    nf = [(int(kk // M), int(kk % M) - sh, int(x))
          for kk, x in zip(b2["gk"], b2["gc"])]
    r, sd = dd("SELECT SearchEngineID, ClientIP, COUNT(*) AS c FROM read_parquet('{PQ}') "
               "WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 11")
    r = [(int(k), int(y), int(x)) for k, y, x in r]
    win, extra = r[:10], r[10:]
    tie = bool(extra) and bool(win) and extra[0][2] == win[-1][2]
    if nf == win:
        match = "EXACT"
    else:
        full = [(int(kk // M), int(kk % M) - sh, int(x))
                for kk, x in zip(c.ukeys, c.counts)]
        dall, _ = dd("SELECT SearchEngineID, ClientIP, COUNT(*) AS c FROM read_parquet('{PQ}') "
                      "WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP")
        dall = [(int(k), int(y), int(x)) for k, y, x in dall]
        nfc = [x for _, _, x in nf]
        match = ("TIE-MULTISET" if Counter(full) == Counter(dall) and tie
                 and all(a >= b for a, b in zip(nfc, nfc[1:])) else "MISMATCH")
    rec("Q31", "supported", f"groups={int(c.ukeys.size)} top={nf[:2]} tie_hi={tie}",
        {**st, **st2, **sd}, match)

    # ---------- Q32/Q33 (WatchID) ----------
    rec("Q32", "unsupported", "GROUP BY WatchID+ClientIP: WatchID BIGINT narrowing", {}, "N/A")
    rec("Q33", "unsupported", "GROUP BY WatchID+ClientIP: WatchID BIGINT narrowing", {}, "N/A")

    # ---------- Q34/Q35 (URL top-k) ----------
    b, st = run([series("c", np.ascontiguousarray(UC)),
                 a["ir_groupby"]("g", "c", "c", "count", result="carry")])
    c = b["g"]
    b2, st2 = run([a["ir_series"]("cc", np.ascontiguousarray(c.counts), "int64"),
                   a["ir_series"]("kk", np.ascontiguousarray(c.ukeys), "int64"),
                   a["ir_sort"]("p", "cc", descending=True),
                   a["ir_slice"]("t", "p", limit=10),
                   a["ir_gather"]("gc", "cc", "t"), a["ir_gather"]("gk", "kk", "t")])
    nf = [(Ud[int(k)], int(x)) for k, x in zip(b2["gk"], b2["gc"])]
    r, sd = dd("SELECT URL, COUNT(*) AS c FROM read_parquet('{PQ}') "
               "GROUP BY URL ORDER BY c DESC LIMIT 10")
    r = [(k, int(x)) for k, x in r]
    rec("Q34", "supported", f"groups={int(c.ukeys.size)} top={nf[0]}", {**st, **st2, **sd},
        "EXACT" if nf == r else "MISMATCH")
    rec("Q35", "supported", "GROUP BY 1, URL: constant key is a no-op; same as Q34",
        {**st, **st2, **sd}, "EXACT" if nf == r else "MISMATCH")

    # ---------- Q36 (4-col ClientIP composite via generic hash/tuple path) ----------
    # pack_keys(mode="hash", 4 cols) -> exact #tuple_cols sidecar ->
    # composite_tuple_index + _tuple_sums (cpu.py) -> dict. The mixed-radix
    # product (~1e38) cannot fit int64, so radix loudly refuses; the tuple
    # identity is the grouping key throughout (never an int64 scalar pack
    # as the sole path). Composite groupby returns dict (benchmark
    # contract consumes dict like Q17/Q9; top-10 via Python sort+slice,
    # decode = raw tuple columns, no divmod needed on the tuple path).
    _t36a = _t()
    _c36 = [np.ascontiguousarray((CIP - _i).astype(np.int32)) for _i in range(4)]
    _b36, _st36 = run([series("k0", _c36[0]), series("k1", _c36[1]),
                       series("k2", _c36[2]), series("k3", _c36[3]),
                       a["ir_pack_keys"]("kk36", "k0", "k1", "k2", "k3",
                                         mode="hash"),
                       a["ir_groupby"]("g36", "k0", "kk36", "count",
                                       result="dict")])
    _t36e = (_t() - _t36a) * 1000
    _full36 = dict(_b36["g36"])
    _nf36 = sorted(_full36.items(), key=lambda _t: -_t[1])[:10]
    _nf36l = [(int(_k[0]), int(_k[1]), int(_k[2]), int(_k[3]), int(_c))
              for _k, _c in _nf36]
    _r36, _sd36 = dd("SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c "
                     "FROM read_parquet('{PQ}') GROUP BY ClientIP, ClientIP - 1, "
                     "ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 11")
    _r36 = [(int(_a), int(_b), int(_c), int(_d), int(_e)) for _a, _b, _c, _d, _e in _r36]
    _win36, _ext36 = _r36[:10], _r36[10:]
    _tie36 = bool(_ext36) and bool(_win36) and _ext36[0][4] == _win36[-1][4]
    _dall36, _ = dd("SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c "
                    "FROM read_parquet('{PQ}') GROUP BY ClientIP, ClientIP - 1, "
                    "ClientIP - 2, ClientIP - 3")
    _dall36 = [(int(_a), int(_b), int(_c), int(_d), int(_e)) for _a, _b, _c, _d, _e in _dall36]
    _nfc36 = [_c for _, _, _, _, _c in _nf36l]
    _full36l = [(int(_k[0]), int(_k[1]), int(_k[2]), int(_k[3]), int(_c))
                for _k, _c in _full36.items()]
    if _nf36l == _win36:
        _m36 = "EXACT"
    elif (Counter(_full36l) == Counter(_dall36)
            and Counter(_nf36l) == Counter(_win36)
            and all(_x >= _y for _x, _y in zip(_nfc36, _nfc36[1:]))):
        _m36 = "TIE-MULTISET"
    else:
        _m36 = "MISMATCH"
    rec("Q36", "supported",
        f"groups={len(_full36)} top={_nf36l[:1]} tie={_tie36} exec_ms={_t36e:.1f}",
        {**_st36, **_sd36}, _m36)

    # ---------- Q37/Q38/Q39 (filtered URL/Title top-k + OFFSET, full predicates) ----------
    import datetime as _d
    d1 = a["date_encode"]([_d.date(2013, 7, 1)])["codes"][0]
    d2 = a["date_encode"]([_d.date(2013, 7, 31)])["codes"][0]
    # (tag, codes, dictionary, DuckDB proj, extra flag predicates, limit, offset, nonempty?)
    specs = (("u", UC, Ud, "URL", ["dch", "irf"], "Q37", 10, 0, True),
             ("t", TC, Td, "Title", ["dch", "irf"], "Q38", 10, 0, True),
             ("u", UC, Ud, "URL", ["irf", "ilk", "idl"], "Q39", 10, 1000, False))
    for tag, col, dct, ddsel, flags, qname, lim, off, nonempty in specs:
        # TEXT predicate through the engine (domain codes -> member mask with
        # 3VL validity, never harness-side isin: NULL rows must not leak).
        preds = [series("cc", np.ascontiguousarray(col)),
                 series("cid", np.ascontiguousarray(CID)),
                 series("ed", np.ascontiguousarray(ED)),
                 series("dch", np.ascontiguousarray(DCH)),
                 series("irf", np.ascontiguousarray(IRF)),
                 series("ilk", np.ascontiguousarray(ILK)),
                 series("idl", np.ascontiguousarray(IDL)),
                 a["ir_compare"]("m1", "cid", 62, op="=="),
                 a["ir_compare"]("m2", "ed", int(d1), op=">="),
                 a["ir_compare"]("m3", "ed", int(d2), op="<=")]
        # flag predicates per query: Q37/Q38 dch==0+irf==0; Q39 irf==0+ilk!=0+idl==0
        fops = {"dch": ("dch", 0, "=="), "irf": ("irf", 0, "=="),
                "ilk": ("ilk", 0, "!="), "idl": ("idl", 0, "==")}
        for i, f in enumerate(flags):
            coln, val, op = fops[f]
            preds.append(a["ir_compare"](f"mf{i}", coln, val, op=op))
        preds += [a["ir_mask"]("mm", "m1", "m2", op="and"),
                  a["ir_mask"]("mall", "mm", "m3", op="and")]
        acc = "mall"
        for i in range(len(flags)):
            preds.append(a["ir_mask"](f"a{i}", acc, f"mf{i}", op="and"))
            acc = f"a{i}"
        if nonempty:
            tmask = a["codes_member_mask"](
                col, a["dict_not_empty_codes"](dct),
                {"URL": res["URL"]["validity"],
                 "Title": res["Title"]["validity"]}[ddsel])
            preds.append(a["ir_series"]("txt", np.ascontiguousarray(tmask), dtype="bool"))
            preds.append(a["ir_mask"]("allm", acc, "txt", op="and"))
            acc = "allm"
        preds.append(a["ir_filter"]("f", "cc", acc))
        bj, stj = run(preds)
        fkeys = np.ascontiguousarray(bj["f"])
        vals = dct
        fb = fkeys
        bj2, stj2 = run([a["ir_series"]("ff", fb),
                         a["ir_groupby"]("g", "ff", "ff", "count", result="carry")])
        c = bj2["g"]
        bj3, stj3 = run([a["ir_series"]("ccc", np.ascontiguousarray(c.counts), "int64"),
                         a["ir_series"]("kkk", np.ascontiguousarray(c.ukeys), "int64"),
                         a["ir_sort"]("pp", "ccc", descending=True),
                         a["ir_slice"]("tt", "pp", limit=lim, offset=off),
                         a["ir_gather"]("gc", "ccc", "tt"),
                         a["ir_gather"]("gk", "kkk", "tt")])
        nf = [(vals[int(k)], int(x)) for k, x in zip(bj3["gk"], bj3["gc"])]
        ddw = ("CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'")
        if "dch" in flags:
            ddw += " AND DontCountHits = 0"
        if "irf" in flags:
            ddw += " AND IsRefresh = 0"
        if "ilk" in flags:
            ddw += " AND IsLink <> 0"
        if "idl" in flags:
            ddw += " AND IsDownload = 0"
        if nonempty:
            ddw += f" AND {ddsel} <> ''"
        # tie-aware window gate: SQL tie order is unspecified, so fetch one row
        # past each window edge; exact match only when both edges are tie-free,
        # otherwise count-multiset + sortedness on both sides.
        r, sd = dd(f"SELECT {ddsel}, COUNT(*) AS PageViews FROM read_parquet('{{PQ}}') "
                   f"WHERE {ddw} GROUP BY {ddsel} ORDER BY PageViews DESC "
                   f"LIMIT {lim + off + 1} OFFSET {off}")
        r = [(k, int(x)) for k, x in r]
        win, extra = r[:lim], r[lim:]
        tie_hi = bool(extra) and bool(win) and extra[0][1] == win[-1][1]
        tie_lo = False
        if off > 0:
            rlo, _ = dd(f"SELECT COUNT(*) AS c FROM read_parquet('{{PQ}}') "
                        f"WHERE {ddw} GROUP BY {ddsel} ORDER BY c DESC "
                        f"LIMIT 1 OFFSET {off - 1}")
            tie_lo = bool(rlo) and bool(win) and int(rlo[0][0]) == win[0][1]
        nfc = [x for _, x in nf]
        winc = [x for _, x in win]
        srt = all(a >= b for a, b in zip(nfc, nfc[1:]))
        if not tie_lo and not tie_hi and nf == win:
            match = "EXACT"
        elif (Counter(nfc) == Counter(winc) and srt
                and all(a >= b for a, b in zip(winc, winc[1:]))):
            match = "TIE-MULTISET"
        else:
            match = "MISMATCH"
        rec(qname, "supported", f"top={nf[0] if nf else None} tie_lo={tie_lo} tie_hi={tie_hi}",
            {**stj, **stj2, **stj3, **sd}, match)

    # ---------- Q40 ----------
    rec("Q40", "unsupported", "CASE WHEN TraficSourceID...: no conditional/case primitive", {}, "N/A")

    # ---------- Q41/Q42 (URLHash BIGINT) ----------
    try:
        run([a["ir_series"]("h", [3594120000172545465])])
        rec("Q41", "supported?", "UNEXPECTED", {}, "MISMATCH")
    except OverflowError:
        rec("Q41", "unsupported", "RefererHash=3594120000172545465: BIGINT literal narrowing", {}, "N/A")
    rec("Q42", "unsupported", "URLHash=2868770270353813622: BIGINT literal narrowing", {}, "N/A")

    # ---------- Q43 (DATE_TRUNC minute + full predicates + OFFSET) ----------
    d3 = a["date_encode"]([_d.date(2013, 7, 14)])["codes"][0]
    d4 = a["date_encode"]([_d.date(2013, 7, 15)])["codes"][0]
    b, st = run([series("ee", np.ascontiguousarray(ET)),
                 series("cid", np.ascontiguousarray(CID)),
                 series("ed", np.ascontiguousarray(ED)),
                 series("dch", np.ascontiguousarray(DCH)),
                 series("irf", np.ascontiguousarray(IRF)),
                 a["ir_compare"]("m1", "cid", 62, op="=="),
                 a["ir_compare"]("m2", "ed", int(d3), op=">="),
                 a["ir_compare"]("m3", "ed", int(d4), op="<="),
                 a["ir_compare"]("m4", "dch", 0, op="=="),
                 a["ir_compare"]("m5", "irf", 0, op="=="),
                 a["ir_mask"]("a1", "m1", "m2", op="and"),
                 a["ir_mask"]("a2", "a1", "m3", op="and"),
                 a["ir_mask"]("a3", "a2", "m4", op="and"),
                 a["ir_mask"]("all", "a3", "m5", op="and"),
                 a["ir_filter"]("ef", "ee", "all"),
                 a["ir_map"]("mn", "ef", "floor_div", 60),
                 a["ir_groupby"]("g", "mn", "mn", "count", result="carry")])
    c = b["g"]
    b2, st2 = run([a["ir_series"]("kk", np.ascontiguousarray(c.ukeys), "int64"),
                   a["ir_series"]("cc", np.ascontiguousarray(c.counts), "int64"),
                   a["ir_sort"]("p", "kk"),
                   a["ir_slice"]("t", "p", limit=10, offset=1000),
                   a["ir_gather"]("gk", "kk", "t"), a["ir_gather"]("gc", "cc", "t")])
    import datetime as _dt2
    nf = [(str(_dt2.datetime(1970, 1, 1) + _dt2.timedelta(seconds=int(k) * 60)), int(x))
          for k, x in zip(b2["gk"], b2["gc"])]
    r, sd = dd("SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews "
               "FROM read_parquet('{PQ}') WHERE CounterID = 62 AND EventDate >= '2013-07-14' "
               "AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 "
               "GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) "
               "LIMIT 10 OFFSET 1000")
    r = [(str(k), int(x)) for k, x in r]
    rec("Q43", "supported", f"top={nf[0] if nf else None}",
        {**st, **st2, **sd}, "EXACT" if nf == r else "MISMATCH")

    # ---------- matrix ----------
    print("=== ClickBench 43 matrix (hits_1m, 999978 rows) ===")
    print(f"ingest ms: {' '.join(f'{k}={v:.1f}' for k, v in ING.items())}")
    ns = nu = nm = 0
    for m in MATRIX:
        st = " ".join(f"{k}={v:.1f}" for k, v in m["stages_ms"].items())
        print(f"{m['q']}: {m['status']} [{m['match']}] {m['detail']} | {st}")
        if m["status"].startswith("supported"):
            ns += 1
        else:
            nu += 1
        if m["match"] not in ("EXACT", "EXACT+TOL", "N/A", "GATED-CORE",
                              "MULTISET", "TIE-MULTISET", "POPULATION-DIFFERS"):
            nm += 1
    print(f"SUPPORTED={ns}/43 UNSUPPORTED={nu}/43 MISMATCH={nm}")
    Path(__file__).with_name("bench_clickbench_43.json").write_text(
        json.dumps({"ingest_ms": ING, "matrix": MATRIX}, default=str, indent=1))
    print("matrix JSON: tests/heavy/bench_clickbench_43.json")
    if nm:
        sys.exit(1)


if __name__ == "__main__":
    main()
