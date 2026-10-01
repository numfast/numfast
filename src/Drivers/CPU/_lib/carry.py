# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""ColumnCarry: primary internal groupby result (spec DELTA result-path).

ukeys (int64 sorted) + counts (int64) + sums per column (int64/float64).
Mean is always derived sum/count (vectorized, cached) — never a second
pass, never a per-key Python stage. Python dict is explicit compatibility
materialization only (to_dict_*), never built on the hot path unless the
caller asks for result='dict'.

STANDALONE module: numpy stdlib only, no Builder/Extension imports
(NO INTERNAL IMPORTS rule). Materializers are the proven _dict_* shapes
(same output, same Python scalar types); threads>1 chunks the key range
for compat builds where the caller needs dicts at scale.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor


class ColumnCarry:
    """Columnar groupby result: ukeys/counts/sums, mean derived."""

    __slots__ = ("ukeys", "counts", "sums", "_means", "mins", "maxs")

    def __init__(self, ukeys, counts, sums, mins=None, maxs=None):
        self.ukeys = np.ascontiguousarray(np.asarray(ukeys, dtype=np.int64))
        self.counts = np.ascontiguousarray(np.asarray(counts, dtype=np.int64))
        self.sums = {c: np.ascontiguousarray(np.asarray(s)) for c, s in sums.items()}
        self._means = {}
        self.mins = ({c: np.ascontiguousarray(np.asarray(s))
                      for c, s in (mins or {}).items()})
        self.maxs = ({c: np.ascontiguousarray(np.asarray(s))
                      for c, s in (maxs or {}).items()})

    @property
    def ngroups(self):
        return int(self.ukeys.size)

    def means(self, col):
        """Derived mean sums/counts (vectorized f64, cached per column)."""
        m = self._means.get(col)
        if m is None:
            m = np.ascontiguousarray(
                self.sums[col].astype(np.float64) / self.counts)
            self._means[col] = m
        return m

    # ---- explicit compatibility materialization (dict shapes) ----

    def to_dict_flat(self, op, threads=1):
        """{key: scalar} for op in sum/count/mean/min/max (groupby legacy shape)."""
        if op == "count":
            return _dict_from_cols(self.ukeys, self.counts, None, None,
                                   "count", threads)
        if op == "mean":
            return _dict_from_cols(self.ukeys, self.counts, None,
                                   self.means(_only_col(self.sums)),
                                   "mean", threads)
        if op == "min":
            col = _only_col(self.mins)
            return _dict_from_cols(self.ukeys, self.counts, self.mins[col],
                                   None, "sum", threads)
        if op == "max":
            col = _only_col(self.maxs)
            return _dict_from_cols(self.ukeys, self.counts, self.maxs[col],
                                   None, "sum", threads)
        col = _only_col(self.sums)
        return _dict_from_cols(self.ukeys, self.counts, self.sums[col],
                               None, "sum", threads)

    def to_dict_single(self, col, ops, threads=1):
        """{key: {op: val}} (groupby_multi single-column legacy shape)."""
        mins = self.mins.get(col) if "min" in ops else None
        maxs = self.maxs.get(col) if "max" in ops else None
        if mins is None and maxs is None:
            return _dict_single_from(self.ukeys, self.counts,
                                     self.sums[col],
                                     self.means(col) if "mean" in ops else None,
                                     ops, threads)
        return _dict_single_minmax(self.ukeys, self.counts,
                                   self.sums.get(col),
                                   self.means(col) if "mean" in ops else None,
                                   mins, maxs, ops, threads)

    def to_dict_multi(self, cols, ops_map, threads=1):
        """{key: {col: {op: val}}} (groupby_multi legacy shape)."""
        if not any("min" in ops_map[c] or "max" in ops_map[c] for c in cols):
            sums_list = [self.sums[c] for c in cols]
            means_list = [self.means(c) if "mean" in ops_map[c] else None
                          for c in cols]
            return _dict_multi_from(self.ukeys, self.counts, cols, sums_list,
                                    means_list, ops_map, threads)
        return _dict_multi_minmax(self.ukeys, self.counts, cols, ops_map,
                                  self.sums, self._means_lazy(cols, ops_map),
                                  self.mins, self.maxs, threads)

    def _means_lazy(self, cols, ops_map):
        return [self.means(c) if "mean" in ops_map[c] else None
                for c in cols]


def _only_col(sums):
    if len(sums) != 1:
        raise ValueError(
            "ColumnCarry.to_dict_flat needs single-column carry, "
            f"got {sorted(sums)}")
    return next(iter(sums))


def _chunks(n, threads):
    threads = max(1, min(int(threads), n or 1))
    bounds = np.linspace(0, n, threads + 1).astype(np.int64)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(threads)
            if int(bounds[i + 1]) > int(bounds[i])]


def _dict_from_cols(ukeys, counts, sums, means, op, threads):
    """Flat {key: scalar}: tolist slices + zip (proven shape)."""
    kl = ukeys.tolist()
    if threads == 1:
        if op == "count":
            cl = counts.tolist()
            return {k: c for k, c in zip(kl, cl)}
        if op == "mean":
            ml = list(means.tolist())
            return {k: float(m) for k, m in zip(kl, ml)}
        sl = sums.tolist()
        return {k: s for k, s in zip(kl, sl)}
    rng = _chunks(len(kl), threads)
    with ThreadPoolExecutor(max_workers=len(rng)) as ex:
        if op == "count":
            parts = list(ex.map(
                lambda b: {k: c for k, c in
                           zip(kl[b[0]:b[1]], counts[b[0]:b[1]].tolist())},
                rng))
        elif op == "mean":
            parts = list(ex.map(
                lambda b: {k: float(m) for k, m in
                           zip(kl[b[0]:b[1]], means[b[0]:b[1]].tolist())},
                rng))
        else:
            parts = list(ex.map(
                lambda b: {k: s for k, s in
                           zip(kl[b[0]:b[1]], sums[b[0]:b[1]].tolist())},
                rng))
    res = {}
    for p in parts:
        res.update(p)
    return res


def _dict_single_from(ukeys, counts, sums, means, ops, threads):
    """Single-column {key: {op: val}} (proven shape)."""
    kl = ukeys.tolist()
    want_sum, want_count, want_mean = ("sum" in ops, "count" in ops,
                                      "mean" in ops)
    if threads == 1:
        sl = sums.tolist() if want_sum else None
        cl = counts.tolist() if want_count else None
        ml = list(means.tolist()) if want_mean else None
        n = len(kl)
        if want_sum and want_count and want_mean:
            return {kl[i]: {"sum": sl[i], "count": cl[i], "mean": ml[i]}
                    for i in range(n)}
        if want_sum and want_count:
            return {kl[i]: {"sum": sl[i], "count": cl[i]} for i in range(n)}
        if want_sum and want_mean:
            return {kl[i]: {"sum": sl[i], "mean": ml[i]} for i in range(n)}
        if want_count and want_mean:
            return {kl[i]: {"count": cl[i], "mean": ml[i]} for i in range(n)}
        if want_sum:
            return {kl[i]: {"sum": sl[i]} for i in range(n)}
        if want_count:
            return {kl[i]: {"count": cl[i]} for i in range(n)}
        return {kl[i]: {"mean": ml[i]} for i in range(n)}
    rng = _chunks(len(kl), threads)

    def _one(b):
        a, e = b
        sl = sums[a:e].tolist() if want_sum else None
        cl = counts[a:e].tolist() if want_count else None
        ml = means[a:e].tolist() if want_mean else None
        kk = kl[a:e]
        out = {}
        for i, k in enumerate(kk):
            sub = {}
            if want_sum:
                sub["sum"] = sl[i]
            if want_count:
                sub["count"] = cl[i]
            if want_mean:
                sub["mean"] = ml[i]
            out[k] = sub
        return out

    with ThreadPoolExecutor(max_workers=len(rng)) as ex:
        parts = list(ex.map(_one, rng))
    res = {}
    for p in parts:
        res.update(p)
    return res


def _dict_multi_from(ukeys, counts, cols, sums_list, means_list, ops_map,
                     threads):
    """Multi-column {key: {col: {op: val}}}: columns tolist once, one pass."""
    kl = ukeys.tolist()
    if threads == 1:
        cl = counts.tolist()
        col_data = []
        for c, sums, means in zip(cols, sums_list, means_list):
            ops = ops_map[c]
            sl = sums.tolist() if "sum" in ops else None
            ml = list(means.tolist()) if means is not None else None
            col_data.append((c, sl, ml, ops))
        n = len(kl)
        res = {}
        for i in range(n):
            cell = {}
            for c, sl, ml, ops in col_data:
                sub = {}
                if "sum" in ops:
                    sub["sum"] = sl[i]
                if "count" in ops:
                    sub["count"] = cl[i]
                if "mean" in ops:
                    sub["mean"] = ml[i]
                cell[c] = sub
            res[kl[i]] = cell
        return res
    rng = _chunks(len(kl), threads)

    def _one(b):
        a, e = b
        cl = counts[a:e].tolist()
        col_data = []
        for c, sums, means in zip(cols, sums_list, means_list):
            ops = ops_map[c]
            sl = sums[a:e].tolist() if "sum" in ops else None
            ml = means[a:e].tolist() if means is not None else None
            col_data.append((c, sl, ml, ops))
        kk = kl[a:e]
        out = {}
        for i, k in enumerate(kk):
            cell = {}
            for c, sl, ml, ops in col_data:
                sub = {}
                if "sum" in ops:
                    sub["sum"] = sl[i]
                if "count" in ops:
                    sub["count"] = cl[i]
                if "mean" in ops:
                    sub["mean"] = ml[i]
                cell[c] = sub
            out[k] = cell
        return out

    with ThreadPoolExecutor(max_workers=len(rng)) as ex:
        parts = list(ex.map(_one, rng))
    res = {}
    for p in parts:
        res.update(p)
    return res


def _dict_single_minmax(ukeys, counts, sums, means, mins, maxs, ops,
                        threads):
    """Single-column {key: {op: val}} with min/max lanes (proven shape)."""
    kl = ukeys.tolist()
    sl = sums.tolist() if sums is not None and "sum" in ops else None
    cl = counts.tolist() if "count" in ops else None
    ml = list(means.tolist()) if means is not None else None
    nl = mins.tolist() if mins is not None else None
    xl = maxs.tolist() if maxs is not None else None
    res = {}
    for i in range(len(kl)):
        sub = {}
        if "sum" in ops:
            sub["sum"] = sl[i]
        if "count" in ops:
            sub["count"] = cl[i]
        if "mean" in ops:
            sub["mean"] = ml[i]
        if "min" in ops:
            sub["min"] = nl[i]
        if "max" in ops:
            sub["max"] = xl[i]
        res[kl[i]] = sub
    return res


def _dict_multi_minmax(ukeys, counts, cols, ops_map, sums_d, means_list,
                       mins_d, maxs_d, threads):
    """Multi-column {key: {col: {op: val}}} with min/max lanes."""
    kl = ukeys.tolist()
    cl = counts.tolist()
    col_data = []
    for c, means in zip(cols, means_list):
        ops = ops_map[c]
        sl = sums_d[c].tolist() if c in sums_d and "sum" in ops else None
        ml = list(means.tolist()) if means is not None else None
        nl = mins_d[c].tolist() if c in mins_d and "min" in ops else None
        xl = maxs_d[c].tolist() if c in maxs_d and "max" in ops else None
        col_data.append((c, sl, ml, nl, xl, ops))
    res = {}
    for i in range(len(kl)):
        cell = {}
        for c, sl, ml, nl, xl, ops in col_data:
            sub = {}
            if "sum" in ops:
                sub["sum"] = sl[i]
            if "count" in ops:
                sub["count"] = cl[i]
            if "mean" in ops:
                sub["mean"] = ml[i]
            if "min" in ops:
                sub["min"] = nl[i]
            if "max" in ops:
                sub["max"] = xl[i]
            cell[c] = sub
        res[kl[i]] = cell
    return res
