"""Backward-anchor features for TWS forecasting (Zindi drought competition).

History = every observed (cell, month, TWS) — train rows + visible test rows,
deduped to one row per (cell, month). Sorted by (cell, month) once, then all
anchor lookups are GPU binary searches (SearchSorted kernel).

Public API:
    cell_ids(lat, lon)                      -> int64 cell ids (1D)
    build_history(cell_id, month_idx, tws, dedupe_months)
                                            -> (months, vals, cell_start, cell_len)
    searchsorted(q_cell, q_month, hist, mode) -> (val, found, month_out) GPU
"""

import numpy as np

from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime import Runtime

from ZindiGaps._lib.searchsorted import describe, cpu, WGSL

_CAPABILITIES = {"streaming": False, "workspace": False, "multi_input": True,
                 "multi_output": True}
_registered = False
_rt = None


def _runtime():
    global _rt, _registered
    if _rt is None:
        _rt = Runtime(driver=WebGpuDriver())
    if not _registered:
        if len(_rt.kernel_table) == 0:
            _rt.register_kernel("SearchSorted", describe=describe, cpu=cpu,
                                wgsl=WGSL, abi_version=1,
                                capabilities=_CAPABILITIES)
        _registered = True
    return _rt


def cell_ids(lat, lon):
    """Encode (lat, lon) grid to unique int64 cell id.

    Grid step 0.5 deg: lat in [-90, 90], lon in [-180, 180].
    id = (lat+90)*2*721 + (lon+180)*2
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return ((lat + 90.0) * 2.0 * 721.0 + (lon + 180.0) * 2.0).astype(np.int64)


def build_history(cell_id, month_idx, tws, dedupe_months=False):
    """Build sorted (cell, month) index from observed rows.

    Args:
        cell_id: 1D int64 array, cell id per row
        month_idx: 1D int array, global month index per row
        tws: 1D float array, observed TWS per row
        dedupe_months: if True, keep one row per (cell, month) — use when
            source has duplicate bands per month (test file)

    Returns:
        (months, vals, cell_start, cell_len) — float64 numpy arrays
    """
    cells = np.asarray(cell_id, dtype=np.int64)
    months0 = np.asarray(month_idx, dtype=np.int64)
    vals0 = np.asarray(tws, dtype=np.float64)
    idx = np.arange(len(cells))
    if dedupe_months:
        key = cells * 100000 + months0
        _, first = np.unique(key, return_index=True)
        idx = first
    cells = cells[idx]
    months = months0[idx]
    vals = vals0[idx]
    order = np.lexsort((months, cells))
    cells = cells[order]
    months = months[order].astype(np.float64)
    vals = vals[order].astype(np.float64)
    unique_cells, counts = np.unique(cells, return_counts=True)
    cell_start = np.zeros(unique_cells.max() + 1, dtype=np.float64)
    cell_len = np.zeros(unique_cells.max() + 1, dtype=np.float64)
    cell_start[unique_cells] = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.float64)
    cell_len[unique_cells] = counts.astype(np.float64)
    return months, vals, cell_start, cell_len


def searchsorted(q_cell, q_month, hist, mode=0):
    """GPU binary search over the sorted history index.

    Args:
        q_cell: 1D int array, cell id per query
        q_month: 1D int array, month index per query
        hist: tuple (months, vals, cell_start, cell_len) from build_history
        mode: 0 = exact month, 1 = rightmost month <= query

    Returns:
        (val, found, month_out): float64 arrays, len == len(q_cell)
    """
    months, vals, cell_start, cell_len = hist
    q_cell = np.asarray(q_cell, dtype=np.float64)
    q_month = np.asarray(q_month, dtype=np.float64)
    rt = _runtime()
    jobs = [{
        "op": "SearchSorted",
        "inputs": ["q_cell", "q_month", "months", "vals", "cell_start", "cell_len"],
        "params": {"mode": float(mode)},
        "out": ["val", "found", "month_out"],
    }]
    data = {"q_cell": q_cell, "q_month": q_month,
            "months": months, "vals": vals,
            "cell_start": cell_start, "cell_len": cell_len}
    tasks = rt.compile(jobs)
    rt.execute(tasks, data)
    return (rt.driver.resolve_output("val"),
            rt.driver.resolve_output("found"),
            rt.driver.resolve_output("month_out"))


def anchors(lat, lon, month_idx, tws_obs=None, dedupe=False):
    """Convenience: build history index from an observed-frame.

    Args:
        lat, lon: observation frame cell coordinates
        month_idx: global month index per row
        tws_obs: observed TWS per row (None -> placeholder ones)
        dedupe: collapse duplicate (cell, month) rows

    Returns:
        (months, vals, cell_start, cell_len) — ready for searchsorted calls
    """
    if tws_obs is None:
        tws_obs = np.ones(len(lat))
    return build_history(cell_ids(lat, lon), month_idx, tws_obs,
                         dedupe_months=dedupe)


__all__ = ["cell_ids", "build_history", "searchsorted", "anchors"]