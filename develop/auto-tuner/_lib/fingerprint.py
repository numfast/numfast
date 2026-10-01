# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Workload fingerprint: pure function of observables (no dataset names)."""
import numpy as np


def fingerprint(keys, values, thresh, seed, split, n):
    keys = np.asarray(keys)
    values = np.asarray(values)
    mask = values > thresh
    sel = float(mask.mean()) if len(mask) else 0.0
    fkeys = keys[mask] if mask.any() else keys[:0]
    nunique = int(np.unique(fkeys).size) if fkeys.size else 0
    if fkeys.size:
        span = int(fkeys.max()) - int(fkeys.min()) + 1
        is_sorted = bool(np.all(fkeys[:-1] <= fkeys[1:]))
    else:
        span = 0
        is_sorted = True
    return {
        "n": int(n),
        "seed": int(seed),
        "split": str(split),
        "thresh": int(thresh),
        "selectivity": round(sel, 4),
        "nunique_filtered": int(nunique),
        "span_filtered": int(span),
        "is_sorted": bool(is_sorted),
        "keys_dtype": str(keys.dtype),
        "values_dtype": str(values.dtype),
        "query": "filter(values>thresh) -> groupby(keys) sum(values)",
    }
