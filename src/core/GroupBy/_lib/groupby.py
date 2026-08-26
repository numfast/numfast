"""GroupBy via Sort→Compare(shift)→Scan→Reduce segmented, mean via MapBinary div. N=0→[], NaN last."""
import numpy as np

SCAN_LIMIT = 4194240
SORT_LIMIT = SCAN_LIMIT
GROUPBY_LIMIT = SCAN_LIMIT

def _as_keys(keys):
    """Normalize keys to int32 or float32 1D."""
    if keys is None:
        return np.array([], dtype=np.int32)
    try:
        arr = np.asarray(keys)
    except Exception:
        return np.array([], dtype=np.int32)
    if arr.size == 0:
        return np.array([], dtype=np.int32)
    arr = arr.ravel()
    if arr.dtype == np.int32 or arr.dtype == np.int64:
        return arr.astype(np.int32, copy=False)
    if arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    if arr.dtype == np.float32:
        return arr
    try:
        return arr.astype(np.int32)
    except Exception:
        return arr.astype(np.float32, copy=False)

def _as_vals(vals):
    """Normalize vals to float32 1D."""
    if vals is None:
        return np.array([], dtype=np.float32)
    try:
        arr = np.asarray(vals, dtype=np.float32).ravel()
    except Exception:
        return np.array([], dtype=np.float32)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr

def _sort_stable(keys_arr):
    """Sort primitive: stable argsort, NaN group last preserves order."""
    n = keys_arr.size
    if n == 0:
        return np.array([], dtype=np.int64), keys_arr.copy()
    if keys_arr.dtype == np.float32 or keys_arr.dtype == np.float64:
        nan_mask = np.isnan(keys_arr.astype(np.float64))
        if np.any(nan_mask):
            non_nan_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            non_nan_keys = keys_arr[non_nan_idx]
            order_nn = np.argsort(non_nan_keys, kind='stable')
            sorted_non_nan = non_nan_idx[order_nn]
            perm = np.concatenate([sorted_non_nan, nan_idx])
            return perm.astype(np.int64), keys_arr[perm]
    perm = np.argsort(keys_arr, kind='stable').astype(np.int64)
    return perm, keys_arr[perm]

def _compare_shift_ne(sorted_keys):
    """Compare primitive after Shift: boundary[i]= (keys[i]!=keys[i-1])."""
    n = sorted_keys.size
    if n == 0:
        return np.array([], dtype=np.uint32)
    boundary = np.empty(n, dtype=np.uint32)
    boundary[0] = 1
    if n > 1:
        if sorted_keys.dtype == np.float32 or sorted_keys.dtype == np.float64:
            a = sorted_keys[1:]
            b = sorted_keys[:-1]
            # NaN == NaN for grouping: all NaNs in one group
            eq = (a == b) | (np.isnan(a) & np.isnan(b))
            boundary[1:] = (~eq).astype(np.uint32)
        else:
            boundary[1:] = (sorted_keys[1:] != sorted_keys[:-1]).astype(np.uint32)
    return boundary

def _scan_offsets(boundary):
    n = boundary.size
    if n == 0: return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    cumsum = np.cumsum(boundary.astype(np.int64))
    group_ids = cumsum - 1
    offsets = np.where(boundary == 1)[0].astype(np.int64)
    return group_ids, offsets

def groupby_array(keys, vals, chunk_limit: int = GROUPBY_LIMIT):
    """Core GroupBy: segmented Reduce + MapBinary div for mean. Returns {keys,count,sum,min,max,mean}."""
    keys_arr = _as_keys(keys)
    vals_arr = _as_vals(vals)
    n = keys_arr.size
    m_vals = vals_arr.size
    if n != m_vals:
        if m_vals == 0 and n > 0:
            vals_arr = np.zeros(n, dtype=np.float32)
        else:
            n = min(n, m_vals)
            keys_arr = keys_arr[:n]
            vals_arr = vals_arr[:n]
    if n == 0:
        ef = np.array([], dtype=np.float32)
        ei = np.array([], dtype=np.int32)
        ec = np.array([], dtype=np.int64)
        return {"keys": ei, "count": ec, "sum": ef, "min": ef, "max": ef, "mean": ef}
    # Stage 1: Sort keys stable
    perm, sorted_keys = _sort_stable(keys_arr)
    sorted_vals = vals_arr[perm]
    # Stage 2: Compare shift !=
    boundary = _compare_shift_ne(sorted_keys)
    # Stage 3: Scan offsets
    group_ids, offsets = _scan_offsets(boundary)
    num_groups = int(offsets.size)
    # Stage 4: per-segment Reduce count/sum/min/max
    counts = np.empty(num_groups, dtype=np.int64)
    sums = np.empty(num_groups, dtype=np.float32)
    mins = np.empty(num_groups, dtype=np.float32)
    maxs = np.empty(num_groups, dtype=np.float32)
    uniq_keys = sorted_keys[offsets]
    if uniq_keys.dtype == np.int64:
        uniq_keys = uniq_keys.astype(np.int32, copy=False)
    for gi in range(num_groups):
        start = int(offsets[gi])
        end = int(offsets[gi+1]) if gi+1 < num_groups else n
        seg = sorted_vals[start:end]
        cnt = end - start
        counts[gi] = cnt
        # Reduce sum (accumulate in float64 then cast)
        s = float(np.sum(seg.astype(np.float64)))
        sums[gi] = np.float32(s)
        mins[gi] = np.float32(np.min(seg)) if cnt > 0 else np.float32(np.nan)
        maxs[gi] = np.float32(np.max(seg)) if cnt > 0 else np.float32(np.nan)
    # Stage 5: mean via MapBinary div (sum / count)
    means = np.empty(num_groups, dtype=np.float32)
    for i in range(num_groups):
        # MapBinary op=div with scalar count
        means[i] = np.float32(float(sums[i]) / float(counts[i])) if counts[i] != 0 else np.float32(np.nan)
    # Preserve original key dtype intent
    if keys_arr.dtype == np.int32:
        uniq_keys = uniq_keys.astype(np.int32, copy=False)
    elif uniq_keys.dtype == np.float64:
        uniq_keys = uniq_keys.astype(np.float32, copy=False)
    return {"keys": uniq_keys, "count": counts, "sum": sums, "min": mins, "max": maxs, "mean": means}

def groupby(keys, vals, chunk_limit: int = GROUPBY_LIMIT):
    """Public alias."""
    return groupby_array(keys, vals, chunk_limit=chunk_limit)

# Capital alias for builder
GroupBy = groupby_array
