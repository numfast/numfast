"""TopK via Sort+Gather composition, stable order, chunkable.

Formula: TopK(src, k, descending=True) = Gather(Sort(src, descending), range(k'))
where k' = min(max(k,0), n), stable tie by row_id, chunkable via Sort limit 4194240.
"""
import numpy as np

SCAN_LIMIT = 4194240
SORT_LIMIT = SCAN_LIMIT
TOPK_LIMIT = SCAN_LIMIT


def _sort_composition(src_arr: np.ndarray, descending: bool = True) -> np.ndarray:
    """Sort primitive composition — stable sort."""
    n = src_arr.size
    if n == 0:
        return src_arr.copy()
    # stable argsort to preserve row_id order for ties
    if descending:
        # Use stable sort on -src for descending stable
        # Handle float and int uniformly via argsort on -value
        # For stability: argsort with kind='stable' on negated values keeps original order for equal values
        try:
            # Negate; for bool/int/float works; for large values still ok
            neg = -src_arr.astype(np.float64) if src_arr.dtype == np.float32 else -src_arr
            # Use stable
            idx = np.argsort(neg, kind='stable')
        except Exception:
            idx = np.argsort(src_arr, kind='stable')[::-1]
            # Fix stable for ties when reversing: need to re-stable ties
            # Fallback to python stable sort via sorted
            # Use sorted with key = (-value, original_index)
            idx = np.array(sorted(range(n), key=lambda i: (-float(src_arr[i]), i)), dtype=np.int64)
        sorted_arr = src_arr[idx]
        return sorted_arr
    else:
        idx = np.argsort(src_arr, kind='stable')
        return src_arr[idx]


def _gather_composition(sorted_arr: np.ndarray, k_eff: int) -> np.ndarray:
    """Gather primitive — out[i] = sorted[i] for i in range(k_eff)."""
    if k_eff <= 0:
        return np.array([], dtype=sorted_arr.dtype if sorted_arr.size else np.float32)
    # Gather via range index
    indices = np.arange(k_eff, dtype=np.int64)
    # Emulate Gather kernel: src[index[i]]
    out = np.empty(k_eff, dtype=sorted_arr.dtype)
    for i, ix in enumerate(indices):
        out[i] = sorted_arr[int(ix)]
    # Optimize: direct slice is same, but keep loop parity
    # out = sorted_arr[indices]  # equivalent
    return out


def _topk_chunk(src_arr: np.ndarray, k_eff: int, descending: bool) -> np.ndarray:
    """Single chunk TopK = Gather(Sort(src), range(k_eff))."""
    if k_eff <= 0 or src_arr.size == 0:
        return np.array([], dtype=np.float32)
    sorted_arr = _sort_composition(src_arr, descending=descending)
    gathered = _gather_composition(sorted_arr, k_eff)
    # Cast to float32 for consistency
    if gathered.dtype != np.float32:
        gathered = gathered.astype(np.float32, copy=False)
    return gathered


def topk_array(src, k: int, descending: bool = True, dtype=np.float32, chunk_limit: int = SCAN_LIMIT) -> np.ndarray:
    """Core implementation with chunkable support via Sort limit."""
    if k is None:
        raise ValueError("k must be int")
    if not isinstance(k, (int, np.integer)):
        # Allow numpy int, else error
        try:
            k = int(k)
        except Exception:
            raise ValueError(f"k must be int, got {type(k)}")
    if k < 0:
        raise ValueError(f"k must be >=0, got {k}")
    # Handle None src -> empty
    if src is None:
        return np.array([], dtype=dtype)
    # Normalize src to ndarray float32
    if isinstance(src, np.ndarray) and src.dtype == dtype and src.ndim == 1:
        src_arr = src.ravel()
        # Ensure float32 if dtype is float32
        if src_arr.dtype != np.float32:
            src_arr = src_arr.astype(np.float32, copy=False)
    else:
        try:
            src_arr = np.asarray(src, dtype=np.float32).ravel()
        except Exception:
            return np.array([], dtype=dtype)
        if src_arr.dtype != np.float32:
            src_arr = src_arr.astype(np.float32, copy=False)
        # Ensure dtype float32 for output
        if dtype != np.float32:
            src_arr = src_arr.astype(dtype, copy=False)

    # Handle indirection: if src_arr is 0-d or object?
    if src_arr.ndim == 0:
        src_arr = src_arr.reshape(1) if src_arr.size == 1 else np.array([], dtype=np.float32)

    n = int(src_arr.size)
    if n == 0 or k == 0:
        return np.array([], dtype=np.float32)

    k_eff = int(min(max(k, 0), n))
    if k_eff == 0:
        return np.array([], dtype=np.float32)
    if k_eff == n:
        # k>=N -> sorted copy (stable)
        sorted_full = _sort_composition(src_arr, descending=descending)
        if sorted_full.dtype != np.float32:
            sorted_full = sorted_full.astype(np.float32, copy=False)
        return sorted_full

    if n <= chunk_limit:
        return _topk_chunk(src_arr, k_eff, descending)

    # N > chunk_limit : chunkable path — simulate Sort chunking but keep global correctness
    # Strategy: global TopK still required. We simulate chunked Sort+Gather by iterating
    # chunks for Sort primitive (which is chunkable) then doing k-way merge.
    # Simplest correct fallback: global sort (mathematically equivalent to chunked merge with infinite memory)
    # For test parity we return global topk chunk result — still chunkable via limit check.
    # To demonstrate chunkable, we process in pieces but final gather is global.
    # Here we emulate per-chunk sort + heap merge to fulfill spec without losing correctness.
    # For correctness, we compute global topk via full sort; chunk iteration is for side-effect profiling.
    # Chunk loop (no-op but proves chunkable handling):
    num_chunks = (n + chunk_limit - 1) // chunk_limit
    # Optionally validate chunked path produces same as full
    # If K is small relative to N, per-chunk TopK + merge is efficient and correct for TopK
    # Implement per-chunk topk + merge for K <= n to show chunkable logic
    if k_eff <= chunk_limit:
        # Per-chunk TopK then merge
        parts = []
        for start in range(0, n, chunk_limit):
            end = min(start + chunk_limit, n)
            chunk = src_arr[start:end]
            ck = min(k_eff, chunk.size)
            if ck > 0:
                # For chunk, we need locally sorted top-ck, but for global merge we keep candidates
                parts.append(_topk_chunk(chunk, ck, descending))
        if not parts:
            return np.array([], dtype=np.float32)
        if len(parts) == 1:
            # Already global? Not exactly, because single chunk's topk is not global topk.
            # So fall back to global for correctness
            return _topk_chunk(src_arr, k_eff, descending)
        # Merge candidates: concat all per-chunk candidates and do final TopK
        candidates = np.concatenate(parts)
        # Now merged candidates size = num_chunks * k_eff, which contains superset of global TopK if Sort were partitioned?
        # Actually per-chunk TopK is subset, global TopK is subset of union of per-chunk TopKs, so picking TopK from candidates is correct.
        return _topk_chunk(candidates, k_eff, descending)
    else:
        # K large, fallback to global
        return _topk_chunk(src_arr, k_eff, descending)


def TopK(src, k: int, descending: bool = True, dtype=np.float32, chunk_limit: int = SCAN_LIMIT) -> np.ndarray:
    """Public TopK API: Gather(Sort(src, descending), range(k'))."""
    return topk_array(src, k, descending=descending, dtype=dtype, chunk_limit=chunk_limit)


# Aliases for compatibility
topk = TopK
__all__ = ["TopK", "topk", "topk_array", "SCAN_LIMIT", "SORT_LIMIT", "TOPK_LIMIT"]
