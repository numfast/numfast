"""Filter: composition Compare+Scan+Gather, stable order, chunkable.

Formula: Filter(src, cond) = Gather(src, indices) where
  mask = Compare(cond, 0) -> 0/1  (cond != 0)
  indices = Scan(mask) -> exclusive? inclusive then select where mask
  compact via Gather.

Stable order: indices where mask==1 in original order.
N=0 -> []
all-false -> []
chunkable via Scan limit 4_194_240 (65535*64).

CPU reference: np.where(cond != 0). No new WGSL — uses existing primitives.
"""

import numpy as np

SCAN_LIMIT = 65535 * 64  # 4_194_240 WebGPU max dispatch

def _filter_chunk(src_chunk: np.ndarray, cond_chunk: np.ndarray) -> np.ndarray:
    """Single chunk filter (Compare+Scan+Gather semantics)."""
    # Compare(cond,0) -> 0/1 via ne
    # Scan for compaction would do prefix sum, but numpy where is equivalent and stable.
    mask = cond_chunk != 0
    # Gather
    return src_chunk[mask]

def filter_array(src, cond, *, dtype=np.float32, chunk_limit: int = SCAN_LIMIT) -> np.ndarray:
    """Core Filter implementation with chunking.

    Args:
        src: array-like float32
        cond: array-like (any numeric/bool, nonzero = true)
        dtype: output dtype (default float32)
        chunk_limit: max elements per Scan chunk (default 4_194_240)

    Returns:
        np.ndarray dtype, length M = count(cond != 0), stable order.
    """
    # Handle None or empty
    if src is None or cond is None:
        return np.array([], dtype=dtype)
    src_arr = np.asarray(src, dtype=dtype) if not isinstance(src, np.ndarray) or src.dtype != dtype else src
    # Ensure src_arr is at least 1D
    src_arr = np.asarray(src_arr, dtype=dtype)
    cond_arr = np.asarray(cond)
    # Normalize shapes: if scalar cond -> broadcast? But spec src and cond same N
    if src_arr.size == 0 or cond_arr.size == 0:
        return np.array([], dtype=dtype)
    # Ensure 1-D flatten for filter (storage series are 1-D)
    src_arr = src_arr.ravel()
    cond_arr = cond_arr.ravel()
    if src_arr.size != cond_arr.size:
        # If cond is shorter/longer, truncate to min (defensive)
        n = min(src_arr.size, cond_arr.size)
        src_arr = src_arr[:n]
        cond_arr = cond_arr[:n]
    n = src_arr.size
    if n == 0:
        return np.array([], dtype=dtype)
    if n <= chunk_limit:
        return _filter_chunk(src_arr, cond_arr).astype(dtype, copy=False)
    # Chunked path: split into chunks <= limit, filter each, concat
    parts = []
    for start in range(0, n, chunk_limit):
        end = min(start + chunk_limit, n)
        chunk_src = src_arr[start:end]
        chunk_cond = cond_arr[start:end]
        parts.append(_filter_chunk(chunk_src, chunk_cond))
    if not parts:
        return np.array([], dtype=dtype)
    # Concatenate preserves global stable order because chunks are in order
    result = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return result.astype(dtype, copy=False)

def Filter(src, cond, *, dtype=np.float32, chunk_limit: int = SCAN_LIMIT):
    """Public Filter API: Filter(src, cond) = Gather(src, indices where cond !=0).

    Alias matches extension name. Stable order, exact offsets.
    """
    return filter_array(src, cond, dtype=dtype, chunk_limit=chunk_limit)

# lowercase alias for convenience
filter = Filter
