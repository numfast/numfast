"""Select(cond,a,b) = Mask: conditional select.

Select is alias for Mask primitive. No new WGSL.
"""

import numpy as np

def select_array(cond, a, b, *, dtype=np.float32) -> np.ndarray:
    """Select(cond,a,b) -> out[i] = cond[i]!=0 ? a[i] : b[i].

    cond: array-like (nonzero = true) or scalar
    a, b: array-like or scalar
    """
    cond_arr = np.asarray(cond)
    # Determine N from cond if array, else from a/b
    # If cond scalar, broadcast
    a_arr = np.asarray(a) if not np.isscalar(a) else a
    b_arr = np.asarray(b) if not np.isscalar(b) else b

    # Handle scalar cond
    if cond_arr.ndim == 0 or cond_arr.size == 1 and np.asarray(cond).ndim == 0:
        is_true = bool(cond_arr) if cond_arr.size == 1 else False
        # need to determine output length from a/b if they are arrays
        if isinstance(a_arr, np.ndarray) and a_arr.size > 1:
            n = a_arr.size
            cond_broadcast = np.full(n, is_true, dtype=bool)
            a_b = np.asarray(a_arr, dtype=dtype).ravel()
            b_b = np.asarray(b_arr, dtype=dtype).ravel() if isinstance(b_arr, np.ndarray) else np.full(n, b_arr, dtype=dtype)
            return np.where(cond_broadcast, a_b, b_b).astype(dtype, copy=False)
        if isinstance(b_arr, np.ndarray) and b_arr.size > 1:
            n = b_arr.size
            cond_broadcast = np.full(n, is_true, dtype=bool)
            b_b = np.asarray(b_arr, dtype=dtype).ravel()
            a_b = np.asarray(a_arr, dtype=dtype).ravel() if isinstance(a_arr, np.ndarray) else np.full(n, a_arr, dtype=dtype)
            return np.where(cond_broadcast, a_b, b_b).astype(dtype, copy=False)
        # all scalars
        aval = float(a_arr) if not isinstance(a_arr, np.ndarray) else float(np.asarray(a_arr, dtype=dtype).ravel()[0])
        bval = float(b_arr) if not isinstance(b_arr, np.ndarray) else float(np.asarray(b_arr, dtype=dtype).ravel()[0])
        return np.array([aval if is_true else bval], dtype=dtype)

    # cond is array
    cond_bool = cond_arr.ravel() != 0
    n = cond_bool.size

    # Prepare a and b as arrays of size n
    def _prep(x, n, dtype):
        if np.isscalar(x):
            return np.full(n, x, dtype=dtype)
        arr = np.asarray(x, dtype=dtype).ravel()
        if arr.size == 1:
            return np.full(n, arr[0], dtype=dtype)
        if arr.size != n:
            # broadcast/truncate
            if arr.size < n:
                # repeat? but spec says a,b same N as cond; fallback to tile
                return np.resize(arr, n)
            return arr[:n]
        return arr

    a_prep = _prep(a_arr, n, dtype)
    b_prep = _prep(b_arr, n, dtype)
    return np.where(cond_bool, a_prep, b_prep).astype(dtype, copy=False)

def Select(cond, a, b, *, dtype=np.float32):
    """Public Select API: alias Mask."""
    return select_array(cond, a, b, dtype=dtype)

# lowercase alias
select = Select
