"""Phase breakdown for _encode_arrow_text with native dedup detection."""
import time, sys, numpy as np

sys.path.insert(0, "C:/App/numfast/numfast/src")
sys.path.insert(0, "C:/App/numfast/app-builder")

import pyarrow.parquet as pq
import pyarrow as pa

from Storage.Dictionary._lib.dictionary import (
    DictionaryBody, ENCODING, TEXT_DTYPE, _err
)
import Storage.Dictionary._lib.dictionary as _dict_mod

# Force native_cpu into sys.modules so _encode_arrow_text can find it
import Drivers.CPU._lib.native_cpu as _native_cpu_mod

_orig = _dict_mod._encode_arrow_text

# Per-column phase accumulators
_phases = {}
_native_active = None

def _patched(arr, validity, format_error):
    global _native_active
    n = len(arr)
    bufs = arr.buffers()
    valid_bitmap, offsets_buf, data_buf = bufs[0], bufs[1], bufs[2]

    # --- Phase 1: validity bitmap unpack ---
    t0 = time.perf_counter()
    nonnull = np.ones(n, dtype=bool)
    if valid_bitmap is not None:
        raw = np.frombuffer(valid_bitmap, dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder='little')
        nonnull = np.ascontiguousarray(bits[:n].astype(bool))
    p_vbit = (time.perf_counter() - t0) * 1000

    # --- Phase 2: validity sidecar merge ---
    t0 = time.perf_counter()
    vb = None
    if validity is not None:
        vb = np.asarray(validity, dtype=bool).reshape(-1)
        if vb.size != n:
            _err(format_error, f"dictionary_encode validity size {vb.size} != {n}.",
                 "pass validity matching values")
    p_vside = (time.perf_counter() - t0) * 1000

    # --- Phase 3: all-empty early return ---
    if not bool(nonnull.any()):
        eff = vb if vb is not None else np.zeros(n, dtype=bool)
        _phases['validity_bitmap'] = _phases.get('validity_bitmap', 0) + p_vbit
        _phases['validity_sidecar'] = _phases.get('validity_sidecar', 0) + p_vside
        _phases['buffer_access'] = _phases.get('buffer_access', 0)
        _phases['native_dedup'] = _phases.get('native_dedup', 0)
        _phases['python_dedup'] = _phases.get('python_dedup', 0)
        _phases['body_build'] = _phases.get('body_build', 0)
        _phases['values_mat'] = _phases.get('values_mat', 0)
        return {"codes": np.zeros(n, dtype=np.int32), "values": [],
                "dictionary": DictionaryBody(b"", np.zeros(1, dtype=np.int32)),
                "dtype": TEXT_DTYPE,
                "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
                "metadata": {"encoding": ENCODING, "d": 0, "n": int(n),
                             "sorted": True, "nulls": int(n - np.count_nonzero(eff))}}

    # --- Phase 4: zero-copy buffer access ---
    t0 = time.perf_counter()
    arrow_offsets = np.frombuffer(offsets_buf, dtype=np.int32).copy()
    data_mv = memoryview(data_buf)
    data_arr = np.ascontiguousarray(np.frombuffer(data_buf, dtype=np.uint8))
    valid_u8 = np.ascontiguousarray(nonnull.view(np.uint8))
    p_buf = (time.perf_counter() - t0) * 1000

    # --- Phase 5: native dedup attempt ---
    t0 = time.perf_counter()
    native_ok = False
    try:
        import sys as _sys
        _mod = _sys.modules.get("Drivers.CPU._lib.native_cpu")
        if _mod is None:
            for _p in (_sys.modules.get(m) for m in list(_sys.modules)):
                if _p is not None and hasattr(_p, "_req_text_dedup"):
                    _mod = _p
                    break
        if _mod is not None and hasattr(_mod, "_req_text_dedup"):
            _lib = _mod._req_text_dedup()
            if _lib is not None:
                if _native_active is None:
                    _native_active = 'ACTIVE'
                codes = np.zeros(n, dtype=np.int32)
                data_len = len(data_buf)
                uniq_data = np.empty(data_len, dtype=np.uint8)
                uniq_offs = np.empty(n + 1, dtype=np.int32)
                ng_arr = np.zeros(1, dtype=np.int32)
                rc = _lib.nf_unique_dict_utf8(
                    data_arr.ctypes.data, data_len,
                    arrow_offsets.ctypes.data, n + 1,
                    valid_u8.ctypes.data, n,
                    codes.ctypes.data, uniq_data.ctypes.data,
                    uniq_offs.ctypes.data, ng_arr.ctypes.data)
                if rc == 0:
                    ng = int(ng_arr[0])
                    native_ok = True
                    body_data = bytes(uniq_data[:int(uniq_offs[ng])].tobytes())
                    body_offs = uniq_offs[:ng + 1].copy()
    except Exception:
        pass
    p_native = (time.perf_counter() - t0) * 1000

    if native_ok:
        # --- Phase 5a: post-native (validity + body + values) ---
        t0 = time.perf_counter()
        valid_row = np.zeros(n, dtype=bool)
        valid_row[nonnull] = True
        eff = (valid_row & vb) if vb is not None else valid_row
        if vb is not None:
            codes[~vb] = 0
        d = ng
        if d > 0:
            mv_body = memoryview(body_data)
            vals = [bytes(mv_body[int(body_offs[j]):int(body_offs[j + 1])]).decode("utf-8")
                    for j in range(d)]
        else:
            vals = []
        nulls = int(n - np.count_nonzero(eff))
        p_post = (time.perf_counter() - t0) * 1000

        _phases['validity_bitmap'] = _phases.get('validity_bitmap', 0) + p_vbit
        _phases['validity_sidecar'] = _phases.get('validity_sidecar', 0) + p_vside
        _phases['buffer_access'] = _phases.get('buffer_access', 0) + p_buf
        _phases['native_dedup'] = _phases.get('native_dedup', 0) + p_native
        _phases['python_dedup'] = _phases.get('python_dedup', 0)
        _phases['body_build'] = _phases.get('body_build', 0)
        _phases['values_mat'] = _phases.get('values_mat', 0) + p_post

        return {"codes": np.ascontiguousarray(codes), "values": vals,
                "dictionary": DictionaryBody(body_data, body_offs),
                "dtype": TEXT_DTYPE,
                "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
                "metadata": {"encoding": ENCODING, "d": d, "n": int(n),
                             "sorted": True, "nulls": nulls}}

    if _native_active is None:
        _native_active = 'NONE'

    # --- Phase 6: Python byte-key dedup ---
    t0 = time.perf_counter()
    pos = np.nonzero(nonnull)[0]
    seen = {}
    uniq_first = []
    codes_raw = np.empty(len(pos), dtype=np.int32)
    for idx, i in enumerate(pos):
        a_off = int(arrow_offsets[i])
        b_off = int(arrow_offsets[i + 1])
        chunk = bytes(data_mv[a_off:b_off])
        c = seen.get(chunk)
        if c is None:
            c = len(uniq_first)
            seen[chunk] = c
            uniq_first.append(chunk)
        codes_raw[idx] = c
    sorted_uniq = sorted(uniq_first)
    byte_to_sorted = {b: i for i, b in enumerate(sorted_uniq)}
    remap = np.array([byte_to_sorted[b] for b in uniq_first], dtype=np.int32)
    inverse = remap[codes_raw]
    codes = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    codes[pos] = inverse.astype(np.int32, copy=False)
    valid[pos] = True
    eff = (valid & vb) if vb is not None else valid
    if vb is not None:
        codes[~vb] = 0
    p_pdedup = (time.perf_counter() - t0) * 1000

    # --- Phase 7: body build ---
    t0 = time.perf_counter()
    d = len(sorted_uniq)
    if d == 0:
        body_data = b""
        body_offs = np.zeros(1, dtype=np.int32)
    else:
        lengths = np.array([len(b) for b in sorted_uniq], dtype=np.int32)
        body_offs = np.zeros(d + 1, dtype=np.int32)
        body_offs[1:] = np.cumsum(lengths)
        body_data = b"".join(sorted_uniq)
    p_body = (time.perf_counter() - t0) * 1000

    # --- Phase 8: values materialization ---
    t0 = time.perf_counter()
    vals = [s.decode("utf-8") for s in sorted_uniq]
    p_vals = (time.perf_counter() - t0) * 1000

    nulls = int(n - np.count_nonzero(eff))
    _phases['validity_bitmap'] = _phases.get('validity_bitmap', 0) + p_vbit
    _phases['validity_sidecar'] = _phases.get('validity_sidecar', 0) + p_vside
    _phases['buffer_access'] = _phases.get('buffer_access', 0) + p_buf
    _phases['native_dedup'] = _phases.get('native_dedup', 0)
    _phases['python_dedup'] = _phases.get('python_dedup', 0) + p_pdedup
    _phases['body_build'] = _phases.get('body_build', 0) + p_body
    _phases['values_mat'] = _phases.get('values_mat', 0) + p_vals

    return {"codes": np.ascontiguousarray(codes), "values": vals,
            "dictionary": DictionaryBody(body_data, body_offs),
            "dtype": TEXT_DTYPE,
            "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
            "metadata": {"encoding": ENCODING, "d": d, "n": int(n),
                         "sorted": True, "nulls": nulls}}

_dict_mod._encode_arrow_text = _patched

# --- Main ---
from Storage.Dictionary._lib.dictionary import dictionary_encode_impl

table = pq.read_table('C:/App/competitions/ClickBench/data/hits_1m.parquet')

text_cols = []
for name in table.column_names:
    col = table.column(name)
    t = col.type
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        combined = col.combine_chunks()
        text_cols.append((name, combined))

text_cols.sort(key=lambda x: len(x[1]), reverse=True)

print(f"{'Column':<20s} {'N':>10s} {'D':>8s} {'total':>8s} {'vbit':>6s} {'vside':>6s} {'buf':>6s} {'nat_d':>7s} {'py_d':>7s} {'body':>7s} {'vals':>6s}")
print("-" * 102)

for name, arr in text_cols[:5]:
    _phases.clear()
    n = len(arr)
    t0 = time.perf_counter()
    result = dictionary_encode_impl(arr)
    t_end = time.perf_counter()
    total_ms = (t_end - t0) * 1000
    d = len(result['values'])
    p = dict(_phases)
    print(f"{name:<20s} {n:>10d} {d:>8d} {total_ms:>7.1f}ms "
          f"{p.get('validity_bitmap',0):>5.1f}ms "
          f"{p.get('validity_sidecar',0):>5.1f}ms "
          f"{p.get('buffer_access',0):>5.1f}ms "
          f"{p.get('native_dedup',0):>6.1f}ms "
          f"{p.get('python_dedup',0):>6.1f}ms "
          f"{p.get('body_build',0):>6.1f}ms "
          f"{p.get('values_mat',0):>5.1f}ms")

print(f"\nNative dedup: {_native_active}")
