# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Dictionary-backed TEXT + int64: codes int32[N] + native body + validity.

Dictionary<T> typed envelope (ABI): TEXT is a UTF-8 body
(DictionaryBody: utf8_data bytes + offsets int32[D+1], dtype "text").
int64 reuses the same envelope shape (dtype "int64", int64[D] sorted-unique
payload, np.unique, D<2^31) -- same codes int32[N] + validity sidecar.
TEXT ABI untouched; int64 is the second real type (no generic Dictionary<T>).

Residency: codes + validity sidecar + native body ONLY. The list values[D]
(str for TEXT, int for int64) are boundary/adaptor materializations
(D-scale: display via dictionary_decode, ingest edges, Series sidecar) --
never resident, never in compute paths. GPU takes codes only
(codes-only contract); int64 gather stays CPU (no WGSL int64 gather).

Deterministic sorted-unique encode (one bulk pass, C-speed via np.unique);
code == sorted rank. NULL contract (DELTA-3): None -> invalid (codes
placeholder 0 + validity False), never an error, never a sentinel in
compute. Decode is display-only boundary restore, never in compute paths.

Numeric rule: SUM/AVG/arithmetic MUST go LUT->gather->numeric kernel
(lut[codes] via existing ir_gather, then ir_reduce/ir_groupby). SUM(codes)
is meaningless and FORBIDDEN (codes are ranks, not values).
"""

import numbers

import numpy as np

try:
    import pyarrow as pa
except ImportError:
    pa = None

ENCODING = "dictionary-sorted-v1"
TEXT_DTYPE = "text"
INT64_DTYPE = "int64"
BODY_FORMAT = "utf8"

# Arrow integer type -> numpy dtype, for the buffer read in
# _coerce_int64_arrow (no pandas, no to_pandas_dtype dependency).
_ARROW_INT_NP = {}


class DictionaryBody:
    """Native TEXT body: utf8_data bytes + offsets int32[D+1], lazy list-like.

    No list[str] held resident: getitem decodes one UTF-8 slice, iteration
    and comparison materialize D strings only (D-scale, never N rows).
    Duck-type contract (no cross-Extension import needed): .dtype == "text",
    .format == "utf8", .utf8_data (bytes), .offsets (int32[D+1]), .d (int),
    .to_list() boundary materialization.
    """

    __slots__ = ("_data", "_offsets", "_vals")
    dtype = TEXT_DTYPE
    format = BODY_FORMAT

    def __init__(self, data, offsets):
        offs = np.ascontiguousarray(np.asarray(offsets, dtype=np.int32))
        if offs.ndim != 1 or offs.size == 0 or int(offs[0]) != 0:
            raise ValueError(
                "DictionaryBody needs offsets int32[D+1] starting at 0. "
                "Fix: build via dictionary_encode. See specs/05-storage-encoding.md"
            )
        if int(offs[-1]) != len(bytes(data)):
            raise ValueError(
                "DictionaryBody offsets must end at len(utf8_data). "
                "Fix: build via dictionary_encode. See specs/05-storage-encoding.md"
            )
        self._data = bytes(data)
        self._offsets = offs
        self._vals = None  # D-scale cache, filled on first full materialization

    @property
    def d(self):
        return int(self._offsets.size - 1)

    @property
    def utf8_data(self):
        return self._data

    @property
    def offsets(self):
        return self._offsets

    def _item(self, k):
        d = self.d
        if k < 0:
            k += d
        if k < 0 or k >= d:
            raise IndexError(
                f"DictionaryBody index {k} out of range D={d}. "
                "Fix: pass codes from dictionary_encode of the same dictionary"
            )
        o = self._offsets
        return bytes(memoryview(self._data)[int(o[k]):int(o[k + 1])]).decode("utf-8")

    def to_list(self):
        """Boundary materialization: [str]*D (D-scale, cached)."""
        if self._vals is None:
            o = self._offsets
            mv = memoryview(self._data)
            self._vals = [bytes(mv[int(a):int(b)]).decode("utf-8")
                          for a, b in zip(o[:-1].tolist(), o[1:].tolist())]
        return list(self._vals)

    def __len__(self):
        return self.d

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._item(k) for k in range(*i.indices(self.d))]
        return self._item(int(i))

    def __iter__(self):
        return iter(self.to_list())

    def __contains__(self, key):
        return any(s == key for s in self.to_list()) if isinstance(key, str) else False

    def index(self, key, *args):
        return self.to_list().index(key, *args)

    def __eq__(self, other):
        if isinstance(other, DictionaryBody):
            return self._data == other._data and bool(
                np.array_equal(self._offsets, other._offsets))
        if isinstance(other, (list, tuple)):
            return self.to_list() == list(other)
        return NotImplemented

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = None

    def __repr__(self):
        return (f"DictionaryBody(text, d={self.d}, "
                f"bytes={len(self._data)})")


def _body_build(vals, format_error=None):
    """[str]*D sorted-unique -> (utf8_data bytes, offsets int32[D+1]).

    Arrow bulk transcode first (C-speed, one pass); byte-identical Python
    fallback owns every Arrow failure (fallback/WASM parity).
    """
    d = len(vals)
    if d == 0:
        return b"", np.zeros(1, dtype=np.int32)
    try:
        import pyarrow as pa
        bufs = pa.array(list(vals), type=pa.string()).buffers()
        data = bytes(bufs[2])
        offs = np.ascontiguousarray(
            np.frombuffer(bufs[1], dtype=np.int32).copy())
        if offs.size != d + 1 or int(offs[0]) != 0 or int(offs[-1]) != len(data):
            raise ValueError("arrow body layout mismatch")
        return data, offs
    except ImportError:
        pass
    except Exception:
        pass
    raw = [s.encode("utf-8") for s in vals]
    total = sum(len(b) for b in raw)
    if total > 2 ** 31 - 1:
        _err(format_error, f"dictionary UTF-8 body {total} bytes exceeds int32 offsets.",
             "split the column to smaller cardinalities")
    offs = np.zeros(d + 1, dtype=np.int32)
    offs[1:] = np.cumsum(
        np.asarray([len(b) for b in raw], dtype=np.int64)).astype(np.int32)
    return b"".join(raw), offs


def _body_to_list(data, offsets, format_error=None):
    """Native body -> [str]*D (D-scale boundary materialization)."""
    o = np.ascontiguousarray(np.asarray(offsets, dtype=np.int32))
    mv = memoryview(bytes(data))
    try:
        return [bytes(mv[int(a):int(b)]).decode("utf-8")
                for a, b in zip(o[:-1].tolist(), o[1:].tolist())]
    except (UnicodeDecodeError, ValueError) as e:
        _err(format_error, f"dictionary body is not valid UTF-8 ({e}).",
             "pass a body from dictionary_encode")


def _coerce_list(values, format_error=None):
    """list[str] | DictionaryBody | body/enc dict -> [str]*D (D-scale).

    int64 LUT carriers are NOT coerced here -- decode/metadata route them
    to the int64 path before reaching this TEXT-only helper.
    """
    if isinstance(values, dict):
        dt = values.get("dtype", TEXT_DTYPE)
        if dt != TEXT_DTYPE:
            _err(format_error,
                 f"dictionary body dtype {dt!r} is not implemented (TEXT only).",
                 "pass a TEXT dictionary body from dictionary_encode")
        if isinstance(values.get("values"), (list, tuple)):
            vals = list(values["values"])
        elif "utf8_data" in values and "offsets" in values:
            return _body_to_list(values["utf8_data"], values["offsets"],
                                 format_error)
        elif "data" in values and "offsets" in values:
            return _body_to_list(values["data"], values["offsets"],
                                 format_error)
        else:
            _err(format_error, "dictionary-domain op needs [str] values.",
                 "pass resident['dictionary'] from resident_prepare")
    elif isinstance(values, (list, tuple)):
        vals = list(values)
    elif hasattr(values, "to_list") and hasattr(values, "offsets") \
            and getattr(values, "dtype", TEXT_DTYPE) == TEXT_DTYPE:
        try:
            return list(values.to_list())
        except (UnicodeDecodeError, ValueError) as e:
            _err(format_error, f"dictionary body is not valid UTF-8 ({e}).",
                 "pass a body from dictionary_encode")
    else:
        _err(format_error, "dictionary-domain op needs [str] values.",
             "pass resident['dictionary'] from resident_prepare")
    if any(not isinstance(s, str) for s in vals):
        _err(format_error, "dictionary-domain op needs [str] values.",
             "pass resident['dictionary'] from resident_prepare")
    return vals


def _is_int_scalar(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _arrow_int_np_dtype(t):
    """pa integer type -> numpy dtype, table built once, no pandas needed."""
    if not _ARROW_INT_NP:
        _ARROW_INT_NP.update({
            pa.int8(): np.int8, pa.int16(): np.int16, pa.int32(): np.int32,
            pa.int64(): np.int64, pa.uint8(): np.uint8, pa.uint16(): np.uint16,
            pa.uint32(): np.uint32, pa.uint64(): np.uint64,
        })
    return _ARROW_INT_NP.get(t)


def _coerce_int64_arrow(values):
    """Arrow integer column -> (flat int64[N], nonnull bool[N]) or None.

    Buffer read only: the Arrow values buffer plus the validity bitmap, both
    through ``np.frombuffer``. The previous route was ``list(values)``, which
    built N Python ints for a column whose bytes Arrow already holds (measured
    1114 ms @ N=2,000,000 against 0.00 ms for the buffer read).

    Semantics are the numpy-int lane of :func:`_coerce_int64_flat` verbatim:
    any integer width, same ``astype(int64)`` narrowing. Boolean and
    non-integer Arrow types return None -- the TEXT path owns them and its
    errors.
    """
    if pa is None:
        return None
    try:
        arr = values.combine_chunks() if isinstance(values, pa.ChunkedArray) \
            else values
        t = arr.type
        if pa.types.is_dictionary(t) or not pa.types.is_integer(t):
            return None
        np_dt = _arrow_int_np_dtype(t)
        if np_dt is None:
            return None
        n = len(arr)
        data_buf = arr.buffers()[1]
        if data_buf is None:
            return None
        flat = np.ascontiguousarray(
            np.frombuffer(data_buf, dtype=np_dt, count=n,
                          offset=arr.offset * np.dtype(np_dt).itemsize
                          ).astype(np.int64, copy=False).reshape(-1))
        nonnull = np.ones(n, dtype=bool)
        vb = arr.buffers()[0]
        if vb is not None:
            bits = np.unpackbits(np.frombuffer(vb, dtype=np.uint8),
                                 bitorder='little')
            nonnull = np.ascontiguousarray(
                bits[arr.offset:arr.offset + n].astype(bool))
        return flat, nonnull
    except Exception:  # noqa: BLE001 -- a non-Arrierable carrier returns None
        return None


def _coerce_int64_flat(values):
    """int64 column -> (flat int64[N], nonnull bool[N]) or None when not int64.

    Accepts int ndarray (any int/uint width, bool excluded), [int|None]
    lists/tuples, object arrays of int|None, Arrow integer Array/ChunkedArray.
    Empty/str/float/bool carriers return None (TEXT path owns them, including
    its errors). No packed-bit stage: int32 codes only.
    """
    if pa is not None and isinstance(values, (pa.Array, pa.ChunkedArray)):
        arrow_probe = _coerce_int64_arrow(values)
        if arrow_probe is not None:
            return arrow_probe
    if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
        flat = np.ascontiguousarray(values.reshape(-1).astype(np.int64, copy=False))
        return flat, np.ones(flat.size, dtype=bool)
    if isinstance(values, np.ndarray) and values.dtype.kind == "U":
        return None
    if isinstance(values, (np.ndarray, list, tuple)):
        arr = values if isinstance(values, np.ndarray) else None
        seq = values if isinstance(values, (list, tuple)) else None
        if arr is not None:
            if arr.dtype.kind in "bU":
                return None
            # Vectorized path for object arrays with int|None
            flat_arr = arr.reshape(-1)
            sz = flat_arr.size
            if sz == 0:
                return None
            try:
                nonnull = ~np.equal(flat_arr, None)  # vectorized None mask
            except (TypeError, ValueError):
                nonnull = None  # fall back to loop
            if nonnull is not None and nonnull.any():
                try:
                    flat = np.zeros(sz, dtype=np.int64)
                    flat[nonnull] = flat_arr[nonnull].astype(np.int64)
                    return np.ascontiguousarray(flat), nonnull
                except (TypeError, ValueError):
                    pass  # non-int objects → fall back to loop
            elif nonnull is not None and not nonnull.any():
                return None  # all None
            # Fallback: per-row loop
            seq = flat_arr.tolist()
        # seq from the list/tuple branch is already a sized, iterable sequence;
        # the removed defensive list() only duplicated N pointers before the
        # probe loop that usually returns None on its first row.
        if not seq:
            return None
        nonnull = np.ones(len(seq), dtype=bool)
        flat = np.zeros(len(seq), dtype=np.int64)
        seen = False
        for i, v in enumerate(seq):
            if v is None:
                nonnull[i] = False
            elif _is_int_scalar(v):
                flat[i] = int(v)
                seen = True
            else:
                return None
        return (flat, nonnull) if seen else None
    try:
        seq = list(values)
    except TypeError:
        return None
    if not seq:
        return None
    nonnull = np.ones(len(seq), dtype=bool)
    flat = np.zeros(len(seq), dtype=np.int64)
    seen = False
    for i, v in enumerate(seq):
        if v is None:
            nonnull[i] = False
        elif _is_int_scalar(v):
            flat[i] = int(v)
            seen = True
        else:
            return None
    return (flat, nonnull) if seen else None


def _encode_int64_impl(flat, nonnull, validity, format_error):
    """Sorted-unique int64 encode: codes int32[N] + lut int64[D] + validity."""
    n = flat.size
    vb = None
    if validity is not None:
        vb = np.asarray(validity, dtype=bool).reshape(-1)
        if vb.size != n:
            _err(format_error, f"dictionary_encode validity size {vb.size} != values {n}.",
                 "pass validity matching values length",
                 doc="specs/delta-3-null-contract.md")
    if not bool(nonnull.any()):
        eff = vb if vb is not None else np.zeros(n, dtype=bool)
        lut = np.zeros(0, dtype=np.int64)
        return {"codes": np.zeros(n, dtype=np.int32), "values": [],
                "dictionary": np.ascontiguousarray(lut),
                "dtype": INT64_DTYPE,
                "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
                "metadata": {"encoding": ENCODING, "d": 0, "n": int(n),
                             "sorted": True, "nulls": int(n - np.count_nonzero(eff))}}
    pos = np.nonzero(nonnull)[0]
    uniq, inverse = np.unique(flat[nonnull], return_inverse=True)
    if uniq.size > 2 ** 31 - 1:
        _err(format_error, f"dictionary_encode cardinality {uniq.size} exceeds int32.",
             "use a wider code dtype path")
    lut = np.ascontiguousarray(uniq.astype(np.int64, copy=False))
    codes = np.zeros(n, dtype=np.int32)
    codes[pos] = inverse.astype(np.int32, copy=False)
    if vb is not None:
        codes[~vb] = 0
    eff = (nonnull & vb) if vb is not None else nonnull
    nulls = int(n - np.count_nonzero(eff))
    return {"codes": np.ascontiguousarray(codes), "values": [int(v) for v in lut.tolist()],
            "dictionary": lut,
            "dtype": INT64_DTYPE,
            "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
            "metadata": {"encoding": ENCODING, "d": int(lut.size), "n": int(n),
                         "sorted": True, "nulls": nulls}}


def _coerce_int64_lut(values, format_error=None):
    """int64 LUT carrier -> int64[D] ndarray or None when not an int64 carrier.

    Accepts int64[D] ndarray, [int]*D lists, or body/enc dicts with
    dtype "int64" (keys "dictionary"/"values"). Empty list and TEXT
    carriers return None (TEXT path owns them).
    """
    if isinstance(values, dict):
        if values.get("dtype", TEXT_DTYPE) != INT64_DTYPE:
            return None
        if "dictionary" in values:
            return np.ascontiguousarray(
                np.asarray(values["dictionary"], dtype=np.int64).reshape(-1))
        if isinstance(values.get("values"), (list, tuple, np.ndarray)):
            seq = list(values["values"]) if not isinstance(values["values"], np.ndarray) \
                else values["values"].reshape(-1).tolist()
            if not seq:
                return np.zeros(0, dtype=np.int64)
            if any(not _is_int_scalar(v) for v in seq):
                _err(format_error, "dictionary-domain op needs [int] int64 LUT.",
                     "pass resident['dictionary'] from resident_prepare")
            return np.ascontiguousarray(np.asarray(seq, dtype=np.int64))
        _err(format_error, "dictionary-domain op needs int64 LUT.",
             "pass resident['dictionary'] from resident_prepare")
    if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
        return np.ascontiguousarray(values.reshape(-1).astype(np.int64, copy=False))
    if isinstance(values, (list, tuple)) and values:
        if any(not _is_int_scalar(v) for v in values):
            return None
        return np.ascontiguousarray(np.asarray(list(values), dtype=np.int64))
    return None


def _err(format_error, what, fix, doc="specs/05-storage-encoding.md"):
    if format_error is not None:
        raise format_error(what, fix=fix, doc=doc)
    raise ValueError(f"{what} Fix: {fix}. See {doc}")


def _arrow_text_layout(arr):
    """True when ``arr`` is a TEXT/binary Arrow array (the 3-buffer layout).

    Gate for the Arrow-native path: a non-text Arrow array carries fewer
    than three buffers, so indexing ``bufs[2]`` on it is a raw IndexError.
    binary/large_binary stay here on purpose -- their offsets buffer is the
    3-buffer layout _encode_arrow_text already reads.
    """
    if pa is None:
        return False
    try:
        t = arr.type
        return (pa.types.is_string(t) or pa.types.is_large_string(t)
                or pa.types.is_binary(t) or pa.types.is_large_binary(t))
    except Exception:
        return False


def _arrow_dict_text(values):
    """Arrow dictionary-typed carrier -> plain text pa.Array, else None.

    Reuses the ONE canonical normalisation already proven in the Arrow
    adapter (``col.cast(pa.string()).combine_chunks()``): Arrow rewrites
    the index stream into string offsets over the dictionary's own value
    buffer, after which the existing Arrow-native text path owns dedup and
    the sorted-unique re-rank. Non-text dictionaries return None (never a
    silent stringify -- an int64 dictionary must reach the int64 path).
    """
    if pa is None:
        return None
    try:
        t = values.type
        if not pa.types.is_dictionary(t):
            return None
        if not (pa.types.is_string(t.value_type)
                or pa.types.is_large_string(t.value_type)):
            return None
        out = values.cast(pa.string())
        # Array.cast -> StringArray (already single chunk); the Arrow adapter
        # casts a ChunkedArray, whose cast stays chunked and needs merging.
        return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out
    except Exception:
        return None


def _arrow_cxx_sorted_encode(arr, validity, format_error):
    """Arrow C++ dense-dict fast path (generic, shared H2O/KXTAQ).

    pc.dictionary_encode is C++ resident (no Python per-row loop, no str
    decode); sorted remap restores the sorted-unique contract
    (code == sorted rank). D-scale Python work only (sort + D remap).
    Returns dict contract or None when unavailable (caller keeps fallback).
    """
    import pyarrow.compute as _pc
    n = len(arr)
    nonnull = np.ones(n, dtype=bool)
    try:
        _nn = arr.is_null().to_numpy(zero_copy_only=False)
        nonnull = np.ascontiguousarray(~np.asarray(_nn, dtype=bool))
    except Exception:
        try:
            _py = arr.to_pylist()
            nonnull = np.ascontiguousarray(
                np.array([v is not None for v in _py], dtype=bool))
        except Exception:
            return None
    vb = None
    if validity is not None:
        vb = np.asarray(validity, dtype=bool).reshape(-1)
        if vb.size != n:
            _err(format_error,
                 f"dictionary_encode validity size {vb.size} != values {n}.",
                 "pass validity matching values length",
                 doc="specs/delta-3-null-contract.md")
    try:
        _d = _pc.dictionary_encode(arr)
    except Exception:
        return None
    try:
        _raw_dict = _d.dictionary.to_pylist()
    except Exception:
        return None
    _sorted_vals = sorted(_raw_dict)
    d = len(_sorted_vals)
    if d > 2 ** 31 - 1:
        _err(format_error,
             f"dictionary_encode cardinality {d} exceeds int32.",
             "use a wider code dtype path")
    if d == 0:
        eff = vb if vb is not None else np.zeros(n, dtype=bool)
        return {"codes": np.zeros(n, dtype=np.int32), "values": [],
                "dictionary": DictionaryBody(b"", np.zeros(1, dtype=np.int32)),
                "dtype": TEXT_DTYPE,
                "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
                "metadata": {"encoding": ENCODING, "d": 0, "n": int(n),
                             "sorted": True, "nulls": int(n - np.count_nonzero(eff))}}
    _v2s = {v: i for i, v in enumerate(_sorted_vals)}
    try:
        _remap = np.array([_v2s[v] for v in _raw_dict], dtype=np.int32)
    except KeyError:
        return None
    try:
        _idx = _d.indices.fill_null(0).to_numpy()
        _idx = np.ascontiguousarray(np.asarray(_idx, dtype=np.int64).reshape(-1))
    except Exception:
        return None
    codes = np.ascontiguousarray(_remap[_idx].astype(np.int32, copy=False))
    valid = np.ascontiguousarray(nonnull)
    eff = (valid & vb) if vb is not None else valid
    codes = np.ascontiguousarray(codes)
    codes[~eff] = 0
    data, offs = _body_build(_sorted_vals, format_error)
    nulls = int(n - np.count_nonzero(eff))
    return {"codes": codes, "values": list(_sorted_vals),
            "dictionary": DictionaryBody(data, offs),
            "dtype": TEXT_DTYPE,
            "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
            "metadata": {"encoding": ENCODING, "d": d, "n": int(n),
                         "sorted": True, "nulls": nulls}}


def _encode_arrow_text(arr, validity, format_error):
    """Arrow-native TEXT encode: pa.Array -> {codes, dictionary, values, ...}.

    Zero-copy buffer access, byte-key dedup (no str decode), byte-sort
    (lexicographic == codepoint for BMP UTF-8), direct body build from
    sorted unique byte slices.
    """
    n = len(arr)
    bufs = arr.buffers()
    valid_bitmap = bufs[0]
    offsets_buf = bufs[1]
    data_buf = bufs[2]

    # 1. Extract validity from Arrow bitmap
    nonnull = np.ones(n, dtype=bool)
    if valid_bitmap is not None:
        raw = np.frombuffer(valid_bitmap, dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder='little')
        nonnull = np.ascontiguousarray(bits[:n].astype(bool))

    # 2. Validity sidecar merge
    vb = None
    if validity is not None:
        vb = np.asarray(validity, dtype=bool).reshape(-1)
        if vb.size != n:
            _err(format_error,
                 f"dictionary_encode validity size {vb.size} != values {n}.",
                 "pass validity matching values length",
                 doc="specs/delta-3-null-contract.md")

    # 3. Handle all-empty case
    if not bool(nonnull.any()):
        eff = vb if vb is not None else np.zeros(n, dtype=bool)
        return {"codes": np.zeros(n, dtype=np.int32), "values": [],
                "dictionary": DictionaryBody(b"", np.zeros(1, dtype=np.int32)),
                "dtype": TEXT_DTYPE,
                "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
                "metadata": {"encoding": ENCODING, "d": 0, "n": int(n),
                             "sorted": True, "nulls": int(n - np.count_nonzero(eff))}}

    # 4. Zero-copy offset + data access (large_string guard: int64 offsets -> fallback)
    _is_large = False
    try:
        _is_large = bool(arr.type == pa.large_string() or arr.type == pa.large_utf8())
    except Exception:
        pass
    if not _is_large and offsets_buf is not None and int(offsets_buf.size) == (n + 1) * 8:
        _is_large = True
    if _is_large:
        arrow_offsets = np.frombuffer(offsets_buf, dtype=np.int64).copy()
    else:
        arrow_offsets = np.frombuffer(offsets_buf, dtype=np.int32).copy()
    data_mv = memoryview(data_buf)

    # 5. Try native dedup path first (canonical native_cpu, memoized there)
    data_arr = np.ascontiguousarray(np.frombuffer(data_buf, dtype=np.uint8))
    valid_u8 = np.ascontiguousarray(nonnull.view(np.uint8))
    native_ok = False
    try:
        if _is_large:
            raise ValueError("large_string -> fallback")
        import importlib as _il
        _req = None
        for _cand in ("_lib.native_cpu",
                      "Drivers.CPU._lib.native_cpu",
                      "src.Drivers.CPU._lib.native_cpu"):
            try:
                _m = _il.import_module(_cand)
                _req = getattr(_m, "_req_text_dedup", None)
                if _req is not None:
                    break
            except ImportError:
                continue
        if _req is None:
            try:
                import importlib.util as _ilu
                import os as _os
                _nc = _os.path.abspath(_os.path.join(
                    _os.path.dirname(_os.path.abspath(__file__)),
                    "..", "..", "Drivers", "CPU", "_lib", "native_cpu.py"))
                if _os.path.isfile(_nc):
                    _spec = _ilu.spec_from_file_location(
                        "_nf_native_cpu_dedup", _nc)
                    if _spec is not None and _spec.loader is not None:
                        _nm = _ilu.module_from_spec(_spec)
                        _spec.loader.exec_module(_nm)
                        _req = getattr(_nm, "_req_text_dedup", None)
            except ImportError:
                pass
        if _req is not None:
            _lib = _req()
            if _lib is not None:
                codes = np.zeros(n, dtype=np.int32)
                # Worst-case: all rows unique, data_len bytes for uniq_data
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
                    codes[~nonnull] = 0
                    # Build DictionaryBody from native output
                    body_data = bytes(uniq_data[:int(uniq_offs[ng])].tobytes())
                    body_offs = uniq_offs[:ng + 1].copy()

                    # Build effective validity + null count
                    valid_row = np.zeros(n, dtype=bool)
                    valid_row[nonnull] = True
                    eff = (valid_row & vb) if vb is not None else valid_row
                    if vb is not None:
                        codes[~vb] = 0

                    d = ng
                    if d > 2 ** 31 - 1:
                        _err(format_error,
                             f"dictionary_encode cardinality {d} exceeds int32.",
                             "use a wider code dtype path")

                    # Boundary values list (D-scale materialization)
                    if d > 0:
                        mv_body = memoryview(body_data)
                        vals = [bytes(mv_body[int(body_offs[j]):int(body_offs[j + 1])]).decode("utf-8")
                                for j in range(d)]
                    else:
                        vals = []

                    nulls = int(n - np.count_nonzero(eff))
                    return {"codes": np.ascontiguousarray(codes), "values": vals,
                            "dictionary": DictionaryBody(body_data, body_offs),
                            "dtype": TEXT_DTYPE,
                            "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
                            "metadata": {"encoding": ENCODING, "d": d, "n": int(n),
                                         "sorted": True, "nulls": nulls}}
    except Exception:
        pass  # Fallback to C++ dense-dict below, then Python dedup

    # 5b. Arrow C++ dense-dict fast path (generic shared primitive).
    try:
        _fast = _arrow_cxx_sorted_encode(arr, validity, format_error)
        if _fast is not None:
            return _fast
    except Exception:
        pass

    # 6. Fallback: Python byte-key dedup (existing proven path)
    pos = np.nonzero(nonnull)[0]
    seen = {}          # bytes -> code (insertion order)
    uniq_first = []    # bytes in insertion order
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

    # 7. Sort unique bytes lexicographically (UTF-8 byte order == codepoint for BMP)
    sorted_uniq = sorted(uniq_first)   # Python bytes sort is byte-lex
    byte_to_sorted = {b: i for i, b in enumerate(sorted_uniq)}
    remap = np.array([byte_to_sorted[b] for b in uniq_first], dtype=np.int32)

    # 8. Build codes array
    inverse = remap[codes_raw]
    codes = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    codes[pos] = inverse.astype(np.int32, copy=False)
    valid[pos] = True

    eff = (valid & vb) if vb is not None else valid
    if vb is not None:
        codes[~vb] = 0

    d = len(sorted_uniq)
    if d > 2 ** 31 - 1:
        _err(format_error,
             f"dictionary_encode cardinality {d} exceeds int32.",
             "use a wider code dtype path")

    # 9. Build DictionaryBody directly from sorted unique byte slices
    if d == 0:
        body_data = b""
        body_offs = np.zeros(1, dtype=np.int32)
    else:
        lengths = np.array([len(b) for b in sorted_uniq], dtype=np.int32)
        body_offs = np.zeros(d + 1, dtype=np.int32)
        body_offs[1:] = np.cumsum(lengths)
        total = int(body_offs[-1])
        if total > 2 ** 31 - 1:
            _err(format_error,
                 f"dictionary UTF-8 body {total} bytes exceeds int32 offsets.",
                 "split the column to smaller cardinalities")
        body_data = b"".join(sorted_uniq)

    # 10. Boundary values list (D-scale materialization)
    vals = [s.decode("utf-8") for s in sorted_uniq]

    nulls = int(n - np.count_nonzero(eff))
    return {"codes": np.ascontiguousarray(codes), "values": vals,
            "dictionary": DictionaryBody(body_data, body_offs),
            "dtype": TEXT_DTYPE,
            "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
            "metadata": {"encoding": ENCODING, "d": d, "n": int(n),
                         "sorted": True, "nulls": nulls}}


def dictionary_encode_impl(values, validity=None, format_error=None):
    """encode([str|None]*N) -> {codes int32[N], dictionary Body, values, validity, metadata}.

    Sorted-unique dictionary (deterministic across runs); codes are positions
    in the sorted values list (lexicographic by codepoint == np.unique order).
    Resident carrier is ``dictionary`` (native UTF-8 body for TEXT,
    int64[D] ndarray for int64); ``values`` stays as the D-scale
    boundary/adaptor materialization only.
    """
    # --- Arrow-native path: skip to_pylist + object array + Python dedup ---
    if pa is not None and isinstance(values, (pa.Table, pa.RecordBatch)):
        _err(format_error,
             f"dictionary_encode got {type(values).__name__}, a table of "
             "columns, not a column.",
             "pass one column: table.column('name') or table['name']")
    if pa is not None and isinstance(values, (pa.Array, pa.ChunkedArray)):
        # Dictionary-encoded text carrier: normalize with the canonical Arrow
        # cast, then the plain text path owns the rest. Without this the
        # index stream reaches bufs[2] below as a bare IndexError.
        _dt = _arrow_dict_text(values)
        if _dt is not None:
            return _encode_arrow_text(_dt, validity, format_error)
    if pa is not None and isinstance(values, pa.Array):
        if not _arrow_text_layout(values):
            _err(format_error,
                 f"dictionary_encode needs a TEXT Arrow array, got {values.type}.",
                 "pass pa.string()/pa.large_string() data, a dictionary-encoded "
                 "string column, or a [str|None] / int64 column")
        return _encode_arrow_text(values, validity, format_error)
    probe = _coerce_int64_flat(values)
    if probe is not None:
        flat, nonnull = probe
        return _encode_int64_impl(flat, nonnull, validity, format_error)
    # --- Generic C++ dense-dict fast path for U/object str columns ---
    # Reuses Arrow resident encode (no Python per-row loop). Falls through
    # to proven paths on any failure (non-str values, no pyarrow).
    if pa is not None and isinstance(values, np.ndarray) and values.dtype.kind in "UO":
        try:
            _flat = values.reshape(-1)
            _nn = None
            if values.dtype.kind == "O":
                try:
                    _nn = np.asarray(_flat != None, dtype=bool)  # noqa: E711
                except TypeError:
                    _nn = None
                if _nn is not None and bool(_nn.any()):
                    _sample = _flat[_nn]
                    _nchk = min(1024, int(_sample.size))
                    if any(not isinstance(v, str) for v in _sample[:_nchk].tolist()):
                        raise ValueError("non-str object")
                elif _nn is not None and not bool(_nn.any()):
                    raise ValueError("all-null")
            _pa = pa.array(_flat, type=pa.string())
            _fast = _arrow_cxx_sorted_encode(_pa, validity, format_error)
            if _fast is not None:
                return _fast
        except Exception:
            pass
    if isinstance(values, np.ndarray) and values.dtype.kind == "U":
        arr = values.reshape(-1) if values.ndim != 1 else values
        if values.ndim != 1:
            _err(format_error, "dictionary_encode needs a rank-1 column, got shape "
                 f"{values.shape}.", "pass a flat string column")
        nonnull = np.ones(arr.size, dtype=bool)
        u_all = arr
    else:
        if not isinstance(values, (np.ndarray, list, tuple)):
            try:
                values = list(values)
            except TypeError:
                _err(format_error, "dictionary_encode needs a string column, got "
                     f"{type(values).__name__}.", "pass a list[str|None] column")
        arr = np.asarray(values, dtype=object)
        if arr.ndim != 1:
            _err(format_error, "dictionary_encode needs a rank-1 column, got shape "
                 f"{arr.shape}.", "pass a flat string column")
        n = arr.size
        if n == 0:
            return {"codes": np.zeros(0, dtype=np.int32), "values": [],
                    "dictionary": DictionaryBody(b"", np.zeros(1, dtype=np.int32)),
                    "dtype": TEXT_DTYPE,
                    "validity": None,
                    "metadata": {"encoding": ENCODING, "d": 0, "n": 0,
                                 "sorted": True, "nulls": 0}}
        try:
            nonnull = np.asarray(arr != None, dtype=bool)  # noqa: E711 -- C loop
        except TypeError as e:
            _err(format_error, f"dictionary_encode needs a string column ({e}).",
                 "pass str or None per row, never raw numbers")
        u_all = None
    n = arr.size
    if validity is not None:
        vb = np.asarray(validity, dtype=bool).reshape(-1)
        if vb.size != n:
            _err(format_error, f"dictionary_encode validity size {vb.size} != values {n}.",
                 "pass validity matching values length",
                 doc="specs/delta-3-null-contract.md")
    else:
        vb = None
    if u_all is None:
        if not bool(nonnull.any()):
            eff = vb if vb is not None else np.zeros(n, dtype=bool)
            return {"codes": np.zeros(n, dtype=np.int32), "values": [],
                    "dictionary": DictionaryBody(b"", np.zeros(1, dtype=np.int32)),
                    "dtype": TEXT_DTYPE,
                    "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
                    "metadata": {"encoding": ENCODING, "d": 0, "n": int(n),
                                 "sorted": True, "nulls": int(n - np.count_nonzero(eff))}}
        # --- object/list path: hash dedup on Python str, skip U conversion ---
        s = arr[nonnull]
        _items = s.tolist()  # already Python str — zero-cost unbox
        _seen = {}
        _uniq_first = []
        _next_code = 0
        _codes_raw = np.empty(len(_items), dtype=np.int32)
        for _i, _v in enumerate(_items):
            if not isinstance(_v, str):
                _err(format_error,
                     f"dictionary_encode needs a string column, got "
                     f"{type(_v).__name__} value {_v!r}.",
                     "pass str or None per row, never raw numbers")
            _c = _seen.get(_v)
            if _c is None:
                _c = _next_code
                _seen[_v] = _c
                _uniq_first.append(_v)
                _next_code += 1
            _codes_raw[_i] = _c
        _sorted_vals = sorted(_uniq_first)
        uniq = np.asarray(_sorted_vals, dtype="U")
        _val_to_sorted = {v: i for i, v in enumerate(_sorted_vals)}
        _remap = np.array([_val_to_sorted[v] for v in _uniq_first], dtype=np.int32)
        inverse = _remap[_codes_raw]
        pos = np.nonzero(nonnull)[0]
    else:
        # --- "U" array path: np.unique (C-speed, no Python loop) ---
        u = np.ascontiguousarray(u_all)
        uniq, inverse = np.unique(u, return_inverse=True)
        pos = np.nonzero(nonnull)[0]
    # --- common tail ---
    if uniq.size > 2 ** 31 - 1:
        _err(format_error, f"dictionary_encode cardinality {uniq.size} exceeds int32.",
             "use a wider code dtype path")
    codes = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    codes[pos] = inverse.astype(np.int32, copy=False)
    valid[pos] = True
    eff = (valid & vb) if vb is not None else valid
    if vb is not None:
        codes[~vb] = 0
    vals = [str(s) for s in uniq.tolist()]
    data, offs = _body_build(vals, format_error)
    nulls = int(n - np.count_nonzero(eff))
    return {"codes": np.ascontiguousarray(codes), "values": vals,
            "dictionary": DictionaryBody(data, offs),
            "dtype": TEXT_DTYPE,
            "validity": None if bool(np.all(eff)) else np.ascontiguousarray(eff),
            "metadata": {"encoding": ENCODING, "d": int(uniq.size), "n": int(n),
                         "sorted": True, "nulls": nulls}}


def _all_null(codes, d, validity, format_error):
    """D==0 -> [None]*N (every row NULL, exactly).

    DELTA-3 gives an all-NULL column a zero-length dictionary, so no value
    can legally sit at code 0 and every code is the NULL placeholder --
    true with or without the sidecar, since there is nothing to gather. A
    negative code is still an error (corrupt carrier, never a NULL row), and
    a sidecar of the wrong length still fails loudly.
    """
    if codes.size and int(codes.min()) < 0:
        bad = int(codes[np.nonzero(codes < 0)[0][0]])
        _err(format_error, f"dictionary_decode code {bad} out of range D={d}.",
             "pass codes from dictionary_encode of the same dictionary")
    if validity is not None:
        v = np.asarray(validity, dtype=bool).reshape(-1)
        if v.size != codes.size:
            _err(format_error,
                 f"dictionary_decode validity size {v.size} != codes {codes.size}.",
                 "pass validity matching codes length",
                 doc="specs/delta-3-null-contract.md")
    return [None] * int(codes.size)


def dictionary_decode_impl(codes, values, validity=None, format_error=None):
    """Display-only boundary restore: codes -> strs|ints (invalid -> None).

    ``values`` accepts the boundary list, the native body (DictionaryBody
    for TEXT, int64[D] ndarray for int64), or a body/enc dict -- all D-scale
    carriers. Never in compute paths. Out-of-range codes -> explicit error,
    never wrap.

    ``validity`` omitted == "every code in ``codes`` is valid" (the encode
    envelope returns ``validity=None`` for an all-valid column): decode is
    then a pure gather over the LUT. It CANNOT restore NULLs, and must not
    pretend to -- under DELTA-3 the NULL placeholder IS code 0, which is
    also the sorted rank of the smallest real value, so NULL and a genuine
    value are indistinguishable once the sidecar is dropped. Pass the
    envelope's ``validity`` to get NULLs back. The one exact exception is
    D=0 (an all-NULL column): no value exists, so every row decodes to None.

    A dict carrier is also the encode ENVELOPE, and an envelope carries its
    own ``validity`` sidecar. Dropping it silently turned every NULL row
    into code 0 (a real sorted rank) with no signal to the caller, so an
    omitted ``validity`` now adopts the envelope's. An explicitly passed
    ``validity`` always wins -- that stays the way to assert "every code is
    valid" over a bare list/body carrier.
    """
    c = np.asarray(codes, dtype=np.int64).reshape(-1)
    if validity is None and isinstance(values, dict):
        validity = values.get("validity")
    lut = _coerce_int64_lut(values)
    if lut is not None:
        d = lut.size
        if d == 0:
            return _all_null(c, d, validity, format_error)
        if c.size and (int(c.min()) < 0 or int(c.max()) >= d):
            bad = int(c[np.nonzero((c < 0) | (c >= d))[0][0]])
            _err(format_error, f"dictionary_decode code {bad} out of range D={d}.",
                 "pass codes from dictionary_encode of the same dictionary")
        out = [None] * int(c.size)
        if validity is None:
            for i, k in enumerate(c.tolist()):
                out[i] = int(lut[int(k)])
            return out
        v = np.asarray(validity, dtype=bool).reshape(-1)
        if v.size != c.size:
            _err(format_error, f"dictionary_decode validity size {v.size} != codes {c.size}.",
                 "pass validity matching codes length",
                 doc="specs/delta-3-null-contract.md")
        for i, k in enumerate(c.tolist()):
            out[i] = int(lut[int(k)]) if bool(v[i]) else None
        return out
    vals = _coerce_list(values, format_error)
    d = len(vals)
    if d == 0:
        return _all_null(c, d, validity, format_error)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= d):
        bad = int(c[np.nonzero((c < 0) | (c >= d))[0][0]])
        _err(format_error, f"dictionary_decode code {bad} out of range D={d}.",
             "pass codes from dictionary_encode of the same dictionary")
    out = [None] * int(c.size)
    if validity is None:
        for i, k in enumerate(c.tolist()):
            out[i] = vals[int(k)]
        return out
    v = np.asarray(validity, dtype=bool).reshape(-1)
    if v.size != c.size:
        _err(format_error, f"dictionary_decode validity size {v.size} != codes {c.size}.",
             "pass validity matching codes length",
             doc="specs/delta-3-null-contract.md")
    for i, k in enumerate(c.tolist()):
        out[i] = vals[int(k)] if bool(v[i]) else None
    return out


def dictionary_metadata_impl(values, format_error=None):
    """metadata([str|int]*D | body) -> {encoding, d, sorted}; validates sorted-unique."""
    lut = _coerce_int64_lut(values)
    if lut is not None:
        d = int(lut.size)
        if d == 0:
            return {"encoding": ENCODING, "d": 0, "sorted": True, "unique": True}
        flat = lut.tolist()
        return {"encoding": ENCODING, "d": d,
                "sorted": flat == sorted(flat), "unique": len(set(flat)) == d}
    if isinstance(values, dict) or hasattr(values, "to_list"):
        vals = _coerce_list(values, format_error)
        return {"encoding": ENCODING, "d": len(vals),
                "sorted": vals == sorted(vals), "unique": len(set(vals)) == len(vals)}
    vals = list(values)
    return {"encoding": ENCODING, "d": len(vals),
            "sorted": vals == sorted(vals), "unique": len(set(vals)) == len(vals)}
