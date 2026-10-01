# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Bulk carrier for a D-scale TEXT dictionary: any accepted shape ->
one pyarrow string Array, with no per-row Python.

Why this exists
---------------
A dictionary-domain predicate has to look at D distinct values, not N rows.
If the D values are handed over as a plain ``[str]`` list, the obvious
NumPy spelling (``np.array(list, dtype=str)``) first materializes a UCS4
array whose itemsize is ``4 x longest value``: for a 302 MB UTF-8 body of
2.62 M URLs that is a multi-gigabyte allocation and a multi-second pass
before a single byte is compared. The bulk transcode has to happen once,
in C, through Arrow -- the same discipline TextOps.text_length already
uses for a whole column.

Lanes, in order of preference:

  1. zero-copy view over a native body (``utf8_data`` + ``offsets``):
     a pyarrow Array is rebuilt from the two buffers, nothing is copied
     and nothing is decoded;
  2. bulk transcode of a flat carrier (Arrow Array/ChunkedArray, ``<U``
     or object ndarray, ``[str]`` sequence) in one C call;
  3. explicit error -- never a per-row Python loop and never a UCS4
     materialization.

Every lane yields the same Array, so the predicate on top of it is
carrier-independent by construction.

Lane 1 is entered through ``native_text_body`` and nowhere else, so the
body is validated exactly once per call no matter how many consumers it
has: ``text_dictionary`` hands the *same* validated ``(uint8 body, int32
offsets)`` pair to the Arrow view and to any consumer that wants the raw
bytes (the byte-scan substring kernel), which is what makes the native
lane a lane and not a special case.
"""

import numbers

import numpy as np

try:  # Arrow is the bulk UTF-8 carrier (Storage/Loaders use it too).
    import pyarrow as pa
    _HAS_PA = True
except ImportError:  # pragma: no cover -- Arrow-less install
    pa = None
    _HAS_PA = False

TEXT_DTYPE = "text"
INT64_DTYPE = "int64"
_NATIVE_KEYS = (("utf8_data", "offsets"), ("data", "offsets"))


def _err(what, fix):
    raise ValueError(f"{what} Fix: {fix}. See specs/05-storage-encoding.md")


def _is_int_scalar(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _reject_int64(values, op):
    """int64 dictionaries have no text predicates -- say so, do not scan."""
    dt = None
    if isinstance(values, dict):
        dt = values.get("dtype", TEXT_DTYPE)
    elif isinstance(values, np.ndarray) and values.dtype.kind in "iu":
        dt = INT64_DTYPE
    if dt == INT64_DTYPE:
        _err(f"{op} is TEXT-only, got an int64 dictionary.",
             "use the numeric range/equality domain path for int64 codes")
    if isinstance(values, (list, tuple, np.ndarray)) and not isinstance(values, str):
        probe = values[:1]
        if len(probe) and _is_int_scalar(probe[0]) and not isinstance(
                probe[0], (str, bytes)):
            _err(f"{op} is TEXT-only, got an int sequence.",
                 "pass the TEXT dictionary carrier (body/enc dict or [str])")


def _body_pair(values):
    """(utf8_data, offsets) for a native-body carrier, else None."""
    if isinstance(values, dict):
        if "dictionary" in values:
            return _body_pair(values["dictionary"])
        for dk, ok in _NATIVE_KEYS:
            if dk in values and ok in values:
                return values[dk], values[ok]
        return None
    for dk, ok in _NATIVE_KEYS:
        dv = getattr(values, dk, None)
        ov = getattr(values, ok, None)
        if dv is not None and ov is not None:
            return dv, ov
    return None


def _as_uint8(data):
    """A flat uint8 view over a UTF-8 body, or None if it is not bytes."""
    if isinstance(data, np.ndarray):
        if data.dtype == np.uint8:
            v = data.reshape(-1)
            return v if v.flags["C_CONTIGUOUS"] else np.ascontiguousarray(v)
        if data.dtype.kind == "S":
            return np.frombuffer(np.ascontiguousarray(data).tobytes(),
                                 dtype=np.uint8)
        return None
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(memoryview(data).cast("B"), dtype=np.uint8)
    return None


def native_text_body(values):
    """``(uint8 body, int32 offsets)`` for a native-body carrier, else None.

    The one gate into the native lane. Soundness is decided here and only
    here: int offsets, ``offsets[0] == 0``, ``offsets[-1] == len(body)``,
    non-decreasing, and a body small enough for the int32 offset buffer
    Arrow needs. Returns ``None`` -- never raises -- for a carrier that is
    not a native body, so the caller falls through to the transcode lane.
    """
    pair = _body_pair(values)
    if pair is None:
        return None
    raw_data, raw_offsets = pair
    try:
        o = np.ascontiguousarray(np.asarray(raw_offsets))
        if o.dtype.kind not in "iu" or o.size < 2:
            return None
        d = int(o.size) - 1
        if d < 1 or int(o[0]) != 0:
            return None
        body = _as_uint8(raw_data)
        if body is None or int(o[-1]) != int(body.size):
            return None
        if body.size >= 2 ** 31:
            return None
        if bool((o[1:] < o[:-1]).any()):
            return None
        o = o.astype(np.int32, copy=False)
    except Exception:  # noqa: BLE001 -- the transcode lane owns it
        return None
    return body, o


def _zero_copy_view(body):
    """pa.Array over a validated ``(body, offsets)``, no copy."""
    d8, o = body
    return pa.Array.from_buffers(pa.string(), int(o.size) - 1,
                                 [None, pa.py_buffer(o), pa.py_buffer(d8)])


def _flat_sequence(values):
    if isinstance(values, dict):
        for key in ("values", "dictionary"):
            v = values.get(key)
            if v is not None:
                return v
        return None
    return values


def text_dictionary(values, op="text dictionary predicate"):
    """``(pyarrow string Array, native body or None)`` for any TEXT carrier.

    The body is handed back so a kernel that wants the raw bytes -- and not
    a pyarrow Array -- can skip the rebuild entirely. It is ``None`` on
    every transcode lane, which is the signal that only the Array exists.

    Arrow is imported lazily-by-guard (see ``_HAS_PA``): an Arrow-less
    install imports this module and registers every DomainLUT alias, and
    only a call into a TEXT predicate -- the one place that truly needs
    an Array -- refuses, with the op in the message. ``native_text_body``
    and ``codes_lut_mask`` stay NumPy-only and keep working there.
    """
    if not _HAS_PA:
        _err(f"{op}: pyarrow is required for the TEXT dictionary carrier "
             "(pip install pyarrow).",
             "install pyarrow, or use codes_lut_mask, which is NumPy-only")
    _reject_int64(values, op)
    body = native_text_body(values)
    if body is not None:
        try:
            return _zero_copy_view(body), body
        except Exception:  # noqa: BLE001 -- the transcode lane owns it
            pass
    flat = _flat_sequence(values)
    if flat is None:
        _err(f"{op}: carrier exposes no body, no offsets and no values.",
             "pass the ENC dict, its dictionary body, or a [str] sequence")
    if isinstance(flat, pa.ChunkedArray):
        return flat.combine_chunks(), None
    if isinstance(flat, pa.Array):
        return flat, None
    if isinstance(flat, (str, bytes)):
        _err(f"{op}: got a single str, expected a dictionary carrier.",
             "pass the ENC dict or a [str] sequence")
    if isinstance(flat, np.ndarray):
        if flat.dtype.kind not in "UO":
            _err(f"{op}: got a {flat.dtype} ndarray, expected a text carrier.",
                 "pass the ENC dict or a [str] sequence")
        return pa.array(flat.reshape(-1), type=pa.string()), None
    if not isinstance(flat, (list, tuple)):
        flat = list(flat)
    return pa.array(flat, type=pa.string()), None


def text_dictionary_array(values, op="text dictionary predicate"):
    """Any TEXT dictionary carrier -> one pyarrow string Array (bulk)."""
    return text_dictionary(values, op)[0]
