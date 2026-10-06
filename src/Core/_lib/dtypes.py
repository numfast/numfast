# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Core/dtypes — canonical dtype table, single source (spec 07).

Types: int32 logical, int64 logical for BIGINT int64-capable ops
(group/filter/sort/compare/agg-payload, stays int64), f32 compute default,
f64 explicit/display only. Execution codes stay int32 (sort positions,
gather indices, pack_keys codes, inv codes, group key codes).
bool = 1-bit packed BoolMask (DELTA-5 bit suitcase, 8 masks/byte).
enum:W (W in 1/2/4/8) = width-W packed small-enum storage.
"""

from .errors import format_error

_TABLE = {
    "int32": {"itemsize": 4, "logical": "int32", "accum": "int64"},
    "int64": {"itemsize": 8, "logical": "int64", "accum": "int64"},
    "float32": {"itemsize": 4, "logical": "float32", "accum": "float64"},
    "f32": {"itemsize": 4, "logical": "float32", "accum": "float64"},
    "float64": {"itemsize": 8, "logical": "float64", "accum": "float64"},
    "f64": {"itemsize": 8, "logical": "float64", "accum": "float64"},
    "bool": {"itemsize": 0.125, "logical": "bool", "accum": "int64",
             "width": 1, "kind": "mask"},
}

_ENUM_WIDTHS = (1, 2, 4, 8)

_INT32_MIN = -(2 ** 31)
_INT32_MAX = 2 ** 31 - 1


def check_int32_range(values, what="series"):
    """check_int32_range(values, what) -> None; raises OverflowError if narrowing overflows.

    Invariant #1: NumFast never changes a value on implicit narrowing without
    a range check. int64/BIGINT stays int64 on int64-capable ops
    (group/filter/sort/compare/agg-payload via dtype='int64'); any
    int64->int32 narrowing that does not fit raises OverflowError with the
    what+range+value+fix contract -- never a silent wrap (e.g. UserID ~9e18
    must not become -6384394). Empty columns pass. Non-numeric inputs are
    left for the cast itself to reject (no masking of its error).
    """
    import numpy as np

    arr = values if isinstance(values, np.ndarray) else np.asarray(values)
    if arr.size == 0:
        return None
    if arr.dtype == np.dtype(np.int32):
        return None
    kind = arr.dtype.kind
    try:
        if kind in "iu":
            lo, hi = int(arr.min()), int(arr.max())
        elif kind == "f":
            if not bool(np.isfinite(arr).all()):
                raise OverflowError(
                    f"{what}: int->int32 narrowing refused: non-finite float "
                    f"cannot narrow to int32 range [{_INT32_MIN}, {_INT32_MAX}]. "
                    "Fix: pass finite values within int32 range, use dtype='int64' "
                    "for BIGINT int64-capable ops, or scaled-int "
                    "via Schema (scale/offset)."
                )
            lo, hi = float(arr.min()), float(arr.max())
        elif kind == "b":
            return None
        else:
            try:
                lo, hi = int(arr.min()), int(arr.max())
            except (TypeError, ValueError):
                return None
    except TypeError:
        return None
    if lo < _INT32_MIN or hi > _INT32_MAX:
        bad = lo if lo < _INT32_MIN else hi
        raise OverflowError(
            f"{what}: int64->int32 narrowing overflow: observed range [{lo}, {hi}] "
            f"exceeds int32 range [{_INT32_MIN}, {_INT32_MAX}] "
            f"(offending value {bad}). Fix: keep values within int32 range, "
            "pass dtype='int64' for BIGINT int64-capable ops, or "
            "use scaled-int via Schema (scale/offset)."
        )
    return None


def canonical_dtype(name):
    """canonical_dtype(name) -> {itemsize, logical, accum} (+width/kind for packed).

    Packed dtypes carry bit-width info; numeric entries unchanged (frozen shape).
    """
    if isinstance(name, str) and name.startswith("enum:"):
        try:
            width = int(name.split(":")[1])
        except (IndexError, ValueError):
            width = -1
        if width not in _ENUM_WIDTHS:
            raise format_error(
                f"unknown dtype '{name}': enum width must be one of "
                f"{list(_ENUM_WIDTHS)} (e.g. 'enum:2'). "
                "int64 logical allowed for BIGINT int64-capable ops "
                "(group/filter/sort/compare/agg-payload, stays int64)",
                fix="pass dtype='bool'/'enum:2'/'enum:4'/'enum:8'",
            )
        return {"itemsize": width / 8, "logical": name, "accum": "int64",
                "width": width, "kind": "enum"}
    info = _TABLE.get(name)
    if info is None:
        raise format_error(
            f"unknown dtype '{name}': use int32/int64 logical (or float32/float64). "
            "int64 stays int64 on int64-capable ops; int32 narrowing is "
            "range-checked (OverflowError, never wrap)",
            fix="pass dtype='int32'/'int64'/'float32'/'float64'",
        )
    return dict(info)


def accum_dtype(logical):
    """accum_dtype(logical) -> wider accumulator (int32->int64, int64->int64, float32->float64)."""
    return canonical_dtype(logical)["accum"]


def is_scaled(scale=1, offset=0):
    """is_scaled(scale, offset) -> True if scaled-int representation applies."""
    return not (scale == 1 and offset == 0)


def check_overflow(values, scale=1, offset=0):
    """check_overflow(values, scale, offset) -> None; raises if int32 overflows.

    Validates physical range of scaled-int encoding before materializing.
    """
    if not is_scaled(scale, offset):
        return None
    import numpy as np

    phys = np.rint((np.asarray(values, dtype=np.float64) - offset) / scale)
    if phys.size and (phys.min() < -(2**31) or phys.max() > 2**31 - 1):
        raise format_error(
            f"scaled-int overflow: physical range [{phys.min()}, {phys.max()}] "
            "exceeds int32",
            fix="choose larger scale or smaller offset",
        )
    return None
