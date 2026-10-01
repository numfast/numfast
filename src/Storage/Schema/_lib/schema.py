# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Schema semantics (spec 05 formula).

logical_value -> schema_transform() -> physical_value -> block_transform().
schema.offset is SEMANTIC (lives in column metadata, same for all blocks).
Agent must never change schema.offset for local compression.
"""

import numpy as np


def column_schema_impl(name, logical, scale, offset, auto, physical, canonical_dtype):
    """ColumnSchema: logical/physical/scale/offset + auto encoding."""
    info = canonical_dtype(logical)
    if auto or physical is None:
        if scale != 1 or offset != 0:
            phys = "int32"
        else:
            phys = info["logical"]
    else:
        phys = physical
    return {
        "name": name,
        "logical": info["logical"],
        "physical": phys,
        "scale": scale,
        "offset": offset,
        "auto": bool(auto),
        "encoding": {"type": "auto" if auto else "manual"},
    }


def schema_transform_impl(values, scale, offset, check_int32_range=None):
    """logical -> physical: (logical - offset) / scale for scaled-int, else identity.

    Invariant #1: the scaled path narrows to int32 only after a range check
    (OverflowError, never a silent wrap). The checker arrives via kernel.alias
    (Core, injected by the entry); None keeps the legacy coerce path for
    standalone use only.
    """
    v = np.asarray(values, dtype=np.float64)
    if scale == 1 and offset == 0:
        return v
    phys = np.rint((v - offset) / scale)
    if check_int32_range is not None:
        check_int32_range(phys, "schema_transform")
    return phys.astype(np.int32)


def schema_untransform_impl(physical, scale, offset):
    """physical -> logical: physical * scale + offset for scaled-int, else identity."""
    p = np.asarray(physical, dtype=np.float64)
    if scale == 1 and offset == 0:
        return p
    return p * scale + offset


def pattern_decode_impl(codes, prefix, validity=None, width=None):
    """Display-only inverse of encode_pattern (DELTA-4): codes -> strings.

    width (from encode sidecar '#pattern') restores zero-padded fixed-width
    patterns losslessly: f"{prefix}{code:0{width}d}". width=None keeps the
    legacy unpadded form. Invalid rows -> None. Never in compute paths.
    """
    out = []
    for i, c in enumerate(codes):
        if validity is not None and not validity[i]:
            out.append(None)
        elif width is not None and int(c) >= 0:
            out.append(f"{prefix}{int(c):0{width}d}")
        else:
            out.append(f"{prefix}{int(c)}")
    return out
