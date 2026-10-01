# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden harness: exact int, tolerance/ULP float.

Thresholds ONLY from specs/conformance-profile.toml (no hardcode).
Numeric contract (spec 08): int exact; float passes if
max_abs_diff <= max(atol, rtol*|ref|) OR ULP distance <= max_ulp.
"""

import struct
import tomllib
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parents[2] / "specs-rebuilt" / "conformance-profile.toml"


def load_profile(path=None):
    with open(path or PROFILE_PATH, "rb") as f:
        return tomllib.load(f)


def tolerance(profile, dtype):
    key = "f32" if dtype in ("f32", "float32") else "f64"
    sec = profile["tolerance"][key]
    return sec["atol"], sec["rtol"], sec["max_ulp"]


def _ordered(bits, nbits):
    mask = (1 << nbits) - 1
    sign = 1 << (nbits - 1)
    return bits ^ (mask if bits & sign else sign)


def ulp_distance(a, b, dtype):
    """ULP distance between two floats (sign-magnitude ordered)."""
    a, b = float(a), float(b)
    if dtype in ("f32", "float32"):
        ia = struct.unpack("<i", struct.pack("<f", a))[0] & 0xFFFFFFFF
        ib = struct.unpack("<i", struct.pack("<f", b))[0] & 0xFFFFFFFF
        return abs(_ordered(ia, 32) - _ordered(ib, 32))
    ia = struct.unpack("<q", struct.pack("<d", a))[0] & 0xFFFFFFFFFFFFFFFF
    ib = struct.unpack("<q", struct.pack("<d", b))[0] & 0xFFFFFFFFFFFFFFFF
    return abs(_ordered(ia, 64) - _ordered(ib, 64))


def assert_int_exact(actual, expected, label=""):
    assert int(actual) == int(expected), f"{label}: {actual!r} != {expected!r} (int must be exact)"


def assert_float_close(actual, expected, profile, dtype="f32", label=""):
    a, e = float(actual), float(expected)
    atol, rtol, max_ulp = tolerance(profile, dtype)
    diff = abs(a - e)
    if diff <= max(atol, rtol * abs(e)):
        return {"diff": diff, "mode": "tolerance"}
    u = ulp_distance(a, e, dtype)
    assert u <= max_ulp, (
        f"{label}: diff {diff} > max({atol}, {rtol}*|ref|) and {u} ULP > {max_ulp}"
    )
    return {"diff": diff, "mode": "ulp", "ulp": u}
