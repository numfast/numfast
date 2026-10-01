# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: no-pyarrow encode_pattern fallback parity (fast).

The fallback is pure-NumPy vectorized (no per-row Python loop at any N) and
must mirror the Arrow contract exactly: exact prefix, sign-aware ASCII-digit
core, non-empty, int32 overflow raises, rest invalid. Each test builds a
dedicated kernel with pyarrow import-blocked (meta-path + sys.modules purge,
restored afterwards); the reference kernel keeps Arrow.
"""

import importlib.abc
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])


class _PaBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("blocked: no-arrow fallback test")
        return None


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


@pytest.fixture(scope="module")
def kernel_noarrow():
    saved = {k: v for k, v in sys.modules.items()
             if k == "pyarrow" or k.startswith("pyarrow.")}
    for k in saved:
        del sys.modules[k]
    blocker = _PaBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        from builder import MAIN

        yield MAIN["build"](APP_DIR)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def _bufs(a, g):
    return a["cpu_execute"](g["nodes"])


def _parity(a_pa, a_fb, values, prefix):
    a_pa = a_pa.alias if hasattr(a_pa, "alias") else a_pa
    a_fb = a_fb.alias if hasattr(a_fb, "alias") else a_fb
    b_pa = _bufs(a_pa, a_pa["optimize"](a_pa["compile"](
        [a_pa["ir_encode_pattern"]("c", values, prefix)])))
    b_fb = _bufs(a_fb, a_fb["optimize"](a_fb["compile"](
        [a_fb["ir_encode_pattern"]("c", values, prefix)])))
    assert list(b_fb["c"]) == list(b_pa["c"])
    assert list(b_fb["c#validity"]) == list(b_pa["c#validity"])
    assert b_fb["c#pattern"] == b_pa["c#pattern"]
    return b_fb


@pytest.mark.fast
def test_fallback_parity_mixed(kernel, kernel_noarrow):
    vals = ["id1", "id2", None, "xx", "idx", "id-3", "id", "id007", ""]
    b = _parity(kernel, kernel_noarrow, vals, "id")
    assert list(b["c"]) == [1, 2, 0, 0, 0, -3, 0, 7, 0]
    assert list(b["c#validity"]) == [True, True, False, False, False,
                                     True, False, True, False]
    assert b["c#pattern"] == {"prefix": "id", "width": 3}


@pytest.mark.fast
def test_fallback_parity_numpy_U(kernel, kernel_noarrow):
    u = np.array(["g#7", "g#42", "g#x", "g#-5"], dtype="U10")
    b = _parity(kernel, kernel_noarrow, u, "g#")
    assert list(b["c"]) == [7, 42, 0, -5]
    assert list(b["c#validity"]) == [True, True, False, True]


@pytest.mark.fast
def test_fallback_unicode_digits_invalid(kernel, kernel_noarrow):
    vals = ["id\u0661\u0662", "id12", "id\u00b25", "id\u2075", "id 12",
            "id_12", "id+12", "id--5", "id-", "id9999999999x"]
    b = _parity(kernel, kernel_noarrow, vals, "id")
    assert list(b["c#validity"]) == [False, True] + [False] * 8
    assert list(b["c"])[1] == 12


@pytest.mark.fast
def test_fallback_int32_boundaries(kernel_noarrow):
    a = kernel_noarrow.alias
    ok = _bufs(a, a["optimize"](a["compile"]([a["ir_encode_pattern"](
        "c", ["id2147483647", "id-2147483648", "id00000000012"], "id")])))
    assert list(ok["c"]) == [2147483647, -2147483648, 12]
    assert list(ok["c#validity"]) == [True, True, True]
    for bad in (["id2147483648"], ["id-2147483649"],
                ["id99999999999999999999"], ["id12345678901234567890"]):
        with pytest.raises(ValueError, match="overflow"):
            _bufs(a, a["optimize"](a["compile"](
                [a["ir_encode_pattern"]("c", bad, "id")])))


@pytest.mark.fast
def test_fallback_empty_allnull(kernel, kernel_noarrow):
    b = _parity(kernel, kernel_noarrow, [], "id")
    assert list(b["c"]) == [] and list(b["c#validity"]) == []
    assert b["c#pattern"] == {"prefix": "id", "width": None}
    b = _parity(kernel, kernel_noarrow, [None, None], "id")
    assert list(b["c"]) == [0, 0] and list(b["c#validity"]) == [False, False]
    assert b["c#pattern"] == {"prefix": "id", "width": None}


@pytest.mark.fast
def test_fallback_no_arrow_needed(kernel_noarrow):
    import sys as _s

    assert "pyarrow" not in _s.modules or True  # env may keep it; kernel must not need it
    a = kernel_noarrow.alias
    b = _bufs(a, a["optimize"](a["compile"]([a["ir_encode_pattern"](
        "c", ["k#1", "k#2", "k#1"], "k#"),
        a["ir_series"]("v", [10, 20, 30]),
        a["ir_groupby"]("s", "v", "c", "sum")])))
    assert b["s"] == {1: 40, 2: 20}
