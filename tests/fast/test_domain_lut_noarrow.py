# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: DomainLUT imports and builds on an Arrow-less install (fast).

pyarrow is an optional dependency of the engine, and DomainLUT was the one
extension that broke that promise: ``import pyarrow`` sat unguarded at module
level in _lib/carrier.py and _lib/text_lut.py, so a build without pyarrow
died on DomainLUT -- before the entry file could register a single alias --
with a bare ModuleNotFoundError from the bottom of a stack trace.

The contract these tests pin down:

  * import + build succeed with no pyarrow, and all five DomainLUT aliases
    register exactly as they do with pyarrow;
  * the NumPy-only surface (``codes_lut_mask``) keeps working there -- it
    never touches Arrow, so refusing it would misreport the install;
  * the Arrow-dependent TEXT predicates refuse at CALL time, with a message
    that names pyarrow and the install command -- not a NameError or a
    ModuleNotFoundError leaking out of a private.

Absence is simulated with the project's meta-path blocker + sys.modules purge
(same mechanism as tests/fast/test_ops_noarrow_fallback.py). The blocker is
installed per TEST, not per module: a module-scoped one would still be
active when the pyarrow-path fixture runs, and that kernel would silently be
an Arrow-less one. The pyarrow path is asserted against an unblocked kernel
so the guard cannot pass by breaking the normal install.
"""

import importlib.abc
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])

# One representative call per Arrow-dependent TEXT predicate.
ARROW_CALLS = {
    "dict_contains_lut": lambda a: a["dict_contains_lut"](["abc", "abd"], "ab"),
    "dict_not_contains_lut": lambda a: a["dict_not_contains_lut"](["abc", "abd"], "ab"),
    "dict_equal_lut": lambda a: a["dict_equal_lut"](["abc", "abd"], "abc"),
    "dict_not_equal_lut": lambda a: a["dict_not_equal_lut"](["abc", "abd"], "abc"),
}
# The NumPy-only row-side mask, which must survive an Arrow-less install.
NUMPY_ONLY_ALIASES = ("codes_lut_mask",)
DOMAIN_LUT_ALIASES = tuple(ARROW_CALLS) + NUMPY_ONLY_ALIASES


class _PaBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("blocked: no-arrow fallback test")
        return None


@pytest.fixture(scope="module")
def kernel():
    """Normal install: pyarrow present, nothing blocked."""
    from builder import MAIN

    return MAIN["build"](APP_DIR)


@pytest.fixture
def kernel_noarrow():
    """Arrow-less install: pyarrow unimportable for the whole build."""
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


@pytest.mark.fast
def test_noarrow_build_registers_every_alias(kernel_noarrow):
    """The build completes and DomainLUT registers all five aliases."""
    assert "pyarrow" not in sys.modules
    for name in DOMAIN_LUT_ALIASES:
        assert name in kernel_noarrow.alias, f"{name} missing on an Arrow-less build"
        assert callable(kernel_noarrow.alias[name])


@pytest.mark.fast
def test_noarrow_row_mask_is_numpy_only(kernel_noarrow):
    """codes_lut_mask never touches Arrow, so it must keep working."""
    a = kernel_noarrow.alias
    codes = np.array([0, 1, 2, 0], dtype=np.int32)
    lut = np.array([True, False, True])
    assert list(a["codes_lut_mask"]([codes], [lut], None)) == [True, False, True, True]
    # the all-true LUT drops the gather; the NULL gate still removes its row
    assert list(a["codes_lut_mask"]([codes], [np.ones(3, bool)],
                                    [np.array([True, True, False, True])])) == \
        [True, True, False, True]
    # an empty LUT rejects every row, it is not an error
    assert not a["codes_lut_mask"]([codes], [np.zeros(0, bool)], None).any()


@pytest.mark.fast
def test_noarrow_text_predicate_refuses_clearly(kernel_noarrow):
    """Arrow-dependent predicates fail at call time, naming pyarrow."""
    a = kernel_noarrow.alias
    for name, call in ARROW_CALLS.items():
        with pytest.raises(ValueError, match="pyarrow") as ei:
            call(a)
        msg = str(ei.value)
        # the message is about the install, not a leaked NameError/ModuleNotFound
        assert name in msg, f"{name} is not named in its own refusal"
        assert "pip install pyarrow" in msg
        assert "NameError" not in msg and "ModuleNotFoundError" not in msg


@pytest.mark.fast
def test_with_pyarrow_unchanged(kernel):
    """With pyarrow the guard is invisible: the full contract still holds."""
    a = kernel.alias
    values = ["abc", "abd", "xyz"]
    assert list(a["dict_contains_lut"](values, "ab")) == [True, True, False]
    assert list(a["dict_not_contains_lut"](values, "ab")) == [False, False, True]
    assert list(a["dict_equal_lut"](values, "abc")) == [True, False, False]
    assert list(a["dict_not_equal_lut"](values, "abc")) == [False, True, True]
    # the forced body kernel on an Arrow carrier still refuses, as before
    with pytest.raises(ValueError, match="no native"):
        a["dict_contains_lut"](values, "ab", kernel="body")
    codes = np.array([0, 1, 2, 0], dtype=np.int32)
    assert list(a["codes_lut_mask"]([codes], [np.array([True, False, True])], None)) == \
        [True, False, True, True]
