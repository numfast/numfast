# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the Proxy module."""

from Proxy._lib.proxy_lib import _LazyLoader


def test_load_module():
    loader = _LazyLoader()
    mod = loader.load("math")
    assert mod.__name__ == "math"
    assert mod.sqrt(4) == 2.0


def test_load_caches():
    loader = _LazyLoader()
    m1 = loader.load("math")
    m2 = loader.load("math")
    assert m1 is m2


def test_multiple_modules():
    loader = _LazyLoader()
    loader.load("math")
    loader.load("json")
    assert "math" in loader._cache
    assert "json" in loader._cache
