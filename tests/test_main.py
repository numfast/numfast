# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the _main (core) module."""

from _main._lib.main_lib import _get_version, _get_status


def test_get_version_returns_string():
    v = _get_version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_get_status_contains_extensions():
    s = _get_status()
    assert s["name"] == "numfast"
    assert "_main" in s["extensions"]
    assert "Series" in s["extensions"]
    assert "Tables" in s["extensions"]
    assert "Stats" in s["extensions"]
    assert "Mods" in s["extensions"]
    assert "Proxy" in s["extensions"]
