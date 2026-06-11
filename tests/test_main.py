"""Тесты модуля _main (ядро)."""

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
    assert "Memory" in s["extensions"]
    assert "Mods" in s["extensions"]
    assert "Proxy" in s["extensions"]
