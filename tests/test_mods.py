"""Тесты модуля Mods."""

from Mods._lib.mods_lib import _Registry


def test_register_and_get():
    reg = _Registry()

    def foo():
        return 1

    reg.add("foo", foo)
    assert reg.get("foo") is foo


def test_get_missing():
    reg = _Registry()
    assert reg.get("nope") is None


def test_items():
    reg = _Registry()

    def a():
        pass

    def b():
        pass

    reg.add("a", a)
    reg.add("b", b)
    items = dict(reg.items())
    assert items["a"] is a
    assert items["b"] is b
