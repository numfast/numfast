"""Тесты модуля Memory."""

from Memory._lib.memory_lib import _CacheStore, _CacheEntry


def test_cache_store_put_get():
    store = _CacheStore()
    entry = _CacheEntry(42, 1000.0)
    store.put("key", entry)
    assert store.get("key") is entry


def test_cache_store_delete():
    store = _CacheStore()
    store.put("a", _CacheEntry(1, 1.0))
    store.delete("a")
    assert store.get("a") is None


def test_cache_store_delete_missing():
    store = _CacheStore()
    store.delete("nonexistent")
