from Memory._lib.memory_lib import _CacheStore, _CacheEntry
import time

_cache = _CacheStore()

def cached_calculation(key: str, func, ttl: int = 60):
    now = time.time()
    entry = _cache.get(key)
    if entry is not None and (now - entry.timestamp) < ttl:
        return entry.value
    result = func()
    _cache.put(key, _CacheEntry(result, now))
    return result

def clear_expired():
    now = time.time()
    expired = [k for k, v in _cache._store.items() if (now - v.timestamp) >= 60]
    for k in expired:
        _cache.delete(k)

def stats() -> dict:
    return {
        "total_entries": len(_cache._store),
        "keys": list(_cache._store.keys()),
    }
