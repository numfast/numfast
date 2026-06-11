import time
from dataclasses import dataclass

@dataclass
class _CacheEntry:
    value: object
    timestamp: float

class _CacheStore:
    def __init__(self):
        self._store: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> _CacheEntry | None:
        return self._store.get(key)

    def put(self, key: str, entry: _CacheEntry):
        self._store[key] = entry

    def delete(self, key: str):
        self._store.pop(key, None)
