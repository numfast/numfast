"""GpuBufferPool — GPU buffer lifecycle manager for WebGpuDriver.

refcount -> reuse -> free-list -> evict.

Map: id(numpy_raw) -> _Entry(buffer, size_bytes, refcount, last_used)
The mapping lives BETWEEN execute_fused() calls, so repeated calls with the
same input array share one GPU buffer (no re-upload).

- acquire(raw, size_bytes, upload_fn): if the raw is already materialized on
  the GPU, bump refcount and return the buffer (upload_fn NOT called).
  Otherwise take a buffer from the free-list (bucket by power-of-2 size) or
  create a new one, then upload data once.
- alloc(raw, size_bytes): reserve a GPU buffer for an output raw. Reuses the
  mapping if present (the shader overwrites it), else takes from the free-list
  or creates a new one. refcount = 1.
- release(raw): refcount-- ; at 0 the buffer stays resident (cached).
- finalize(raw_ids=None): end of an execution call — reset refcounts to 0
  (buffers become cached, last_used bumped), then evict if over the limit.
- get(raw): mapped buffer or None (for readback).
- stats(): counters for reporting (uploads, allocs, evicts, resident_bytes).
- destroy(): free everything (driver.release()).

All buffers are created with STORAGE | COPY_SRC | COPY_DST so every buffer can
be reused from the free-list (COPY_DST allows overwriting a reused buffer).
"""

import time
from dataclasses import dataclass

import numpy as np
import wgpu

_MIN_BUCKET = 1024
_FREE_SEARCH_STEPS = 16
_USAGE = (
    wgpu.BufferUsage.STORAGE
    | wgpu.BufferUsage.COPY_SRC
    | wgpu.BufferUsage.COPY_DST
)


@dataclass
class _Entry:
    """One mapped or free GPU buffer.

    buf: wgpu.GPUBuffer
    raw: the numpy raw object this buffer belongs to (identity check,
        prevents id() reuse collisions)
    size_bytes: logical size of the buffer in this mapping
    refcount: 0 = cached (evictable), >0 = in use
    last_used: monotonic clock, bumped on every finalize/release-to-zero
    bucket: allocated size (power of 2, >= _MIN_BUCKET) — true byte cost
    """
    buf: object
    raw: object
    size_bytes: int
    refcount: int
    last_used: int
    bucket: int


class GpuBufferPool:
    """GPU buffer lifecycle manager: refcount -> reuse -> release."""

    def __init__(self, device, queue, max_bytes: int = 512 * 1024 * 1024):
        self._device = device
        self._queue = queue
        self.max_bytes = max_bytes
        self._map: dict[int, _Entry] = {}
        self._free: dict[int, list[_Entry]] = {}
        self._clock: int = 0
        self.upload_count: int = 0
        self.alloc_count: int = 0
        self.evict_count: int = 0
        self.destroy_count: int = 0

    # ── helpers ─────────────────────────────────────────────

    @staticmethod
    def _bucket(size_bytes: int) -> int:
        """Round up to a power of 2, min _MIN_BUCKET."""
        b = 1 << (size_bytes - 1).bit_length()
        return b if b >= _MIN_BUCKET else _MIN_BUCKET

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @property
    def total_resident_bytes(self) -> int:
        total = 0
        for e in self._map.values():
            total += e.bucket
        for lst in self._free.values():
            for e in lst:
                total += e.bucket
        return total

    def _pop_free(self, size_bytes: int):
        """Take the first free buffer that fits; search up the bucket chain.

        The bucket is a power of 2 rounded UP, so an entry can be physically
        smaller than the requested size (e.g. 25888768 bytes in the 32MB
        bucket). Such entries are skipped and the search continues — a buffer
        smaller than size_bytes would overflow the WGSL binding.
        """
        b = self._bucket(size_bytes)
        for _ in range(_FREE_SEARCH_STEPS):
            lst = self._free.get(b)
            if lst:
                for i, entry in enumerate(lst):
                    if entry.size_bytes < size_bytes:
                        continue  # actual size too small, keep looking
                    del lst[i]
                    for e in self._map.values():
                        if e.buf is entry.buf:
                            raise RuntimeError(
                                "pool invariant violated: free buffer still mapped")
                    return entry
            b <<= 1
        return None

    def _push_free(self, entry: _Entry):
        self._free.setdefault(entry.bucket, []).append(entry)

    def _reuse_free(self, entry: _Entry, raw, size_bytes: int) -> _Entry:
        """Rebind a free-list buffer to a new raw."""
        entry.raw = raw
        entry.size_bytes = size_bytes
        entry.refcount = 1
        entry.last_used = 0
        return entry

    # ── public API ──────────────────────────────────────────

    def acquire(self, raw, size_bytes: int, upload_fn) -> object:
        """Input buffer: reuse mapping if present, else upload once.

        upload_fn: Callable[[], bytes] — called only when real data must be
        transferred (first upload of this raw, or fill of a reused buffer).
        """
        key = id(raw)
        e = self._map.get(key)
        if e is not None and e.raw is raw and e.size_bytes >= size_bytes:
            e.refcount += 1
            return e.buf

        data = upload_fn()
        free_e = self._pop_free(size_bytes)
        if free_e is not None:
            self._queue.write_buffer(free_e.buf, 0, data)
            entry = self._reuse_free(free_e, raw, size_bytes)
            self.upload_count += 1
        else:
            buf = self._device.create_buffer_with_data(data=data, usage=_USAGE)
            entry = _Entry(buf=buf, raw=raw, size_bytes=size_bytes,
                           refcount=1, last_used=0, bucket=self._bucket(size_bytes))
            self.upload_count += 1
            self.alloc_count += 1
        self._map[key] = entry
        return entry.buf

    def alloc(self, raw, size_bytes: int, zero: bool = False) -> object:
        """Output buffer: reuse mapping, free-list or create (no upload).

        zero=True: reused buffers are cleared before binding (accumulating
        kernels with partial writes need a clean buffer). Fresh buffers are
        guaranteed zero (wgpu create_buffer initializes to 0) and are never
        cleared. zero=False: reused buffers are returned as-is — the shader
        fully overwrites them, stale contents are fine.
        """
        key = id(raw)
        e = self._map.get(key)
        if e is not None and e.raw is raw and e.size_bytes >= size_bytes:
            e.refcount = 1
            return e.buf

        free_e = self._pop_free(size_bytes)
        if free_e is not None:
            if zero:
                self._queue.write_buffer(
                    free_e.buf, 0, np.zeros(free_e.buf.size, dtype=np.uint8))
            entry = self._reuse_free(free_e, raw, size_bytes)
        else:
            buf = self._device.create_buffer(size=size_bytes, usage=_USAGE)
            entry = _Entry(buf=buf, raw=raw, size_bytes=size_bytes,
                           refcount=1, last_used=0, bucket=self._bucket(size_bytes))
            self.alloc_count += 1
        self._map[key] = entry
        return entry.buf

    def get(self, raw):
        """Mapped buffer for readback, or None."""
        e = self._map.get(id(raw))
        if e is not None and e.raw is raw:
            return e.buf
        return None

    def release(self, raw):
        """refcount-- ; at 0 the buffer becomes cached (stays resident)."""
        e = self._map.get(id(raw))
        if e is not None and e.raw is raw:
            e.refcount = max(0, e.refcount - 1)
            if e.refcount == 0:
                e.last_used = self._tick()

    def finalize(self, raw_ids=None):
        """End of an execution call: reset refcounts to 0 (cached), evict.

        raw_ids: optional set[int] of touched raw ids; None = all mapped.
        """
        if raw_ids is None:
            for e in self._map.values():
                e.refcount = 0
                e.last_used = self._tick()
        else:
            for key in raw_ids:
                e = self._map.get(key)
                if e is not None:
                    e.refcount = 0
                    e.last_used = self._tick()
        self._evict()

    def _evict(self):
        """Keep resident memory <= limit: cache->free-list, then destroy old.

        Evicts only cached (refcount == 0) entries, oldest last_used first.
        """
        target = int(self.max_bytes * 0.9)
        while self.total_resident_bytes > target:
            cached = [e for e in self._map.values() if e.refcount == 0]
            if not cached:
                break
            oldest = min(cached, key=lambda e: e.last_used)
            self._map.pop(id(oldest.raw))
            self._push_free(oldest)
            self.evict_count += 1
        while self.total_resident_bytes > target and self._free:
            bucket = min(self._free)
            lst = self._free[bucket]
            if not lst:
                del self._free[bucket]
                continue
            entry = lst.pop()
            entry.buf.destroy()
            self.destroy_count += 1

    def stats(self) -> dict:
        """Counters for reporting (ASCII names)."""
        return {
            "uploads": self.upload_count,
            "allocs": self.alloc_count,
            "evicts": self.evict_count,
            "destroys": self.destroy_count,
            "resident_bytes": self.total_resident_bytes,
            "mapped": len(self._map),
            "cached": sum(1 for e in self._map.values() if e.refcount == 0),
            "free": sum(len(lst) for lst in self._free.values()),
        }

    def destroy(self):
        """Release all GPU buffers (driver.release())."""
        for e in self._map.values():
            e.buf.destroy()
        for lst in self._free.values():
            for e in lst:
                e.buf.destroy()
        self._map.clear()
        self._free.clear()
