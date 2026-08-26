"""GpuBufferPool unit tests (mock device/queue, no GPU needed).

Covers the free-list bucket-chain logic:

1. test_reuse_same_bucket_smaller_size — regression for the overflow bug:
   alloc(25888768) -> finalize -> alloc(25952256). Both requests land in the
   same 32MB bucket; the pool must NOT hand out the smaller buffer (it would
   overflow the WGSL binding of 25952256 bytes).
2. test_reuse_exact_size — same-size reuse actually reuses the buffer
   (alloc_count does not grow).
3. test_reuse_skip_chain — alloc(150000) skips a too-small free buffer
   (70000) and takes the fitting one (200000).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))

from Runtime._lib.Drivers.WebGPU._lib.gpu_buffer_pool import GpuBufferPool, _Entry


class _MockBuf:
    def __init__(self, size):
        self.size = size
        self._destroyed = False

    def destroy(self):
        self._destroyed = True


class _MockDevice:
    def __init__(self):
        self.buffers = []

    def create_buffer_with_data(self, data, usage):
        buf = _MockBuf(len(data))
        self.buffers.append(buf)
        return buf

    def create_buffer(self, size, usage):
        buf = _MockBuf(size)
        self.buffers.append(buf)
        return buf


class _MockQueue:
    def __init__(self):
        self.writes = []

    def write_buffer(self, buf, offset, data):
        self.writes.append((buf, offset, data))


def _make_pool(max_bytes):
    return GpuBufferPool(_MockDevice(), _MockQueue(), max_bytes=max_bytes)


def test_reuse_same_bucket_smaller_size():
    """alloc(25952256) after freeing 25888768 must NOT reuse the smaller buf.

    Both sizes live in the 32MB bucket (bucket(25888768) == bucket(25952256)
    == 33554432). The freed buffer (25888768 bytes) is too small for the new
    binding (25952256 bytes); the pool must skip it and create a fresh buffer.
    """
    # target = 36MB: the 32MB bucket survives eviction, a smaller sacrificial
    # bucket absorbs the destroy phase and leaves the 25888768 buffer free.
    pool = _make_pool(max_bytes=40_000_000)
    raw1 = object()
    b1 = pool.alloc(raw1, 25888768)
    assert b1.size == 25888768
    raw_other = object()
    pool.alloc(raw_other, 3_000_000)  # pushes total over target, gets destroyed
    pool.finalize([id(raw1), id(raw_other)])
    assert any(e.buf is b1 for lst in pool._free.values() for e in lst)

    raw2 = object()
    b2 = pool.alloc(raw2, 25952256)
    assert b2.size >= 25952256  # never the smaller 25888768 buffer
    assert b2 is not b1
    assert pool.alloc_count == 3  # raw1, raw_other, raw2 (no fitting reuse)
    # the too-small buffer stays in the free-list for future smaller requests
    assert any(e.buf is b1 for lst in pool._free.values() for e in lst)


def test_reuse_exact_size():
    """alloc(65536) -> finalize -> alloc(65536) reuses the same buffer."""
    pool = _make_pool(max_bytes=1 << 30)  # no eviction interference
    raw = object()
    b1 = pool.alloc(raw, 65536)
    assert b1.size == 65536
    pool.finalize([id(raw)])

    b2 = pool.alloc(raw, 65536)
    assert b2 is b1  # mapping reuse: same buffer
    assert pool.alloc_count == 1  # reuse works, no new allocation


def test_reuse_skip_chain():
    """alloc(150000) skips free 70000 (bucket 131072) and takes 200000."""
    dev = _MockDevice()
    q = _MockQueue()
    pool = GpuBufferPool(dev, q, max_bytes=1 << 30)  # no auto-evict noise

    b_small = dev.create_buffer(size=70000, usage=0)
    b_fit = dev.create_buffer(size=200000, usage=0)
    pool._push_free(_Entry(buf=b_small, raw=None, size_bytes=70000,
                           refcount=0, last_used=0, bucket=pool._bucket(70000)))
    pool._push_free(_Entry(buf=b_fit, raw=None, size_bytes=200000,
                           refcount=0, last_used=0, bucket=pool._bucket(200000)))

    raw = object()
    buf = pool.alloc(raw, 150000)
    assert buf is b_fit  # 200000 chosen, 70000 skipped
    assert buf.size >= 150000
    # the too-small buffer is still cached in the free-list
    assert len(pool._free[pool._bucket(70000)]) == 1
    assert pool._free[pool._bucket(70000)][0].buf is b_small
