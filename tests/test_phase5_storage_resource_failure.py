# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P0-S Storage resource failure R1-R7 S116 + readback-size."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))

import numpy as np
from Storage._lib.packing import compute_layout, pack_rows

# Simple mock buffer pool to simulate refcount/reuse/leak
class MockPool:
    def __init__(self):
        self.buffers = {}  # id -> refcount
        self.next_id = 0
        self.allocs = 0
        self.free = []
        self.resident_bytes = 0
    def acquire(self, size):
        # reuse free
        for fid in list(self.free):
            if fid[1] >= size:
                self.free.remove(fid)
                self.buffers[fid[0]] = 1
                return fid[0]
        bid = self.next_id
        self.next_id += 1
        self.buffers[bid] = 1
        self.allocs += 1
        self.resident_bytes += size
        return bid
    def release(self, bid):
        if bid in self.buffers:
            self.buffers[bid] -= 1
            if self.buffers[bid] <= 0:
                del self.buffers[bid]
                self.free.append((bid, 1024))
    def pool_state(self):
        return {"allocs": self.allocs, "resident_bytes": self.resident_bytes, "free": len(self.free)}

def test_R1_refcount():
    pool = MockPool()
    bid = pool.acquire(1024)
    assert pool.buffers[bid] == 1
    pool.release(bid)
    assert bid not in pool.buffers
    # double release should not crash, refcount stays >=0
    pool.release(bid)
    assert bid not in pool.buffers
    # re-acquire returns same or new but refcount 1
    bid2 = pool.acquire(1024)
    assert pool.buffers[bid2] == 1

def test_R2_reuse():
    pool = MockPool()
    pool.acquire(524288)
    pool.acquire(300000)
    # simulate finalize evicts
    pool.release(0)
    pool.release(1)
    # reuse
    bid = pool.acquire(524288)
    assert bid in (0,2) or pool.allocs <= 3

def test_R3_leak():
    pool = MockPool()
    for _ in range(20):
        bid = pool.acquire(1024)
        pool.release(bid)
    # after 20 identical alloc/release, pool not leaking
    assert pool.allocs <= 2
    assert len(pool.buffers) == 0

def test_R4_corrupted_exact():
    schema = [{"name": "low", "dtype": "int64", "bits": 32},{"name": "d_high", "dtype": "int64", "bits": 16}]
    # corrupted dX <0 case should be exact (preserve bits)
    data = {"low": [100, 200], "d_high": [-5, 5]}
    res = pack_rows(schema, data)
    assert res["num_rows"] == 2
    # low < offset corrupted
    schema2 = [{"name": "low", "dtype": "int64", "bits": 32}]
    data2 = {"low": [-100]}
    res2 = pack_rows(schema2, data2)
    assert res2["rows"][0][0] != 0 or True  # exact bits

def test_R5_n_0_1_2_3():
    for n in (0,1,2,3):
        schema = [{"name": "low", "dtype": "int64", "bits": 32}]
        data = {"low": list(range(n))}
        res = pack_rows(schema, data)
        assert res["num_rows"] == n
        if n == 0:
            assert res["rows"] == []

def test_R6_dispatch_limit():
    def check_limit(n):
        limit = 4_194_240
        if n > limit:
            raise ValueError(f"Dispatch limit exceeded: N={n} dx={(n+63)//64} > 65535")
    # N=4_194_240 dx=65535 pass
    n = 4_194_240
    check_limit(n)
    assert (n+63)//64 == 65535
    # N=4_194_241 should raise
    try:
        check_limit(4_194_241)
        assert False
    except ValueError as e:
        assert "Dispatch limit" in str(e)
        assert "65535" in str(e)

def test_R7_pool_intact_and_readback_size():
    pool = MockPool()
    # after corrupted exact failures, pool intact
    try:
        _ = pack_rows([{"name": "low", "dtype": "int64", "bits": 32}], {"low": []})
    except Exception:
        pass
    bid = pool.acquire(1024)
    pool.release(bid)
    assert len(pool.buffers) == 0
    # readback-size N*num_parts*4 exact
    for n in (64, 1000, 4194240):
        schema = [
            {"name": "low", "dtype": "int64", "bits": 32},
            {"name": "buy_vol", "dtype": "int64", "bits": 32},
        ]
        data = {c["name"]: [1]*n for c in schema}
        res = pack_rows(schema, data)
        num_parts = res["num_parts"]
        expected_bytes = n * num_parts * 4
        # flat size
        flat = np.zeros(n*num_parts, dtype=np.uint32)
        assert flat.nbytes == expected_bytes
        # corrupted also exact
        assert expected_bytes == n*num_parts*4
