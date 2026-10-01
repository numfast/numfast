# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU standalone Scan/Reduction golden: portable WGSL prefix scan (int32/uint32).

Hybrid default (GPU block-scan + host W-prefix + GPU fixup) vs full-device
resident variant: bit-exact each other and vs wraparound reference.
Gates: exact vs numpy cumsum/sum; empty; 1 element; non-power-of-two;
0/1/all patterns; big N; inclusive/exclusive; overflow wrap policy explicit.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def G():
    spec = importlib.util.spec_from_file_location(
        "nfgpu_scan", str(APP_DIR / "src" / "Drivers" / "GPU" / "_lib"
                          / "gpu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.fast
def test_scan_matches_numpy_cumsum(G):
    rng = np.random.default_rng(42)
    x = rng.integers(-5000, 5000, 100000, dtype=np.int32)
    ref64 = np.cumsum(x.astype(np.int64))
    wrap = (ref64 % 2 ** 32).astype(np.uint32).view(np.int32)
    assert (G.scan_inclusive(x) == wrap).all()
    assert (G.scan_inclusive(x) == G.scan_ref(x, "int32", True)).all()
    assert (G.scan_exclusive(x) == G.scan_ref(x, "int32", False)).all()
    assert (G.scan_exclusive_gpu(x) == G.scan_ref(x, "int32", False)).all()
    assert G.reduce_full(x, "int32", "sum") == int(
        x.astype(np.int64).sum() % 2 ** 32 - 2 ** 32 * (
            int(x.astype(np.int64).sum() % 2 ** 32) >= 2 ** 31))
    assert G.reduce_full(x, "int32", "min") == int(x.min())
    assert G.reduce_full(x, "int32", "max") == int(x.max())
    assert G.reduce_full(x, "int32", "count") == x.size


@pytest.mark.fast
def test_scan_200_random_exact(G):
    """200 random cases (seed 42): dtypes x modes x incl/excl, exact."""
    rng = np.random.default_rng(42)
    for t in range(200):
        n = int(rng.integers(0, 900))
        if t % 2 == 0:
            x = rng.integers(-10000, 10000, n, dtype=np.int32)
            dt = "int32"
        else:
            x = rng.integers(0, 2 ** 32 - 1, n, dtype=np.uint32)
            dt = "uint32"
        assert (G.scan_inclusive(x, dt) == G.scan_ref(x, dt, True)).all(), t
        assert (G.scan_exclusive(x, dt) == G.scan_ref(x, dt, False)).all(), t
        assert (G.scan_exclusive_gpu(x, dt)
                == G.scan_ref(x, dt, False)).all(), t
        ops = ("sum", "count") if n == 0 else ("sum", "min", "max", "count")
        for op in ops:
            assert G.reduce_full(x, dt, op) == G.reduce_ref(x, dt, op), (t, op)


@pytest.mark.fast
def test_scan_edges(G):
    z = np.zeros(0, dtype=np.int32)
    assert G.scan_inclusive(z).size == 0
    assert G.scan_exclusive(z).size == 0
    assert G.scan_exclusive_gpu(z).size == 0
    assert G.reduce_full(z, "int32", "sum") == 0
    assert G.reduce_full(z, "int32", "count") == 0
    with pytest.raises(ValueError, match="empty"):
        G.reduce_full(z, "int32", "min")
    with pytest.raises(ValueError, match="empty"):
        G.reduce_full(z, "int32", "max")
    one = np.array([7], dtype=np.int32)
    assert G.scan_inclusive(one).tolist() == [7]
    assert G.scan_exclusive(one).tolist() == [0]
    assert G.scan_exclusive_gpu(one).tolist() == [0]
    assert G.reduce_full(one, "int32", "sum") == 7
    for n in (3, 5, 255, 256, 257, 511, 512, 513, 1000, 4095, 4096, 4097):
        x = np.arange(1, n + 1, dtype=np.int32)
        assert (G.scan_inclusive(x) == np.cumsum(
            x.astype(np.int64)).astype(np.int32)).all(), n
        assert (G.scan_exclusive(x) == G.scan_ref(x, "int32", False)).all(), n


@pytest.mark.fast
def test_scan_patterns_01_all(G):
    n = 5000
    pats = {
        "zeros": np.zeros(n, dtype=np.int32),
        "ones": np.ones(n, dtype=np.int32),
        "all_max": np.full(n, 2147483647, dtype=np.int32),
        "all_min": np.full(n, -2147483648, dtype=np.int32),
        "alternating": np.tile(np.array([1, -1], dtype=np.int32), n // 2),
        "u32_max": np.full(n, 2 ** 32 - 1, dtype=np.uint32),
    }
    for name, x in pats.items():
        dt = "uint32" if x.dtype == np.uint32 else "int32"
        assert (G.scan_inclusive(x, dt) == G.scan_ref(x, dt, True)).all(), name
        assert (G.scan_exclusive(x, dt)
                == G.scan_ref(x, dt, False)).all(), name
        assert (G.scan_exclusive_gpu(x, dt)
                == G.scan_ref(x, dt, False)).all(), name
        for op in ("sum", "min", "max", "count"):
            assert G.reduce_full(x, dt, op) == G.reduce_ref(x, dt, op), name


@pytest.mark.fast
def test_scan_overflow_policy_wrap(G):
    """Overflow = wraparound mod 2**32 (u32 lanes), never saturate/trap."""
    x = np.full(70000, 2147483647, dtype=np.int32)  # sum overflows i32/i64? no, wraps u32
    got = G.scan_inclusive(x)
    ref = G.scan_ref(x, "int32", True)
    assert (got == ref).all()
    assert got[-1] == G.reduce_full(x, "int32", "sum") == G.reduce_ref(
        x, "int32", "sum")
    # sequential-prefix property under wrap: out[i]-out[i-1] == x[i] (mod 2**32)
    d = (got[1:].astype(np.int64) - got[:-1].astype(np.int64)) % 2 ** 32
    assert (d == (2147483647 % 2 ** 32)).all()
    assert got[0] == 2147483647


@pytest.mark.fast
def test_scan_big_n_exact(G):
    rng = np.random.default_rng(42)
    x = rng.integers(-10 ** 6, 10 ** 6, 1_000_000, dtype=np.int32)
    assert (G.scan_inclusive(x) == G.scan_ref(x, "int32", True)).all()
    assert (G.scan_exclusive_gpu(x) == G.scan_ref(x, "int32", False)).all()
    for op in ("sum", "min", "max", "count"):
        assert G.reduce_full(x, "int32", op) == G.reduce_ref(x, "int32", op)


@pytest.mark.fast
def test_scan_rejects(G):
    with pytest.raises(ValueError, match="int32/uint32"):
        G.scan_inclusive(np.ones(4, dtype=np.float32), "float32")
    with pytest.raises(ValueError, match="sum/min/max/count"):
        G.reduce_full(np.ones(4, dtype=np.int32), "int32", "mean")
    with pytest.raises(ValueError, match="16M"):
        G.scan_exclusive_gpu(np.ones(17 * 1024 * 1024, dtype=np.int32))
