# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""P0-L Loader functional S121 — empty N=0, N=1, small, chunk boundaries, N=86400 real, signed delta, corrupted, round-trip."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))

import tempfile
import csv
import io
import numpy as np
import pyzstd

from Loader._lib.dzst import load_dzst
from Storage._lib.packing import compute_layout, pack_rows, extract_column

DATA_DIR = os.environ.get("NUMFAST_DATA_DIR", "")
REAL = (
    os.path.join(DATA_DIR, "bybit", "BTCUSDT", "BTCUSDT_d1_2024-01-01.csv.zst")
    if DATA_DIR else ""
)


def _write_dzst(path, rows_deltas, power=5):
    """Write deltas (N,6) to .csv.zst. rows_deltas are delta values forcols 0-3, volumes absolute."""
    header = "Open,High,Low,Close,Buy_Volume,Sell_Volume\n"
    lines = [header]
    for r in rows_deltas:
        # write blank for 0 to mimic real file null handling, but keep 0 also valid
        parts = []
        for v in r:
            if v == 0:
                parts.append("")
            else:
                parts.append(str(int(v)))
        lines.append(",".join(parts) + "\n")
    data = "".join(lines).encode("utf-8")
    comp = pyzstd.compress(data, 3)
    with open(path, "wb") as f:
        f.write(comp)


def _abs_to_deltas(abs_arr):
    """Convert absolute (N,6) ticks to delta rows for writing."""
    n = abs_arr.shape[0]
    if n == 0:
        return np.empty((0, 6), dtype=np.int64)
    deltas = np.zeros((n, 6), dtype=np.int64)
    # Open, High, Low, Close are in 0..3, volumes 4..5 absolute
    deltas[0, 0] = abs_arr[0, 0]  # Open[0] - 0
    deltas[0, 1] = abs_arr[0, 1] - abs_arr[0, 0]
    deltas[0, 2] = abs_arr[0, 2] - abs_arr[0, 1]
    deltas[0, 3] = abs_arr[0, 3] - abs_arr[0, 2]
    deltas[0, 4] = abs_arr[0, 4]
    deltas[0, 5] = abs_arr[0, 5]
    for i in range(1, n):
        prev_close = abs_arr[i-1, 3]
        deltas[i, 0] = abs_arr[i, 0] - prev_close
        deltas[i, 1] = abs_arr[i, 1] - abs_arr[i, 0]
        deltas[i, 2] = abs_arr[i, 2] - abs_arr[i, 1]
        deltas[i, 3] = abs_arr[i, 3] - abs_arr[i, 2]
        deltas[i, 4] = abs_arr[i, 4]
        deltas[i, 5] = abs_arr[i, 5]
    return deltas


def _make_abs(n, seed=42, base=10000, power=1):
    rng = np.random.RandomState(seed)
    abs_arr = np.zeros((n, 6), dtype=np.int64)
    if n == 0:
        return abs_arr
    # random walk for close
    close = base
    for i in range(n):
        # generate OHLC around close
        o = close + rng.randint(-5, 6)
        h = max(o, close) + rng.randint(0, 5)
        l = min(o, close) - rng.randint(0, 5)
        c = l + rng.randint(0, h - l + 1) if h > l else l
        # ensure >=1
        o = max(1, o); h = max(1, h); l = max(1, l); c = max(1, c)
        # ensure high >= low
        if h < l:
            h, l = l, h
        abs_arr[i, 0] = o
        abs_arr[i, 1] = h
        abs_arr[i, 2] = l
        abs_arr[i, 3] = c
        abs_arr[i, 4] = rng.randint(0, 1000)
        abs_arr[i, 5] = rng.randint(0, 1000)
        close = c
    return abs_arr


def test_empty_N0():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "BTCUSDT_d5_2024-01-01.csv.zst")
        abs_arr = np.empty((0, 6), dtype=np.int64)
        deltas = _abs_to_deltas(abs_arr)
        _write_dzst(p, deltas)
        arr, mult, power = load_dzst(p)
        assert arr.shape == (0, 6)
        assert arr.dtype == np.int64
        assert mult == 100000
        assert power == 5


def test_N1():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "BTCUSDT_d5_2024-01-01.csv.zst")
        abs_arr = _make_abs(1, seed=42)
        deltas = _abs_to_deltas(abs_arr)
        _write_dzst(p, deltas)
        arr, mult, power = load_dzst(p)
        assert arr.shape == (1, 6)
        assert np.array_equal(arr, abs_arr)


def test_small():
    for n in (2, 3, 10, 63):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "BTCUSDT_d1_2024-01-01.csv.zst")
            abs_arr = _make_abs(n, seed=42)
            deltas = _abs_to_deltas(abs_arr)
            _write_dzst(p, deltas)
            arr, mult, power = load_dzst(p)
            assert arr.shape[0] == n
            assert np.array_equal(arr, abs_arr), f"mismatch n={n}"


def test_chunk_boundaries():
    for n in (64, 65, 256, 1024):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, f"TEST_d2_2024-01-01.csv.zst")
            abs_arr = _make_abs(n, seed=42, base=5000)
            deltas = _abs_to_deltas(abs_arr)
            _write_dzst(p, deltas)
            arr, mult, power = load_dzst(p)
            assert arr.shape[0] == n
            assert np.array_equal(arr, abs_arr)
            assert mult == 100


def test_N86400_real():
    real = REAL
    if not os.path.exists(real):
        # fallback: skip if not found but still check synthetic 86400
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "BTCUSDT_d1_2024-01-01.csv.zst")
            abs_arr = _make_abs(86400, seed=42)
            deltas = _abs_to_deltas(abs_arr)
            _write_dzst(p, deltas)
            arr, mult, power = load_dzst(p)
            assert arr.shape[0] == 86400
            assert np.array_equal(arr, abs_arr)
        return
    arr, mult, power = load_dzst(real)
    assert arr.shape[0] == 86400
    assert arr.shape[1] == 6
    assert mult == 10
    assert power == 1
    # signed delta invariants: high >= low
    assert (arr[:, 1] >= arr[:, 2]).all()
    # prices >=1
    assert (arr[:, :4] >= 1).all()


def test_signed_delta():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "BTCUSDT_d5_2024-01-01.csv.zst")
        # craft case with negative deltas: price goes down
        abs_arr = np.array([
            [100, 105, 95, 100, 10, 20],
            [100, 102, 98, 99, 5, 5],  # Open 100 from prev Close 100, High 102, Low 98, Close 99 (down)
            [99, 99, 90, 95, 1, 2],   # down further, Low delta negative large
        ], dtype=np.int64)
        deltas = _abs_to_deltas(abs_arr)
        # ensure Low delta is negative
        assert (deltas[:, 2] <= 0).all()
        _write_dzst(p, deltas)
        arr, _, _ = load_dzst(p)
        assert np.array_equal(arr, abs_arr)


def test_corrupted_truncated():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "BTCUSDT_d5_2024-01-01.csv.zst")
        abs_arr = _make_abs(10, seed=42)
        deltas = _abs_to_deltas(abs_arr)
        _write_dzst(p, deltas)
        # truncate file
        with open(p, "rb") as f:
            data = f.read()
        with open(p, "wb") as f:
            f.write(data[:10])
        try:
            load_dzst(p)
            assert False, "should raise ValueError"
        except ValueError as e:
            msg = str(e)
            assert "corrupted file" in msg or "truncated" in msg or "dX out of i32 range" in msg
        except FileNotFoundError:
            assert False


def test_corrupted_dX_negative_price():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "BTCUSDT_d5_2024-01-01.csv.zst")
        # Create corrupted deltas that decode to negative price (<1) or high<low
        # We write deltas that will make Low > High after decode
        # Craft absolute that violates high<low -> loader should detect and raise
        # Instead craft deltas directly that are inconsistent: set Low delta positive (should be <=0)
        # That will make Low > High
        deltas = np.array([[100, 5, 10, 5, 0, 0]], dtype=np.int64)  # Low delta 10 positive -> Low = 115 > High 105
        _write_dzst(p, deltas)
        try:
            load_dzst(p)
            assert False, "should raise"
        except ValueError as e:
            assert "corrupted file" in str(e) or "dX out of i32 range" in str(e)


def test_roundtrip_load_pack_unpack_exact():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "BTCUSDT_d5_2024-01-01.csv.zst")
        n = 1000
        abs_arr = _make_abs(n, seed=42)
        deltas = _abs_to_deltas(abs_arr)
        _write_dzst(p, deltas)
        arr, mult, power = load_dzst(p)
        assert np.array_equal(arr, abs_arr)
        # pack via Storage: use low, dHigh etc
        # schema for Storage packing: low (int64 32), d_high/d_open/d_close 16 each, volumes 32
        # Build data dict for packing: need to compute deltas for packing separate from loader deltas
        # Pack low absolute and delta columns
        low = arr[:, 2].tolist()  # Low
        d_high = (arr[:, 1] - arr[:, 2]).tolist()
        d_open = (arr[:, 0] - np.concatenate([[0], arr[:-1, 3]])).tolist() if n>0 else []
        # but for pack test we just pack low and volumes exact
        schema = [
            {"name": "low", "dtype": "int64", "bits": 32},
            {"name": "d_high", "dtype": "int64", "bits": 16},
            {"name": "buy_vol", "dtype": "int64", "bits": 32},
        ]
        data = {"low": low, "d_high": d_high, "buy_vol": arr[:, 4].tolist()}
        res = pack_rows(schema, data)
        layout = res["layout"]
        vals_low = extract_column(res["rows"], "low", layout)
        vals_dh = extract_column(res["rows"], "d_high", layout)
        assert vals_low == low
        assert vals_dh == d_high
        # high == low + dHigh globally
        high_recon = [l + dh for l, dh in zip(vals_low, vals_dh)]
        assert high_recon == arr[:, 1].tolist()
