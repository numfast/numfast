# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import math

from _core.compression import pack_scaled, unpack_scaled, pack_enum, unpack_enum
from _core import kernel
from _core.context import create_context
from _core.series import make_series
from Stats.Stats import total, minimum, maximum, mean, var


def test_pack_scaled_int8_small_range():
    data = [10.0, 10.5, 11.0, 10.25, 9.75]
    packed, meta = pack_scaled(data, "int8")
    assert meta["dtype"] == "int8"
    assert len(packed) == len(data)
    # Verify roundtrip
    restored = unpack_scaled(packed, meta)
    for a, b in zip(data, restored):
        assert abs(a - b) < 0.02


def test_pack_scaled_int16_wide_range():
    data = [0.0, 500.0, 1000.0, 32767.0, -32768.0]
    packed, meta = pack_scaled(data, "int16")
    assert meta["dtype"] == "int16"
    restored = unpack_scaled(packed, meta)
    for a, b in zip(data, restored):
        assert abs(a - b) / max(1.0, abs(a)) < 0.01


def test_pack_scaled_auto_chooses_int16_for_large_range():
    data = [0.0, 1000.0]
    packed, meta = pack_scaled(data, "auto")
    assert meta["dtype"] == "int16"


def test_pack_scaled_auto_chooses_int8_for_small_range():
    data = [0.0, 1.0]
    packed, meta = pack_scaled(data, "auto")
    assert meta["dtype"] == "int8"


def test_pack_scaled_constant():
    data = [42.0, 42.0, 42.0]
    packed, meta = pack_scaled(data, "auto")
    assert meta["dtype"] == "int8"
    assert all(v == 0 for v in packed)
    restored = unpack_scaled(packed, meta)
    assert all(abs(v - 42.0) < 1e-9 for v in restored)


def test_pack_scaled_empty():
    packed, meta = pack_scaled([], "auto")
    assert packed == []
    assert meta["type"] == "scaled"


def test_unpack_scaled_matches_original():
    data = [15.0, 20.0, 25.0, 30.0]
    packed, meta = pack_scaled(data, "int16")
    restored = unpack_scaled(packed, meta)
    for a, b in zip(data, restored):
        assert abs(a - b) < 0.01


def test_pack_enum_simple():
    data = ["AAPL", "MSFT", "TSLA", None]
    packed, meta = pack_enum(data)
    assert meta["type"] == "enum"
    assert meta["null_slot"] == 2
    assert len(meta["vocab"]) == 3
    # NULL=2, AAPL=0, MSFT=1, TSLA=3 (skips slot 2)
    assert meta["vocab"][0] == "AAPL"
    assert meta["vocab"][1] == "MSFT"
    assert meta["vocab"][3] == "TSLA"
    assert packed[0] == 0
    assert packed[1] == 1
    assert packed[2] == 3
    assert packed[3] == 2


def test_pack_enum_null_handling():
    data = ["AAPL", None, "MSFT", None]
    packed, meta = pack_enum(data)
    assert packed[0] == 0
    assert packed[1] == 2
    assert packed[2] == 1
    assert packed[3] == 2


def test_unpack_enum_roundtrip():
    data = ["AAPL", "MSFT", None, "TSLA", "AAPL"]
    packed, meta = pack_enum(data)
    restored = unpack_enum(packed, meta)
    assert restored == data


def test_scaled_series_in_kernel():
    kernel.clear_all()
    data = [10.0, 10.5, 11.0, 10.25, 9.75]
    from _core.compression import pack_scaled
    packed, meta = pack_scaled(data, "int16")
    ctx = create_context("test")
    sid = kernel.register_series(packed, ctx["_id"], compression=meta)
    entry = kernel.get_series(sid)
    assert entry["compression"] is not None
    assert entry["compression"]["type"] == "scaled"


def test_scaled_series_stats():
    kernel.clear_all()
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    from _core.compression import pack_scaled
    _, meta = pack_scaled(data, "int16")
    comp = {"bits": 16, "scale": meta["scale"], "offset": meta["offset"]}
    ctx = create_context("test")
    s = make_series(data, ctx, compression=comp)
    import numpy as np
    np_ref = np.array(data, dtype=np.float64)
    assert abs(total(s) - float(np.sum(np_ref))) / float(np.sum(np_ref)) < 0.01
    assert abs(minimum(s) - float(np.min(np_ref))) < 0.01
    assert abs(maximum(s) - float(np.max(np_ref))) < 0.01
    assert abs(mean(s) - float(np.mean(np_ref))) < 0.01


def test_enum_series_in_kernel():
    kernel.clear_all()
    data = ["AAPL", "MSFT", None, "TSLA"]
    from _core.compression import pack_enum
    packed, meta = pack_enum(data)
    ctx = create_context("test")
    sid = kernel.register_series(packed, ctx["_id"], compression=meta)
    entry = kernel.get_series(sid)
    assert entry["compression"]["type"] == "enum"
    restored = kernel.series_data(sid)
    assert restored == data


def test_memory_savings_scaled():
    import sys
    data = [float(i) for i in range(1000)]
    packed, meta = pack_scaled(data, "int16")
    # int16 vs float64: 2 bytes vs 8 bytes per element
    # In Python, list overhead dominates, but the logical size is 4x smaller
    assert meta["dtype"] == "int16"
    assert len(packed) == 1000
    # Verify precision loss within 0.1% of range
    restored = unpack_scaled(packed, meta)
    max_err = max(abs(a - b) for a, b in zip(data, restored))
    data_range = max(data) - min(data) if data else 1.0
    assert max_err / data_range < 0.001


def test_bit_width_auto_detection():
    small = ["A", "B"]
    packed, meta = pack_enum(small)
    assert meta["bit_width"] >= 2

    medium = [f"V{i}" for i in range(10)]
    packed, meta = pack_enum(medium + [None])
    # 10 values + null slot + reserved => at least 13, ceil(log2(13)) = 4
    assert meta["bit_width"] >= 4
