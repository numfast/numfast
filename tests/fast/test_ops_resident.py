# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Resident typed columns: encode-once, queries take int32 directly (fast golden)."""

from pathlib import Path

import numpy as np
import pytest

from harness import assert_int_exact

APP_DIR = str(Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


@pytest.mark.fast
def test_resident_prepare_pattern_once_exact(kernel):
    a = kernel.alias
    raw = ["id1", "id2", "id1", "id10", "oops", None]
    res = a["resident_prepare"]({"c": {"values": raw, "prefix": "id"}})
    col = res["c"]
    assert [int(v) for v in col["codes"]] == [1, 2, 1, 10, 0, 0]
    assert [bool(v) for v in col["validity"]] == [True, True, True, True, False, False]
    assert col["physical"] == "int32" and col["n"] == 6 and col["kind"] == "pattern"


@pytest.mark.fast
def test_resident_prepare_series_zero_copy(kernel):
    a = kernel.alias
    src = np.arange(8, dtype=np.int32)
    res = a["resident_prepare"]({"v": {"values": src, "dtype": "int32"}})
    assert np.shares_memory(res["v"]["codes"], src)
    assert res["v"]["validity"] is None and res["v"]["kind"] == "series"


@pytest.mark.fast
def test_resident_query_matches_querytime_encode(kernel):
    a = kernel.alias
    strs = ["id1", "id2", "id1", "id2", "id1"]
    vals = [5, 1, 7, 3, 2]
    jobs_a = [a["ir_encode_pattern"]("k", strs, "id"),
              a["ir_series"]("v", vals), a["ir_groupby"]("g", "v", "k", "sum")]
    ra = a["evaluate"](a["optimize"](a["compile"](jobs_a)), "cpu", 5)["result"]
    res = a["resident_prepare"]({"k": {"values": strs, "prefix": "id"}})
    jobs_b = [a["ir_series"]("k", res["k"]["codes"]),
              a["ir_series"]("v", vals), a["ir_groupby"]("g", "v", "k", "sum")]
    rb = a["evaluate"](a["optimize"](a["compile"](jobs_b)), "cpu", 5)["result"]
    assert ra == rb == {1: 14, 2: 4}
    assert_int_exact(sum(rb.values()), 18, label="resident groupby checksum")
