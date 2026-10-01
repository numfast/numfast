# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: storage persist/load round-trip, NFS-minimum (spec 09).

int — exact; float — tolerance/ULP из conformance-profile.toml.
"""

import time
from pathlib import Path

import pytest

from harness import assert_float_close, assert_int_exact, load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


@pytest.mark.fast
def test_nfs_registered_with_depends(kernel):
    assert "persist_table" in kernel.alias and "load_table" in kernel.alias
    assert kernel.metadata["NFS"]["version"] == "0.1.0"


@pytest.mark.fast
def test_persist_load_int_roundtrip_exact(kernel, tmp_path):
    a = kernel.alias
    s = time.perf_counter()
    table = {"id": {"values": [1, 2, 3, 4], "dtype": "int32"},
             "key": {"values": [0, 1, 0, 1], "dtype": "int32"}}
    meta = a["persist_table"](table, str(tmp_path / "t.npz"))
    t_persist = (time.perf_counter() - s) * 1000
    assert meta["format"] == "npz-staging-v0" and meta["n"] == 4
    assert [c["name"] for c in meta["columns"]] == ["id", "key"]
    s = time.perf_counter()
    back = a["load_table"](str(tmp_path / "t.npz"))
    t_load = (time.perf_counter() - s) * 1000
    print(f"\nstages ms: persist={t_persist:.3f} load={t_load:.3f}")
    assert back["id"]["dtype"] == "int32"
    for got, want in zip(back["id"]["values"], [1, 2, 3, 4]):
        assert_int_exact(got, want, label="int round-trip")
    assert [int(v) for v in back["key"]["values"]] == [0, 1, 0, 1]


@pytest.mark.fast
def test_persist_load_float_roundtrip_tolerance(kernel, tmp_path):
    a = kernel.alias
    table = {"price": {"values": [1.5, 2.5, 3.25], "dtype": "float32"}}
    meta = a["persist_table"](table, str(tmp_path / "f.npz"))
    assert meta["columns"] == [{"name": "price", "dtype": "float32", "n": 3}]
    back = a["load_table"](str(tmp_path / "f.npz"))
    for got, want in zip(back["price"]["values"], [1.5, 2.5, 3.25]):
        assert_float_close(got, want, PROFILE, "f32", label="float round-trip")


@pytest.mark.fast
def test_load_missing_file_rejected(kernel, tmp_path):
    a = kernel.alias
    with pytest.raises(ValueError, match="not found.*persist"):
        a["load_table"](str(tmp_path / "nope.npz"))


@pytest.mark.fast
def test_persist_int64_raw_rejected(kernel, tmp_path):
    a = kernel.alias
    with pytest.raises(ValueError, match="int64"):
        a["persist_table"]({"x": {"values": [1], "dtype": "int64"}}, str(tmp_path / "x.npz"))


@pytest.mark.fast
def test_persist_ragged_columns_rejected(kernel, tmp_path):
    a = kernel.alias
    with pytest.raises(ValueError, match="length"):
        a["persist_table"](
            {"x": {"values": [1, 2], "dtype": "int32"},
             "y": {"values": [1], "dtype": "int32"}}, str(tmp_path / "r.npz"))
