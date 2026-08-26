# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import pytest
from _core import kernel


def setup_method():
    kernel.clear_all()
    kernel.configure(chunk_size=1000)


def test_register_and_get():
    kernel.clear_all()
    sid = kernel.register_series([1.0, 2.0, 3.0], "ctx1")
    entry = kernel.get_series(sid)
    assert entry is not None
    assert entry["length"] == 3
    assert entry["context_id"] == "ctx1"


def test_remove_series():
    kernel.clear_all()
    sid = kernel.register_series([1, 2], "ctx1")
    assert kernel.remove_series(sid) is True
    assert kernel.get_series(sid) is None


def test_series_data_roundtrip():
    kernel.clear_all()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    sid = kernel.register_series(data, "ctx1")
    result = kernel.series_data(sid)
    assert result == data


def test_chunk_exact_256_multiple():
    kernel.clear_all()
    kernel.configure(chunk_size=768)
    data = list(range(768))
    sid = kernel.register_series(data, "ctx1")
    chunks = kernel.get_chunks(sid)
    assert chunks is not None
    assert len(chunks) == 1
    c = chunks[0]
    assert c["size"] == 768
    assert c["valid_elements"] == 768
    assert len(c["buf"]) == 768


def test_chunk_with_remainder():
    kernel.clear_all()
    kernel.configure(chunk_size=1000)
    data = list(range(500))
    sid = kernel.register_series(data, "ctx1")
    chunks = kernel.get_chunks(sid)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["size"] == 512
    assert c["valid_elements"] == 500
    assert len(c["buf"]) == 512
    assert c["buf"][500:] == [0.0] * 12


def test_chunk_multiple_chunks():
    kernel.clear_all()
    kernel.configure(chunk_size=1000)
    chunk_size = kernel._calc_chunk_size()
    assert chunk_size == 768
    data = list(range(2000))
    sid = kernel.register_series(data, "ctx1")
    chunks = kernel.get_chunks(sid)
    assert len(chunks) == 3
    assert chunks[0]["valid_elements"] == 768
    assert chunks[0]["size"] == 768
    assert chunks[1]["valid_elements"] == 768
    assert chunks[1]["size"] == 768
    remaining = 2000 - 768 - 768
    assert chunks[2]["valid_elements"] == remaining
    padded = ((remaining + 255) // 256) * 256
    assert chunks[2]["size"] == padded


def test_chunk_padding_is_zero():
    kernel.clear_all()
    data = [42.0] * 500
    sid = kernel.register_series(data, "ctx1")
    chunks = kernel.get_chunks(sid)
    c = chunks[0]
    for i in range(c["valid_elements"], c["size"]):
        assert c["buf"][i] == 0.0


def test_empty_data():
    kernel.clear_all()
    sid = kernel.register_series([], "ctx1")
    chunks = kernel.get_chunks(sid)
    assert chunks == []


def test_single_element():
    kernel.clear_all()
    sid = kernel.register_series([99.0], "ctx1")
    chunks = kernel.get_chunks(sid)
    assert len(chunks) == 1
    assert chunks[0]["valid_elements"] == 1
    assert chunks[0]["size"] == 256
    assert chunks[0]["buf"][0] == 99.0


def test_calc_chunk_size_returns_multiple_of_256():
    kernel.configure(chunk_size=1000)
    cs = kernel._calc_chunk_size()
    assert cs % 256 == 0


def test_padded_size():
    assert kernel._padded_size(0) == 0
    assert kernel._padded_size(1) == 256
    assert kernel._padded_size(256) == 256
    assert kernel._padded_size(257) == 512
