# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Parity: PairInsert (Rust `nf_pair_insert_i64` vs bit-exact fallback) +
BitmaskSweep (WGSL sweep vs bit-exact numpy fallback). Seed 42.

Generic lanes only, no domain vocabulary.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

APP_DIR = str(Path(__file__).resolve().parents[2])
LIB_PI = Path(APP_DIR) / "src" / "Relational" / "PairInsert" / "_lib" / "pair_insert.py"
LIB_BS = Path(APP_DIR) / "src" / "Relational" / "BitmaskSweep" / "_lib" / "bitmask_sweep.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(
        name, str(path),
        submodule_search_locations=[str(path.parent)])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def test_kernel_alias_pairinsert_bitmasksweep(kernel):
    assert kernel.metadata["PairInsert"]["types"] == ["pair_insert", "pair_insert_available"]
    assert kernel.metadata["BitmaskSweep"]["types"] == ["bitmask_sweep", "bitmask_sweep_available"]


def test_pair_insert_parity_native_fallback():
    pi = _load("pi_parity", LIB_PI)
    rng = np.random.default_rng(42)
    n, cap = 2000, 4096
    keys = rng.integers(-2 ** 62, 2 ** 62, size=n, dtype=np.int64)
    vals = rng.integers(-2 ** 62, 2 ** 62, size=n, dtype=np.int64)
    keys[0] = np.int64(0)
    keys[1] = np.int64(-9223372036854775808)
    keys[2] = np.int64(9223372036854775807)
    tk, tv, used, ng = pi.pair_insert(keys, vals, cap)
    tk2, tv2, used2, ng2 = pi._fb_pair_insert(
        pi._as_i64(keys, "keys"), pi._as_i64(vals, "vals"), cap)
    assert ng == ng2
    assert np.array_equal(tk, tk2)
    assert np.array_equal(tv, tv2)
    assert np.array_equal(used, used2)
    assert pi.pair_insert_available()


def test_bitmask_sweep_parity_wgsl_fallback():
    bs = _load("bs_parity", LIB_BS)
    rng = np.random.default_rng(42)
    masks = rng.integers(0, 2 ** 63, size=512, dtype=np.uint64)
    probes = np.zeros(8, dtype=np.uint64)
    probes[1:] = rng.integers(0, 2 ** 63, size=7, dtype=np.uint64)
    hits, counts, popcnt = bs.bitmask_sweep(masks, probes)
    ref = bs._fb_hits(np.ascontiguousarray(masks), np.ascontiguousarray(probes))
    assert np.array_equal(hits, ref)
    assert np.array_equal(counts, ref.reshape(512, 8).sum(axis=0).astype(np.uint64))
    assert int(popcnt[0]) == bin(int(masks[0])).count("1")


def test_pair_insert_bench_smoke():
    pi = _load("pi_bench", LIB_PI)
    rng = np.random.default_rng(42)
    keys = rng.integers(-2 ** 62, 2 ** 62, size=2000, dtype=np.int64)
    vals = rng.integers(-2 ** 62, 2 ** 62, size=2000, dtype=np.int64)
    s = time.perf_counter()
    pi.pair_insert(keys, vals, 4096)
    assert (time.perf_counter() - s) * 1000 < 5000
