# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 1 gate: Dictionary-backed Series (codes int32[N] + values str[D]).

unit + differential (seed 42) + zero-copy resident/GPU-split + ClickBench
hits_1m equality queries (exact vs pandas). Correctness exact everywhere.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from harness import load_profile  # noqa: F401  (profile-parity with suite)

APP_DIR = str(Path(__file__).resolve().parents[2])
HITS = Path("C:/App/competitions/ClickBench/data/hits_1m.parquet")
SEED = 42


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


# ---------- unit ----------

@pytest.mark.fast
def test_dictionary_registered(kernel):
    a = kernel.alias
    assert "dictionary_encode" in a and "dictionary_decode" in a
    assert "dictionary_metadata" in a
    assert kernel.metadata["Dictionary"]["version"] == "0.1.0"


@pytest.mark.fast
def test_encode_small_exact(kernel):
    a = kernel.alias
    r = a["dictionary_encode"](["b", "a", "b", None, "a"])
    assert r["codes"].dtype == np.int32
    assert list(r["codes"]) == [1, 0, 1, 0, 0]
    assert r["values"] == ["a", "b"]
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, True, True, False, True]
    assert r["metadata"] == {"encoding": "dictionary-sorted-v1", "d": 2, "n": 5,
                             "sorted": True, "nulls": 1}


@pytest.mark.fast
def test_encode_all_valid_validity_none(kernel):
    a = kernel.alias
    r = a["dictionary_encode"](["x", "y"])
    assert r["validity"] is None
    assert a["dictionary_decode"](r["codes"], r["values"], r["validity"]) == ["x", "y"]


@pytest.mark.fast
def test_encode_empty_and_all_null(kernel):
    a = kernel.alias
    r = a["dictionary_encode"]([])
    assert r["codes"].dtype == np.int32 and r["codes"].size == 0
    assert r["values"] == [] and r["validity"] is None
    r2 = a["dictionary_encode"]([None, None])
    assert r2["values"] == [] and r2["metadata"]["nulls"] == 2
    assert list(np.asarray(r2["validity"], dtype=bool)) == [False, False]


@pytest.mark.fast
def test_encode_unicode_and_determinism(kernel):
    a = kernel.alias
    vals = ["ёлка", "apple", "ёлка", "Z", None, "apple"]
    r1 = a["dictionary_encode"](vals)
    r2 = a["dictionary_encode"](list(vals))
    assert r1["values"] == sorted(set(v for v in vals if v is not None))
    assert list(r1["codes"]) == list(r2["codes"])
    assert r1["values"] == r2["values"]
    back = a["dictionary_decode"](r1["codes"], r1["values"], r1["validity"])
    assert back == vals


@pytest.mark.fast
def test_encode_explicit_validity_and_reject_non_str(kernel):
    a = kernel.alias
    r = a["dictionary_encode"](["a", "b", "c"], validity=[1, 0, 1])
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, False, True]
    assert a["dictionary_decode"](r["codes"], r["values"], r["validity"]) == ["a", None, "c"]
    with pytest.raises(ValueError, match="string column"):
        a["dictionary_encode"](["a", 5, "b"])
    with pytest.raises(ValueError, match="validity size"):
        a["dictionary_encode"](["a"], validity=[1, 0])
    with pytest.raises(ValueError, match="out of range"):
        a["dictionary_decode"](np.array([7], dtype=np.int32), ["a"])


@pytest.mark.fast
def test_metadata_sorted_flag(kernel):
    a = kernel.alias
    assert a["dictionary_metadata"](["a", "b"]) == {
        "encoding": "dictionary-sorted-v1", "d": 2, "sorted": True, "unique": True}
    assert a["dictionary_metadata"](["b", "a"])["sorted"] is False


# ---------- M4b: NULL sentinel + Arrow dictionary input (regressions) ----------
# BUG 1: dictionary_decode with D=0 raised "code 0 out of range D=0" instead
# of returning NULLs -- on BOTH the validity-supplied and validity-omitted
# paths. BUG 2: an Arrow dictionary-typed column reached bufs[2] as a bare
# IndexError. The validity-omitted D>0 contract is UNCHANGED: it is a pure
# gather, because code 0 is also the sorted rank of the smallest real value,
# so a NULL row and a real code-0 row cannot be told apart without the
# sidecar. The tests below pin both.


@pytest.mark.fast
def test_decode_all_null_column_d0_returns_nulls_not_error(kernel):
    a = kernel.alias
    r = a["dictionary_encode"]([None, None, None])
    assert r["values"] == [] and r["metadata"]["d"] == 0
    assert a["dictionary_decode"](r["codes"], r["values"]) == [None, None, None]
    assert a["dictionary_decode"](r["codes"], r["values"], r["validity"]) == [None, None, None]
    assert a["dictionary_decode"](r["codes"], r["dictionary"]) == [None, None, None]
    assert a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"]) == [None, None, None]
    e = a["dictionary_encode"]([])
    assert a["dictionary_decode"](e["codes"], e["values"]) == []
    assert a["dictionary_decode"](e["codes"], e["values"], e["validity"]) == []


@pytest.mark.fast
def test_decode_validity_omitted_is_a_pure_gather_not_a_null_restore(kernel):
    a = kernel.alias
    r = a["dictionary_encode"](["a", None, "b", None, "a"])
    assert a["dictionary_decode"](r["codes"], r["values"], r["validity"]) == \
        ["a", None, "b", None, "a"]
    # omitting the sidecar is a caller assertion that every code is valid
    assert a["dictionary_decode"](r["codes"], r["values"]) == \
        [r["values"][int(k)] for k in r["codes"]]
    assert a["dictionary_decode"](r["codes"], r["dictionary"]) == \
        [r["values"][int(k)] for k in r["codes"]]


@pytest.mark.fast
def test_decode_code0_is_a_real_value_not_a_null_sentinel(kernel):
    a = kernel.alias
    r = a["dictionary_encode"](["", None, "z"])
    assert r["values"] == ["", "z"] and list(r["codes"]) == [0, 0, 1]
    assert a["dictionary_decode"](r["codes"], r["values"], r["validity"]) == ["", None, "z"]
    assert a["dictionary_decode"](r["codes"], r["values"]) == ["", "", "z"]


@pytest.mark.fast
def test_decode_d0_still_rejects_negative_code_and_bad_validity(kernel):
    a = kernel.alias
    with pytest.raises(ValueError, match="out of range"):
        a["dictionary_decode"](np.array([-1, 0], dtype=np.int32), [])
    with pytest.raises(ValueError, match="validity size"):
        a["dictionary_decode"](np.array([0, 0], dtype=np.int32), [], validity=[False])


@pytest.mark.fast
def test_decode_envelope_dict_carries_its_own_sidecar(kernel):
    # P0-7: the encode envelope is a documented dictionary_decode carrier
    # ("body/enc dict"). Its validity sidecar must NOT be silently dropped --
    # decode(codes, envelope) must equal decode(codes, envelope['values'],
    # envelope['validity']), otherwise NULL rows come back as code 0 (a real
    # sorted rank) with no signal to the caller.
    a = kernel.alias
    r = a["dictionary_encode"](["b", "a", None, "b"])
    assert a["dictionary_decode"](r["codes"], r) == \
        a["dictionary_decode"](r["codes"], r["values"], r["validity"])
    assert a["dictionary_decode"](r["codes"], r) == ["b", "a", None, "b"]


@pytest.mark.fast
def test_decode_envelope_dict_carries_its_own_sidecar_int64(kernel):
    a = kernel.alias
    r = a["dictionary_encode"]([10, 20, None, 10])
    assert r["dtype"] == "int64"
    assert a["dictionary_decode"](r["codes"], r) == \
        a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"])
    assert a["dictionary_decode"](r["codes"], r) == [10, 20, None, 10]


@pytest.mark.fast
def test_decode_explicit_validity_wins_over_envelope_sidecar(kernel):
    # The caller assertion "every code is valid" stays available for a bare
    # carrier; when it is passed explicitly it overrides the envelope.
    a = kernel.alias
    r = a["dictionary_encode"](["b", "a", None, "b"])
    assert a["dictionary_decode"](r["codes"], r, validity=[True] * 4) == \
        ["b", "a", "a", "b"]
    assert a["dictionary_decode"](r["codes"], r["values"]) == ["b", "a", "a", "b"]


@pytest.mark.fast
def test_encode_arrow_dictionary_array_accepted(kernel):
    pa = pytest.importorskip("pyarrow")
    a = kernel.alias
    d = pa.DictionaryArray.from_arrays(pa.array([0, 1, None, 1, 0], type=pa.int32()),
                                       pa.array(["bb", "aa"]))
    r = a["dictionary_encode"](d)
    assert r["dtype"] == "text"
    assert r["values"] == ["aa", "bb"]          # sorted-unique contract restored
    assert r["codes"].dtype == np.int32
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, True, False, True, True]
    assert a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"]) == \
        ["bb", "aa", None, "aa", "bb"]


@pytest.mark.fast
def test_encode_arrow_dictionary_chunked_and_unsorted_dict(kernel):
    pa = pytest.importorskip("pyarrow")
    a = kernel.alias
    d1 = pa.DictionaryArray.from_arrays(pa.array([0, 1, None], type=pa.int32()),
                                        pa.array(["bb", "aa"]))
    d2 = pa.DictionaryArray.from_arrays(pa.array([2, 0, 1], type=pa.int32()),
                                        pa.array(["z", "m", "a"]))
    r = a["dictionary_encode"](pa.chunked_array([d1, d2]))
    assert r["values"] == ["a", "aa", "bb", "m", "z"]
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, True, False, True, True, True]
    assert a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"]) == \
        ["bb", "aa", None, "a", "z", "m"]


@pytest.mark.fast
def test_encode_arrow_rejects_bad_types_with_a_named_error(kernel):
    pa = pytest.importorskip("pyarrow")
    a = kernel.alias
    for bad, needle in (
            (pa.array([5, 3, 5]), "TEXT Arrow array"),          # was IndexError
            (pa.array([True, False]), "TEXT Arrow array"),      # was IndexError
            (pa.array([None], type=pa.null()), "TEXT Arrow array"),   # was IndexError
            (pa.DictionaryArray.from_arrays(pa.array([0, 1], type=pa.int32()),
                                            pa.array([10, 20])), "TEXT Arrow array"),
            (pa.table({"c": pa.array(["a", "b"])}), "Table")):
        with pytest.raises(ValueError, match=needle):
            a["dictionary_encode"](bad)


@pytest.mark.fast
def test_encode_arrow_plain_string_path_unchanged(kernel):
    pa = pytest.importorskip("pyarrow")
    a = kernel.alias
    r = a["dictionary_encode"](pa.array(["b", None, "a", "b"]))
    assert r["values"] == ["a", "b"]
    assert list(r["codes"]) == [1, 0, 0, 1]
    assert list(np.asarray(r["validity"], dtype=bool)) == [True, False, True, True]
    assert a["dictionary_decode"](r["codes"], r["dictionary"], r["validity"]) == \
        ["b", None, "a", "b"]


# ---------- zero-copy + GPU-split ----------

@pytest.mark.fast
def test_codes_zero_copy_through_series(kernel):
    a = kernel.alias
    r = a["dictionary_encode"](["m", "n", "m", "o"])
    codes = r["codes"]
    assert codes.dtype == np.int32 and codes.flags["C_CONTIGUOUS"]
    # values stay RAM-side Python strs — never a GPU buffer
    assert isinstance(r["values"], list) and all(isinstance(s, str) for s in r["values"])
    bufs = a["cpu_execute"]([a["ir_series"]("c", codes, dtype="int32")])
    assert np.shares_memory(bufs["c"], codes) or bufs["c"] is codes
    out = a["cpu_execute"]([a["ir_series"]("c", codes, dtype="int32"),
                            a["ir_compare"]("m", "c", 1, op="==")])
    assert out["m"].to_array().tolist() == [False, True, False, False]


# ---------- differential (seed 42) ----------

@pytest.mark.fast
def test_differential_vs_python_reference(kernel):
    a = kernel.alias
    rng = np.random.default_rng(SEED)
    pool = [f"s-{i:04d}" for i in range(500)] + ["юникод", "x" * 40]
    vals = [None if rng.random() < 0.05 else pool[int(rng.integers(0, len(pool)))]
            for _ in range(20_000)]
    s = time.perf_counter()
    r = a["dictionary_encode"](vals)
    t_enc = (time.perf_counter() - s) * 1000
    ref_vals = sorted({v for v in vals if v is not None})
    ref_idx = {v: i for i, v in enumerate(ref_vals)}
    assert r["values"] == ref_vals
    assert [int(c) for c in r["codes"]
            [np.asarray([(v is not None) for v in vals])]] == [
        ref_idx[v] for v in vals if v is not None]
    back = a["dictionary_decode"](r["codes"], r["values"], r["validity"])
    assert back == vals
    print(f"\nstages ms: diff-encode-20K={t_enc:.2f} D={len(ref_vals)}")


# ---------- ClickBench hits_1m (1-2 queries, exact) ----------

def _cb_col(col):
    import pandas as pd

    df = pd.read_parquet(HITS, columns=[col])
    return [None if (v is None or (isinstance(v, float) and v != v)) else str(v)
            for v in df[col].tolist()]


@pytest.mark.fast
def test_clickbench_url_equality_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    t0 = time.perf_counter()
    url = _cb_col("URL")
    t_load = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    r = a["dictionary_encode"](url)
    t_enc = (time.perf_counter() - t0) * 1000
    target = r["values"][len(r["values"]) // 2]
    want = sum(1 for v in url if v == target)
    t0 = time.perf_counter()
    code = r["values"].index(target)
    bufs = a["cpu_execute"]([a["ir_series"]("codes", r["codes"], dtype="int32"),
                             a["ir_compare"]("m", "codes", int(code), op="==")])
    n = int(np.count_nonzero(bufs["m"].to_array()))
    t_cmp = (time.perf_counter() - t0) * 1000
    assert n == want, f"URL equality: {n} != pandas {want}"
    print(f"\nstages ms: load={t_load:.1f} encode={t_enc:.1f} lookup+compare={t_cmp:.2f} "
          f"N={len(url)} D={r['metadata']['d']} hits={n}")


@pytest.mark.fast
def test_clickbench_searchphrase_equality_with_nulls_exact(kernel):
    if not HITS.exists():
        pytest.skip("hits_1m.parquet absent")
    a = kernel.alias
    col = _cb_col("SearchPhrase")
    r = a["dictionary_encode"](col)
    assert r["metadata"]["nulls"] == sum(1 for v in col if v is None)
    nonnull = [v for v in col if v is not None]
    target = sorted(set(nonnull))[len(set(nonnull)) // 3]
    want = sum(1 for v in col if v == target)
    code = r["values"].index(target)
    nodes = [a["ir_series"]("codes", r["codes"], dtype="int32",
                            validity=None if r["validity"] is None
                            else np.asarray(r["validity"], dtype=np.int8).tolist()),
             a["ir_compare"]("m", "codes", int(code), op="=="),
             a["ir_filter"]("f", "codes", "m")]
    bufs = a["cpu_execute"](nodes)
    assert int(bufs["f"].size) == want, f"SearchPhrase: {bufs['f'].size} != pandas {want}"
    print(f"\nSearchPhrase N={len(col)} D={r['metadata']['d']} nulls={r['metadata']['nulls']} hits={want}")
