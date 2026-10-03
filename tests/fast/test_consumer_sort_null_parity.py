# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""`sort()` on a NULL key agrees with pandas -- the false refusal, removed.

GATE_semantics.md §9 asserted that `ir_sort` puts a NULL key FIRST, on the
strength of one rendered order `[1, 2, 3, 0]`. That order is the NULL-LAST
order: the snapshot's own evidence refutes its own conclusion. The consumer
facade believed §9 and shipped `_refuse_null_sort_key`, a refusal whose stated
premise was false.

Measured (2026-10-04, this file):

* The CPU driver partitions the permutation by validity --
  `perm[:n_valid]` then `perm[n_valid:]` (`Drivers/CPU/_lib/cpu.py:1228-1229`),
  and its own docstring says "Invalid rows (valid_mask False) go last in input
  order, always" (`:1185`).
* The GPU driver concatenates the same two index sets
  (`Drivers/GPU/_lib/gpu.py:3149`).
* That is pandas `sort_values(na_position="last", kind="stable")`, and the
  facade matches it row for row on every fixture below, in both directions,
  on the key column AND every payload column, with NULL-ness preserved.

So the guard was a FALSE REFUSAL and is deleted. The `group()` NULL-key guard
STAYS: `ir_groupby*` compacts NULL keys away (`cpu.py:3325`), so a NULL group
is an ABSENT value -- a different defect, pinned by
`test_group_refuses_null_key_loudly` in `test_consumer_surface.py`.

The GPU driver's own limits are unchanged and are NOT a parity failure: it
refuses a sort whose VALID row count is not a power of two, and it refuses
int64 and float64 keys. Those cases are pinned in `test_ops_gpu_sort.py`.
"""

import numpy as np
import pandas as pd
import pytest

# (fixture name, key column, frame). Every frame has exactly one nullable
# column unless the name says otherwise.
_FIXTURES = [
    ("i64_null_mid", "k", pd.DataFrame(
        {"k": pd.array([1, 2, None, 1, 3], dtype="Int64"),
         "v": np.array([10.0, 20.0, 30.0, 40.0, 50.0])})),
    ("f64_null_mid", "k", pd.DataFrame(
        {"k": pd.array([1.5, 2.5, None, 1.5, 3.5], dtype="Float64"),
         "v": np.array([10.0, 20.0, 30.0, 40.0, 50.0])})),
    ("f64_with_text_sidecar", "k", pd.DataFrame(
        {"k": pd.array([1.0, 2.0, None, 1.0, 3.0], dtype="Float64"),
         "tag": pd.array(["x", "y", None, "y", "z"], dtype="string")})),
    ("nullable_text", "k", pd.DataFrame(
        {"k": pd.array(["b", "a", None, "a", "c"], dtype="string"),
         "v": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})),
    ("all_null_i64", "k", pd.DataFrame(
        {"k": pd.array([None, None, None], dtype="Int64"),
         "v": np.array([1.0, 2.0, 3.0])})),
    ("all_null_text", "k", pd.DataFrame(
        {"k": pd.array([None, None], dtype="string"),
         "v": np.array([1.0, 2.0])})),
    ("null_free_i32", "k", pd.DataFrame(
        {"k": np.array([3, 1, 2, 1, 0], np.int32),
         "v": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})),
    ("null_free_f64", "k", pd.DataFrame(
        {"k": np.array([3.0, 1.0, 2.0], np.float64),
         "v": np.array([1.0, 2.0, 3.0])})),
    ("null_extremes", "k", pd.DataFrame(
        {"k": pd.array([None, 5, 1, None, -3, 0], dtype="Int64"),
         "v": np.arange(6, dtype=np.float64)})),
    ("null_among_negatives", "k", pd.DataFrame(
        {"k": pd.array([-1, None, -5, 0, None], dtype="Int64"),
         "v": np.arange(5, dtype=np.float64)})),
    ("null_next_to_a_real_zero", "k", pd.DataFrame(
        {"k": pd.array([0, None, 0, 1], dtype="Int64"),
         "v": np.arange(4, dtype=np.float64)})),
    ("two_nulls_same_value", "k", pd.DataFrame(
        {"k": pd.array([2, None, 1, None], dtype="Int64"),
         "v": np.arange(4, dtype=np.float64)})),
    ("null_in_second_key", "b", pd.DataFrame(
        {"a": np.array([1, 1, 1, 2, 2], np.int32),
         "b": pd.array([9.0, None, 2.0, None, 1.0], dtype="Float64")})),
]


def _pandas_shaped(series):
    """A numfast Series as pandas would show it: NA folded into None."""
    raw = series.to_numpy().tolist()
    validity = series.validity
    out = []
    for i, v in enumerate(raw):
        if validity is not None and not bool(validity[i]):
            out.append(None)
        elif isinstance(v, str):
            out.append(v)
        else:
            out.append(v.item() if hasattr(v, "item") else v)
    return out


@pytest.mark.fast
@pytest.mark.parametrize("name,key,frame", _FIXTURES,
                         ids=[f[0] for f in _FIXTURES])
@pytest.mark.parametrize("desc", [False, True], ids=["asc", "desc"])
def test_sort_null_key_matches_the_pandas_oracle(name, key, frame, desc):
    """Every column of the sorted result equals pandas, NA -> None included.

    The reference permutation is pandas' OWN index order, so the payload
    column is checked through the same perm the key column chose.
    """
    import numfast as nf
    out = nf.from_pandas(frame).query().sort(key, desc=desc).compile()
    order = frame[key].sort_values(ascending=not desc, kind="stable").index
    ref = frame.iloc[list(order)]
    for col in frame.columns:
        want = [None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
                for v in ref[col].tolist()]
        assert _pandas_shaped(out.column(col)) == want, f"{name}/{col}"


@pytest.mark.fast
def test_sort_puts_null_rows_last_in_both_directions():
    """The one-line statement the false guard contradicted."""
    import numfast as nf
    frame = pd.DataFrame({"k": pd.array([2, None, 1, None], dtype="Int64")})
    t = nf.from_pandas(frame)
    asc = t.query().sort("k").compile().column("k").to_numpy().tolist()
    assert asc[:2] == [1, 2] and not asc[2] and not asc[3], asc
    desc = t.query().sort("k", desc=True).compile().column("k").to_numpy().tolist()
    assert desc[:2] == [2, 1] and not desc[2] and not desc[3], desc


@pytest.mark.fast
def test_raw_ir_sort_perm_puts_nulls_last_on_cpu_and_gpu():
    """`ir_sort` itself, both drivers, against pandas' permutation.

    The GPU driver refuses a sort whose valid row count is not a power of two,
    so the fixture sizes are chosen to keep the GPU in play: 4 valid rows out
    of 8 (half NULL) and 8 valid rows out of 16 (half NULL).
    """
    import numfast as nf
    a = nf.get_kernel().alias
    cases = [
        ("i32_8_4null", np.array([3, 1, 3, 1, 9, 0, 2, 2], np.int32),
         np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)),
        ("i32_16_8null", np.array([0, 3, 1, 4, 0, 3, 1, 4, 2, 5, 2, 5,
                                   1, 0, 4, 2], np.int32),
         (np.arange(16) % 2 == 0)),
        ("f32_8_4null", np.array([1.5, -2.0, 0.0, 3.25, 7.5, 7.5, -9.0, 0.5],
                                 np.float32),
         np.array([1, 1, 1, 0, 1, 0, 0, 0], bool)),
    ]
    for name, values, validity in cases:
        nullable = [v if validity[i] else None
                    for i, v in enumerate(values.tolist())]
        dt = "Int64" if values.dtype.kind == "i" else "Float64"
        ref_series = pd.Series(pd.array(nullable, dtype=dt))
        for desc in (False, True):
            jobs = [a["ir_series"]("k", values, str(values.dtype),
                                   np.asarray(validity, dtype=bool)),
                    a["ir_sort"]("p", "k", descending=desc)]
            nodes = a["compile"](jobs)["nodes"]
            cpu = np.asarray(a["cpu_execute"](nodes)["p"]).tolist()
            gpu = np.asarray(a["gpu_execute"](nodes)["p"]).tolist()
            ref = [int(i) for i in ref_series.sort_values(
                ascending=not desc, kind="stable").index]
            assert cpu == ref, f"{name} asc={not desc} cpu={cpu} pandas={ref}"
            assert gpu == ref, f"{name} asc={not desc} gpu={gpu} pandas={ref}"


@pytest.mark.fast
def test_group_null_key_still_refuses():
    """The SIBLING guard is justified and stays: a NULL group is an ABSENT
    value (`cpu.py:3325` drops NULL keys), which the consumer cannot tell
    apart from the answer. Kept here so the sort deletion is visibly scoped."""
    import numfast as nf
    frame = pd.DataFrame({"k": pd.array([1, None, 1], dtype="Int64"),
                          "v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="key column 'k' has 1 NULL rows"):
        nf.from_pandas(frame).query().group("k", {"v": ("sum",)}).compile()