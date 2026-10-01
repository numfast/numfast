# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Packaging + adapters boundary tests (fast).

Covers: clean boot (no dev-tree coupling), NumPy/Pandas/Arrow round-trips
(dtype/shape/names/values/order/validity/empty/single/nullable/
non-contiguous), native enabled/disabled parity, clean-process import,
controlled optional-dependency errors. Fixed seed 42 wherever RNG is used.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# The dev venv has the sibling `numfast` repo installed editable, which would
# shadow this fork's packaged boundary. Test-only path surgery (tooling, not
# application code): drop ONLY the sibling numfast editable finder, keep
# everything else (e.g. framework_builder) intact for the engine suite.
REPO = Path(__file__).resolve().parents[2]
SRC = str(REPO / "src")
sys.meta_path = [f for f in sys.meta_path
                 if "__editable___numfast_" not in type(f).__module__]
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numfast as nf  # noqa: E402

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def kernel():
    return nf.get_kernel(fresh=True)


# -- boot / packaging static gates --------------------------------------

def test_boot_resolves_inside_package():
    f = Path(nf.__file__).resolve()
    assert f.parent.name == "numfast"
    assert (f.parent / "_builder").is_dir()
    assert (f.parent / "adapters").is_dir()


def test_native_dll_package_relative():
    info = nf.native_info()
    assert info["dll"] is not None and info["dll_exists"]
    dll = Path(info["dll"]).resolve()
    pkg = Path(nf.__file__).resolve().parent.resolve()
    assert dll.is_relative_to(pkg), f"{dll} outside {pkg}"


def test_boundary_sources_have_no_hardcoded_roots():
    pats = ["C:\\App", "C:/App", "app-builder-ponytail"]
    bad = []
    pkg = Path(nf.__file__).resolve().parent
    for f in (list((pkg / "_builder").rglob("*.py")) + [pkg / "__init__.py"]
              + list((pkg / "adapters").glob("*.py"))
              + list((pkg / "_lib").glob("*.py"))):
        src = f.read_text(encoding="utf-8")
        for p in pats:
            if p in src:
                bad.append((f.name, p))
    assert not bad, f"hardcoded dev references: {bad}"


def test_kernel_alias_surface(kernel):
    for name in ("ir_series", "ir_map", "ir_compare", "ir_filter", "ir_sort",
                 "ir_gather", "ir_reduce", "ir_groupby", "compile", "optimize",
                 "evaluate", "cpu_execute", "column_schema", "canonical_dtype",
                 "check_int32_range", "dictionary_encode", "dictionary_decode"):
        assert name in kernel.alias, f"missing alias {name}"


# -- NumPy round-trips ----------------------------------------------------

@pytest.mark.parametrize("dtype", ["int32", "float32", "float64", "bool"])
def test_numpy_roundtrip_dtypes(dtype):
    arr = (np.arange(8) % 3 == 0) if dtype == "bool" else np.arange(8)
    arr = np.ascontiguousarray(arr.astype(dtype))
    s = nf.from_numpy(arr, name="c")
    assert s.dtype == ("bool" if dtype == "bool" else dtype)
    back = nf.to_numpy(s)
    assert back.dtype == arr.dtype
    np.testing.assert_array_equal(back, arr)
    assert s.validity is None


def test_numpy_zero_copy_when_possible():
    arr = np.arange(6, dtype=np.int32)
    s = nf.from_numpy(arr)
    assert getattr(s, "_shared", False) is True


def test_numpy_copies_documented_cases():
    assert nf.from_numpy(np.arange(6)[::2])._shared is False  # non-contiguous
    assert nf.from_numpy(np.arange(4, dtype=np.int16))._shared is False  # narrow
    assert nf.from_numpy(np.array([1.0, np.nan]))._shared is False  # NaN
    assert nf.from_numpy(np.arange(4, dtype=np.int64))._shared is True  # int64 keeps width


def test_numpy_int64_stays_int64_and_uint64_overflow():
    s = nf.from_numpy(np.array([2 ** 30, -(2 ** 30)], dtype=np.int64))
    assert s.dtype == "int64"
    np.testing.assert_array_equal(nf.to_numpy(s), [2 ** 30, -(2 ** 30)])
    big = nf.from_numpy(np.array([2 ** 40, -(2 ** 40)], dtype=np.int64))
    assert big.dtype == "int64"
    np.testing.assert_array_equal(nf.to_numpy(big), [2 ** 40, -(2 ** 40)])
    with pytest.raises(OverflowError):
        nf.from_numpy(np.array([2 ** 63], dtype=np.uint64))


@pytest.mark.parametrize("bad", ["float16", "complex64"])
def test_numpy_rejected_dtypes(bad):
    with pytest.raises(ValueError):
        nf.from_numpy(np.zeros(3, dtype=bad))


def test_numpy_nan_validity_contract():
    s = nf.from_numpy(np.array([1.0, float("nan"), 3.0]))
    np.testing.assert_array_equal(s.validity, [True, False, True])
    back = nf.to_numpy(s)
    assert back[0] == 1.0 and np.isnan(back[1]) and back[2] == 3.0


def test_numpy_masked_and_explicit_validity():
    m = np.ma.MaskedArray([1, 2, 3], mask=[0, 1, 0])
    s = nf.from_numpy(m)
    np.testing.assert_array_equal(s.validity, [True, False, True])
    s2 = nf.from_numpy([10, 20, 30], validity=[1, 0, 1])
    np.testing.assert_array_equal(s2.validity, [True, False, True])


def test_numpy_empty_single_random():
    e = nf.from_numpy(np.array([], dtype=np.float32))
    assert len(e) == 0 and nf.to_numpy(e).shape == (0,)
    one = nf.from_numpy(np.array([7], dtype=np.int32))
    assert nf.to_numpy(one * 6).tolist() == [42]
    r = RNG.integers(-1000, 1000, size=513).astype(np.int32)
    np.testing.assert_array_equal(nf.to_numpy(nf.from_numpy(r) + 1), r + 1)


def test_numpy_table_names_order():
    t = nf.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.int32), names=["b", "a"])
    assert t.names == ["b", "a"]
    back = nf.to_numpy(t)
    assert back.tolist() == [[1, 2], [3, 4]]
    assert t.column("a").dtype == "int32"


def test_numpy_text_roundtrip_sort_filter():
    s = nf.from_numpy(np.array(["b", "a", None], dtype=object))
    assert s.dtype == "text"
    assert nf.to_numpy(s).tolist() == ["b", "a", None]
    assert nf.to_numpy(s.sort()).tolist() == ["a", "b", None]
    keep = nf.from_numpy([3, 1, 2]).compare(">", 1)
    assert nf.to_numpy(nf.from_numpy(np.array(["a", "b", "c"])).filter(keep)).tolist() == ["a", "c"]


def test_series_ops_through_engine():
    s = nf.from_numpy([1, 2, 3, 4])
    np.testing.assert_array_equal(nf.to_numpy(s * 2), [2, 4, 6, 8])
    assert s.reduce("sum") == 10
    assert s.reduce("mean") == 2.5
    np.testing.assert_array_equal(nf.to_numpy(s.sort()), [1, 2, 3, 4])
    np.testing.assert_array_equal(
        nf.to_numpy(nf.from_numpy([10, 20, 30]).filter(
            nf.from_numpy([1, 2, 3]).compare(">", 1))), [20, 30])


# -- native enabled/disabled parity ---------------------------------------

def _parity(fn):
    old = os.environ.get("NUMFAST_NATIVE_DISABLE")
    try:
        os.environ.pop("NUMFAST_NATIVE_DISABLE", None)
        a = fn()
        os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
        b = fn()
    finally:
        if old is None:
            os.environ.pop("NUMFAST_NATIVE_DISABLE", None)
        else:
            os.environ["NUMFAST_NATIVE_DISABLE"] = old
    return a, b


def _norm(v):
    if isinstance(v, np.ndarray):
        return ("nd", v.dtype.str, v.tolist())
    if isinstance(v, dict):
        return ("dict", sorted((k, _norm(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return ("seq", [_norm(x) for x in v])
    if isinstance(v, float):
        return ("f", v)
    return ("v", v)


def test_native_parity_series_ops():
    def run():
        r = np.random.default_rng(42).integers(0, 50, size=2000).astype(np.int32)
        s = nf.from_numpy(r)
        m = nf.from_numpy(r.astype(np.float32))
        return [nf.to_numpy(s * 3).tolist(), s.reduce("sum"),
                nf.to_numpy(s.sort()).tolist(),
                nf.to_numpy(s.filter(s.compare(">", 25))).tolist(),
                m.reduce("mean"),
                nf.to_numpy(nf.from_numpy(
                    np.array(["b", "a", "c"])).sort()).tolist()]
    a, b = _parity(run)
    assert _norm(a) == _norm(b)


def test_native_parity_groupby_kernel(kernel):
    def run():
        a = kernel.alias
        r = np.random.default_rng(42).integers(0, 37, size=5000).astype(np.int32)
        v = np.random.default_rng(7).integers(0, 100, size=5000).astype(np.int32)
        jobs = [a["ir_series"]("v", v), a["ir_series"]("k", r),
                a["ir_groupby"]("g", "v", "k", "sum")]
        g = a["optimize"](a["compile"](jobs))
        return a["evaluate"](g, "cpu", 5000)["result"]
    a, b = _parity(run)
    assert _norm(a) == _norm(b)
    assert sum(a.values()) == int(np.random.default_rng(7).integers(
        0, 100, size=5000).sum())


# -- Pandas ---------------------------------------------------------------

pd = pytest.importorskip("pandas")


def test_pandas_nullable_roundtrip():
    import pandas as _pd
    df = _pd.DataFrame({
        "i": _pd.array([1, 2, None], dtype="Int64"),
        "f": _pd.array([1.5, float("nan"), 3.0], dtype="Float64"),
        "b": _pd.array([True, False, None], dtype="boolean"),
        "s": _pd.array(["x", "y", None], dtype="string"),
    })
    t = nf.from_pandas(df)
    assert t.names == ["i", "f", "b", "s"]
    back = nf.to_pandas(t)
    assert back["i"].tolist() == [1, 2, _pd.NA]
    assert back["s"].tolist() == ["x", "y", _pd.NA]
    assert back["b"].tolist() == [True, False, _pd.NA]
    assert back["f"].isna().tolist() == [False, True, False]
    # double round-trip stability
    again = nf.to_pandas(nf.from_pandas(back))
    assert again["i"].tolist() == back["i"].tolist()
    assert again["s"].tolist() == back["s"].tolist()


def test_pandas_plain_and_series():
    import pandas as _pd
    df = _pd.DataFrame({"a": [1, 2, 3], "z": [0.5, 1.5, 2.5]})
    t = nf.from_pandas(df)
    assert t.names == ["a", "z"]
    out = nf.to_pandas(t)
    assert out["a"].tolist() == [1, 2, 3]
    s = nf.from_pandas(df["a"])
    assert nf.to_pandas(s).tolist() == [1, 2, 3]


# -- Arrow ------------------------------------------------------------------

pa = pytest.importorskip("pyarrow")


def test_arrow_roundtrip_names_order_validity():
    import pyarrow as _pa
    tbl = _pa.table({
        "x": _pa.array([1, 2, None], type=_pa.int64()),
        "s": _pa.array(["a", None, "b"]),
        "f": _pa.array([1.0, None, 3.0], type=_pa.float32()),
        "b": _pa.array([True, None, False]),
    })
    t = nf.from_arrow(tbl)
    assert t.names == ["x", "s", "f", "b"]
    back = nf.to_arrow(t)
    assert back.schema.names == ["x", "s", "f", "b"]
    assert back.column("x").to_pylist() == [1, 2, None]
    assert back.column("s").to_pylist() == ["a", None, "b"]
    assert back.column("f").to_pylist() == [1.0, None, 3.0]
    assert back.column("b").to_pylist() == [True, None, False]


def test_arrow_empty_and_single():
    import pyarrow as _pa
    e = nf.from_arrow(_pa.table({"v": _pa.array([], type=_pa.int32())}))
    assert len(e) == 0
    one = nf.from_arrow(_pa.array([None], type=_pa.string()))
    assert nf.to_arrow(one).to_pylist() == [None]


def test_arrow_rejects_timestamps():
    import pyarrow as _pa
    import datetime as _dt
    with pytest.raises(ValueError, match="cast"):
        nf.from_arrow(_pa.array([_dt.datetime(2024, 1, 1)],
                                type=_pa.timestamp("s")))


def test_arrow_int64_overflow_raises_not_wraps():
    # Invariant #1: the int64->int32 narrowing check must see the PRE-narrow
    # int64 array. It used to run on the already-narrowed int32 buffer, where
    # check_int32_range short-circuits, so out-of-range values wrapped silently
    # (2**33+7 -> 7, -2**33 -> 0) with dtype still reported as "int32".
    import pyarrow as _pa
    overflow = [2 ** 31, 2 ** 33 + 7, -(2 ** 33)]
    for call in (lambda a: nf.from_arrow(a),
                 lambda a: nf.from_arrow(_pa.table({"x": a}))):
        with pytest.raises(OverflowError, match="narrowing overflow"):
            call(_pa.array(overflow, type=_pa.int64()))
    # nulls are excluded from the check (Arrow stores garbage there)
    with pytest.raises(OverflowError, match="narrowing overflow"):
        nf.from_arrow(_pa.array([1, None, 2 ** 33], type=_pa.int64()))
    # in-range int64 still narrows to int32 bit-exactly
    i32_min, i32_max = -(2 ** 31), 2 ** 31 - 1
    ok = [1, 2, 3, i32_min, i32_max, 0, -1]
    for col in (nf.from_arrow(_pa.array(ok, type=_pa.int64())),
                nf.from_arrow(_pa.table({"x": _pa.array(ok, type=_pa.int64())}))
                .column("x")):
        assert col.dtype == "int32"
        assert nf.to_arrow(col).to_pylist() == ok


def test_arrow_missing_dep_controlled_error(monkeypatch):
    import importlib.abc
    import numfast.adapters.arrow as _mod

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("blocked: no-arrow boundary test")
            return None

    saved = {k: v for k, v in sys.modules.items()
             if k == "pyarrow" or k.startswith("pyarrow.")}
    for k in saved:
        del sys.modules[k]
    monkeypatch.setattr(_mod, "pa", None)
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(ImportError, match="numfast\\[arrow\\]"):
            nf.from_arrow([[1]])
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
        import pyarrow as _pa
        monkeypatch.setattr(_mod, "pa", _pa)


# -- clean processes --------------------------------------------------------

def test_clean_process_no_cwd_coupling(tmp_path):
    script = (
        "import sys; "
        "sys.meta_path=[f for f in sys.meta_path "
        "if '__editable___numfast_' not in type(f).__module__]; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "import numfast as nf; "
        "s=nf.from_numpy([1,2,3]); "
        "assert nf.to_numpy(s*2).tolist()==[2,4,6]; "
        "assert s.reduce('sum')==6; "
        "print('clean-process-ok')"
    )
    p = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "clean-process-ok" in p.stdout


def test_clean_process_installed_wheel():
    for venv in ("vfull", "vbare"):
        exe = REPO / "scratch" / ".venvs" / venv / "Scripts" / "python.exe"
        if not exe.exists():
            pytest.skip(f"wheel venv {venv} not built here")
        script = ("import numfast as nf;"
                  "s=nf.from_numpy([4,5,6]);"
                  "assert nf.to_numpy(s+1).tolist()==[5,6,7];"
                  "assert s.reduce('sum')==15;"
                  "print('wheel-ok')")
        p = subprocess.run([str(exe), "-I", "-c", script],
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, f"{venv}: {p.stderr[-2000:]}"
        assert "wheel-ok" in p.stdout
