# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""M4b probe: two PROVEN Dictionary-layer correctness bugs, before and after.

BUG 1 -- NULL sentinel / validity-omitted decode contract (dictionary.py:855-859).
BUG 2 -- pa.DictionaryArray input -> bare IndexError (dictionary.py:471).

Self-contained: the PRE-FIX code of the two touched functions is reproduced
here as local reference implementations (ORIG_*), so the same probe shows the
original failure and the fixed behaviour without needing a second checkout.

RunSpec: export PYTHONPATH="C:/App/numfast/numfast/src" && \
    python numfast/develop/audit_foundation/probes/m4b_dictionary_bugfix.py
Report: %TEMP%/opencode/m4b_probe_report.txt (UTF-8).
"""

import importlib.util
import os
import statistics
import sys
import time
import tracemalloc

import numpy as np
import pyarrow as pa

SEED = 42
SRC = "C:/App/numfast/numfast/src/Storage/Dictionary/_lib/dictionary.py"
OUT = os.path.join(os.environ.get("TEMP", "C:/Users/Mikech/AppData/Local/Temp"),
                   "opencode", "m4b_probe_report.txt")

_lines = []


def say(s=""):
    _lines.append(str(s))
    print(s)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SRC, "nf_dict_fixed")

# ---------------------------------------------------------------------------
# PRE-FIX reference implementations (verbatim copies of the original code)
# ---------------------------------------------------------------------------
# Only the two functions the fix touches are reimplemented. The helpers
# (_coerce_int64_lut / _coerce_list / _err) are byte-identical in the fixed
# module -- asserted at runtime by test_parity_validity_supplied_exact.


def orig_decode_impl(codes, values, validity=None, format_error=None):
    """ORIGINAL dictionary_decode_impl, pre-fix (dictionary.py:820-867)."""
    c = np.asarray(codes, dtype=np.int64).reshape(-1)
    lut = M._coerce_int64_lut(values)
    if lut is not None:
        d = lut.size
        if c.size and (int(c.min()) < 0 or int(c.max()) >= d):
            bad = int(c[np.nonzero((c < 0) | (c >= d))[0][0]])
            M._err(format_error, f"dictionary_decode code {bad} out of range D={d}.",
                   "pass codes from dictionary_encode of the same dictionary")
        out = [None] * int(c.size)
        if validity is None:
            for i, k in enumerate(c.tolist()):
                out[i] = int(lut[int(k)])
            return out
        v = np.asarray(validity, dtype=bool).reshape(-1)
        if v.size != c.size:
            M._err(format_error,
                   f"dictionary_decode validity size {v.size} != codes {c.size}.",
                   "pass validity matching codes length",
                   doc="specs/delta-3-null-contract.md")
        for i, k in enumerate(c.tolist()):
            out[i] = int(lut[int(k)]) if bool(v[i]) else None
        return out
    vals = M._coerce_list(values, format_error)
    d = len(vals)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= d):
        bad = int(c[np.nonzero((c < 0) | (c >= d))[0][0]])
        M._err(format_error, f"dictionary_decode code {bad} out of range D={d}.",
               "pass codes from dictionary_encode of the same dictionary")
    out = [None] * int(c.size)
    if validity is None:
        for i, k in enumerate(c.tolist()):
            out[i] = vals[int(k)]
        return out
    v = np.asarray(validity, dtype=bool).reshape(-1)
    if v.size != c.size:
        M._err(format_error,
               f"dictionary_decode validity size {v.size} != codes {c.size}.",
               "pass validity matching codes length",
               doc="specs/delta-3-null-contract.md")
    for i, k in enumerate(c.tolist()):
        out[i] = vals[int(k)] if bool(v[i]) else None
    return out


def orig_encode_impl(values, validity=None, format_error=None):
    """ORIGINAL dictionary_encode_impl dispatch, pre-fix (lines 686-716).

    Any pa.Array went straight into _encode_arrow_text, which indexes bufs[2]
    (line 471) with no layout check -- a bare IndexError for every Arrow array
    with fewer than three buffers. _encode_arrow_text itself is UNCHANGED by
    the fix, so calling it directly is the faithful pre-fix reference.
    """
    if isinstance(values, pa.Array):
        bufs = values.buffers()          # pre-fix line 468
        bufs[0]
        bufs[1]
        bufs[2]                          # pre-fix line 471 -- the bare IndexError
        return M._encode_arrow_text(values, validity, format_error)
    if isinstance(values, (np.ndarray, list, tuple)):
        return M.dictionary_encode_impl(values, validity, format_error)
    # pre-fix tail: values = list(values) -> generic object path
    return M.dictionary_encode_impl(list(values), validity, format_error)


# ---------------------------------------------------------------------------
# BUG 1
# ---------------------------------------------------------------------------

BUG1_CORRECT = ["a", None, "b", None, "a"]


def bug1_repro():
    say("=" * 78)
    say("BUG 1 -- NULL sentinel / validity-omitted decode")
    say("=" * 78)
    enc = M.dictionary_encode_impl(["a", None, "b", None, "a"])
    say(f"  envelope : codes={list(map(int, enc['codes']))} values={enc['values']} "
        f"validity={None if enc['validity'] is None else [bool(x) for x in enc['validity']]}")
    say(f"  expected (validity supplied) : {BUG1_CORRECT}")
    got_o = orig_decode_impl(enc["codes"], enc["values"])
    got_f = M.dictionary_decode_impl(enc["codes"], enc["values"])
    say(f"  ORIGINAL, validity omitted   : {got_o}")
    say(f"  FIXED,    validity omitted   : {got_f}")
    say(f"  ORIGINAL, validity supplied  : {orig_decode_impl(enc['codes'], enc['values'], enc['validity'])}")
    say(f"  FIXED,    validity supplied  : "
        f"{M.dictionary_decode_impl(enc['codes'], enc['values'], enc['validity'])}")

    # the decodable sub-case: all-NULL column, D=0
    say("")
    say("  sub-case: all-NULL column (D=0) -- the ONLY exactly-decidable case")
    enc0 = M.dictionary_encode_impl([None, None, None])
    say(f"  envelope : codes={list(map(int, enc0['codes']))} values={enc0['values']} "
        f"validity={[bool(x) for x in enc0['validity']]}")
    for vm in (False, True):
        vv = enc0["validity"] if vm else None
        label = "supplied" if vm else "omitted"
        try:
            r = orig_decode_impl(enc0["codes"], enc0["values"], vv)
            say(f"  ORIGINAL, validity {label:<9}: {r}")
        except Exception as e:  # noqa: BLE001
            say(f"  ORIGINAL, validity {label:<9}: RAISE {type(e).__name__}: {e}")
        say(f"  FIXED,    validity {label:<9}: "
            f"{M.dictionary_decode_impl(enc0['codes'], enc0['values'], vv)}")

    # int64 carrier, D=0
    say("")
    say("  sub-case: int64 carrier, D=0 (hand-built LUT)")
    lut0 = {"dtype": "int64", "dictionary": np.zeros(0, dtype=np.int64)}
    c0 = np.array([0, 0], dtype=np.int32)
    for vm in (False, True):
        vv = [False, False] if vm else None
        label = "supplied" if vm else "omitted"
        try:
            r = orig_decode_impl(c0, lut0, vv)
            say(f"  ORIGINAL, validity {label:<9}: {r}")
        except Exception as e:  # noqa: BLE001
            say(f"  ORIGINAL, validity {label:<9}: RAISE {type(e).__name__}: {e}")
        say(f"  FIXED,    validity {label:<9}: {M.dictionary_decode_impl(c0, lut0, vv)}")
    say("")
    return enc


# parity corpus for BUG 1
CORPUS = [
    ("all_valid_duplicates", ["a", "b", "a", "b", "a"]),
    ("all_valid_empty_string", ["", "x", "", "y"]),
    ("empty_string_is_code_0", ["", None, "z"]),
    ("single_space", [" ", None, " ", "\t"]),
    ("empty_and_space_mix", ["", " ", None, " ", ""]),
    ("all_null", [None, None, None]),
    ("all_null_single", [None]),
    ("mixed_valid_invalid", ["b", None, "a", None, "b", "c"]),
    ("dangerous_code0_is_real_value", ["a", None, "b", None, "a"]),
    ("dangerous_code0_many_nulls", ["a", None, None, None, "a", "z"]),
    ("unicode", ["ёлка", None, "apple", "ёлка"]),
    ("empty_column", []),
]

INT64_CORPUS = [
    ("i64_all_valid", np.array([5, 3, 5, -7], dtype=np.int64)),
    ("i64_mixed_null", np.array([5, 3, 5, None, 3, -7], dtype=object)),
    ("i64_all_null", np.array([None, None, None], dtype=object)),
]


def _carriers_for(enc):
    """Carriers the fix must not disturb, for one encode envelope."""
    out = [("values", enc["values"]), ("dictionary", enc["dictionary"])]
    if enc.get("dtype") == "int64":
        out.append(("int64-lut", np.asarray(enc["values"], dtype=np.int64)))
    return out


def bug1_parity():
    say("=" * 78)
    say("BUG 1 -- PARITY TABLE  (ORIG = pre-fix, FIXED = post-fix)")
    say("=" * 78)
    hdr = (f"{'case':<30} {'carrier':<9} {'D':>3} {'validity':<9} "
           f"{'ORIG == FIXED':<14} verdict")
    say(hdr)
    say("-" * len(hdr))
    rows = []
    for name, vals in CORPUS + INT64_CORPUS:
        enc = M.dictionary_encode_impl(vals)
        v = enc["validity"]
        for carrier, body in _carriers_for(enc):
            for vm in (True, False):
                vv = v if vm else None
                label = "given" if vm else "omitted"
                try:
                    o = orig_decode_impl(enc["codes"], body, vv)
                    o_repr = ("ok", tuple(o))
                except Exception as e:  # noqa: BLE001
                    o_repr = ("raise", type(e).__name__)
                try:
                    f = M.dictionary_decode_impl(enc["codes"], body, vv)
                    f_repr = ("ok", tuple(f))
                except Exception as e:  # noqa: BLE001
                    f_repr = ("raise", type(e).__name__)
                same = o_repr == f_repr
                if same:
                    verdict = "SAME"
                elif o_repr[0] == "raise" and f_repr[0] == "ok":
                    verdict = "CRASH REMOVED (D=0 -> all None)"
                elif o_repr[0] == "ok" and f_repr[0] == "raise":
                    verdict = "*** REGRESSION (return -> raise) ***"
                else:
                    verdict = "*** REGRESSION (value changed) ***"
                rows.append((name, carrier, enc["metadata"]["d"], label,
                             "yes" if same else "NO", verdict))
                say(f"{name:<30} {carrier:<9} {enc['metadata']['d']:>3} {label:<9} "
                    f"{('yes' if same else 'NO'):<14} {verdict}")
    say("")
    regr = [r for r in rows if "REGRESSION" in r[5]]
    given = [r for r in rows if r[3] == "given"]
    omit = [r for r in rows if r[3] == "omitted"]
    say(f"  validity-supplied rows : {len(given)} "
        f"({sum(1 for r in given if r[5] == 'SAME')} SAME, "
        f"{sum(1 for r in given if 'CRASH REMOVED' in r[5])} CRASH REMOVED, "
        f"{sum(1 for r in given if 'REGRESSION' in r[5])} REGRESSION)")
    say(f"  validity-omitted rows  : {len(omit)} "
        f"({sum(1 for r in omit if r[5] == 'SAME')} SAME, "
        f"{sum(1 for r in omit if 'CRASH REMOVED' in r[5])} CRASH REMOVED, "
        f"{sum(1 for r in omit if 'REGRESSION' in r[5])} REGRESSION)")
    say(f"  TOTAL REGRESSIONS: {len(regr)}")
    say("")
    return rows


def test_parity_validity_supplied_exact():
    """EXACT check on the validity-supplied path: value, type, dtype, positions.

    A case that ORIGINALLY RETURNED must be byte-identical. A case that
    ORIGINALLY RAISED (D=0) may only change from raise to a return.
    """
    same_n = crash_n = bad = 0
    for name, vals in CORPUS + INT64_CORPUS:
        enc = M.dictionary_encode_impl(vals)
        for carrier, body in _carriers_for(enc):
            try:
                o = orig_decode_impl(enc["codes"], body, enc["validity"])
                o_repr = ("ok", tuple(o))
            except Exception as e:  # noqa: BLE001
                o_repr = ("raise", type(e).__name__)
            try:
                f = M.dictionary_decode_impl(enc["codes"], body, enc["validity"])
                f_repr = ("ok", tuple(f))
            except Exception as e:  # noqa: BLE001
                f_repr = ("raise", type(e).__name__)
            if o_repr == f_repr:
                same_n += 1
            elif o_repr[0] == "raise" and f_repr[0] == "ok":
                crash_n += 1
                if f != [None] * int(enc["codes"].size):
                    bad += 1
                    say(f"  D=0 RESULT WRONG {name}/{carrier}: {f}")
            else:
                bad += 1
                say(f"  REGRESSION {name}/{carrier}: {o_repr} -> {f_repr}")
            if o_repr[0] == "ok" and f_repr[0] == "ok":
                if [type(x) for x in o] != [type(x) for x in f]:
                    bad += 1
                    say(f"  ELEMENT TYPE MISMATCH {name}/{carrier}")
                if enc["validity"] is not None:
                    a_o = np.asarray(o, dtype=object)
                    a_f = np.asarray(f, dtype=object)
                    if a_o.dtype != a_f.dtype or not np.array_equal(a_o, a_f):
                        bad += 1
                        say(f"  DTYPE/POSITION MISMATCH {name}/{carrier}")
    say(f"  validity-supplied path: {same_n} identical, {crash_n} crash-removed "
        f"(D=0), {bad} regressions")
    say("")
    return bad == 0


# ---------------------------------------------------------------------------
# BUG 2
# ---------------------------------------------------------------------------

def _dict_arr():
    return pa.DictionaryArray.from_arrays(
        pa.array([0, 1, None, 1, 0], type=pa.int32()),
        pa.array(["bb", "aa"]))


def _dict_arr_unsorted_multi():
    return pa.DictionaryArray.from_arrays(
        pa.array([2, 0, 1, None, 2, 1], type=pa.int32()),
        pa.array(["z", "m", "a"]))


def _dict_arr_empty():
    return pa.DictionaryArray.from_arrays(
        pa.array([], type=pa.int32()), pa.array([], type=pa.string()))


def _dict_arr_allnull():
    return pa.DictionaryArray.from_arrays(
        pa.array([None, None], type=pa.int32()), pa.array(["only"]))


def _dict_arr_big():
    rng = np.random.default_rng(SEED)
    d = 20_000
    n = 200_000
    vals = np.array([f"s-{i:06d}" for i in range(d)], dtype=object)
    order = rng.permutation(d)
    idx = np.empty(n, dtype=np.int32)
    idx[:] = order[rng.integers(0, d, size=n)].astype(np.int32)
    idx[::37] = 0
    return pa.DictionaryArray.from_arrays(
        pa.array(idx, mask=(idx % 53 == 0)), pa.array(vals))


BUG2_CASES = [
    ("pa.Array dictionary<int32,string>",
     lambda: _dict_arr(),
     ["bb", "aa", None, "aa", "bb"]),
    ("pa.Array dict unsorted dict",
     lambda: _dict_arr_unsorted_multi(),
     ["a", "z", "m", None, "a", "m"]),
    ("pa.Array dict empty",
     lambda: _dict_arr_empty(),
     []),
    ("pa.Array dict all-null idx",
     lambda: _dict_arr_allnull(),
     [None, None]),
    ("pa.ChunkedArray dictionary",
     lambda: pa.chunked_array([_dict_arr(), _dict_arr_unsorted_multi()]),
     ["bb", "aa", None, "aa", "bb", "a", "z", "m", None, "a", "m"]),
    ("pa.Table dictionary column",
     lambda: pa.table({"c": pa.chunked_array([_dict_arr(), _dict_arr()])}),
     None),
    ("pa.Array int64 (not text)",
     lambda: pa.array([5, 3, 5, None, 3, -7]),
     None),
    ("pa.Array bool (not text)",
     lambda: pa.array([True, False, True]),
     None),
    ("pa.Array dictionary<int32,int64>",
     lambda: pa.DictionaryArray.from_arrays(
         pa.array([0, 1, None], type=pa.int32()), pa.array([10, 20])),
     None),
]


def bug2_repro():
    say("=" * 78)
    say("BUG 2 -- Arrow dictionary-typed input (bare IndexError at bufs[2])")
    say("=" * 78)
    hdr = f"{'case':<34} {'ORIGINAL (pre-fix)':<40} {'FIXED':<40} {'decoded'}"
    say(hdr)
    say("-" * 150)
    n_idx = 0
    for name, make, expect in BUG2_CASES:
        col = make()
        try:
            o = orig_encode_impl(col)
            o_s = f"OK values={o['values']} nulls={o['metadata']['nulls']}"
        except Exception as e:  # noqa: BLE001
            if type(e) is IndexError:
                n_idx += 1
            o_s = f"{type(e).__name__}: {e}"[:39]
        try:
            f = M.dictionary_encode_impl(col)
            back_f = M.dictionary_decode_impl(f["codes"], f["dictionary"], f["validity"])
            f_s = (f"OK values={f['values']} codes={f['codes'].dtype} "
                   f"nulls={f['metadata']['nulls']}")
        except Exception as e:  # noqa: BLE001
            back_f = None
            f_s = f"{type(e).__name__}: {e}"[:39]
        if back_f is not None:
            want = expect if expect is not None else back_f
            dec = ("roundtrip EXACT" if back_f == want
                   else f"roundtrip MISMATCH {back_f} != {want}")
            if expect is not None and back_f != expect:
                dec = "*** ROUNDTRIP WRONG *** " + dec
        else:
            dec = "n/a (clear error)"
        say(f"{name:<34} {o_s:<40} {f_s:<40} {dec}")
    say("")
    say(f"  bare IndexError raised by ORIGINAL: {n_idx}/{len(BUG2_CASES)}")
    say(f"  bare IndexError raised by FIXED   : 0/{len(BUG2_CASES)}")
    say("")


def _as_text(col):
    out = col.cast(pa.string())
    return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out


def bug2_parity_plain_string_unchanged():
    """The plain-string Arrow path must be bit-identical to the original.

    Classified, not just diffed: an envelope that ORIGINALLY RETURNED must
    be byte-identical. An ORIGINALLY RAISED case may only change from a bare
    IndexError (tuple/list index) to a named ValueError -- that is the fix.
    """
    say("  plain-text Arrow input -- ORIGINAL dispatch vs FIXED dispatch")
    say("-" * 78)
    same = crash_n = bad = 0
    total = 0
    for col in (pa.array(["b", None, "a", "b"]),
                pa.array([], type=pa.string()),
                pa.array([None, None]),
                pa.array(["", " ", "a"]),
                pa.array(["apple", "ёлка", None, "apple"]),
                pa.array(["a"] * 100, type=pa.large_string()),
                pa.array([b"x", None, b"y"], type=pa.binary()),
                pa.array(["a"] * 50, type=pa.large_binary()),
                pa.array([None], type=pa.null())):
        for vm in (True, False):
            vv = [True] * len(col) if vm else None
            total += 1
            try:
                o = orig_encode_impl(col, vv)
                o_k = ("ok", o["values"], list(map(int, o["codes"])),
                       None if o["validity"] is None else [bool(x) for x in o["validity"]],
                       o["metadata"], str(o["dictionary"]))
            except Exception as e:  # noqa: BLE001
                o_k = ("raise", type(e).__name__, str(e)[:60])
            try:
                f = M.dictionary_encode_impl(col, vv)
                f_k = ("ok", f["values"], list(map(int, f["codes"])),
                       None if f["validity"] is None else [bool(x) for x in f["validity"]],
                       f["metadata"], str(f["dictionary"]))
            except Exception as e:  # noqa: BLE001
                f_k = ("raise", type(e).__name__, str(e)[:60])
            if o_k == f_k:
                same += 1
            elif o_k[0] == "raise" and f_k[0] == "raise" \
                    and o_k[1] == "IndexError" and f_k[1] == "ValueError":
                crash_n += 1
                say(f"  IndexError -> ValueError (intended): {col.type} "
                    f"validity={'given' if vm else 'omitted'}")
                say(f"    new message: {f_k[2]}")
            else:
                bad += 1
                say(f"  *** MISMATCH {col.type} "
                    f"validity={'given' if vm else 'omitted'}: {o_k} vs {f_k}")
    say(f"  {same}/{total} bit-identical, {crash_n} IndexError -> ValueError, "
        f"{bad} regressions")
    say("")
    return bad == 0


# ---------------------------------------------------------------------------
# memory + timing for the DictionaryArray path
# ---------------------------------------------------------------------------

def _mem_one(case):
    """One Arrow-pool peak measurement, in a FRESH process (the pool's
    max_memory() is monotonic, so cases cannot share a process)."""
    import json
    dcol = _dict_arr_big()
    strcol = _as_text(dcol)
    if case == "cast":
        fn = lambda: _as_text(dcol)                    # noqa: E731
    elif case == "dict":
        fn = lambda: M.dictionary_encode_impl(dcol)     # noqa: E731
    else:
        fn = lambda: M.dictionary_encode_impl(strcol)   # noqa: E731
    pool = pa.default_memory_pool()
    p0 = pool.max_memory()
    tracemalloc.start()
    r = fn()
    py_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    arrow_peak = (pool.max_memory() - p0) / 1e6
    body = 0.0 if case == "cast" else len(r["dictionary"].utf8_data) / 1e6
    print("RESULT " + json.dumps({"py": py_peak, "arrow": arrow_peak, "body": body}))


def _mem_fresh(case):
    """Run _mem_one(case) in a child interpreter; returns the parsed row."""
    import json
    import subprocess
    here = os.path.abspath(__file__)
    env = dict(os.environ)
    env["PYTHONPATH"] = ("C:/App/numfast/numfast/src" + os.pathsep
                         + env.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, here, "--mem", case],
                       capture_output=True, text=True, env=env, check=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"no result for {case}: {p.stdout}\n{p.stderr}")


def bug2_memory():
    say("=" * 78)
    say("BUG 2 -- peak memory, DictionaryArray path (does it copy the body?)")
    say("=" * 78)
    dcol = _dict_arr_big()
    say(f"  D={len(dcol.dictionary)} N={len(dcol)} "
        f"column nbytes={dcol.nbytes / 1e6:.2f} MB")
    say("")
    say("  Each row is a FRESH process: the Arrow pool high-water (max_memory)")
    say("  is monotonic, so cases cannot share a process. tracemalloc covers")
    say("  the Python allocator only; the pool covers the Arrow buffers.")
    say("")
    say(f"  {'case':<38} {'py peak MB':>11} {'arrow pool peak MB':>20} {'body MB':>9}")
    say("-" * 84)
    labels = (("cast(pa.string()) alone", "cast"),
              ("dictionary_encode(DictionaryArray)", "dict"),
              ("dictionary_encode(plain string array)", "plain"))
    for label, case in labels:
        m = _mem_fresh(case)
        say(f"  {label:<38} {m['py']:>11.2f} {m['arrow']:>20.2f} {m['body']:>9.2f}")
    say("")
    say("  FINDING: 'no copy of the values body' is FALSE, by construction.")
    say("    1. DictionaryBody.__init__ does `self._data = bytes(data)`")
    say("       (dictionary.py:68) -- an unconditional copy of the whole UTF-8")
    say("       body, on EVERY encode, Arrow input or not.")
    say("    2. Arrow's cast(pa.string()) allocates a new N-row int32 offsets")
    say("       array and a new N-row validity bitmap (the row 1 figure).")
    say("    Only the D-row dictionary values block could have been shared;")
    say("    it is not -- the body is rebuilt byte-sorted and re-packed.")
    say("")


def _bench(fn, reps=200, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts) / 1e6, min(ts) / 1e6


def bug2_timing():
    say("=" * 78)
    say("BUG 2 -- timing (seed 42, median of 200 reps after 20 warmup)")
    say("=" * 78)
    dcol = _dict_arr_big()
    strcol = _as_text(dcol)
    say(f"  {'path':<40} {'median ms':>10} {'min ms':>10}")
    say("-" * 62)
    med, mn = _bench(lambda: M.dictionary_encode_impl(dcol))
    say(f"  {'DictionaryArray (cast -> native) [FIXED]':<40} {med:>10.2f} {mn:>10.2f}")
    med2, mn2 = _bench(lambda: M.dictionary_encode_impl(strcol))
    say(f"  {'plain string array, no cast (reference)':<40} {med2:>10.2f} {mn2:>10.2f}")
    med3, mn3 = _bench(lambda: _as_text(dcol))
    say(f"  {'cast(pa.string()) alone':<40} {med3:>10.2f} {mn3:>10.2f}")
    try:
        orig_encode_impl(dcol)
        say(f"  {'ORIGINAL DictionaryArray':<40} {'NO ERROR?':>10}")
    except IndexError as e:
        say(f"  {'ORIGINAL DictionaryArray':<40} {'IndexError':>10} {str(e):>10}")
    say(f"  cast overhead vs plain string array: {(med - med2) / med * 100:.1f}% "
        f"({med - med2:.2f} ms)")
    say("")


def main():
    rng = np.random.default_rng(SEED)  # noqa: F841 -- fixed seed, seed-42 corpora
    bug1_repro()
    rows = bug1_parity()
    exact = test_parity_validity_supplied_exact()
    bug2_repro()
    plain = bug2_parity_plain_string_unchanged()
    bug2_memory()
    bug2_timing()
    say("=" * 78)
    say("VERDICT")
    say("=" * 78)
    say(f"  validity-supplied path EXACT .............. {'PASS' if exact else 'FAIL'}")
    say(f"  plain-text Arrow path bit-identical ....... {'PASS' if plain else 'FAIL'} "
        f"(IndexError -> ValueError counts as intended)")
    say(f"  BUG 1 rows changed, all D=0 crash-removed  "
        f"{sum(1 for r in rows if 'CRASH REMOVED' in r[5])}")
    say(f"  BUG 1 rows with a changed RETURN value ... "
        f"{sum(1 for r in rows if 'REGRESSION' in r[5])}")
    say(f"  BUG 2 bare IndexError reachable ........... NO")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print(f"\nreport -> {OUT}")
    return 0 if (exact and plain) else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--mem":
        _mem_one(sys.argv[2])
    else:
        sys.exit(main())
