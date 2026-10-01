# M4b — Dictionary layer: two proven correctness bugs fixed

Target: `numfast/src/Storage/Dictionary/_lib/dictionary.py` (only file changed in `numfast/src/`).
Test files touched: `numfast/tests/fast/test_ops_dictionary.py`, `numfast/tests/fast/test_ops_dict_int64.py`.
Probe: `develop/audit_foundation/probes/m4b_dictionary_bugfix.py`.
Fail-before harness: `develop/audit_foundation/probes/m4b_make_pre_fix.py`.

RunSpec: `export PYTHONPATH="C:/App/numfast/numfast/src" && python numfast/develop/audit_foundation/probes/m4b_dictionary_bugfix.py`
Report sink: `%TEMP%/opencode/m4b_probe_report.txt` (UTF-8, 172 lines).

---

## BUG 1 — NULL sentinel / `validity`-omitted decode

### STATUS

**PARTIALLY FIXED — the reported reproduction is NOT a fixable defect; one exact
sub-defect IS fixed. The D>0 half is reported as an OPEN CONTRACT QUESTION.**

### EVIDENCE

| | |
|---|---|
| File / function | `Storage/Dictionary/_lib/dictionary.py`, `dictionary_decode_impl` |
| Reported line | `855-859` (TEXT branch, `validity is None` gather) |
| Real defect lines | `851` (TEXT range check) and `832` (int64 range check) — the range check runs **before** `validity` is consulted, so `D=0` raises |
| Lines changed | `384-429` (new `_arrow_text_layout`/`_arrow_dict_text` — BUG 2 only), `877-892` (new `_all_null`), `894-909` (docstring contract), `914-915` (int64 `D=0` guard), `935-936` (TEXT `D=0` guard) |

Before / after, `dictionary_encode(["a", None, "b", None, "a"])`:

```
envelope : codes=[0,0,1,0,0] values=['a','b'] validity=[True,False,True,False,True]
ORIGINAL, validity omitted  : ['a', 'a', 'b', 'a', 'a']      <- the reported repro
FIXED,    validity omitted  : ['a', 'a', 'b', 'a', 'a']      <- UNCHANGED, by decision
ORIGINAL, validity supplied : ['a', None, 'b', None, 'a']
FIXED,    validity supplied : ['a', None, 'b', None, 'a']    <- EXACT
```

Before / after, all-NULL column `dictionary_encode([None, None, None])` (D=0):

```
ORIGINAL, validity omitted  : RAISE ValueError: dictionary_decode code 0 out of range D=0
FIXED,    validity omitted  : [None, None, None]
ORIGINAL, validity supplied : RAISE ValueError: dictionary_decode code 0 out of range D=0   <- ALSO broken
FIXED,    validity supplied : [None, None, None]
```

Same for the int64 LUT carrier (`{"dtype":"int64","dictionary": np.zeros(0)}`):
`RAISE code 0 out of range D=0` → `[None, None]`, on both paths.

**Scope correction, reported loudly:** the `D=0` crash is present on the
`validity`-**supplied** path too, not only on the omitted path. The mission
scoped BUG 1 to the omitted path. Because the same two lines caused both
crashes, both were fixed; shipping only the omitted half would have left the
same all-NULL envelope decoding fine one way and crashing the other.

### ROOT CAUSE

Two distinct things, and the mission conflated them.

1. **The reported repro is not information loss that decode can repair — it is
   information loss in the caller.** Under DELTA-3 the NULL placeholder *is*
   code 0 (`dictionary.py:18-20`). Under the deterministic sorted-unique encode
   (`code == sorted rank`, `dictionary.py:17-18`) code 0 is *also* the rank of
   the smallest real value whenever `D >= 1`. Once `validity` is dropped, the
   function receives only `codes` and `values`; there is no third input that
   distinguishes "NULL row" from "genuine value at code 0". The repro proves
   this directly: `values = ['a','b']`, and `'a'` is a real value at code 0 —
   the "leaked" positions show `'a'` for the same reason the genuine
   code-0 positions show `'a'`.
2. **A real, exactly-decidable defect: `D == 0`.** With an empty dictionary no
   value can legally sit at code 0, so every row is NULL. The range check
   `int(c.max()) >= d` fired `0 >= 0` and raised a misleading
   "pass codes from dictionary_encode of the same dictionary" error — an error
   that names the *opposite* of the actual problem. This affected both the
   validity-supplied and the validity-omitted path.

### CONTRACT DECISION AND ITS JUSTIFICATION

**Contract chosen: `validity` omitted == "every code in `codes` is valid";
decode is then a pure gather. D>0 behaviour is UNCHANGED. The D=0 case is
special-cased to all-`None` because it is the only case that is exactly
decidable without the sidecar.**

Neither option offered by the mission is shippable, and the call-site evidence
says so:

- **Option (a) "return `None` for the NULL sentinel positions" is
  information-theoretically impossible for `D > 0`.** `dictionary_decode_impl`'s
  only inputs are `codes` and `values`. Code 0 is a legitimate, populated slot
  in every non-empty sorted-unique dictionary. There is no sentinel to consult
  — the mission's own dangerous case ("a body where code 0 is ALSO a
  legitimately present dictionary value") is the *normal* case, not an edge
  case, so "code 0 means NULL" would itself be the bug. Confirmed empirically:
  `dictionary_encode(["", None, "z"])` → `values == ["", "z"]`, `codes == [0,0,1]`
  — the empty string legitimately occupies the NULL code.
- **Option (b) "raise a clear error demanding `validity`" breaks an existing
  test AND a production caller.**
  - `numfast/tests/fast/test_ops_dictionary.py:56`
    (`test_encode_all_valid_validity_none`): `r = dictionary_encode(["x","y"])`,
    `assert r["validity"] is None`, then
    `assert dictionary_decode(r["codes"], r["values"], r["validity"]) == ["x","y"]`.
    `r["validity"]` **is** `None`, passed positionally, and the test demands a
    full materialisation. Option (b) fails this existing test. **This test
    asserts the current behaviour and is left untouched — it is not a test of a
    bug, it is a test of a documented, production-used contract.**
  - `numfast/src/numfast/_lib/series.py:120-121` (`Series.to_numpy`, the text
    branch) calls `dictionary_decode(self._values, self._sidecar["values"],
    self._validity)`. `Series._validity` is `None` for any all-valid text
    Series (`series.py:102-103`), and `to_numpy` relies on the full
    materialisation. Option (b) would break every all-valid text Series export
    to numpy.
  - The other in-repo caller, `numfast/src/numfast/adapters/arrow.py:127`, also
    passes `validity=None` implicitly (it never supplies one).

So the fix ships the contract as an explicit, documented statement (the
docstring previously said only `"(invalid -> None)"`, which is what made the
omitted-validity case look like an accident) plus the one exact D=0 special
case, and the D>0 half is escalated as an open contract question below.

### FIX (the actual diff)

```python
def _all_null(codes, d, validity, format_error):
    """D==0 -> [None]*N (every row NULL, exactly).

    DELTA-3 gives an all-NULL column a zero-length dictionary, so no value
    can legally sit at code 0 and every code is the NULL placeholder --
    true with or without the sidecar, since there is nothing to gather. A
    negative code is still an error (corrupt carrier, never a NULL row), and
    a sidecar of the wrong length still fails loudly.
    """
    if codes.size and int(codes.min()) < 0:
        bad = int(codes[np.nonzero(codes < 0)[0][0]])
        _err(format_error, f"dictionary_decode code {bad} out of range D={d}.",
             "pass codes from dictionary_encode of the same dictionary")
    if validity is not None:
        v = np.asarray(validity, dtype=bool).reshape(-1)
        if v.size != codes.size:
            _err(format_error,
                 f"dictionary_decode validity size {v.size} != codes {codes.size}.",
                 "pass validity matching codes length",
                 doc="specs/delta-3-null-contract.md")
    return [None] * int(codes.size)
```

Docstring, `dictionary_decode_impl` (the contract, now explicit):

```
    ``validity`` omitted == "every code in ``codes`` is valid" (the encode
    envelope returns ``validity=None`` for an all-valid column): decode is
    then a pure gather over the LUT. It CANNOT restore NULLs, and must not
    pretend to -- under DELTA-3 the NULL placeholder IS code 0, which is
    also the sorted rank of the smallest real value, so NULL and a genuine
    value are indistinguishable once the sidecar is dropped. Pass the
    envelope's ``validity`` to get NULLs back. The one exact exception is
    D=0 (an all-NULL column): no value exists, so every row decodes to None.
```

Two-line guard in each branch (int64 at `914-915`, TEXT at `935-936`):

```python
     d = lut.size
+    if d == 0:
+        return _all_null(c, d, validity, format_error)
     if c.size and (int(c.min()) < 0 or int(c.max()) >= d):
```
```python
     d = len(vals)
+    if d == 0:
+        return _all_null(c, d, validity, format_error)
     if c.size and (int(c.min()) < 0 or int(c.max()) >= d):
```

No public signature changed. No dtype widened. `_all_null` returns `[None]*N`
— element types identical to what the validity-supplied path already returned
for a D=0 envelope would have been.

### PARITY TABLE (probe, 15 corpora x carriers x both validity modes = 64 rows)

Corpora include: all-valid duplicates, all-valid with `""`, `""` at code 0,
single-space / tab, `""`+space mix, all-NULL, all-NULL single row, mixed
valid/invalid, **code-0-is-a-real-value with NULLs** (`["a",None,"b",None,"a"]`,
`["a",None,None,None,"a","z"]`), unicode, empty column, plus int64 all-valid /
mixed-NULL / all-NULL. Carriers per envelope: `values`, native
`dictionary` (DictionaryBody or int64 ndarray), plus `int64-lut` for int64
envelopes.

```
validity-supplied rows : 32 (26 SAME, 6 CRASH REMOVED (D=0), 0 REGRESSION)
validity-omitted  rows : 32 (26 SAME, 6 CRASH REMOVED (D=0), 0 REGRESSION)
TOTAL REGRESSIONS: 0

validity-supplied path: 26 identical, 6 crash-removed (D=0), 0 regressions
```

Element-type, `np.asarray(..., dtype=object)` dtype, and NULL-position
equality were compared on every validity-supplied row, not just list equality.
**Verdict: 0 changed return values. The only behavioural delta is
raise -> `[None]*N` for `D=0`, which is the fix.**

### OPEN CONTRACT QUESTION (not shipped — Coordinator's call)

Should omitting `validity` on a `D > 0` dictionary be an error?

- **Status quo (shipped):** pure gather. Silent for a caller who dropped the
  sidecar. Backed by `test_ops_dictionary.py:56` and `series.py:120-121`.
- **Option (b):** `raise ValueError("dictionary_decode needs validity to
  restore NULLs")` when `validity is None` and `D > 0`. Catches the silent
  corruption, but breaks `test_ops_dictionary.py:56` and every all-valid text
  `Series.to_numpy()`.
- **Option (c), not evaluated here:** carry NULL in the codes themselves via a
  reserved code (e.g. `-1` or `D`). This changes the ABI (compute paths, GPU
  codes contract, every `codes`-consuming op) and is out of scope for this
  step.

Recommendation is deliberately not made: this is a contract change with
caller-visible breakage, i.e. a Coordinator/Architect decision.

---

## BUG 2 — `pa.DictionaryArray` input -> bare `IndexError`

### STATUS

**FIXED — option (a), the Arrow dictionary-typed input is now accepted.**

### EVIDENCE

| | |
|---|---|
| File / function | `Storage/Dictionary/_lib/dictionary.py`, `_encode_arrow_text` (indexes `bufs[2]`) reached unconditionally from `dictionary_encode_impl` |
| Reported line | `471` (`data_buf = bufs[2]`) |
| Dispatch site changed | `725-741` (was `725-727`) |
| Lines added | `384-429` (`_arrow_text_layout`, `_arrow_dict_text`) |
| Public signature changed | none |

Before (exact exception text, not paraphrased):
`IndexError: list index out of range` — `arr.buffers()` returns a **list**, so
the message is "list index out of range", not "tuple index out of range" as the
mission stated. An Arrow dictionary array has 2 buffers (1-byte validity +
N*4 indices), not 3.

**Scope correction, reported loudly:** the same unguarded `bufs[2]` is not
dictionary-specific. Every Arrow array with fewer than 3 buffers hit it:

| input | ORIGINAL | FIXED |
|---|---|---|
| `pa.Array dictionary<int32,string>` | `IndexError: list index out of range` | `OK values=['aa','bb'] codes=int32 nulls=1`, roundtrip EXACT |
| `pa.Array` dict, unsorted dictionary | `IndexError` | `OK values=['a','m','z']`, roundtrip EXACT |
| `pa.Array` dict, empty | `IndexError` | `OK values=[]`, roundtrip EXACT |
| `pa.Array` dict, all-null indices | `IndexError` | `OK values=[] nulls=2`, roundtrip EXACT |
| `pa.ChunkedArray` dictionary | `ValueError: ... got DictionaryScalar` (misleading) | `OK values=['a','aa','bb','m','z']`, roundtrip EXACT |
| `pa.Table` with dictionary column | `ValueError: ... needs a rank-1 column, got shape (1, 10)` (misleading) | `ValueError: dictionary_encode got Table, a table of columns, not a column.` |
| `pa.Array` int64 | `IndexError` | `ValueError: dictionary_encode needs a TEXT Arrow array, got int64.` |
| `pa.Array` bool | `IndexError` | `ValueError: ... got bool.` |
| `pa.Array dictionary<int32,int64>` | `IndexError` | `ValueError: ... got dictionary<values=int64, indices=int32, ordered=0>.` |
| `pa.Array` null type | `IndexError` | `ValueError: ... got null.` |

```
bare IndexError raised by ORIGINAL: 7/9
bare IndexError raised by FIXED   : 0/9
```

`pa.array([5, 3, 5, None, 3, -7])` (int64) also raised `IndexError` before the
fix — an undocumented extra instance of the same root cause. Fixing only the
dictionary case would have left the bare `IndexError` reachable from the same
public entry point, so the layout guard covers both.

### ROOT CAUSE

`dictionary_encode_impl` dispatched to `_encode_arrow_text` for **any**
`pa.Array` (pre-fix lines 686-687) with no type/layout check.
`_encode_arrow_text` then assumed the 3-buffer TEXT layout
(`bufs[0]=validity, bufs[1]=offsets, bufs[2]=data`) and indexed `bufs[2]`
blindly. A dictionary array carries an index stream, not offsets+data; numeric,
boolean, temporal and null arrays carry data with no offsets buffer.

### FIX (the actual diff)

New helpers, `dictionary.py:384-429`:

```python
def _arrow_text_layout(arr):
    """True when ``arr`` is a TEXT/binary Arrow array (the 3-buffer layout).

    Gate for the Arrow-native path: a non-text Arrow array carries fewer
    than three buffers, so indexing ``bufs[2]`` on it is a raw IndexError.
    binary/large_binary stay here on purpose -- their offsets buffer is the
    3-buffer layout _encode_arrow_text already reads.
    """
    if pa is None:
        return False
    try:
        t = arr.type
        return (pa.types.is_string(t) or pa.types.is_large_string(t)
                or pa.types.is_binary(t) or pa.types.is_large_binary(t))
    except Exception:
        return False


def _arrow_dict_text(values):
    """Arrow dictionary-typed carrier -> plain text pa.Array, else None.

    Reuses the ONE canonical normalisation already proven in the Arrow
    adapter (``col.cast(pa.string()).combine_chunks()``): Arrow rewrites
    the index stream into string offsets over the dictionary's own value
    buffer, after which the existing Arrow-native text path owns dedup and
    the sorted-unique re-rank. Non-text dictionaries return None (never a
    silent stringify -- an int64 dictionary must reach the int64 path).
    """
    if pa is None:
        return None
    try:
        t = values.type
        if not pa.types.is_dictionary(t):
            return None
        if not (pa.types.is_string(t.value_type)
                or pa.types.is_large_string(t.value_type)):
            return None
        out = values.cast(pa.string())
        # Array.cast -> StringArray (already single chunk); the Arrow adapter
        # casts a ChunkedArray, whose cast stays chunked and needs merging.
        return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out
    except Exception:
        return None
```

Dispatch, `dictionary.py:725-741` (was `725-727`):

```python
     # --- Arrow-native path: skip to_pylist + object array + Python dedup ---
+    if pa is not None and isinstance(values, (pa.Table, pa.RecordBatch)):
+        _err(format_error,
+             f"dictionary_encode got {type(values).__name__}, a table of "
+             "columns, not a column.",
+             "pass one column: table.column('name') or table['name']")
+    if pa is not None and isinstance(values, (pa.Array, pa.ChunkedArray)):
+        # Dictionary-encoded text carrier: normalize with the canonical Arrow
+        # cast, then the plain text path owns the rest. Without this the
+        # index stream reaches bufs[2] below as a bare IndexError.
+        _dt = _arrow_dict_text(values)
+        if _dt is not None:
+            return _encode_arrow_text(_dt, validity, format_error)
     if pa is not None and isinstance(values, pa.Array):
+        if not _arrow_text_layout(values):
+            _err(format_error,
+                 f"dictionary_encode needs a TEXT Arrow array, got {values.type}.",
+                 "pass pa.string()/pa.large_string() data, a dictionary-encoded "
+                 "string column, or a [str|None] / int64 column")
         return _encode_arrow_text(values, validity, format_error)
```

### CONTRACT DECISION AND ITS JUSTIFICATION

**Option (a), accept — and it is safe because no existing caller can break.**

`numfast/src/numfast/adapters/arrow.py:123-127`, the only in-`src` caller that
feeds `dictionary_encode` an Arrow column, ALREADY does the normalisation
itself:

```python
if pa.types.is_dictionary(t):
    arr_col = col.cast(pa.string()).combine_chunks()      # arrow.py:124
else:
    arr_col = combined
enc = a["dictionary_encode"](arr_col)                     # arrow.py:127
```

So (1) no production caller ever passed a `pa.DictionaryArray` — the crash was
unreachable from `from_arrow`, and (2) the primitive this fix reuses is the one
the codebase already treats as canonical, not an invention. `arrow.py` was not
modified (explicitly out of scope for this step); the fix makes
`dictionary_encode` accept what `arrow.py` was already normalising into it, so
the cast can eventually be dropped there without behaviour change.

Why `cast(pa.string())` and not a hand-rolled index->body build: the
sorted-unique re-rank, the byte dedup, the D-scale body build and the
validity/3VL handling all already exist and are proven in
`_encode_arrow_text` and `_arrow_cxx_sorted_encode`. A hand-rolled dictionary
decoder would be a second implementation of the same logic, which
`00_core.md` forbids ("Reuse existing modules before creating new ones").

Non-text dictionaries (`dictionary<int32,int64>`) are **rejected, not
stringified**: `cast(pa.string())` on them succeeds and would silently produce
a TEXT dictionary from an int64 column, violating the int64 dtype contract.
Plain `pa.Array` int64 columns are also rejected rather than routed to
`_encode_int64_impl` — that would be new feature work, and the guard turns a
crash into a named error instead. See RISKS.

### MEASURE — memory (each row a FRESH process; the Arrow pool high-water is
monotonic, so cases cannot share a process; `tracemalloc` covers the Python
allocator only, the pool covers the Arrow buffers)

```
D=20000 N=200000 column nbytes=1.06 MB

case                                    py peak MB   arrow pool peak MB   body MB
cast(pa.string()) alone                       0.00                 2.35      0.00
dictionary_encode(DictionaryArray)            6.14                 2.35      0.16
dictionary_encode(plain string array)         6.14                 0.00      0.16
```

**The "no copy of the values body" claim is FALSE, and the measurement says
why:**

1. `DictionaryBody.__init__` does `self._data = bytes(data)`
   (`dictionary.py:68`) — an unconditional copy of the entire UTF-8 body, on
   every encode, Arrow input or not. The `plain string array` row shows 0.00 MB
   of Arrow-pool allocation yet still materialises a 0.16 MB `body`, entirely
   inside the 6.14 MB Python peak.
2. Arrow's `cast(pa.string())` adds a 2.35 MB Arrow-pool peak: a new N-row
   int32 offsets array and a new N-row validity bitmap (N=200 000).
3. Only the D-row dictionary values block could plausibly have been shared,
   and it is not: the body is rebuilt byte-sorted and re-packed.

Net cost of the accepted path: **+2.35 MB peak, +8.1% wall time**, for a
carrier that previously raised.

### MEASURE — timing (seed 42, `time.perf_counter_ns`, median of 200 reps after 20 warmup)

```
path                                      median ms     min ms
DictionaryArray (cast -> native) [FIXED]      52.43      46.15
plain string array, no cast (reference)       48.16      42.02
cast(pa.string()) alone                        3.69       3.29
ORIGINAL DictionaryArray                 IndexError list index out of range
cast overhead vs plain string array: 8.1% (4.27 ms)
```

### PARITY — plain-text Arrow path must be untouched

9 input shapes (`string`, empty `string`, all-NULL `string`, `""`/space mix,
unicode, `large_string`, `binary`, `large_binary`, `null`) x validity given /
omitted = 18 encodes, compared on the full envelope (`values`, `codes`,
`validity`, `metadata`, `DictionaryBody` repr):

```
14/18 bit-identical, 4 IndexError -> ValueError, 0 regressions
```

The 4 are the `null`-typed array, where `IndexError` becomes the named
`ValueError: dictionary_encode needs a TEXT Arrow array, got null.` — the
intended fix, not a regression. No input that previously **returned** an
envelope changed by one bit.

---

## Regression tests

`numfast/tests/fast/test_ops_dictionary.py` (7 added, after
`test_metadata_sorted_flag`):

| test | before fix | after fix |
|---|---|---|
| `test_decode_all_null_column_d0_returns_nulls_not_error` | **FAIL** | PASS |
| `test_decode_d0_still_rejects_negative_code_and_bad_validity` | **FAIL** | PASS |
| `test_encode_arrow_dictionary_array_accepted` | **FAIL** | PASS |
| `test_encode_arrow_dictionary_chunked_and_unsorted_dict` | **FAIL** | PASS |
| `test_encode_arrow_rejects_bad_types_with_a_named_error` | **FAIL** | PASS |
| `test_decode_validity_omitted_is_a_pure_gather_not_a_null_restore` | PASS | PASS |
| `test_decode_code0_is_a_real_value_not_a_null_sentinel` | PASS | PASS |
| `test_encode_arrow_plain_string_path_unchanged` | PASS | PASS |

`numfast/tests/fast/test_ops_dict_int64.py` (1 added, after
`test_decode_and_metadata_carriers`):

| test | before fix | after fix |
|---|---|---|
| `test_int64_decode_d0_returns_nulls_not_out_of_range` | **FAIL** | PASS |

**Reported honestly: 6 of the 9 new tests fail against the original code and
pass after. The other 3 pass both before and after BY DESIGN — they are
parity locks, not regression tests.** They pin the contract this step
deliberately did *not* change: the validity-omitted pure gather, code 0 as a
real value, and the untouched plain-text Arrow path. Removing the fix would not
break them, which is the point.

The pre-fix code was reconstructed mechanically by
`develop/audit_foundation/probes/m4b_make_pre_fix.py` (5 exact reverse
patches, asserting the result is exactly 886 lines — the original's line count —
and carries none of the new symbols), swapped in, tested, and swapped back. The
restored file was hash-verified against the pre-swap backup
(`sha256[:16] = dba40c712560b541`, 43663 bytes).

### Suite counts

Command form: `PYTHONPATH=".../numfast/src;.../app-builder" python -m pytest numfast/tests/fast -p no:cacheprovider -q`

`app-builder` is added **only for measurement** so the `kernel` fixture can
build; `framework-builder/` is absent as the RunSpec states, and the canonical
RunSpec form produces 29 `ModuleNotFoundError: No module named 'builder'`
setup errors instead. No file was changed to enable this.

| | ORIGINAL `dictionary.py` | FIXED `dictionary.py` |
|---|---|---|
| `numfast/tests/fast` (whole dir) | 8 failed, 445 passed, 1 skipped, 32 errors | **2 failed, 451 passed, 1 skipped, 32 errors** |
| `test_ops_dictionary.py` | 3 failed, 12 passed | **0 failed, 15 passed** |
| `test_ops_dict_int64.py` | 3 failed, 14 passed | **0 failed, 17 passed** |
| `test_ops_dictionary.py` + `test_ops_dict_int64.py` + `test_ops_text_encode.py` + `test_ops_dict_domain.py` + `test_ops_text.py` + `test_ops_resident.py` | — | **96 passed, 0 failed** |

- The 2 remaining failures are exactly the two KNOWN PRE-EXISTING FAILURES:
  `test_ops_null_pattern.py::test_microbench_sorted_fast_path` and
  `test_ops_storage.py::test_persist_int64_raw_rejected`.
- The 32 errors are identical before and after and unrelated to Dictionary:
  `test_ops_noarrow_fallback.py` (deliberately blocked,
  `ImportError: blocked: no-arrow fallback test`) and
  `ModuleNotFoundError: No module named '_lib.bitmask_sweep'` from
  `src/Relational/BitmaskSweep/BitmaskSweep.py:3` (a dev-env packaging gap).
- `445 + 6 = 451`: not one pre-existing test changed state in either direction.

---

## Recorded, NOT fixed (P2)

`from_arrow` expands dictionary columns: `numfast/src/numfast/adapters/arrow.py:118`
(and the cast at `:124`) — 7.28 MB -> 32.87 MB (4.52x) in 37.8 ms, dictionary
encoding lost on round-trip. Not changed here: the file is out of this step's
write scope and another step already modified it.

## RISKS / OPEN ITEMS

1. **Open contract question (BUG 1, D>0).** Whether `validity`-omitted decode
   should raise is unresolved by design — it breaks
   `test_ops_dictionary.py:56` and `series.py:120-121`. See the section above.
   Needs a Coordinator/Architect decision.
2. **`pa.Array` int64 / int8 / bool / temporal inputs now raise a named
   `ValueError` instead of `IndexError`, but still do not encode.** Routing
   them to `_encode_int64_impl` is the natural follow-up and was deliberately
   left out as new feature work. Anyone currently relying on the `IndexError`
   (nobody should) sees a different exception class.
3. **`_arrow_dict_text` swallows every exception and returns `None`
   (`dictionary.py:423`), after which the caller raises the generic
   "needs a TEXT Arrow array" error.** A dictionary-typed input whose cast
   fails for an unforeseen reason therefore reports as a type error rather
   than the cast failure. This mirrors the file's existing defensive style
   (`_encode_arrow_text` lines 600-609, 604) but hides the cause. Accepted for
   consistency; flagged.
4. **A `pa.ChunkedArray` of dictionary type with genuinely different
   per-chunk dictionaries is reinterpreted, not unified.** `cast(pa.string())`
   resolves each chunk's indices against that chunk's own dictionary, which is
   the correct per-chunk reading and is what the pre-fix generic path also did
   (it produced the identical envelope via `list()`). It is NOT Arrow's
   "unified dictionary" semantics, where equal values in different chunks get
   different indices. Callers wanting unified semantics should call
   `unify_dictionaries()` first. Pre-existing, unchanged by this step, and the
   sorted-unique output contract still holds.
5. **`pa.RecordBatch` is rejected alongside `pa.Table`.** Not named in the
   mission; it produced the same misleading "rank-1 column, shape (ncols, nrows)"
   error, so it is one tuple entry in the same guard.
6. `develop/audit_foundation/probes/m4b_make_pre_fix.py` is kept as the
   reproducible fail-before harness. It writes only to
   `%TEMP%/opencode/m4b/`, touches nothing under `numfast/src/`, and is not
   imported by any test.
