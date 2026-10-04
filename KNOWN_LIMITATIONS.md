# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# KNOWN LIMITATIONS — what does not work, and how sure we are

This file exists so that nothing in NumFast has to be discovered by failing.
Every entry carries an evidence tag from `DESIGN_consumer_api_v0.md` §10,
adapted to the five tags below. **A tag without a reproduction next to it is
not a claim, it is a guess** — that is the rule §10 exists to enforce, and it
exists because a number taken by eye in that document was once a lie.

| Tag | Meaning |
|---|---|
| **PROVEN** | a command or a test is given, and it passes today |
| **MEASURED** | a named fixture was run; the numbers below are what it printed |
| **INFERENCE** | derived from another entry here, with the reasoning shown |
| **HYPOTHESIS** | believed, not checked. Do not rely on it. |
| **UNKNOWN** | nobody has established this. Treated as broken. |

Audit date: **2026-10-04**, commit `f2af608`, Windows, Python 3.14.6,
numpy 2.5.1, pandas 3.0.3, CPU backend.
Reproduction for every **PROVEN**/**MEASURED** row:

```bash
export PYTHONPATH="C:/App/numfast/numfast/src;C:/App/numfast/app-builder"
python -m pytest tests/fast -q -p no:cacheprovider
```

---

## 1. `window` is absent from the surface — **PROVEN**

`app.window` does not exist and is not in the v0 registry.
Test: `tests/fast/test_consumer_surface.py::test_window_is_absent_from_the_chain_surface`.

**Why.** `ir_rolling_sum` accumulates in float64, so on an int64 column above
2^53 it loses exactness **silently** — no exception, wrong number. A capability
that can only be wrong quietly is not shipped as if it were right. The fix is
in the CPU driver, which is FROZEN.

**Consequence for you.** No rolling / windowed aggregate is reachable from the
public API. `cumsum` is available and is exact for the integer widths it
accepts.

## 2. `or_` is absent from the surface — **PROVEN**

`(app.c('a') > 1) | (app.c('b') > 10)` raises `ValueError: or_ is not in v0`.
Tests: `test_consumer_surface.py::test_or_is_absent_from_the_v0_registry_and_refuses_loudly`.

**Why.** `ir_mask(..., 'or')` AND-s the two operands' validity sidecars, so the
mask is UNKNOWN on any row where either side was NULL, and `ir_filter` then
drops that row. `filter(a | b)` would silently return fewer rows than either
predicate alone. FROZEN; refused loudly instead.

**Consequence for you.** Combine predicates with `and_` only, or run the two
filters as two separate steps and re-check that the row count is what you
expect.

## 3. `join` is absent from the consumer surface — **PROVEN**

Neither `app` nor a chain exposes any `join*` method. **MEASURED** 2026-10-04:
the join primitives exist in the engine (`src/Relational/Join`) and are
reachable through the kernel, but there is no consumer-facing spelling.

**Consequence for you.** Do joins through the kernel-level API, or outside
NumFast.

## 4. `replace` is absent — **PROVEN**

No `replace*` method on `app` or a chain. **MEASURED** 2026-10-04.

**Consequence for you.** Use `derive` with `isin` / `str_eq` to build the
substitution yourself.

## 5. NULL keys in `group` are refused, loudly — **PROVEN**

`group('k', ...)` raises when column `k` has any NULL rows:
`group: key column 'k' has 1 NULL rows; v0 does not group NULL keys -- NULL
keys are silently dropped (Drivers/CPU/_lib/cpu.py:3325) ...`
Test: `test_consumer_surface.py::test_group_refuses_null_key_loudly`.

**Why this one is justified and `sort`'s was not.** `ir_groupby*` *compacts
NULL keys away*: the group is not wrong, it is **absent**, and an absent group
is indistinguishable from an answer. The refusal is the correct call.

**What to do.** `filter(q.c('k').is_null().not_())` first — `is_null()` exists
precisely so this advice is executable.

**Contrast — the guard that was removed.** `sort()` used to refuse a NULL sort
key on the stated ground that NULL rows would come **first**. They come
**last**. That refusal was a false premise and was deleted on 2026-10-04;
parity is now pinned by `tests/fast/test_consumer_sort_null_parity.py`.

## 6. `to_numpy()` and `to_pandas()` disagree on NULL for integer columns — **PROVEN**

**MEASURED** 2026-10-04, column `[1, None, 3]` as pandas `Int64`:

```
to_numpy()   -> [1, 0, 3]        <- the NULL reads as 0
to_pandas()  -> [1, <NA>, 3]      <- correct
```

`Series.to_numpy()` restores float invalid rows as `NaN` but has nowhere to put
an integer NULL, so the underlying buffer value shows through. `to_pandas()`
reads the validity sidecar and is correct.

**Consequence for you.** If a column can hold NULL, compare with
`to_pandas()` or read `Series.validity` — never with `to_numpy()`.

## 7. Arithmetic on a text column computed on dictionary codes — **FIXED 2026-10-04; kept here because the before-state is the evidence**

**MEASURED 2026-10-04, before the fix.** Column `k` = `["b","a","c"]` (text),
column `v` = `[1.0, 2.0, 3.0]`. **No guard fired on any of these**, and the
answer was a correct-shaped wrong number:

| Op | Spelled | Result | What it actually did |
|---|---|---|---|
| `+` | `q.c('k') + q.c('v')` | `[2, 2, 5]` | dictionary codes + values |
| `-` | `q.c('k') - q.c('v')` | `[0, -2, -1]` | dictionary codes − values |
| `*` | `q.c('k') * q.c('v')` | `[1, 0, 6]` | dictionary codes × values |
| `/` | `q.c('k') / q.c('v')` | `[1, 0, 1]` | dictionary codes ÷ values |
| `%` | `q.c('k') % q.c('v')` | `[0.0, 0.0, 2.0]` | dictionary codes mod values |
| `**` | `q.c('k') ** q.c('v')` | raises | blocked by an unrelated driver guard ("array exponent"), **not** by a text guard |

Five of the six arithmetic operators were unguarded on text; `pow` was blocked
for a different reason and needed its own check. Six of six are now guarded —
measured 6 of 6 on the `["b","a","c"] + [1,2,3]` fixture, the codes being
`[1, 0, 2]` because the sidecar is sorted. Note that `==`/`!=` were already
guarded against text misuse, and so are `<`/`<=`/`>`/`>=` (refused: ordering on
text needs a collation the engine does not define).

**FIXED 2026-10-04.** `Chain._refuse_text_bin`
(`src/Semantic/TableExpr/_lib/chain.py`) refuses the whole `bin` family when
either operand is a TEXT column, **before any node is emitted**, naming the
column, the side it was on, and the cause: a text column is a
`dictionary_encode` code vector in the DAG and those codes are **ranks, not
values**. All six now raise `ValueError` in the engine's own error shape;
`tests/fast/test_release_blockers.py::test_3_arithmetic_on_text_refuses_loudly`
and `::test_3_text_column_against_a_numeric_column_refuses` pin it.

**Consequence for you.** Arithmetic on a text column is no longer possible by
accident. The honest route is to measure the text first: `q.c('k').str_len()`
returns an int32 code-point **count**, and arithmetic on a count is defined —
`q.c('k').str_len() + 1` means what it says and matches `df['k'].str.len() + 1`
exactly. `ir_map` has no text mode: it is one numeric buffer plus a scalar, so
there is no lowering that would have been correct.

## 8. `group` silently drops a key whose measure is entirely NULL — **PROVEN, and it is silent**

**MEASURED** 2026-10-04. `k = [1, 1, 2]` (no NULLs in the key, so the §5 guard
does not fire), `v = [NULL, NULL, 5.0]`:

```
group('k', {'v': ('sum', 'count')})  ->  k=[2],  v.sum=[5.0],  v.count=[1]
```

Key `1` is **gone**. Its measure was NULL on every one of its rows, and the
group vanished with no exception and no warning. `count` reports 1, not 0, so
the surviving row's own numbers are right; what is missing is the group.

**Consequence for you.** After a `group`, check that the number of output rows
equals the number of distinct keys you expected. `count` alone will not tell
you — compare the key set.

**Not fixed here.** Same reason as §7: the fix is in the grouping path, outside
the publication gate's write scope. Recorded, not fixed.

## 8b. `group` with more than one measure column raises — **PROVEN**

**MEASURED** 2026-10-04. `group('region', {'revenue': ('sum',), 'units': ('sum',)})`
— two measures — raises:

```
ValueError: groupby_multi multi-col ops must be {col: (ops)}
for ['s_revenue_2', 's_units_3'], got {'revenue': ('sum',), 'units': ('sum',)}
```

The error is loud and names the offending keys, which is the good case, but the
capability the message promises ("pass e.g. {'v1': ('sum',), 'v3': ('mean',)}")
does not work. Verified pre-existing at commit `f2af608`, i.e. not introduced by
the `sort` guard removal.

**Consequence for you.** Aggregate one measure column per `group()` call.

**Not fixed here.** The lowering of a multi-measure `group` is outside the
publication gate's write scope. Recorded, not fixed.

## 9. `sort` is stable, but a tie is not broken by anything — **PROVEN (FROZEN)**

Rows with equal keys come back in input order, in both directions, on both
drivers. Pinned by `test_consumer_sort_null_parity.py::test_raw_ir_sort_perm_puts_nulls_last_on_cpu_and_gpu`
(ties-only fixture).

Two caveats, both real:

* If your data changes, or you push the sort through a different planner path,
  the order among tied rows is **not** a contract. Do not depend on it.
* The CPU gather has a contiguous fast path that slices instead of indexing
  (`Drivers/CPU/_lib/cpu.py:1134`). It is guarded to produce the same result,
  but it is the place to look first if tied-row order ever changes.

## 10. `fill_null` is not implemented — **PROVEN**

No `fill_null` on the surface, and it is not in the v0 registry.
**MEASURED** 2026-10-04: absent from `Chain` and from `Expr`.

**Why.** Filling is not chunkable over the current layout: a fill value has to
be applied consistently across chunk boundaries, and the chunk planner cannot
express that. Until it can, `fill_null` would be a row-count or value trap.

**What to do instead.** `filter(q.c('k').is_null().not_())` to drop NULL rows,
or `derive` a replacement column with `isin` / `str_eq`.

---

## Also true, not enumerated above

* **NULL sort keys are supported and match pandas** (NULL last, input order).
  This is the opposite of what §5 used to say about `sort`, and it is pinned by
  a test rather than a promise.
* **GPU sort refuses** a valid-row count that is not a power of two, and refuses
  int64 and float64 keys. That is the driver's own documented limit, not a
  parity failure: where the GPU sort does run, it returns the same permutation
  as the CPU and as pandas.
* **int64 survives storage exactly.** `persist_table` keeps an int64 column
  int64 and `load_table` reads it back exact; an `ir_reduce(sum)` over it is
  exact. What is checked is int32: an out-of-int32 value in an int32 column
  raises rather than wrapping. Pinned by
  `tests/fast/test_ops_storage.py::test_persist_int64_stays_int64_and_roundtrips_exact`.
* **BIGINT keys and literals raise** rather than narrowing — see
  `BENCHMARKS.md` §1 for which ClickBench queries that costs.
* **From-pandas dtype round-trip is not identity.** NumFast returns a *nullable*
  dtype where pandas' own `.loc[...] = None` upcast produced a plain float64
  with NaN. Values and order match; the dtype label may not. Use
  `pd.testing.assert_frame_equal(..., check_dtype=False)` when comparing.
* **CI runs a subset, not the whole suite.** See `.github/workflows/ci.yml`: most
  tests need `builder` from a separate repository whose published `main` is
  behind what this tree needs. The gating job covers 223 of the 651 tests.