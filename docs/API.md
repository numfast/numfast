# NumFast API — the surface as it is today

Everything here was run against this tree. Where a claim is a count, the count is
asserted in the test suite rather than maintained by hand.

Version 0.2.1 · Python 3.11+ · AGPL-3.0-only.

---

## The three layers

| Layer | What it is | Stability |
|---|---|---|
| **Consumer facade** | the 44-name v0 registry: `app()`, `from_numpy`, `query()`, `filter` / `derive` / `group` / `sort` / `limit` / `reduce`, `compile()`, the expression vocabulary | **New in 0.2.1.** Small and deliberate. This is the layer to build on. |
| **Package boundary** | `Series`, `Table`, `from_pandas`, `to_arrow`, `native_info()`, `gpu_capabilities()`, `get_kernel()` | Stable within 0.2.x |
| **Kernel-level names** | `nf.cumsum`, `nf.shift`, `nf.unique`, `nf.lookup`, `nf.map_round`, `nf.rolling_mean`, `nf.returns`, `nf.rng_*` | **Unstable.** Lower-level, not in the v0 registry, may change shape. |

---

## Layer 1 — the consumer facade (44 names)

The count is asserted: `tests/fast/test_consumer_surface.py::test_v0_is_44_names_and_has_no_window_or_or`.

### Boundary adapters (6)

```python
nf.app()                 -> App         # the expression factory + capability query
nf.from_numpy(a, name=)  -> Series
nf.from_arrow(t)         -> Series
nf.to_numpy(x)           -> np.ndarray  # see the NULL warning below
nf.to_pandas(x)          -> pd.DataFrame
nf.to_arrow(x)           -> pa.Table
```

`nf.from_pandas(frame)` also exists and is what the quickstart uses. It is a
pandas entry point, not one of the 44 registry entries.

### App (5)

| Name | What it does |
|---|---|
| `c(name)` | build an expression over a column: `app.c("price")`, `app.c("price") > 10` |
| `query()` | open a lazy chain from a `Table` |
| `capabilities()` | the op set, per-op chunkability, dispatch and buffer limits, and the GPU/CPU split |
| `schema()` | the table's schema |
| `open_stream(path)` | open a columnar file as a streaming source |

### Chain (10)

`filter` · `derive` · `group` · `reduce` · `sort` · `limit` · `compile` ·
`jobs` · `explain` · `nrows`

```python
t = nf.from_pandas(frame)
ch = (t.query()
        .filter(app.c("units") > 2)
        .derive("gross", app.c("units") * app.c("price"))
        .sort("gross", desc=True)
        .limit(10))
ch.jobs()      # the IR: [{op, inputs, params, out}, ...]
ch.explain()   # EXPLAIN / GRAPH / PROFILE / ESTIMATED
out = ch.compile()   # -> Table
```

`jobs()` is the semantic IR made visible: one node per column, no second IR
anywhere. `Chain.material(col)` also exists on the class and takes a column; it
is an internal helper and is not one of the 44 names.

### Expression (23)

| Group | Names |
|---|---|
| Arithmetic | `add` `sub` `mul` `truediv` `mod` `pow` — also spelled `+ - * / % **` |
| Comparison | `eq` `ne` `lt` `le` `gt` `ge` — also spelled `== != < <= > >=` |
| Logic / test | `and_` (`&`), `not_` (`~`), `isin(list)`, `is_null()` |
| Positional | `cumsum()`, `shift(n)` |
| Text | `str_len()`, `str_contains(s)`, `str_startswith(s)`, `str_endswith(s)`, `str_eq(s)` |

Notes that matter:

* **`or_` is not in the registry and raises.** `(a > 1) | (b > 10)` raises
  `ValueError: or_ is not in v0: …`. The reason is in the message: the mask's
  validity is AND-ed, so the filter would drop rows Kleene logic keeps. Use
  `and_`, or write the two filters you actually mean.
* **`is_null()` is a NULL test, not a negation.** It answers hard True/False on
  every row. `not_` is the correct three-valued negation of a predicate. They
  share no code path and both are in the registry; `is_null()` is what makes
  `filter(app.c("k").is_null().not_())` the documented way to drop NULL keys
  before a `group`.
* **`pow` is scalar-exponent only.** `app.c("v") ** 2` works;
  `app.c("base") ** app.c("exp")` raises and tells you to put the column on the
  base.
* **`isin` is point membership, never a substring hit**, and never matches NULL.

### Absent, on purpose

`window` (rolling), `or_`, `join`, `replace`, `fill_null`. Each is documented
with its reason in [KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md) and summarised
in the README. `or_` is the only one that exists as a method: it raises.

---

## Layer 2 — the package boundary

### `Series`

`nf.from_numpy(...)` returns a `Series`: one named buffer plus an optional
validity sidecar. Methods: `backend`, `compare`, `cumsum`, `dtype`, `filter`,
`name`, `reduce`, `returns`, `rolling_mean`, `schema`, `shift`, `sort`,
`to_arrow`, `to_masked`, `to_numpy`, `to_pandas`, `unique`, `validity`.

### `Table`

`nf.Table(kernel, {name: series})`, or the result of `compile()`. Methods:
`column(name)`, `names`, `ncols`, `to_arrow`, `to_numpy`, `to_pandas`, `query()`.

**`to_numpy()` and `to_pandas()` disagree on NULL for integer columns.**

```
column [1, None, 3] as pandas Int64:

  to_numpy()   -> [1, 0, 3]        <- the NULL reads as 0
  to_pandas()  -> [1, <NA>, 3]      <- correct
```

`Series.to_numpy()` restores float invalid rows as `NaN` but has nowhere to put
an integer NULL, so the underlying buffer value shows through. **If a column can
hold NULL, compare with `to_pandas()` or read `Series.validity`.**

### Disclosure functions

Three functions exist so that nothing about the runtime has to be guessed:

```python
>>> nf.native_info()
{'disabled': False, 'dll': '...\\numfast\\_native\\numfast_native.dll', 'dll_exists': True}

>>> caps = nf.app().capabilities()
>>> caps["op_count"], caps["gpu_op_count"], caps["cpu_only_op_count"]
(33, 15, 18)

>>> nf.gpu_capabilities()["ops"]          # the 15
['series', 'pack_keys', 'compare', 'mask', 'filter', 'gather', 'reduce',
 'groupby', 'groupby_multi', 'sort', 'slice', 'shift', 'map', 'cumsum',
 'rng_fill_i32']
```

`native_info()` is the honest answer to "is the native path live here?". On Linux
and macOS it reports `{'disabled': False, 'dll': None, 'dll_exists': False}` —
the native directory is absent, not broken.

`capabilities()["gpu_ops"]` and `["cpu_only_ops"]` name the split. Asking for
`backend='gpu'` on a graph that uses a CPU-only operation raises and names the
operation; there is no silent fallback. `backend='auto'` resolves to the CPU
unless a measured calibration reports a strictly lower GPU host cost.

---

## Layer 3 — kernel-level names (unstable)

These are in `numfast.__all__`, are not part of the 44-name registry, and may
change shape between releases. They work on a single `Series`.

```python
s = nf.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.int32), name="v")

nf.to_numpy(s).tolist()                  # [3, 1, 4, 1, 5]
nf.to_numpy(nf.cumsum(s)).tolist()        # [3, 4, 8, 9, 14]
nf.to_numpy(nf.shift(s, 1)).tolist()      # [0, 3, 1, 4, 1]

u = nf.unique(s)                         # -> dict
u["ng"]                                   # 4
list(nf.to_numpy(u["uniq"]))              # [1, 3, 4, 5]
```

`nf.lookup(build, probe)`, `nf.map_round(series, ndigits)`, `nf.rolling_mean(s,
window)`, `nf.returns(s)` and the `nf.rng_*` family (`rng_seed`, `rng_fill_f64`,
`rng_fill_i32`, `rng_permutation`, `rng_sample`, `rng_compat`) are in the same
layer. Every RNG entry point takes an explicit `seed`, and the same seed with
the same stream and offset reproduces the same values.

---

## The guards

"This refuses rather than lies" is a property of this engine, and these are the
refusals you can rely on. Each is pinned by a named test.

| Situation | Behaviour |
|---|---|
| `group` on a key column containing NULL | `ValueError`, names the column and the NULL count, and says to filter first. A NULL group is *absent*, not wrong. |
| `group` with more than one measure column | `ValueError`, naming the offending keys. One measure column per call. |
| arithmetic on a TEXT column, all six operators | `ValueError` **before any node is emitted**. The DAG holds a `dictionary_encode` code vector and codes are ranks, not values. |
| ordering a TEXT column (`< <= > >=`) | `ValueError` at `filter`: it needs a collation the engine does not define. |
| array exponent (`col ** col`) | `ValueError`: `pow` is scalar-exponent only. |
| an out-of-`int32` value in an `int32` column | `ValueError` rather than wrap-around. |
| a BIGINT key or literal above 2³¹ | `ValueError` rather than silent narrowing. This is what blocks 8 ClickBench queries. |
| `backend='gpu'` on a CPU-only operation | `RuntimeError` naming `op:<name>`. |
| a fused GPU indicator plan with a parameter key no operation consumes | `ValueError` naming the output, the key, the op, and that op's accepted keys. |

One known silent case, documented rather than fixed:
**`group` drops a key whose measure is entirely NULL.** No exception, no warning.
After a `group`, check that the output row count equals the number of distinct
keys you expected; `count` alone will not tell you, because it reports the
surviving rows, not the missing group. See
[KNOWN_LIMITATIONS.md §8](../KNOWN_LIMITATIONS.md).