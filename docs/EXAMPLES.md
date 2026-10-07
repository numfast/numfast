# NumFast examples

A small number of real programs. Every output block below was produced by running
the code against this tree; nothing here is illustrative pseudocode.

The quickstart is in the [README](../README.md) and as a runnable file at
[`examples/quickstart.py`](../examples/quickstart.py).

**Want to run it rather than read it?** The five notebooks in
[`showcase/`](../showcase/README.md) demonstrate the same surface end to end —
`Table`/`Series`, the GPU path and its honest crossover, the WASM package, the
comparative matrix including where NumFast loses, and a full task cross-checked
against pandas. Every notebook output was produced by executing it against this
tree.

```python
import pandas as pd
import numfast as nf

orders = pd.DataFrame({
    "region":  ["emea", "apac", "emea", "amer", "apac", "emea"],
    "revenue": [120.0, 80.0, 240.5, 60.0, 95.5, 310.0],
})

result = (nf.from_pandas(orders)
          .query()
          .group("region", {"revenue": ("sum", "count")})
          .sort("revenue.sum", desc=True)
          .compile())

print(result.to_pandas())
```

```
region  revenue.sum  revenue.count
  emea        670.5              3
  apac        175.5              2
  amer         60.0              1
```

---

## 1. Filter and derive

```python
import pandas as pd
import numfast as nf

app = nf.app()

t = nf.from_pandas(pd.DataFrame({
    "region": ["emea", "apac", "emea", "amer"],
    "units":  [10, 3, 7, 1],
    "price":  [2.5, 9.0, 4.0, 11.0],
}))

out = (t.query()
       .filter(app.c("units") > 2)
       .derive("gross", app.c("units") * app.c("price"))
       .compile())

print(out.to_pandas().to_string(index=False))
```

```
region  units  price  gross
  emea     10    2.5     25
  apac      3    9.0     27
  emea      7    4.0     28
```

`derive` adds a column; it does not replace one. The chain is lazy until
`compile()`, and `jobs()` shows the IR without executing anything.

---

## 2. A NULL key refuses — and the fix is one line

Grouping on a column that contains NULL **raises**, because a NULL group would be
*absent* rather than wrong, and an absent group cannot be told apart from an
answer:

```python
nullable = nf.from_pandas(pd.DataFrame({
    "plan": pd.array(["a", None, "b", "a"], dtype="string"),
    "rev":  [1.0, 2.0, 3.0, 4.0],
}))

nullable.query().group("plan", {"rev": ("sum",)}).compile()
```

```
ValueError: group: key column 'plan' has 1 NULL rows; v0 does not group NULL keys --
NULL keys are silently dropped (Drivers/CPU/_lib/cpu.py:3325), so a NULL group is an
ABSENT value, not a wrong one, and cannot be told apart from the answer. Fix: filter
those rows out before group().
```

`is_null()` exists precisely so that advice is executable:

```python
clean = (nullable.query()
         .filter(app.c("plan").is_null().not_())
         .group("plan", {"rev": ("sum", "count")})
         .compile())

print(clean.to_pandas().to_string(index=False))
```

```
plan  rev.sum  rev.count
   a      5.0          2
   b      3.0          1
```

---

## 3. Text columns: measure the text, then compute

Arithmetic on a text column is refused, because a text column in the DAG is a
`dictionary_encode` **code** vector and those codes are ranks, not values:

```python
txt = nf.from_pandas(pd.DataFrame({"utm": pd.array(["ab", "cde", None, "fghij"],
                                               dtype="string")}))

txt.query().derive("bad", app.c("utm") + 1).compile()
```

```
ValueError: derive: add on TEXT column 'utm' (the left operand) is refused: a text
column is a dictionary_encode CODE vector in the DAG, and those codes are RANKS, not
values.
```

`str_len()` is the honest route. It returns an int32 code-point count, and
arithmetic on a count is defined:

```python
lens = (txt.query()
        .derive("utm_len_plus1", app.c("utm").str_len() + 1)
        .filter(app.c("utm").isin(["ab", "cde"]))
        .compile())

print(lens.to_pandas().to_string(index=False))
```

```
utm  utm_len_plus1
 ab              3
cde              4
```

`isin` is point membership — the exact values, never a substring hit — and it
never matches NULL, so the NULL row is not in the result.

---

## 4. Sorting with NULL keys

NULL sort keys are **supported**, and they come back last in input order, in both
directions — the same answer pandas gives with `na_position="last"`:

```python
print(nullable.query().sort("plan").compile().to_pandas().to_string(index=False))
```

```
plan  rev
   a  1.0
   a  4.0
   b  3.0
<NA>  2.0
```

The sort is stable: rows with equal keys keep their input order. That is pinned by
a test, and it is still not a contract you should depend on across data changes or
a different planner path.

---

## 5. `or_` refuses loudly

```python
app.c("units") > 1 | app.c("price") > 5
```

```
ValueError: or_ is not in v0: ir_mask(..., 'or') AND-s the validities of its two
operands, so ir_filter drops every row either side was NULL on -- filter((a>1)|(b>10))
returns an empty frame where Kleene keeps both rows. The 3VL data vector is right; the
validity is AND-ed. The fix is FROZEN (IR + CPU_Driver), so the operation is deferred,
not documented as is.
```

Combine predicates with `and_` / `&`, or write the two filters you actually mean.

---

## 6. Kernel-level operations on a `Series`

These names are below the consumer facade and are **not** part of the 44-name v0
registry:

```python
import numpy as np

s = nf.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.int32), name="v")

nf.to_numpy(s).tolist()                 # [3, 1, 4, 1, 5]
nf.to_numpy(nf.cumsum(s)).tolist()       # [3, 4, 8, 9, 14]
nf.to_numpy(nf.shift(s, 1)).tolist()     # [0, 3, 1, 4, 1]

u = nf.unique(s)                        # a dict, not a Series
u["ng"]                                  # 4
list(nf.to_numpy(u["uniq"]))             # [1, 3, 4, 5]
```

---

## 7. Ask the engine what it can do

Nothing about the runtime has to be guessed:

```python
caps = app.capabilities()
print(caps["op_count"], caps["gpu_op_count"], caps["cpu_only_op_count"])
# 33 15 18

print(caps["gpu_ops"])
# ['compare', 'cumsum', 'filter', 'gather', 'groupby', 'groupby_multi', 'map',
#  'mask', 'pack_keys', 'reduce', 'rng_fill_i32', 'series', 'shift', 'slice', 'sort']

print(nf.native_info())
# {'disabled': False, 'dll': '.../numfast/_native/numfast_native.dll', 'dll_exists': True}
```

On Linux and macOS `native_info()` reports `dll: None, dll_exists: False`: the
native path is absent by construction, not broken. On the same machine, a GPU
operation that is CPU-only raises rather than falling back:

```
ValueError: GPU driver: op 'unique' CPU-only in v0.2 (encode/gather-text need CPU
path) Fix: run this op with backend='cpu'.
```

---

## 8. Comparing a result with pandas

```python
got  = result.to_pandas()
want = (orders.groupby("region", as_index=False)["revenue"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
        .rename(columns={"sum": "revenue.sum", "count": "revenue.count"}))

pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True),
                              check_dtype=False)
```

Use `check_dtype=False`: NumFast returns a **nullable** dtype where pandas' own
assignment of `None` upcast to a plain `float64` with `NaN`. Values and order
must match; the dtype label need not. That is the one systematic difference, and
it is documented rather than smoothed over.