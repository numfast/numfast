# NumFast

A columnar compute engine for Python with a small, NumPy- and pandas-shaped
surface. Hand it columns; it hands you columns back. Integer columns are computed
exactly in `int32`, NULLs are carried explicitly rather than as `NaN`, and a
query is one lazy chain that a single planner call turns into one execution graph.

```bash
pip install numfast
pip install "numfast[pandas]"   # + pandas adapters
pip install "numfast[gpu]"      # + wgpu, required to execute on the GPU path
```

Python 3.11+. `numpy>=1.24` is the only hard dependency.

---

## Quick start

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

The vocabulary is the whole of it: `query` → `filter` / `derive` / `group` /
`sort` / `limit` / `reduce` → `compile`, with expressions built from
`app.c("col")` and `app.c("col") > 10`. No query language to learn, no index to
declare.

---

## Read this before you rely on it

**The API is 44 names.** It is small on purpose, and it is **new in 0.2.1**.
`window`, `or_`, `join`, `replace` and `fill_null` are **not** in it. Each absence
has a stated reason; `Expr.or_` exists and raises rather than returning an empty
frame.

**Where NumFast is slower than DuckDB.** On high-cardinality grouping and
distinct-count shapes it is **several times slower, not a few percent** — a
property of the CPU driver being a NumPy reference executor, not of missing
effort. **If your workload is high-cardinality grouping, DuckDB is the better
tool today.** Where NumFast wins, it wins on small aggregate-shaped queries,
because one planner call replaces a fixed per-query setup: read that as *low
fixed cost*, not as a large speedup.

This distribution quotes **no wall-clock numbers**. A benchmark claim is only
worth reading if the artefact and the command that produced it are both in the
package, and neither is: the inputs are external and too large to ship. Benchmark
the operation mix you actually have.

**What raises rather than guesses.** A BIGINT key or literal above `int32` —
`UserID`-shaped identifiers, for instance — raises and names the column, because
logical values are `int32` and narrowing them silently would be a wrong answer
rather than a slow one. A query needing a conditional (`CASE`) cannot be
expressed at all: there is no such primitive.

**The GPU does 15 of 33 operations**, measured on one RTX 2060 over Vulkan. The
other 18 run on the CPU and are listed by `nf.app().gpu_capabilities()`. The
recorded GPU timings at the sizes measured were **slower than the CPU path**; the
GPU claim is parity and residency, **not speedup**. Asking for `backend='gpu'` on
a CPU-only operation raises rather than falling back.

**One known silent case.** `group` drops a key whose measure is entirely NULL: no
exception, no warning, one group missing. Check the output row count against the
distinct key count. Also: `to_numpy()` reads an integer NULL as `0` — use
`to_pandas()`.

The complete register, with a certainty tag and a reproduction on every entry, is
[`KNOWN_LIMITATIONS.md`](https://github.com/numfast/numfast/blob/main/KNOWN_LIMITATIONS.md).
It is not a short list, and it is not apologetics.

---

## What is in the box

- **A 44-name consumer facade** — `filter`, `derive`, `group`, `sort`, `limit`,
  `reduce`, `compile`, a 23-name expression vocabulary including `isin`,
  `is_null`, `cumsum`, `shift` and five text predicates.
- **30 Builder Extensions** forming the engine: a semantic IR of one node per
  column, a cost-calibrated planner, a NumPy reference CPU driver, a wgpu-py/WGSL
  GPU driver, and the storage layer.
- **A Rust kernel layer** (`numfast-native`, no third-party dependencies) loaded
  through `ctypes`: sort, join, group-by, carry, segmented reduce, shortest
  paths, cost travel, RNG. It is **kernels, not an executor** — it does not
  schedule, stream or own memory.
- **A WebAssembly build of the same kernels** (`@numfast/kernels` on npm): a
  portable kernel library, 86 function exports, one memory, zero imports, 17
  typed wrappers. **Not a compute core** — there is no executor inside the `.wasm`.
- **Refusals you can rely on**: NULL group keys, arithmetic or ordering on a text
  column, out-of-`int32` values, BIGINT keys and literals, CPU-only operations on
  the GPU. Each raises and names the cause.

## Two wheels, one version

| Wheel | Native library | Who gets it |
|---|---|---|
| `py3-none-win_amd64` | yes | Windows x86-64: full engine, native path live |
| `py3-none-any` | no | Linux, macOS: full engine, native path absent |

Linux and macOS users get the complete engine and **no** native library. That is
degraded, not broken, and it is visible rather than latent:

```python
>>> import numfast as nf
>>> nf.native_info()
{'disabled': False, 'dll': None, 'dll_exists': False}
```

The native directory is simply absent and every Rust-backed call falls back to the
NumPy CPU path — not a binary that fails to load at run time.

## Licence

**AGPL-3.0-only.** The wheel conveys `numfast_native.dll` — object code — so
AGPL-3.0 §6 is discharged by construction: the Corresponding Source (47 `.rs`
files, the crate manifest and lockfile, the cargo config, the link shims, and a
`SHA256SUMS` over exactly those bytes) ships inside the wheel under
`numfast/_corresp_src/`. The written-offer route is deliberately not used.

There is **no "paid AGPL"**. A separate proprietary commercial licence is
announced but its **terms do not exist yet** — not written, not priced, not an
offer. The AGPL grant is free, unconditional and complete. Support and consulting
are available now and need no further paperwork.

Full text: <https://github.com/numfast/numfast/blob/main/LICENSE>

## Documentation

- [README](https://github.com/numfast/numfast/blob/main/README.md) — the front door
- [docs/API.md](https://github.com/numfast/numfast/blob/main/docs/API.md) — the 44-name surface and every guard
- [docs/ARCHITECTURE.md](https://github.com/numfast/numfast/blob/main/docs/ARCHITECTURE.md) — how it fits together
- [docs/EXAMPLES.md](https://github.com/numfast/numfast/blob/main/docs/EXAMPLES.md) — runnable programs with real output
- [KNOWN_LIMITATIONS.md](https://github.com/numfast/numfast/blob/main/KNOWN_LIMITATIONS.md) — what does not work, tagged by certainty
- [SECURITY.md](https://github.com/numfast/numfast/blob/main/SECURITY.md) — how to report a vulnerability

---

*Publishing note:* this file is the prepared PyPI long description. Publishing to
PyPI is a **manual** step — there is no publish workflow in this repository and no
tag trigger in CI. Note also that `pyproject.toml` currently points `readme` at
`README.packaging.md`, so PyPI renders that file instead of this one until that
pointer is changed.