# NumFast

NumFast is a columnar compute engine for Python with a small, NumPy- and
pandas-shaped surface. You hand it columns; it hands you columns back. Integer
columns are computed exactly in `int32`, NULLs are carried explicitly rather
than as `NaN`, and a query is one lazy chain that a single planner call turns
into one execution graph.

**Version 0.2.1.** AGPL-3.0-only. Python 3.11+. The native compute kernels are a
Rust library that currently ships for Windows x86-64 only.

| | |
|---|---|
| Engine, Python | PyPI: <https://pypi.org/project/numfast/> · npm kernels: <https://www.npmjs.com/package/@numfast/kernels> |
| **How to install, all five channels** | **[docs/INSTALL.md](docs/INSTALL.md)** |
| Source | <https://github.com/numfast/numfast> |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Problems | <https://github.com/numfast/numfast/issues> |

Read this page to decide whether it is useful to you in thirty seconds:

* **What it does well, and where it is slower than DuckDB** — where it loses, it
  loses on high-cardinality grouping by several times rather than a few percent.
  That is not a footnote, it is the main performance fact.
* **What works today, and what does not** — the 33 operations, the guards, and
  the one primitive (`CASE`) whose absence costs a real query shape.
* **What is absent** — `window`, `or_`, `join` and `replace` are not in the API.
  The reasons are named too.
* **What the GPU does** — 15 of 33 operations, measured on one RTX 2060 over
  Vulkan. The other 18 run on the CPU. No GPU speedup is claimed anywhere.
* **[Known limitations](KNOWN_LIMITATIONS.md)** — every entry tagged by how sure
  we are, with a reproduction. Read it before you rely on anything.
* **[Licence](#licence)** — AGPL-3.0-only, with the Corresponding Source inside
  the wheel. There is no paid tier of the AGPL.

---

## Install

**[docs/INSTALL.md](docs/INSTALL.md) is the single installation page** — Python,
JavaScript, Browser, Linux native and Windows native, each with what it gives
you and how to check it. The short version:

```bash
pip install numfast            # the engine
pip install "numfast[pandas]"  # + pandas adapters
pip install "numfast[arrow]"   # + pyarrow adapters
pip install "numfast[gpu]"     # + wgpu, required to execute on the GPU path
```

```bash
npm install @numfast/kernels   # the same kernels as WebAssembly, Node host only
```

`numpy>=1.24` is the only hard dependency.

Two wheels are published at the same version:

| Wheel | Tag | Carries the native library | Who gets it |
|---|---|---|---|
| `numfast-0.2.1-py3-none-win_amd64.whl` | `py3-none-win_amd64` | yes | Windows x86-64: full engine, native path live |
| `numfast-0.2.1-py3-none-any.whl` | `py3-none-any` | no | Linux, macOS: full engine, native path absent |

**Linux and macOS get the complete engine and no native library.** That is
degraded, not broken, and it is visible rather than latent:

```python
>>> import numfast as nf
>>> nf.native_info()
{'disabled': False, 'dll': None, 'dll_exists': False}
```

`numfast/_native/` is simply not there, and every Rust-backed call falls back to
the NumPy CPU path. It is not a binary that fails to load at run time. Set
`NUMFAST_NATIVE_DISABLE=1` to force the same path off on Windows, and
`NUMFAST_NATIVE_DLL` to point at a locally built binary.

Nothing in the distribution is a CPython extension module: the `.dll` is loaded
through `ctypes`, so the implementation tag stays `py3` and neither wheel is
pinned to one interpreter version.

To run from a clone instead of installing:

```bash
export PYTHONPATH="<repo>/src"
python examples/quickstart.py
```

---

## Quick start

The smallest program that produces a real result — group, sort, and check the
answer against pandas:

```python
import pandas as pd
import numfast as nf

orders = pd.DataFrame({
    "region":  ["emea", "apac", "emea", "amer", "apac", "emea"],
    "revenue": [120.0, 80.0, 240.5, 60.0, 95.5, 310.0],
})

result = (nf.from_pandas(orders)              # or nf.from_numpy(...)
          .query()                            # start a lazy chain
          .group("region", {"revenue": ("sum", "count")})
          .sort("revenue.sum", desc=True)
          .compile())                         # -> a numfast Table

print(result.to_pandas())
```

```
region  revenue.sum  revenue.count
  emea        670.5              3
  apac        175.5              2
  amer         60.0              1
```

[`examples/quickstart.py`](examples/quickstart.py) is the same program with the
pandas comparison and the assertion inside it. It runs on a clean clone with only
`<repo>/src` on `PYTHONPATH`, needs nothing else from this repository, and its
output is the block above.

The vocabulary is the whole of it: `query` → `filter` / `derive` / `group` /
`sort` / `limit` / `reduce` → `compile`, with expressions built from
`app.c("col")`, `app.c("col") > 0`, `.and_()`, `.isin([...])`. There is no query
language to learn and no index to declare.

More examples, each one runnable and each one with its real output:
**[docs/EXAMPLES.md](docs/EXAMPLES.md)**.

---

## What works today

**33 operations, all of them exact on `int32`. 15 have a GPU implementation; 18
are CPU-only and say so rather than falling back silently.**

| | Count | Detail |
|---|---|---|
| Implemented | **33** | Every one raises rather than guessing: a fused indicator plan carrying a parameter key no operation consumes, an array exponent, a CPU-only op requested on the GPU — each refuses and names the cause. |
| Absent | 4 names | `window`, `or_`, `join`, `replace` — reasons above. |
| One shape we cannot express | — | a **conditional / `CASE` primitive**. There is no way to say "value if predicate else value", so a conditional aggregate or a pivot has to be expressed as several queries. This is a real gap, not a preference. |

The one design decision that shows up as a limitation: logical values are
`int32`. A BIGINT key or literal above 2³¹ **raises**, naming the column, rather
than narrowing silently. A query over `UserID`-shaped identifiers costs you; a
wrong answer on any table would cost more.

---

## Capabilities

**The consumer surface is 44 names, and that is all of it.** It is a new layer —
first released in 0.2.1 — and it is small on purpose.

| Group | Count | Names |
|---|---|---|
| Boundary adapters | 6 | `app`, `from_numpy`, `from_arrow`, `to_numpy`, `to_pandas`, `to_arrow` |
| App and column reference | 5 | `c`, `capabilities`, `open_stream`, `query`, `schema` |
| Chain | 10 | `compile`, `derive`, `explain`, `filter`, `group`, `jobs`, `limit`, `nrows`, `reduce`, `sort` |
| Expression | 23 | `add` `and_` `cumsum` `eq` `ge` `gt` `is_null` `isin` `le` `lt` `mod` `mul` `ne` `not_` `pow` `shift` `str_contains` `str_endswith` `str_eq` `str_len` `str_startswith` `sub` `truediv` |

The count is asserted in the test suite, not maintained by hand.

`nf.from_pandas`, `nf.Series`, `nf.Table`, `nf.native_info()`,
`nf.gpu_capabilities()`, `nf.get_kernel()` and `nf.__version__` are also part of
`import numfast`, and every example in this repository uses them. They are simply
not entries in the 44-name v0 registry, which governs the *consumer* surface: the
pandas entry point, the two public types, and the three disclosure functions.

**Absent from the surface, deliberately:**

| Absent | Why |
|---|---|
| `window` (rolling) | the lowering loses exactness silently on int64 above 2⁵³. An operation that can only be wrong quietly is not shipped as if it were right. The fix is in frozen code. |
| `or_` | `ir_mask(..., 'or')` AND-s the two operands' validity, so the filter drops rows Kleene logic keeps. `Expr.or_` exists and **raises**. Use `and_`, or run the two filters. |
| `join` | the join primitives exist in the engine and are reachable through the kernel-level API. There is no consumer-facing spelling. |
| `replace` | no method on `app` or a chain. Use `derive` with `isin` / `str_eq`. |
| `fill_null` | filling is not chunkable over the current layout; it would be a row-count trap. |

**"This refuses rather than lies" is a real feature, and here is the list of
refusals you can rely on:**

| Situation | What happens |
|---|---|
| `group` on a column with NULL rows | raises, names the column and the count, and tells you to filter first. A NULL group would be *absent*, not wrong, and absence cannot be told apart from an answer. |
| `group` with more than one measure column | raises and names the offending keys. Aggregate one measure column per call. |
| arithmetic on a TEXT column (`+ - * / % **`) | raises before any node is emitted. The DAG holds a dictionary *code* vector, and codes are ranks, not values. |
| ordering a TEXT column (`< <= > >=`) | `filter` raises: ordering on text needs a collation the engine does not define. Use `str_eq` / `isin` / `str_len`. |
| an `int64` value outside `int32` in an `int32` column | raises rather than wrapping. |
| a BIGINT key or literal (ClickBench's `UserID`, `URLHash`, …) | raises rather than narrowing. |
| `backend='gpu'` on a graph using one of the 18 CPU-only ops | `RuntimeError` naming `op:<name>`. There is no silent CPU fallback. |
| a fused GPU indicator plan with a parameter key no operation consumes | raises, naming the output, the key, the op and that op's accepted keys. Dropping the key silently used to re-root the body to the wrong input. |

Full surface, with signatures and worked output:
**[docs/API.md](docs/API.md)**.

---

## Architecture in one page

**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** is the longer version, and
normative for the semantic contract.

```
  you
   │  query() → filter/derive/group/sort/limit → compile()
   ▼
┌─────────────────────────────────────────────────────────────┐
│ Semantic/IR        one node per column, one DAG per chain    │
│ Semantic/TableExpr the consumer facade; 44 public names      │
├─────────────────────────────────────────────────────────────┤
│ Runtime/Planner    capability query + measured cost model    │
│                    → ONE execution graph per chain           │
│ Runtime/Runtime    job → IR → optimise → plan → execute     │
├─────────────────────────────────────────────────────────────┤
│ Drivers/CPU        NumPy-backed reference executor (FROZEN)  │
│ Drivers/GPU        wgpu-py / WGSL, 15 of 33 ops             │
│ Drivers/GroupedHash, GroupedHashMT                           │
├─────────────────────────────────────────────────────────────┤
│ Storage            schema, dictionary coding, NFS            │
│ Core               canonical dtypes, Series / Table          │
└─────────────────────────────────────────────────────────────┘
        │ ctypes
        ▼
  numfast-native/     Rust kernels: sort, join, group-by, map,
                      carry, segmented reduce, SSSP, cost travel
```

* **The IR** is a list of jobs, `{op, inputs, params, out}`, one node per
  column. Expressions lower to nodes; a chain holds jobs, never a second IR.
* **Extensions.** The engine is 30 Builder Extensions, each a folder with a
  `.toml` manifest, a `.py` entry point and a `_lib/` implementation. The
  manifest declares its aliases and dependencies; `setup.py` vendors all 30 into
  the wheel as `numfast/_ext/` — **133 files, byte-identical to the tree**. A
  manifest that names a directory which does not exist is a hard build error.
* **The CPU driver** is the reference executor, NumPy-backed, and is FROZEN.
  Several limitations below are consequences of that, not of missing effort.
* **The native Rust layer is kernels, not an executor.** `numfast_native.dll` is
  a set of column-at-a-time compute kernels reached over `ctypes`. It does not
  schedule, does not stream, does not own memory. The executor is the Python
  planner and the CPU driver.
* **The GPU split** is by operation, declared at run time: `gpu_capabilities()`
  reports **15 of 33** operations, with a `chunkable` hint and dispatch limits
  per operation. The other **18** are CPU-only and say so.
* **The WASM path** compiles the same Rust kernels to
  `@numfast/kernels` (npm): a portable kernel library, **86 function exports, one
  memory, zero imports, 17 typed wrappers**. It is **not a compute core** — there
  is no executor, no graph runtime and no allocator inside the `.wasm`. The
  **`.wasm`** is host-independent; the **published npm package is Node-only** and
  has no browser entry point — see [docs/INSTALL.md](docs/INSTALL.md).
* **Where the semantic contract lives.** In `docs/ARCHITECTURE.md`, and in the
  test suite that pins it. Where the two disagree, the test is right and the
  document is corrected.

---

## Performance

**No wall-clock number is claimed in this repository.** Not one, and the absence
is deliberate: a benchmark claim is only worth reading if the artefact it came
from and the command that produced it are both in the tree, and this release
ships no benchmark harness. The benchmark *inputs* are not the scarce thing —
they are large, external, and cannot be committed — so what a number would need
to be quotable is exactly what a library repository cannot carry. Rather than
publish figures a reader cannot re-run, this README states the performance
characteristics as design facts and leaves the measurement to you.

What can be said without a harness, and is:

* **Where the fixed cost sits.** One planner call turns a whole chain into one
  execution graph, so per-query overhead does not scale with the number of
  operations in the query. Against a system that pays a fixed setup per query
  this favours small and aggregate-shaped queries, and it is the only reason
  NumFast wins on anything.
* **Where it loses, and by how much, structurally.** High-cardinality grouping
  and distinct-count shapes are several times behind DuckDB, not a few percent
  behind. That is a property of the CPU driver being a NumPy reference executor
  (see the next section), not of missing effort. **If your workload is
  high-cardinality grouping, DuckDB is the better tool today.**
* **The GPU buys parity and residency, not speed.** See below.

Benchmark the operation mix you actually have. `examples/quickstart.py` runs in
under a second and prints its own timings.

### The GPU

The GPU path runs through `wgpu-py` 0.31.1 (WebGPU over Vulkan/DX12; no CUDA).
All GPU cost measurements in this repository were taken on **one** RTX 2060
driving Vulkan.

* **15 of 33** operations have a GPU implementation; **18 are CPU-only** and are
  listed by `nf.app().gpu_capabilities()`.
* The recorded GPU measurements at the sizes measured were **slower than the CPU
  path**. What the GPU buys today is **parity and residency**: the same
  permutation, the same sums. **No GPU speedup is claimed.**
* Asking for `backend='gpu'` on a graph that uses a CPU-only operation raises
  rather than falling back.

### What was removed, and why

Two classes of claim were dropped from this README rather than re-verified,
because the artefact that backed them is gone:

* **Every per-query benchmark figure**, including the ClickBench tables, the
  H2O CPU/GPU comparisons and the derived ratios. Their scripts and results
  have been removed from this repository as development history. The inputs were
  never here and could not be; the scripts were the only thing this repository
  uniquely held about them, and they are superseded. Nothing is lost that a
  reader could reproduce — nothing could be reproduced from here.
* **The internal audit record** (`docs/audit_history/`) — internal audit reports
  from a previous generation, several of them substantially Russian and
  deliberately untranslated. They are not user documentation and they describe a
  design that no longer exists.

What survives is the reasoning those reports produced, stated as design facts
where it is still true of this tree (see the architectural limits below), and
dropped where it was not.

---

## Known limitations

**[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)** is the complete list. Every entry
is tagged **PROVEN / MEASURED / INFERENCE / HYPOTHESIS / UNKNOWN** and carries a
reproduction; a tag without a reproduction is not a claim. The short version:

| | Limitation |
|---|---|
| Absent | `window` (rolling), `or_`, `join`, `replace`, `fill_null` — reasons above |
| Refused | `group` on a NULL key — a NULL group is *absent*, not wrong, and cannot be told apart from an answer |
| Refused | arithmetic on a TEXT column — it computed on dictionary codes (ranks, not values); all six operators now refuse, and `str_len()` is the honest route |
| Silent | `group` drops a key whose measure is entirely NULL. Documented, not fixed: fixing it means changing the grouping path. |
| Mismatch | `to_numpy()` reads an integer NULL as `0`; use `to_pandas()` |
| Refused | `group` with more than one measure column raises |
| Driver | GPU sort needs a power-of-two valid-row count, and refuses `int64`/`float64` keys |

Two architectural limits are stated as consequences of the design rather than as
apologies. Both are structural and readable in the code, so neither depends on a
measurement this repository does not ship:

* **There are no cheap strided or windowed views.** A window of width *W* is
  composed as *W* × `shift`, and every `shift` writes a full-length buffer — so
  a width-*W* window costs *W* full-length materialisations plus the (M, W)
  matrix itself. `numpy.sliding_window_view` is a view that writes zero bytes;
  the composition is not. This is why `window` is absent from the API rather
  than merely slow.
* **There is no resident GPU execution.** `gpu_execute` is a batch execute with
  per-node read-back, so a graph that stays on device is not expressible. Through
  the consumer surface the GPU is unreachable at all: `Chain.compile()` runs the
  CPU path and `Chain.explain()` prints `backend=n/a`. The GPU is reachable
  through the kernel-level API, where it buys parity and residency.

---

## Licence

**AGPL-3.0-only.** The verbatim text is [`LICENSE`](LICENSE); third-party
attributions are in [`NOTICE`](NOTICE). Every source file carries an
`AGPL-3.0-only` SPDX header, and `pyproject.toml` declares
`license = "AGPL-3.0-only"` with `license-files = ["LICENSE", "NOTICE"]`.

**Corresponding Source ships in the wheel.** Because the wheel conveys
`numfast_native.dll` — object code — AGPL-3.0 §6 is discharged by construction,
not by a written offer: the wheel carries all **47** `.rs` files plus `Cargo.toml`,
`Cargo.lock`, `REUSE.md`, `.cargo/config.toml` and the two link shims under
`numfast/_corresp_src/numfast-native/`, with a `SHA256SUMS` (53 entries) over
exactly those bytes. Verified on the 0.2.1 release candidate: 53 of 53 checksums
match in the wheel and again in the installed package, and `LICENSE` + `NOTICE`
land in `numfast-0.2.1.dist-info/licenses/` in both wheel flavours with
`License-Expression: AGPL-3.0-only` in `METADATA`. Details:
[`CORRESPONDING-SOURCE.md`](CORRESPONDING-SOURCE.md).

**There is no "paid AGPL".** [`COMMERCIAL-LICENCE.md`](COMMERCIAL-LICENCE.md)
announces that a separate, proprietary commercial licence is planned. **Its terms
do not exist yet** — they are not written, not priced and not an offer. The AGPL
grant is free, unconditional and complete, and nothing in this repository adds a
fee, a field-of-use restriction or a non-commercial condition to it. A closed-source
deployment is only possible under a separate licence from the copyright holder,
and that document is not here yet.

**Support and consulting are available now** and need no further paperwork
(AGPL-3.0 §4 permits charging for warranty, support or indemnity).

Security policy: [`SECURITY.md`](SECURITY.md).

---

## Repository layout

```
src/Core/            canonical dtypes, the public Series / Table types
src/Semantic/IR/     the semantic IR: one node per column
src/Semantic/TableExpr/  the consumer facade (44 public names)
src/Runtime/         planner + runtime  (FROZEN)
src/Drivers/CPU/     the NumPy-backed reference driver (FROZEN)
src/Drivers/GPU/     the wgpu-py / WGSL driver
src/Relational/      join, group-by, sort, dictionary, lookup, segmented
src/Storage/         schema, dictionary coding, NFS
src/numfast/         the importable package and its vendored builder
numfast-native/      the Rust kernel crate + the @numfast/kernels TS package
docs/                user-facing documentation: API, architecture, examples
tests/fast/          the test suite
tests/oracles/       independent reference implementations the tests check against
examples/quickstart.py
```

---

## Tests and CI

```bash
export PYTHONPATH="<repo>/src;<path-to-app-builder>"
python -m pytest tests/ -q
```

There is one tier: `tests/fast/`, run by default. It has no `heavy` marker and
no memory-scale cases.

Measured on the tree this README describes — a `git archive HEAD` extraction
into an empty directory, with no `build/`, `dist/`, `*.egg-info` or
`__pycache__` present, and no Rust build product either (Windows, Python
3.14.6): **772 passed, 4 skipped, 0 failed in 156.20 s**, with the optional
extras `wgpu` and `psutil` installed. Without `wgpu` the same run is **707
passed, 5 skipped, 64 failed** — the GPU lanes are load-bearing, not optional.
Every skip, and why it skips:

| Skipped | Reason |
|---|---|
| `test_ops_null_pattern.py` | one wall-clock ratio assertion, marked `heavy`; run it with `pytest -m heavy`. A ratio on this box is not a property of the engine. |
| `test_ops_regexp_deferred.py` | needs `duckdb`, which is not a dependency and is not installed |
| `test_packaging_adapters.py` | one case needs a second wheel venv built first |
| `test_rng_gate.py` | its R reference implementation is not in the tree |
| `test_gpu_disclosure.py` | one case skips when `wgpu` is not importable |

So two of these are a real coverage hole: the RNG-gate reference and the
packaging-venv case need material that does not ship. They run in a developer's
checkout and skip in CI. That is named here rather than hidden behind a green
line.

Against an **installed** wheel in a fresh venv, with the test tree taken from
the sdist and the package from site-packages, the same suite gives **503 passed,
5 skipped, 0 failed in 122.76 s**, with 20 test modules left uncollected
because they read engine source text that an installed package does not lay out
as a source tree. Each one is named, with its reason, at the end of the run.

On **Linux** (WSL2, Ubuntu 24.04, CPython 3.12) with the native library built
from its own Corresponding Source and `wgpu` installed, the same checkout gives
**770 passed, 0 failed, 21 skipped**, every test file in its own process.

In **one** process — `pytest tests/` — it is a different number, and the
difference is a real defect rather than a test artefact: the run **dies with
SIGSEGV** in the native TEXT lane, reproducibly, 3 runs out of 3. With that one
file deselected the single-process run is green: 729 passed, 0 failed, 21
skipped. The cause is in a frozen component and is not yet fixed; it is written
up in the report that accompanies this release. No Linux native artefact is
published; see [docs/INSTALL.md](docs/INSTALL.md).

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) has three Windows jobs:

1. **build the wheel, install it into a clean venv, import it** — gating.
2. **the tests that need nothing but this repository** — gating. The rest build
   their kernel through `from builder import MAIN`, and `builder` lives in a
   separate repository whose published `main` is behind what this tree needs, so
   that job is present, non-gating, and says so in its own step name.
3. **the whole suite** — non-gating, for the reason above.

There is **no publish workflow in this repository and no `v*` tag trigger**.
There is no Linux CI job either, and no Linux wheel is published: only a
Windows binary is committed. A Linux wheel **builds and installs** — measured
here as `numfast-0.2.1-py3-none-linux_x86_64.whl`, verified by `packaging` and
by installing it into a clean venv and running the installed-package suite
(488 passed, 20 skipped) — but the single-process suite run above is not yet
green, so it is not published.

---

## More reading

| Document | What it is |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | **how to install, all five channels**, and what each one gives you |
| [docs/API.md](docs/API.md) | the 44-name surface, the kernel-level names, and every guard |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the pieces fit: IR, Builder, CPU, native, GPU, WASM |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | runnable programs with their real output |
| [docs/README.md](docs/README.md) | what each document covers |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | what does not work, tagged by certainty |
| [README_PYPI.md](README_PYPI.md) | the PyPI long description, verbatim |
| [README.packaging.md](README.packaging.md) | what ships in the wheel, and what a user gets |
| [numfast-native/ts/README.md](numfast-native/ts/README.md) | `@numfast/kernels` — the npm package |
| [CHANGELOG.md](CHANGELOG.md) | what 0.2.1 is |
| [SECURITY.md](SECURITY.md) | how to report a vulnerability |
| [CORRESPONDING-SOURCE.md](CORRESPONDING-SOURCE.md) | AGPL §6: where the source is in each artefact |
| [COMMERCIAL-LICENCE.md](COMMERCIAL-LICENCE.md) | the announced commercial licence, and what it is not |

---

## Published artefacts

| | |
|---|---|
| PyPI | <https://pypi.org/project/numfast/> — `numfast-0.2.1-py3-none-win_amd64.whl`, `numfast-0.2.1-py3-none-any.whl`, `numfast-0.2.1.tar.gz` |
| npm | <https://www.npmjs.com/package/@numfast/kernels> — `@numfast/kernels-0.2.1` |

Publishing is a **manual** step. There is no publish workflow in this
repository and no `v*` tag trigger in CI.

**One name, one description.** The Python package and the npm package are two
artefacts of one project at one version, described consistently on PyPI, on npm
and here. Note also that the PyPI project name `numfast` and the npm scope
`@numfast` already carry three earlier releases (`0.0.1` yanked, `1.0.0a1`,
`1.0.0a2` on PyPI; `1.0.0-alpha.1/2/3` as `@numfast/numfast` on npm) from a
superseded design generation. `pip install numfast` and
`npm install @numfast/kernels` resolve to 0.2.1 and 0.2.1 respectively; a
pre-release pin (`pip install numfast --pre`, or `npm install
@numfast/numfast`) installs the earlier generation, not this one.