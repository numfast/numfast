# NumFast showcase

Five notebooks that demonstrate **only what this tree actually does**. Every
output in them was produced by executing the notebook against this checkout; none
is typed by hand.

They live in `showcase/` and deliberately **not** in `examples/`: `MANIFEST.in`
carries `graft examples`, so everything under `examples/` enters the sdist. These
are documentation for readers, not part of the distribution. The sdist is
byte-identical with and without this directory, and that is checked rather than
assumed.

| Notebook | What it shows |
|---|---|
| [`01_table.ipynb`](01_table.ipynb) | `Table` / `Series`, arithmetic between columns, derived columns, an across-tables lookup, group/reduce, and the layout the architecture actually exposes |
| [`02_gpu.ipynb`](02_gpu.ipynb) | adapter detection, the real GPU operation set, one task on CPU and on GPU, and the crossover taken from the repository's own recorded measurements |
| [`03_wasm.ipynb`](03_wasm.ipynb) | the `@numfast/kernels` WASM path: one loaded module, repeated calls, 86 exports / 1 memory / 0 imports / 17 wrappers |
| [`04_benchmark.ipynb`](04_benchmark.ipynb) | the comparative matrix, **including where NumFast loses by 250x** |
| [`05_real_task.ipynb`](05_real_task.ipynb) | a 200,000-row revenue roll-up, cross-checked against pandas, on the CPU path with no GPU present |

## Running them

```bash
export PYTHONPATH="<repo>/src;<path-to-app-builder>"
python -m jupyter notebook showcase/
```

* `01`, `04` and `05` need only `numpy`, `pandas` and `numfast`.
* `02` needs the optional `wgpu` extra **only** to print adapter facts. Without
  it the notebook says so and runs the CPU path — it does not fail.
* `03` needs **Node 22.18+** and drives the built `dist/` artefacts through a
  subprocess. It is not a Python API.

The notebooks were executed on Windows 11, Python 3.14.6, with an RTX 2060 over
Vulkan. `02` and `03` print the host they ran on.

## What these notebooks deliberately do not do

* **No wall-clock timing is quoted as a measurement anywhere.** The host is
  shared with other benchmark work, so a number taken here would describe this
  box at this moment. `04_benchmark.ipynb` reads its figures from
  `calibration_dataset.json`, which was in the repository before these notebooks
  existed.
* **No GPU speedup is claimed.** `02` shows the GPU losing for eight of ten
  operations at the sizes measured.
* **No GPU residency is claimed.** `gpu_execute` reads each node back to host
  memory. `tests/fast/test_doc_gpu_residency.py` scans **all five notebooks** —
  markdown, source *and* committed outputs — for such a claim, and the guard's
  own negative controls are in that file.
* **No private API is introduced to make a point.** `01` describes the block
  layout through `app.schema()` and `explain()`, both public.

## «Для AI-агентов»

Guidance for an agent that wants to use NumFast as a library — as opposed to an
agent working on NumFast's *source*, which is what the dev-environment's
`AGENTS.md` covers. There is deliberately no `AGENTS.md` in this repository.

**Read the API, do not guess it.** The public surface is **44 names** and it is
asserted as 44 in
`tests/fast/test_consumer_surface.py::test_v0_is_44_names_and_has_no_window_or_or`.
The registry is `src/Semantic/TableExpr/_lib/names.py`; a name is public iff it is
listed there. Before you write a line:

1. `docs/API.md` states the surface with worked output.
2. `nf.app().capabilities()` and `nf.gpu_capabilities()` answer "what can this
   build do" **at run time** — prefer them to any prose.
3. `KNOWN_LIMITATIONS.md` is tagged **PROVEN / MEASURED / INFERENCE / HYPOTHESIS
   / UNKNOWN**. Check the tag before you rely on a limitation.

**Four names are absent on purpose.** `window` (rolling), `or_`, `join` and
`replace`. If your task needs one, it cannot be done in this version — say so,
do not approximate it with something that silently differs.

**Do not use these, they are not public:** anything under `numfast._*`. That is
internal by the package's own contract and carries no compatibility promise.
`nf.__all__` is the boundary, plus `Series`, `Table` and the three disclosure
functions.

**The engine refuses rather than lies, and you should let it.** A `ValueError`
from NumFast is the feature working. Do not catch it and substitute a NumPy
computation — that reintroduces exactly the silent divergence the refusal exists
to prevent. The refusals you can rely on are tabulated in `docs/API.md`; the most
common are a NULL group key, arithmetic on a TEXT column, `or_`, an array
exponent, and a CPU-only operation requested on the GPU.

**dtypes decide answers.** Logical values are `int32`/`int64`; an out-of-range
value raises rather than wrapping. But note the case in «Находки» below: mixing an
integer column into arithmetic with a float column can produce a truncated
integer result **without raising**. If a number looks wrong, check the schema
with `app.schema(table)` before assuming the engine.

**GPU residency does not exist.** There is no `to_gpu`, no `gpu_resident`, no
`stay_on_gpu`; the assertion is
`test_doc_gpu_residency.py::test_the_surface_has_no_gpu_residency_verb`. Every
result returns to host memory. Do not generate code that assumes otherwise.

## Находки — findings, reported and deliberately **not** fixed

Nothing here was auto-fixed. Each entry says what, whether it blocks a release,
and what a fix would cost.

### 1. `int × float` arithmetic silently truncates the result

**What.** `derive` of `int32/int64` column × `float64` column returns an
**integer** column with the fractional part truncated toward zero. No exception,
no warning.

```python
units = nf.from_numpy(np.array([10, 3, 7, 1], dtype=np.int32))      # int32
price = nf.from_numpy(np.array([2.5, 9.0, 4.0, 11.0]))             # float64
t = nf.Table(nf.get_kernel(), {"units": units, "price": price})
t.query().derive("gross", app.c("units") * app.c("price")).compile()
# -> gross = 25, 27, 28, 11   instead of  25.0, 27.0, 28.0, 11.0
```

**Verified** on this checkout: pandas gives `25.0, 27.0, 28.0, 11.0`; NumFast
gives `25, 27, 28, 11`. With **both** columns `float64` the product is exact and
bit-identical to pandas, so the fault is specific to mixed-dtype `mul`.

**Blocks a release?** **Yes — this is the most serious finding.** It is a silent
wrong answer in the most ordinary operation the library offers, and it directly
contradicts the README's *"This refuses rather than lies"* claim, which
`docs/API.md` tabulates operation by operation. Every other mixed-dtype hazard in
this engine refuses with a precise message naming the cause; this one does not.
A downstream aggregate over such a column is silently wrong, and no
cross-check exists to notice.

**Cost to fix.** The refusal is in `Semantic/TableExpr/_lib/chain.py` alongside
the existing `truediv` dtype guard, which already refuses the *related* case of
a derived left operand with the same "no dtype-promotion verb" reasoning. Adding
a matching guard on `mul`/`add`/`sub` when the operands' declared dtypes differ
is a small, local change with an existing precedent. Changing the lowering to
promote would be larger and would need a `where`/cast capability that v0
deliberately does not have.

**Workaround until fixed.** Load arithmetic columns as `float64` at the source.
`01_table.ipynb` §"A deliberate note on dtypes" and `05_real_task.ipynb` §5 pin
the defect instead of hiding it.

### 2. The WASM function-export count is 86, but three comments in-tree still say 85

**What.** `numfast-native/ts/wasm-info.mjs` says "85 kernels" and "20-of-85",
and `test/artifact.mjs` says "the current 85-function build". The shipped bytes
export **86** functions, and `dist/BUILD.json`, `package.json`,
`numfast-native/ts/README.md`, `docs/INSTALL.md` and the root `README.md` all say
86. `artifact.mjs`'s own comment explains the 85 → 86 move, so the two are
contradictions *within the same file*.

**Verified:** `node numfast-native/ts/wasm-info.mjs dist/numfast_native.wasm`
reports `funcCount: 86`, `exportCount: 87` (= 86 func + 1 memory),
`importCount: 0`, `i64Count: 21`.

**Blocks a release?** **No.** The *documented* count is correct everywhere a
reader meets it; only two stale comments disagree, and one of them is in the file
that would have caught the drift. Nothing user-facing is wrong.

**Cost to fix.** Two comment edits. Deliberately not made here: this is a
documentation/test commit set and the correct number is already what every
published surface says. Recorded so the discrepancy is not rediscovered as a
worry.

### 3. `truediv` is refused on a derived operand regardless of dtype

**What.** `derive("gross", a*b)` then `derive("r", app.c("gross") / app.c("u")`
raises even when both source columns are `float64`, because a derived expression
has no declared dtype for the planner to reason about.

**Blocks a release?** **No.** This is the engine refusing a promise it cannot
keep, and the message names the fix. It is a real ergonomic constraint, though:
the suggested "divide a column that is float at the source" only works when the
numerator is a source column.

**Cost to fix.** A dtype-propagation rule for `mul`/`add`/`sub` results — which
would land in the same place as finding 1 and should be designed together with it.

### 4. `nf.returns` and `nf.rolling_mean` on int32 produce surprising values

**What.** `nf.returns(s)` on `[3,1,4,1,5]` returns `[2147483647, -1, 3, -1, 4]` —
the first element is `int32` wraparound on a divide-by-zero, and the values are
integer-divided. `nf.rolling_mean(s, 2)` returns `[nan, 2.0, 2.5, 2.5, 3.0]`, where
the leading element is NaN rather than a partial-window value.

**Blocks a release?** **No.** Both are kernel-level names, explicitly outside the
44-name v0 registry and documented as unstable. The `RuntimeWarning: divide by
zero` is emitted rather than swallowed.

**Cost to fix.** Out of scope for this tree; noted so no agent reaches for these
two names expecting pandas semantics.

### 5. `nf.lookup` requires `int32` keys and refuses `int64`

**What.** `lookup(build, probe)` raises for `int64` keys with a clear message and
a fix. **This is correct behaviour**, listed only because it is the one
across-tables operation available and an agent will reach for it.

**Blocks a release?** No.

**Cost to fix.** None — the refusal is the documented design.

### 6. Two channels needed to run the `showcase` build, and `setuptools` is not installed by default

**What.** Building the sdist to prove `showcase/` does not enter it requires
`setuptools`, which is absent from this environment's site-packages; the wheel
came from a separate extracted copy. The WASM notebook additionally needs
`node` and a previously-built `numfast-native/ts/dist/`.

**Blocks a release?** **No.** Neither is a defect in the artefact.

**Cost to fix.** None. Recorded because a reader reproducing the sdist proof
needs to know `setuptools` must be present, and `python -m build` fails with a
confusing `BackendUnavailable` without it.

## What the notebooks were verified against

| Claim | Pinned by |
|---|---|
| the surface is 44 names | `test_v0_is_44_names_and_has_no_window_or_or` |
| no GPU-residency verb exists | `test_the_surface_has_no_gpu_residency_verb` |
| no residency claim in prose, notebooks or notebook outputs | `test_no_public_document_claims_gpu_residency`, `test_no_public_notebook_claims_gpu_residency` |
| the notebook guard itself fires | `test_the_notebook_guard_catches_a_claim_on_every_channel` |
| 15 of 33 ops on GPU, 18 CPU-only | `test_gpu_disclosure.py`, measured at run time |
| 86 WASM exports / 1 memory / 0 imports / 17 wrappers | `numfast-native/ts/test/abi.test.mjs`, `readme.test.mjs` |
| the `.wasm` is the build `BUILD.json` records | `numfast-native/ts/test/artifact.mjs` sha256 check, reproduced in `03` |
| the documented CPU/GPU figures | `calibration_dataset.json`, `source: "measured:seed42"` |

## Licence

`AGPL-3.0-only`, the same as the rest of this repository. Every notebook and this
file are part of the work.