# NumFast architecture — orientation

Enough to understand the design and where each decision lives. Where the
design is *normative* — the three-valued logic, NULL handling, tie order — this
document states it and says so explicitly, in "Where the semantic contract
lives" below. Where it only orients, it says that too.

---

## The shape of a query

```
orders (pandas / numpy / arrow)
   │  nf.from_pandas(...)
   ▼
Table ── .query() ──► Chain          a list of jobs: {op, inputs, params, out}
   │                    │
   │                    ├─ filter(expr)      emit nodes
   │                    ├─ derive(name, e)  emit nodes
   │                    ├─ group(key, {m: (ops,)})  emit nodes
   │                    ├─ sort / limit          emit nodes
   │                    └─ .jobs() / .explain()   inspect without executing
   ▼
 .compile()  ──►  one planner call  ──►  ExecutionGraph
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
             cpu_execute                gpu_execute              (r_* resident
            Drivers/CPU                 Drivers/GPU               primitives,
            NumPy reference             wgpu-py / WGSL             not the default)
                    │                         │
                    └──────────┬──────────────┘
                               ▼
                      numfast_native.dll  (ctypes, Rust kernels)
```

One chain is one planner call is one graph. There is no second IR, no per-query
scheduler, and no path that bypasses the planner.

---

## The IR

`src/Semantic/IR/` defines the intermediate representation: a list of jobs, each
one `{op, inputs, params, out}`, one node per column. Nodes are the vocabulary
the drivers implement — `series`, `compare`, `mask`, `filter`, `gather`, `reduce`,
`groupby`, `groupby_multi`, `sort`, `slice`, `shift`, `map`, `cumsum`,
`pack_keys`, `unique`, `lookup`, `where`, the `text_*` family, the `rng_*` family
and a few specialised ones.

`app().capabilities()["ops"]` enumerates the live set at run time:
**33 operations**, each with a dtype contract and a `chunkable` hint.

Two properties worth knowing:

* **A chain holds jobs, not a second IR.** Lowering an expression to a node list
  is the only path into the engine.
* **Indicators are compositions, not monolithic loops.** `sma`, `rsi`,
  `boll_up` / `boll_lo`, `stoch_k`, `rsum`, `rmin`, `rmax`, `mom` and `returns`
  are expressed as *fused elementwise plans* over the primitive kernels rather
  than as hand-written loops, and the test suite holds each fused plan
  **bit-exact against the separate one-output-per-dispatch plan** built from the
  same emitters, at the same dispatch count. `groupby`, `sort` and `ema` are
  refused rather than fused silently.

## The Builder Extension model

The engine is assembled from **30 Extensions**, each a folder with three things:

```
src/Relational/Sort/
    Sort.toml        name, version, depends, alias[], mods[], [metadata]
    Sort.py          entry point: imports from _lib/ and calls setup(kernel)
    _lib/            the implementation, one responsibility per file
```

`setup(kernel)` may only fill `kernel.metadata`. The manifest declares the
Extension's aliases and its dependencies; the Builder resolves them and mounts
Extensions by name. `full.toml` at the repository root lists all 30.

Two consequences a user can observe:

* **Packaging is explicit.** `setup.py` vendors every Extension into the wheel as
  `numfast/_ext/<Name>/` — 133 files, byte-identical to the tree — and a
  `[[extensions]]` entry whose directory is missing is a **hard build error**.
  (That check exists because an earlier build shipped a partial Extension set
  without anyone noticing.)
* **The installed package carries its own builder.** `src/numfast/_builder/` is a
  self-contained Builder with no filesystem roots baked in, so `numfast` resolves
  its own Extension set from `numfast/full.toml` beside `numfast/_ext/`.

## The planner

`src/Runtime/Planner/` answers three questions with data rather than constants:

1. **Can this run?** Capability and limits are read from the driver's own
   `capabilities()`. There is no Planner-side duplicate table.
2. **What will it cost?** A measured, fitted cost model, `cost = a·n + b`, with
   separate CPU and GPU coefficients per operation.
3. **Which backend?** `backend='auto'` picks the GPU only when a measured
   calibration reports a strictly lower GPU host cost. An unmeasured or
   uncovered graph resolves to the CPU — never to a fabricated GPU choice.

The shipped `calibration.toml` is a measured profile: `source = "measured:seed42"`,
fitted on one machine (Intel Family 6 Model 79, Windows 11, Python 3.14.6, GPU
NVIDIA GeForce RTX 2060 over Vulkan). It travels with both wheels, and
`NUMFAST_CALIBRATION_DIR` overrides it. When no profile is found the planner says
so — `version: "stub"`, `source: "none"`, `warning: "no measured calibration
profile; stub costs"` — rather than inventing numbers. When the device changes it
says that too: `device changed (…); recalibrate, costs are stale`.

One thing to be precise about: **the calibration profile is loaded and honoured by
the kernel-level path, but the consumer facade does not surface a backend
decision.** `Chain.explain()` prints `backend=n/a`, and `Chain.compile()` runs the
CPU path. If you need the GPU, call the kernel API directly.

## The CPU driver

`src/Drivers/CPU/` is the reference executor, NumPy-backed, and it is **FROZEN**
— as are the Runtime, the Planner, the GPU/CPU Driver ABIs and the
Executor/Kernel ABIs. Several documented limitations are consequences of that
freeze rather than of missing work; [KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md)
says which.

## The native Rust kernel layer

`numfast-native/` is a Rust crate with **no third-party dependencies**
(`Cargo.lock` holds exactly one package). It is loaded through `ctypes` from
`numfast/_native/numfast_native.dll`.

**It is kernels, not an executor.** It provides column-at-a-time compute
primitives — sort and lexsort, hash join, group-by variants, pack/unpack, carry,
segmented reduce, row-wise k-way reductions, single-source shortest paths, cost
travel, bounded select, pair insert, text encode, RNG. It does not schedule, does
not parallelise across queries, does not stream, and does not own memory. The
executor is the Python planner and the CPU driver.

Because the wheel conveys this binary as object code, AGPL-3.0 §6 is discharged by
shipping the Corresponding Source inside the wheel: 47 `.rs` files, the crate
manifest and lockfile, the cargo config, the link shims, and a `SHA256SUMS` over
exactly those bytes. See
[CORRESPONDING-SOURCE.md](../CORRESPONDING-SOURCE.md).

## The GPU split

The split is by operation, declared at run time, not by wish:

```python
>>> caps = nf.app().capabilities()
>>> caps["op_count"], caps["gpu_op_count"], caps["cpu_only_op_count"]
(33, 15, 18)
```

* **15 operations have a GPU implementation.** `series`, `pack_keys`, `compare`,
  `mask`, `filter`, `gather`, `reduce`, `groupby`, `groupby_multi`, `sort`,
  `slice`, `shift`, `map`, `cumsum`, `rng_fill_i32`.
* **18 are CPU-only**, and asking for the GPU on a graph that uses one raises and
  names it.
* Several GPU operations are **hybrids**: `reduce` computes GPU partials and
  finishes in int64/float64 on the host; `groupby` accumulates WGSL partial
  histograms and merges them on the host; `filter` compacts on the GPU with a
  small host prefix. `sort` is a bitonic stable permutation restricted to a
  single int32 or finite-float key over a power-of-two row count.
* All GPU cost measurements in this repository come from **one** RTX 2060 over
  Vulkan through `wgpu-py`. The recorded GPU timings at the sizes measured were
  slower than the CPU path. The GPU claim is parity and residency, **not
  throughput**.

## The WASM kernel path

`cargo build --target wasm32-unknown-unknown` produces a `.wasm` of the same
kernels, and `numfast-native/ts/` packages it as `@numfast/kernels`.

| Property | Value | Where it comes from |
|---|---|---|
| function exports | **86** | derived from the `#[no_mangle]` bodies in `src/{lib.rs,router.rs}` and recorded in `numfast-native/abi/census.json`, each with the `FuncType` the compiler emitted |
| memory | **1** | the module's memory section |
| imports | **0** | the module imports nothing |
| typed wrappers | **17** | `WRAPPED` in `ts/kernels.ts`, each validating lengths before the call and resolving the per-symbol return code |
| reachable by name | **69 more** | `callRaw(bridge, symbol, ...)`, deliberately unwrapped |

**This is a portable kernel library, not a compute core.** There is no executor,
no graph runtime, no IR dispatch and no buffer allocator inside the `.wasm` —
orchestration, buffer ownership and the type layer stay in the host. 21 of the
86 exports touch `i64`, and the wrappers refuse a JavaScript `Number` for those
rather than truncating it.

Eleven of the 86 exports trap on every non-empty call on wasm32: the join and
multi-threaded hash kernels call `std::thread::scope`, which `wasm32-unknown-unknown`
does not support, and the crate is built `panic = "abort"`. They are reachable
through `callRaw` and will trap. The source comments above some of them claim
"native-only", which is wrong; `numfast-native/abi/census.json` hazard H4 records
the measurement.

The `.wasm` is **not committed** — it is a build product, and `.github/workflows/
wasm-kernels.yml` builds it from source on every push and regenerates the Python
parity fixtures on the same commit.

## Where the semantic contract lives

In two places, in this order of authority:

1. **The code and its tests.** Three-valued logic, NULL handling and tie order are
   pinned by named tests. Where a test and a document disagree, the test is right
   and the document is corrected — that happened to the `sort` NULL guard, whose
   stated premise (that NULL rows would come *first*) was false; they come
   *last*, matching pandas `na_position="last"`. The guard was removed and the
   behaviour is now pinned by a parity test.
2. **`docs/`** — this document, plus [API.md](API.md) for the surface and
   [EXAMPLES.md](EXAMPLES.md) for runnable programs.

**This document is normative for the semantic contract.** Where a rule is stated
here as a rule, it is the rule; the tests enforce it and a change to either
without the other is the defect. The numeric tolerances that back it are
machine-readable in
[`specs-rebuilt/conformance-profile.toml`](../specs-rebuilt/conformance-profile.toml),
which `tests/fast/harness.py` reads rather than hardcoding, so there is one
number per tolerance rather than two.

`KNOWN_LIMITATIONS.md` is the register of everything the contract does *not*
cover, with a certainty tag on each entry.