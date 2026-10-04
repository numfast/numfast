# @numfast/kernels

NumFast's compute kernels, compiled to WebAssembly, wrapped for JavaScript and
TypeScript.

```
npm install @numfast/kernels
```

```js
import { loadKernels, ssspCsr } from "@numfast/kernels";

const k = await loadKernels();               // the packaged .wasm, verified
// 0 -> 1 -> 2, and 2 -> 3
const dist = ssspCsr(k,
  new Uint32Array([0, 1, 2, 3]),             // indptr
  new Uint32Array([1, 2, 0]),                // indices
  new Uint32Array([10, 20, 30]),             // weights
  0);                                        // source
// Uint32Array [ 0, 10, 30, Infinity-as-0xffffffff ]
console.log(dist[1], dist[2]);               // 10 30
```

Five lines, and the answer is checkable by hand.

---

## What this is, stated exactly

> NumFast compiles its compute kernels to WebAssembly: an **85-function** module
> with **one memory and no imports**, runnable in any host with a WebAssembly
> runtime. **This is a portable kernel library, not a compute core** —
> orchestration, buffer management and the type layer stay in the host. This
> package wraps **17 of those 85 kernels** with typed wrappers; the remaining
> 68 are reachable by name through `callRaw` and have no documented argument
> order.

Every number in that paragraph is asserted against the bytes of the shipped
`.wasm` by `npm test`. None of them is a claim.

**What this package is not.** There is no executor, no graph runtime, no IR
dispatch and no buffer allocator inside the `.wasm`. It exports functions and a
memory. That is the whole surface, and it is the Rust source's own position:
`numfast-native/src/router.rs:18` — *"Python owns orchestration/data/ABI"*.
Building the missing executor was measured at **3,515–5,875 lines of Rust**; it
is a separate project and this package does not start it.

**Licence.** AGPL-3.0-only, inherited from the Rust core. A JavaScript consumer
of this package is a consumer of AGPL-licensed software. `LICENSE` and `NOTICE`
travel with the package.

---

## Coverage is a constant, not a paragraph

```js
import { WRAPPED, TOTAL_EXPORTS, SCOPE } from "@numfast/kernels";
WRAPPED.length;   // 17
TOTAL_EXPORTS;    // 85
SCOPE.statement;  // the paragraph above, generated from those two numbers
```

17 wrapped:

| family | kernels |
|---|---|
| graph | `nf_sssp_csr`, `nf_sssp_csr_pred`, `nf_sssp_batch`, `nf_cost_travel_batch`, `nf_cost_intern`, `nf_rowwise_kway_time_argmin_gather`, `nf_adjacency_slice`, `nf_adjacency_gather` |
| elementwise `map` | `nf_map_i32`, `nf_map_scalar_i32`, `nf_map_fscalar_i32`, `nf_map_f32`, `nf_map_f32_divpow`, `nf_map_scalar_f32`, `nf_map_scalar_f32_divpow`, `nf_map_f64`, `nf_map_scalar_f64` |

68 **not** wrapped. They are listed by `WRAPPED`'s absence and reachable through
`callRaw(bridge, "nf_group_sum_count", ...)`. They are unwrapped on purpose: the
pointer-vs-length order of each argument is recorded nowhere in the repository,
so a wrapper written from the signature alone would be a guess dressed as an
API. `test/abi.test.mjs` fails if a kernel is wrapped without an ABI entry, or
documented with a signature that disagrees with the `.wasm`.

Parity fixtures cover **9** of the 17. The eight graph kernels are wrapped but
have **no** Python parity fixture yet, and `test/parity.test.mjs` prints their
names on every run so the gap cannot be mistaken for coverage.

---

## Three rules a caller must know

### 1. The caller owns every buffer

There is no allocator in the module. `Bridge` hands out byte offsets in the one
exported `WebAssembly.Memory`; you fill them and read them back. Every wrapper
copies its output out, so the array you get back is yours.

### 2. `memory.grow` detaches every view you already have

Not theoretical: it detached a view during this package's own audit. If you grow
the memory — and every large call does — **every `TypedArray` you took before
the growth is detached** and reads from it return zeros. `Bridge.ensure()` grows
and returns a fresh view; that is why the wrappers use it. If you hold offsets
and build views yourself, re-fetch the view after any allocation.

### 3. Caller buffers must sit above the module's own data

This is the rule that is easy to get wrong, and getting it wrong is **silent**.

The `.wasm` initialises `[0x100000, 0x103203)` of its own linear memory — the
float constants that `powf` reads live there — and the module's shadow stack
grows *down* from `0x100000`. A caller buffer placed inside that region
overwrites the constants, and `powf` then returns plausible, wrong numbers with
`rc = 0`. Measured: at a `0x101000` buffer base, **any input longer than 8632
bytes** did it; the same input at `0x20000` was correct, and the native ctypes
path was correct at every size.

So `Bridge` does not use a constant base. It probes the instantiated memory for
the top of the module's initialised image and places the first buffer above it:

```js
k.staticDataEnd;  // 1061379 on the current build
k.base;           // 1114112
```

`test/abi.test.mjs` asserts `base > staticDataEnd` and `base >= stackPointer`
against the artefact. **If you bypass `Bridge` and place buffers yourself, you
own this rule.**

---

## The error contract: two channels, both closed

The kernels report failure in two shapes. Neither is an unrecoverable abort.

**A return code.** `0` is ok. Negative codes are frozen by the C ABI
(`numfast-native/src/core/errors.rs`) but their *meaning is per symbol*: `-1` is
"null pointer" for `nf_group_*` and "unsorted abort" for `nf_sorted_run_*`. No
bare number is ever surfaced. Every code the package can meet is resolved
through the per-symbol table in `abi.ts` and thrown as:

```js
class NumFastError extends Error { symbol; code; reason; }
```

A code with no entry in the table is reported as *not in this package's table*
— never as a number pretending to mean something.

**A trap.** `WebAssembly.RuntimeError`. Measured: **63 of the 85 kernels trap
on a negative length**, because the kernels build a slice with `n: usize` and a
negative `i32` becomes a ~4·10⁹ element slice. Catchable in JavaScript, but an
unhandled one terminates the host process.

This package **prevents** that rather than catching it: every wrapper validates
its lengths before the call, so for the wrapped surface the trap channel is
unreachable. Where it cannot be prevented — the unvalidated `callRaw` escape
hatch — it is caught and named:

```js
class NumFastTrap extends Error { symbol; cause; }   // cause is the RuntimeError
```

Catch `NumFastError | NumFastTrap` and you have caught everything this package
can throw.

**On any failure, outputs past the abort point are caller-owned garbage.**
`errors.rs` documents partial writes, and nothing in WebAssembly rolls linear
memory back — `test/traps.test.mjs` asserts that a trap leaves what it wrote.
That is the reason prevention beats catching, and the reason no wrapper returns
a buffer on a failure path.

---

## i64 and the BigInt boundary

**20 of the 85 exports** declare `i64` somewhere in their `FuncType`. On the
JavaScript side that is a hard type boundary, not a coercion:

```js
nf_rng_fill_i32(out, n, 12345, state, ...)   // TypeError: Cannot convert 12345 to a BigInt
nf_rng_fill_i32(out, n, 12345n, state, ...)  // accepted
```

A `Number` is **rejected**, not truncated. The one wrapped kernel on the i64
side is `nf_cost_intern`, whose *return* is the group count:

```js
const { ng, ids, uniq } = costIntern(k, vecs, n, width);   // ng is a number here
```

It crosses the boundary as a `bigint` and is narrowed explicitly. `callGuarded`
and `callTrapped` are kept separate for exactly this reason: `nf_cost_intern`
answers with a **count**, so a successful call returns `3` and running it
through a return-code check would turn every success into an error.

The list of the 20 is in `dist/BUILD.json`, written by every build.

---

## Parity with Python

```bash
npm run build
npm test
```

`npm test` runs three suites and **refuses to run at all** unless `dist/` holds
the build it was written against — no fallback, no skip.

* `test/abi.test.mjs` — the published numbers, the ABI table, and the memory
  layout, all checked against the bytes of the `.wasm`.
* `test/traps.test.mjs` — prevention, the two channels, the BigInt rule, and the
  partial-write property.
* `test/parity.test.mjs` — 203 cases of Python ↔ JS/WASM.

The parity expectations are **not** written in JavaScript. They come from
`test/fixtures/parity.json`, produced by `test/gen_fixtures.py` from **two**
Python paths that were compared against each other first:

```
NumPy + the documented int32 round-trip   (the frozen public semantic)
ctypes into the committed native DLL      (the engine's own path)
```

If those two disagreed the generator aborts, so a fixture never records a
disagreement as if it were truth. Regenerate with:

```bash
python numfast-native/ts/test/gen_fixtures.py
```

Inputs are reproduced from an explicit 32-bit LCG rather than stored, so the
fixture file is 121 kB instead of 31 MB. Small cases carry their expected bytes
verbatim so a reader can check them by hand; large cases carry a sha256 of the
expected bytes plus the head and tail lanes.

**Comparison rules.** Integer lanes: byte-exact, no tolerance ever. Float lanes:
exact, except inside the project's own published budget —
`specs-rebuilt/conformance-profile.toml` grants f32 4 ULP and f64 2 ULP. The
budget is read by the generator and applied by the test; it is not chosen here,
and lanes that need it are **counted and printed**.

Today exactly one family needs it: `pow`, and `div` on f32, where the wasm32
build uses Rust's bundled `libm` and the host uses the platform `libm`. The two
differ by **1 ULP** on a few percent of lanes — for example `nf_map_f64 pow` at
`n=64`: 2 of 64 lanes, both exactly 1 ULP, both inside the 2-ULP budget. That is
a real measured difference, reported rather than hidden.

**The known divergence that was not a kernel bug.** An earlier revision recorded
WASM i32 `map pow` with a fractional exponent as agreeing at `n=6` and diverging
at `n=100000`, while the native path agreed at both, and suspected a stale
`.wasm`. Re-measured on the current 85-function build: **it was neither a stale
artefact nor a `powf` difference.** It was the caller writing its buffer over
the module's constant pool — see rule 3. Native agreed with NumPy at every size
and every exponent, and so does WASM once the buffers are placed correctly.
`test/parity.test.mjs` pins parity at `n=6`, `n=64` **and** `n=100000` for five
fractional exponents, and asserts those lanes are bit-exact rather than merely
inside a budget.

---

## Building

```bash
cargo build --manifest-path numfast-native/Cargo.toml \
      --target wasm32-unknown-unknown --release
cd numfast-native/ts && npm install && npm run build
```

`npm run build` runs `tsc`, copies the release `.wasm` into `dist/`, and writes
`dist/BUILD.json`:

```json
{ "sha256": "...", "bytes": 196149, "exportCount": 86, "funcCount": 85,
  "importCount": 0, "i64Count": 20, "i64Exports": [...],
  "initialMemoryPages": 17, "stackPointer": 1048576,
  "dataSectionEnd": 1048587, "version": "0.2.1", "builtFrom": "<git sha>" }
```

Every test prints that line, so a failure is attributable to one build.

The build **fails** — it does not warn — if the export count has moved, if the
module has gained an import, if the i64 count has moved, or if any `abi.ts`
entry disagrees with the `.wasm`. Those checks are the reason a 56-function
build cannot sit in git unnoticed again: `numfast-native/tools/` still carries
such a copy, 29 symbols behind, and four parity scripts used to validate it.

**The `.wasm` is not committed.** `numfast-native/.gitignore` excludes `target/`
and that is correct — it is a build product, and a committed copy is precisely
how a stale one survived. `dist/` is likewise not committed; it is produced by
the command above.

**Version.** `0.2.1`, read from the engine's `pyproject.toml` at build time and
cross-checked against `package.json`. A kernel package that reported a different
version from the engine could not be traced to a commit, so the build refuses to
produce one.

---

## Coverage of the acceptance criteria

| criterion | status |
|---|---|
| W1 obtain the artefact | met — one documented cargo line, verified by CI |
| W2 export introspection | met — 86 = 85 Func + 1 Memory, 0 imports, asserted from the bytes |
| W3 no unrecoverable abort | met **for the wrapped surface** — lengths are validated before the call, so the trap channel is unreachable through a typed wrapper; `callRaw` normalises it to `NumFastTrap` |
| W4 per-symbol error table | met for the 17 wrapped kernels (`abi.ts`), checked against the `.wasm`. The other 68 are not tabulated, and are not wrapped |
| W5 parity with Python | met for the 9 elementwise kernels (203 cases, fixture-backed, CI-gated). The 8 graph kernels are **not** parity-fixtured |
| W6 installable artefact | met — `package.json`, `exports`, declarations, `dist/`, CI |
| W7 not a compute core | held — no doc calls it an executor, a graph runtime or a compute core |

## Layout

```
ts/
  package.json  tsconfig.json  build.mjs   wasm-info.mjs
  bridge.ts     raw ABI, allocator, memory probe
  errors.ts     NumFastError / NumFastTrap / NumFastArgumentError, callGuarded
  abi.ts        per-symbol argument order and return-code meanings
  kernels.ts    the 17 typed wrappers
  index.ts      public surface, loadKernels, SCOPE
  test/
    artifact.mjs     the artefact guard
    abi.test.mjs     published numbers vs the bytes
    traps.test.mjs   the error contract
    parity.test.mjs  Python <-> JS/WASM
    gen_fixtures.py  produces test/fixtures/parity.json from the Python host
  qlookup.ts    a benchmark driver, outside the tsc program
  demo.html     a browser demo
```