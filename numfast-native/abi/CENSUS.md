# numfast-native ABI census

**All 86 exported symbols. Derived from the Rust source, verified against the compiler's own
output, and reconciled against the 17 wrappers the `@numfast/kernels` package already ships.**

Produced 2026-10-04 by reading `numfast-native/src/*.rs`. Documentation only: **no Rust kernel,
no TypeScript, no build script and no production file was modified.** Nothing here is fixed; where
the census found a defect it is recorded as a finding, and section 6 lists the hazards left alone.

Machine-readable form: [`census.json`](census.json), same directory.

---

## 0. Read this first

The reason 68 of the exports had no wrapper is not that their signatures are missing. They are
everywhere in the source. The reason is that **which `i32` is a pointer and which is a length is
not recoverable from the signature**, and getting it wrong does not produce an error - it produces
a wrong answer or an out-of-bounds write.

Measured over this census: there are **346 pointer parameters** across the 86 symbols. **35 of them
(10%) are immediately preceded by the scalar that determines their length.** Every one of the 86
symbols has at least one pointer whose sizing scalar sits somewhere else in the argument list. The
"pointer, then its length" reading is right about one pointer in ten.

So this census does not guess the convention. For each parameter it records the **length
expression the symbol's own body passes to `borrow()` / `borrow_mut()`** - the literal second
argument of an unchecked `slice::from_raw_parts`. That is the ABI, read out of the code rather
than inferred from a name.

### How the order was then confirmed

Argument order and value types were not taken on trust from that reading. They were checked
against the compiler:

| check | result |
|---|---|
| Rust-derived params + return vs the `FuncType` of a **fresh** `cargo build --target wasm32-unknown-unknown --release` | **86 / 86 symbols, 0 mismatches** |
| the same vs the **shipped** `ts/dist/numfast_native.wasm` (sha256 `9594fe62`) | 85 / 85 present symbols, 0 mismatches; 1 symbol absent (section 2) |
| `FuncType` carries i32 / i64 / f32 / f64 per parameter, positionally | yes, so order is compiler-confirmed for every argument of every symbol |

The remaining fields - role, direction, trap behaviour, allocation, ownership, concurrency - come
from the bodies and the doc comments, and are flagged where they are read from behaviour rather
than from a declaration.

---

## 1. Export count, verified

| quantity | value | how it was counted |
|---|---|---|
| `#[no_mangle]` symbols in the **source** | **86** | 63 literal `extern "C"` fns in `lib.rs` + 1 in `router.rs` + 22 macro instantiations |
| literal `#[no_mangle]` in `lib.rs` | 63 | `grep -c '#\[no_mangle\]' lib.rs` reports **69**; six of those are the `#[no_mangle]` *inside* the six `macro_rules!` templates, which expand to 22 symbols, so 69 = 63 + 6 templates |
| literal `#[no_mangle]` in `router.rs` | 1 | `nf_router_route`, `router.rs:95` |
| macro-generated | 22 | every invocation resolved individually, table below |
| function exports in the **shipped** `.wasm` | **85** | parsed from the export section of `ts/dist/numfast_native.wasm` |
| function exports in a **fresh build of this tree** | **86** | built into a scratch `CARGO_TARGET_DIR`; 87 exports total (86 functions + `memory`), 0 imports |
| wrapped by the TS package | 17 | `WRAPPED` in `ts/kernels.ts` |
| unwrapped | **69** (source) / 68 (shipped artefact) | 86 - 17, or 85 - 17 |

### The six macro shapes, all 22 instantiations resolved

| macro | template arity | instantiations | invocation lines |
|---|---|---|---|
| `select_scatter_ffi` | `(src, mask, n, out) -> i64` | `nf_select_scatter_i32`, `_i64`, `_f32`, `_f64`, `_u8` | 675, 680, 685, 690, 695 |
| `shift_ffi` | `(src, n, periods, out) -> i32` | `nf_shift_i32`, `_f32`, `_f64` | 741, 746, 751 |
| `cumsum_ffi` | `(src, n, out) -> i32` | `nf_cumsum_i32`, `_f32`, `_f64` | 802, 808, 814 |
| `map_ffi` | `(a, b, n, op, out) -> i32` | `nf_map_i32`, `nf_map_f32`, `nf_map_f32_divpow`, `nf_map_f64` | 931, 955, 971, 987 |
| `map_scalar_ffi` | `(a, n, s, op, out) -> i32` | `nf_map_scalar_i32`, `nf_map_fscalar_i32`, `nf_map_scalar_f32`, `nf_map_scalar_f32_divpow`, `nf_map_scalar_f64` | 939, 947, 963, 979, 995 |
| `segment_reduce_ffi` | `(values, n, bounds, m, op, out) -> i32` | `nf_segment_reduce_f32`, `nf_segment_reduce_i32` | 2048, 2055 |

The substitution is where a name-based reading goes wrong. In `map_scalar_ffi` the third argument
is `$sty`, and the five instantiations pass `i32` once and `f64` four times - so
`nf_map_scalar_i32` takes an **i32** scalar while the four symbols beside it take **f64**. The
`FuncType` check confirms exactly that split, per symbol.

---

## 2. The shipped artefact is one symbol behind the source

**`nf_pair_insert_i64` (`src/lib.rs:2972`) carries no `cfg` gate and is absent from the shipped
`.wasm`.** A fresh build of the working tree exports it.

- source: 86 `#[no_mangle]` symbols
- shipped `ts/dist/numfast_native.wasm`, sha256 `9594fe62...`: 85 exports, i64 boundary count 20
- fresh build of the same tree: 86 exports, i64 boundary count **21** (`nf_pair_insert_i64` returns
  `i64`), and **zero** `FuncType` differences on the 85 symbols the two artefacts share

The shipped artefact's sha256 matches `ts/dist/BUILD.json` exactly, so the package is internally
consistent - it was simply built from an older tree, and the change that added
`nf_pair_insert_i64` did not rebuild it. Consequences, none of them fixed here:

- `ts/build.mjs` `EXPECTED_FUNCS = 85` -> `node build.mjs` **fails on a clean clone**:
  "the artefact exports 86 functions, this package is built against 85"
- `EXPECTED_I64 = 20` -> fails on the same build (21, not 20)
- `TOTAL_EXPORTS = 85` in `kernels.ts`, and the figure quoted from it in `index.ts` and the
  README, is one behind
- `abi.ts`'s scope note "17 of 85 kernels. The other 68" should read 17 of 86 and 69
- `errors.ts`'s "63 of the 85 kernels trap" has a stale denominator

The BigInt figure this task quotes - "20 of 85" - is correct **for the shipped artefact**. Against
the source it is **21 of 86**, and the extra one is `nf_pair_insert_i64`.

One source comment points the other way and is wrong for the wrong reason: the `nf_pair_insert_i64`
block claims "no WASM export (native CPU research surface, same gate as `sssp`)". There is no such
gate on it, and `nf_sssp_csr` / `nf_sssp_csr_pred` / `nf_sssp_batch` are all exported by the
shipped artefact. See hazard H9.

---

## 3. Cross-check of the 17 shipped wrappers

The argument **order and arity recorded for all 17 is correct** - each is confirmed positionally
against the `FuncType`. Two of the 17 record a wrong **meaning**; six have incomplete return-code
tables. One of those is a live wrong-results bug. Severity below is about shipped code.

### CRITICAL - `nf_sssp_csr_pred`: an output was shipped as an input

`ts/abi.ts` records *"nf_sssp_csr plus a per-edge predicate byte; pred[i]==0 skips edge i"*, and
`ts/kernels.ts` `ssspCsrPred(bridge, indptr, indices, weights, source, pred: Uint8Array)` treats
argument 7 as a per-edge predicate **input** of `pred.length` bytes.

The truth: argument 7 is `pred: *mut i32`, an **output** of `np - 1 = V` int32 lanes - the
shortest-path tree parent per vertex, `-1` for root/unreached. `lib.rs:2885-2903` borrows it for
`v` lanes, and `sssp.rs:86-90` **fills** it with `SSSP_INF` before the search begins. It is never
read as an input. There is no per-edge predicate on this symbol.

Reproduced on the shipped artefact (sha256 `9594fe62`), V = 3, E = 3, `pred` passed as the
3-byte `Uint8Array` the wrapper would pass:

```
rc   = 0
dist = [4294967295, 4294967295, 4294967295]      expected [0, 2, 7]
```

The kernel's 12-byte `pred` fill runs past the 3-byte caller buffer and over the 12-byte `dist`
output that the bridge allocated immediately after it. **Success code, no trap, no guard-page
hit, silently wrong distances.** With `pred` sized correctly as `i32[V]` the same call returns
`dist = [0, 2, 5]`, `pred = [-1, 0, 0]`, which is the documented behaviour.

This finding outranks the documentation task. `ssspCsrPred` must allocate `pred` itself, return it
alongside `dist`, and drop the caller's `pred` argument; its doc comment describes a parameter that
does not exist. It has no parity test and no trap test - the only 9 of the 17 with parity tests are
the map family.

### MAJOR - `nf_adjacency_gather`: wrong return code in the ABI table

`abi.ts` records `-2: "sum(ends-begins) != total"`. Measured against the shipped artefact:

| call | actual | recorded |
|---|---|---|
| `total` smaller than `sum(ends-begins)` | **-3** MALFORMED | -2 |
| `total` larger than the sum | **0**, only the real lanes written | -2, as an error |
| `ends[g] > e`, or `begins[g] > ends[g]` | -2 BAD_RANGE | -2, but described as something else |

`adjacency.rs:103-134`. The table tells a caller that over-allocating `out` is an error when it is
in fact the safe direction, and omits `-3`, which is the code they will actually receive.

### MAJOR - `nf_cost_travel_batch`: reachable code missing

`abi.ts` lists only `-1`. A `k[i] == 0` lane returns **-2** (measured: `rc = -2`), documented at
`lib.rs:2180-2181`. `errors.ts` resolves codes through `RC_TABLE`, so a caller meets a code with no
table entry.

### MINOR - the rest

| symbol | defect |
|---|---|
| `nf_sssp_csr` | `-3` MALFORMED omitted (`sssp.rs:125-160`) |
| `nf_sssp_csr_pred` | `-3` MALFORMED omitted, on top of the CRITICAL semantics error |
| `nf_sssp_batch` | `abi.ts` note and the `kernels.ts` doc comment say `out[k*np]`; it is `out[k*V]` with `V = np-1` (`lib.rs:2945`). Over-allocates one row and mislabels the stride. |
| `nf_adjacency_slice` | `-3` omitted; `-2` described far more narrowly than `lib.rs:2062-2069` |
| `nf_cost_intern` | `codes: {}` although `-2` (null, or `width == 0`) and `-3` (`total < n*width`) are both reachable. The wrapper handles a negative return itself, so nothing crashes today. |
| `errors.ts` | cites `RC_MEANING` in `abi.ts`; `abi.ts` exports `RC_TABLE`. Dead cross-reference. |

### OK - verified correct

- the **9 map wrappers**, in full. `series/map.rs` confirms every recorded op-coverage range:
  `lane_i32` and `lane_iscalar_i32` cover all 7 codes; `lane_scalar_i32` returns `None` for
  `FLOOR_DIV | MOD`, so "0..=4" is right for `nf_map_fscalar_i32`; `lane_f32` /
  `lane_scalar_f32` cover add/sub/mul/floor_div/mod and `None` for div/pow; `lane_divpow_f32` /
  `lane_scalar_divpow_f32` cover div/pow only; `lane_f64` / `lane_scalar_f64` cover all 7. The
  `i32`-vs-`f64` scalar split per symbol also checks out.
- `nf_rowwise_kway_time_argmin_gather`: "`tPtrs`/`dPtrs` are u32 arrays OF OFFSETS, not lane data"
  is exactly right.

Full machine-readable audit: `census.json` -> `_meta.shipped_wrapper_audit`.
## 4. The census

Every row below is machine-readable in `census.json` (same directory). Columns:

`#` arg position &nbsp;|&nbsp; `name` Rust parameter name &nbsp;|&nbsp; `wasm` the value type the
engine actually sees &nbsp;|&nbsp; `dir` `in` / `out` / `scalar` &nbsp;|&nbsp; `len` the length
expression the body itself passes to `borrow()`/`borrow_mut()` &nbsp;|&nbsp; `role` what it is.

`len` is the decisive column. It is not a convention, it is the literal second argument of the
unchecked slice construction inside that symbol's own body.

### 4.1 `adjacency` - 2 symbols

#### `nf_adjacency_gather` &nbsp;*(7 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2136` (`#[no_mangle]`), origin: literal
- rust: `nf_adjacency_gather(indices: *const u32, e: usize, begins: *const u32, ends: *const u32, k: usize, out: *mut u32, total: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(indicesPtr i32, e i32, beginsPtr i32, endsPtr i32, k i32, outPtr i32, total i32) -> i32`
- what: flatten k half-open slices of indices into out, storage order inside each slice
- returns `i32` (number): 0 ok; -1 null; -2 bad range; -3 bad geometry (out too short)
- BigInt: no
- allocates: no
- result ownership: caller; total is a CAPACITY, not a checksum
- concurrency: reentrant, row-independent
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: begins[g] > ends[g], or ends[g] > e
    - `-3` = MALFORMED: out.len() < sum(ends-begins)
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `indices` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 1 | `e` | i32 | scalar | `-` | edge count; indices has e lanes |
| 2 | `begins` | i32 | in | `k` | input buffer of Uint32Array, k lane(s) |
| 3 | `ends` | i32 | in | `k` | input buffer of Uint32Array, k lane(s) |
| 4 | `k` | i32 | scalar | `-` | number of (begins, ends) slice pairs to flatten |
| 5 | `out` | i32 | out | `total` | output buffer of Uint32Array, total lane(s) (overwritten) |
| 6 | `total` | i32 | scalar | `-` | CAPACITY in lanes; a capacity, NOT a checksum |

#### `nf_adjacency_slice` &nbsp;*(8 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2084` (`#[no_mangle]`), origin: literal
- rust: `nf_adjacency_slice(indptr: *const u32, np: usize, indices: *const u32, e: usize, query: *const u32, k: usize, begins: *mut u32, ends: *mut u32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(indptrPtr i32, np i32, indicesPtr i32, e i32, queryPtr i32, k i32, beginsPtr i32, endsPtr i32) -> i32`
- what: CSR slice: indptr/indices/query -> begins/ends in query order
- returns `i32` (number): 0 ok; -1 null; -2 bad range; -3 bad geometry
- BigInt: no
- allocates: no
- result ownership: caller allocates k lanes for begins and ends
- concurrency: reentrant, row-independent
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: UINT32_MAX reserved lane, non-monotone indptr, out-of-range query, non-empty query on an empty graph
    - `-3` = MALFORMED: indptr[np-1] != e, lane outside [0,e], non-empty indices with empty indptr
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `indptr` | i32 | in | `np` | input buffer of Uint32Array, np lane(s) |
| 1 | `np` | i32 | scalar | `-` | indptr length in lanes; np == 0 is the empty-graph encoding |
| 2 | `indices` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 3 | `e` | i32 | scalar | `-` | edge count; indices has e lanes (content unread, reserved-lane checked) |
| 4 | `query` | i32 | in | `0` | input buffer of Uint32Array, 0 lane(s) |
| 5 | `k` | i32 | scalar | `-` | query count; query has k lanes, begins/ends have k lanes each |
| 6 | `begins` | i32 | out | `0` | output buffer of Uint32Array, 0 lane(s) (overwritten) |
| 7 | `ends` | i32 | out | `0` | output buffer of Uint32Array, 0 lane(s) (overwritten) |

### 4.2 `bounded-select` - 1 symbol

#### `nf_bounded_select_2i32` &nbsp;*(9 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2710` (`#[no_mangle]`), origin: literal
- rust: `nf_bounded_select_2i32(a: *const i32, b: *const i32, n: usize, offs: *const i32, g: usize, k: usize, out_a: *mut i32, out_b: *mut i32, out_idx: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, bPtr i32, n i32, offsPtr i32, g i32, k i32, outAPtr i32, outBPtr i32, outIdxPtr i32) -> i32`
- what: per-group bounded top-K over paired int32 keys; infeasible rows are exactly a == INT32_MAX
- returns `i32` (number): 0 ok; -1 null; -2 bad range (k == 0); -3 bad geometry
- BigInt: no
- allocates: no
- result ownership: caller allocates g*k lanes x3
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: k == 0
    - `-3` = MALFORMED: non-monotone offs, offs[g] != n, offs[0] != 0
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - outputs need g*k lanes of EACH of the three buffers; short groups pad with (INT32_MAX, INT32_MAX, -1)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `b` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `offs` | i32 | in | `g + 1` | input buffer of Int32Array, g + 1 lane(s) |
| 4 | `g` | i32 | scalar | `-` | group count; offs has g+1 lanes |
| 5 | `k` | i32 | scalar | `-` | runtime top-K bound, >= 1; the caller allocates g*k lanes in EACH of the three output buffers |
| 6 | `out_a` | i32 | out | `total` | output buffer of Int32Array, total lane(s) (overwritten) |
| 7 | `out_b` | i32 | out | `total` | output buffer of Int32Array, total lane(s) (overwritten) |
| 8 | `out_idx` | i32 | out | `total` | output buffer of Int32Array, total lane(s) (overwritten) |

### 4.3 `carry` - 2 symbols

#### `nf_carry_build_f64` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:491` (`#[no_mangle]`), origin: literal
- rust: `nf_carry_build_f64(counts_m: *const i64, sums_m: *const f64, m: usize, ukeys: *mut i64, counts: *mut i64, sums: *mut f64) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i64`
- js: `(countsMPtr i32, sumsMPtr i32, m i32, ukeysPtr i32, countsPtr i32, sumsPtr i32) -> bigint`
- what: float64 variant of nf_carry_build_i64
- returns `i64` (bigint): ng (>= 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0+` = ng
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `counts_m` | i32 | in | `m` | input buffer of BigInt64Array, m lane(s) |
| 1 | `sums_m` | i32 | in | `m` | input buffer of Float64Array, m lane(s) |
| 2 | `m` | i32 | scalar | `-` | dense lane count; ukeys/counts/sums each need m lanes |
| 3 | `ukeys` | i32 | out | `m` | output buffer of BigInt64Array, m lane(s) (overwritten) |
| 4 | `counts` | i32 | out | `m` | output buffer of BigInt64Array, m lane(s) (overwritten) |
| 5 | `sums` | i32 | out | `m` | output buffer of Float64Array, m lane(s) (overwritten) |

#### `nf_carry_build_i64` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:461` (`#[no_mangle]`), origin: literal
- rust: `nf_carry_build_i64(counts_m: *const i64, sums_m: *const i64, m: usize, ukeys: *mut i64, counts: *mut i64, sums: *mut i64) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i64`
- js: `(countsMPtr i32, sumsMPtr i32, m i32, ukeysPtr i32, countsPtr i32, sumsPtr i32) -> bigint`
- what: compact (ukeys, counts, sums) from M-lane dense int64 state, ascending code order
- returns `i64` (bigint): ng (>= 0; 0 for m == 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller allocates m lanes
- concurrency: reentrant
- return codes:
    - `0+` = ng
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `counts_m` | i32 | in | `m` | input buffer of BigInt64Array, m lane(s) |
| 1 | `sums_m` | i32 | in | `m` | input buffer of BigInt64Array, m lane(s) |
| 2 | `m` | i32 | scalar | `-` | dense lane count; ukeys/counts/sums each need m lanes |
| 3 | `ukeys` | i32 | out | `m` | output buffer of BigInt64Array, m lane(s) (overwritten) |
| 4 | `counts` | i32 | out | `m` | output buffer of BigInt64Array, m lane(s) (overwritten) |
| 5 | `sums` | i32 | out | `m` | output buffer of BigInt64Array, m lane(s) (overwritten) |

### 4.4 `cost` - 2 symbols

#### `nf_cost_intern` &nbsp;*(6 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2218` (`#[no_mangle]`), origin: literal
- rust: `nf_cost_intern(vecs: *const u32, n: usize, width: usize, ids: *mut u32, uniq: *mut u32, total: usize) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i64`
- js: `(vecsPtr i32, n i32, width i32, idsPtr i32, uniqPtr i32, total i32) -> bigint`
- what: stateless row interning of u32 cost vectors -> first-appearance ids + compact table
- returns `i64` (bigint): ng (>= 0) = number of distinct rows; -2 null or width == 0 with n > 0; -3 short uniq buffer
- BigInt: **required** - RETURN is BigInt; a Number is rejected
- allocates: YES - std HashMap<Vec<u32>, u32> with_capacity(n) plus one Vec per distinct row (cost.rs:250-262)
- result ownership: kernel frees its own temporaries; caller owns ids[0..n] and uniq[0..total]
- concurrency: reentrant; ids order is first-appearance and is deterministic (hash is lookup-only)
- return codes:
    - `0+` = ng, a COUNT not a status
    - `-2` = NULL_STATE, or BAD_RANGE (width == 0)
    - `-3` = MALFORMED: total < n*width
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `vecs` | i32 | in | `need` | input buffer of Uint32Array, need lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `width` | i32 | scalar | `-` | columns per row |
| 3 | `ids` | i32 | out | `n` | output buffer of Uint32Array, n lane(s) (overwritten) |
| 4 | `uniq` | i32 | out | `total` | output buffer of Uint32Array, total lane(s) (overwritten) |
| 5 | `total` | i32 | scalar | `-` | CAPACITY in lanes for `uniq`; must be >= n*width or -3 (MALFORMED) |

#### `nf_cost_travel_batch` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2182` (`#[no_mangle]`), origin: literal
- rust: `nf_cost_travel_batch(dist: *const u32, speed: *const u32, k: *const u16, n: usize, out: *mut u32) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(distPtr i32, speedPtr i32, kPtr i32, n i32, outPtr i32) -> i32`
- what: out[i] = (dist[i]*k[i] + speed[i]/2) / speed[i] with an INF guard
- returns `i32` (number): 0 ok; -1 null; -2 bad range (a k[i] == 0 lane)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant, row-independent
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: K == 0 lane (lanes before it may be written)
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `dist` | i32 | in | `n` | input buffer of Uint32Array, n lane(s) |
| 1 | `speed` | i32 | in | `n` | input buffer of Uint32Array, n lane(s) |
| 2 | `k` | i32 | in | `n` | input buffer of Uint16Array, n lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `out` | i32 | out | `n` | output buffer of Uint32Array, n lane(s) (overwritten) |

### 4.5 `cumsum` - 3 symbols

#### `nf_cumsum_f32` &nbsp;*(3 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:808` (`#[no_mangle]`), origin: macro `cumsum_ffi`
- rust: `nf_cumsum_f32(src: *const f32, n: usize, out: *mut f32) -> i32`
- wasm: `(i32, i32, i32) -> i32`
- js: `(srcPtr i32, n i32, outPtr i32) -> i32   // lanes: Float32Array`
- what: inclusive prefix sum over f32 lanes (out[i] = sum(src[0..=i])); validity travels host-side
- returns `i32` (number): 0 ok; -1 null (n == 0 returns 0 without dereferencing)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - i32 wraps mod 2^32 by contract (release build: overflow-checks off), never traps

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `out` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |

#### `nf_cumsum_f64` &nbsp;*(3 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:814` (`#[no_mangle]`), origin: macro `cumsum_ffi`
- rust: `nf_cumsum_f64(src: *const f64, n: usize, out: *mut f64) -> i32`
- wasm: `(i32, i32, i32) -> i32`
- js: `(srcPtr i32, n i32, outPtr i32) -> i32   // lanes: Float64Array`
- what: inclusive prefix sum over f64 lanes (out[i] = sum(src[0..=i])); validity travels host-side
- returns `i32` (number): 0 ok; -1 null (n == 0 returns 0 without dereferencing)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - i32 wraps mod 2^32 by contract (release build: overflow-checks off), never traps

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |

#### `nf_cumsum_i32` &nbsp;*(3 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:802` (`#[no_mangle]`), origin: macro `cumsum_ffi`
- rust: `nf_cumsum_i32(src: *const i32, n: usize, out: *mut i32) -> i32`
- wasm: `(i32, i32, i32) -> i32`
- js: `(srcPtr i32, n i32, outPtr i32) -> i32   // lanes: Int32Array`
- what: inclusive prefix sum over i32 lanes (out[i] = sum(src[0..=i])); validity travels host-side
- returns `i32` (number): 0 ok; -1 null (n == 0 returns 0 without dereferencing)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - i32 wraps mod 2^32 by contract (release build: overflow-checks off), never traps

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

### 4.6 `ghash` - 6 symbols

#### `nf_ghash_count_i64` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2298` (`#[no_mangle]`), origin: literal
- rust: `nf_ghash_count_i64(fp: *const i64, n: usize, tc: *mut i64, t: usize, p: usize, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(fpPtr i32, n i32, tcPtr i32, t i32, p i32, nthreads i32) -> i32`
- what: MT per-thread histogram: partition ids -> t*p int64 counts
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry
- BigInt: no
- allocates: no (spawns host threads natively)
- result ownership: caller must WARM-zero tc[0..t*p]
- concurrency: MT internally; caller-owned state is tiled by [t][p] so shards are disjoint
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the body calls std::thread::scope + scope.spawn, which is unsupported on wasm32-unknown-unknown, and panic=abort turns the panic into an unreachable trap. The same symbol works natively.

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `fp` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `tc` | i32 | out | `t * p` | output buffer of BigInt64Array, t * p lane(s) (overwritten) |
| 3 | `t` | i32 | scalar | `-` | fan-out rows; tc has t*p lanes and must be WARM-zeroed by the caller |
| 4 | `p` | i32 | scalar | `-` | partition count, power of two |
| 5 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_ghash_fp_i64` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2265` (`#[no_mangle]`), origin: literal
- rust: `nf_ghash_fp_i64(k: *const i64, v: *const i64, n: usize, fp: *mut i64, pmask: i64, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i64, i32) -> i32`
- js: `(kPtr i32, vPtr i32, n i32, fpPtr i32, pmask bigint, nthreads i32) -> i32`
- what: MT pair fingerprint (k,v) -> partition id (h & pmask)
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry
- BigInt: **required** - pmask is i64 -> a BigInt argument
- allocates: no (spawns host threads natively)
- result ownership: caller
- concurrency: MT internally; caller-owned state is tiled by [t][p] so shards are disjoint
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: p not pow2 / p == 0 / p > 256
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the body calls std::thread::scope + scope.spawn, which is unsupported on wasm32-unknown-unknown, and panic=abort turns the panic into an unreachable trap. The same symbol works natively.

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `k` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `v` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `fp` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 4 | `pmask` | i64 | scalar | `-` | P-1 for a power-of-two partition count P in [16,256] |
| 5 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_ghash_occ_count` &nbsp;*(5 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2418` (`#[no_mangle]`), origin: literal
- rust: `nf_ghash_occ_count(used: *const u8, toff: *const i64, p: usize, cnt: *mut i64, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(usedPtr i32, toffPtr i32, p i32, cntPtr i32, nthreads i32) -> i32`
- what: MT occupancy count over slot tables -> per-partition counts
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry
- BigInt: no
- allocates: no (spawns host threads natively)
- result ownership: caller
- concurrency: MT internally; caller-owned state is tiled by [t][p] so shards are disjoint
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the body calls std::thread::scope + scope.spawn, which is unsupported on wasm32-unknown-unknown, and panic=abort turns the panic into an unreachable trap. The same symbol works natively.
    - m is DERIVED inside the body: m = toff[p]

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `used` | i32 | in | `m` | input buffer of Uint8Array, m lane(s) |
| 1 | `toff` | i32 | in | `p + 1` | input buffer of BigInt64Array, p + 1 lane(s) |
| 2 | `p` | i32 | scalar | `-` | partition count, power of two |
| 3 | `cnt` | i32 | out | `p` | output buffer of BigInt64Array, p lane(s) (overwritten) |
| 4 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_ghash_occ_fill` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2446` (`#[no_mangle]`), origin: literal
- rust: `nf_ghash_occ_fill(tk: *const i64, used: *const u8, toff: *const i64, ostart: *const i64, ok: *mut i64, p: usize, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(tkPtr i32, usedPtr i32, toffPtr i32, ostartPtr i32, okPtr i32, p i32, nthreads i32) -> i32`
- what: MT occupancy fill: occupied keys in prefix order
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry
- BigInt: no
- allocates: no (spawns host threads natively)
- result ownership: caller; ranges are disjoint by prefix construction
- concurrency: MT internally; caller-owned state is tiled by [t][p] so shards are disjoint
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the body calls std::thread::scope + scope.spawn, which is unsupported on wasm32-unknown-unknown, and panic=abort turns the panic into an unreachable trap. The same symbol works natively.
    - m = toff[p] and nocc = ostart[p] are DERIVED inside the body

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `tk` | i32 | in | `m` | input buffer of BigInt64Array, m lane(s) |
| 1 | `used` | i32 | in | `m` | input buffer of Uint8Array, m lane(s) |
| 2 | `toff` | i32 | in | `p + 1` | input buffer of BigInt64Array, p + 1 lane(s) |
| 3 | `ostart` | i32 | in | `p + 1` | input buffer of BigInt64Array, p + 1 lane(s) |
| 4 | `ok` | i32 | out | `nocc` | output buffer of BigInt64Array, nocc lane(s) (overwritten) |
| 5 | `p` | i32 | scalar | `-` | partition count, power of two |
| 6 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_ghash_pins_i64` &nbsp;*(10 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2372` (`#[no_mangle]`), origin: literal
- rust: `nf_ghash_pins_i64(pk: *const i64, pv: *const i64, starts: *const i64, toff: *const i64, tm: *const i64, tk: *mut i64, tv: *mut i64, used: *mut u8, p: usize, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(pkPtr i32, pvPtr i32, startsPtr i32, toffPtr i32, tmPtr i32, tkPtr i32, tvPtr i32, usedPtr i32, p i32, nthreads i32) -> i32`
- what: MT pair-hash insert: partitioned rows -> slot tables (tk/tv/used)
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry
- BigInt: no
- allocates: no (spawns host threads natively)
- result ownership: caller; used[0..m] is WARM-zeroed then overwritten
- concurrency: MT internally; caller-owned state is tiled by [t][p] so shards are disjoint
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the body calls std::thread::scope + scope.spawn, which is unsupported on wasm32-unknown-unknown, and panic=abort turns the panic into an unreachable trap. The same symbol works natively.
    - n and m are DERIVED INSIDE the body: n = starts[p], m = toff[p]. There is no n/m parameter - misreading this as (pk, n) is the single easiest mistake here (HAZARD H6).

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `pk` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `pv` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 2 | `starts` | i32 | in | `p + 1` | input buffer of BigInt64Array, p + 1 lane(s) |
| 3 | `toff` | i32 | in | `p + 1` | input buffer of BigInt64Array, p + 1 lane(s) |
| 4 | `tm` | i32 | in | `p` | input buffer of BigInt64Array, p lane(s) |
| 5 | `tk` | i32 | out | `m` | output buffer of BigInt64Array, m lane(s) (overwritten) |
| 6 | `tv` | i32 | out | `m` | output buffer of BigInt64Array, m lane(s) (overwritten) |
| 7 | `used` | i32 | out | `m` | output buffer of Uint8Array, m lane(s) (overwritten) |
| 8 | `p` | i32 | scalar | `-` | partition count, power of two |
| 9 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_ghash_scatter_i64` &nbsp;*(10 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2332` (`#[no_mangle]`), origin: literal
- rust: `nf_ghash_scatter_i64(k: *const i64, v: *const i64, fp: *const i64, n: usize, tpos: *const i64, t: usize, p: usize, pk: *mut i64, pv: *mut i64, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(kPtr i32, vPtr i32, fpPtr i32, n i32, tposPtr i32, t i32, p i32, pkPtr i32, pvPtr i32, nthreads i32) -> i32`
- what: MT reservation scatter: rows -> partitioned (pk, pv) lanes
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry
- BigInt: no
- allocates: no (spawns host threads natively)
- result ownership: caller supplies the tpos prefix
- concurrency: MT internally; caller-owned state is tiled by [t][p] so shards are disjoint
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the body calls std::thread::scope + scope.spawn, which is unsupported on wasm32-unknown-unknown, and panic=abort turns the panic into an unreachable trap. The same symbol works natively.

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `k` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `v` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 2 | `fp` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `tpos` | i32 | in | `t * p` | input buffer of BigInt64Array, t * p lane(s) |
| 5 | `t` | i32 | scalar | `-` | fan-out rows; tpos has t*p lanes |
| 6 | `p` | i32 | scalar | `-` | partition count, power of two |
| 7 | `pk` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 8 | `pv` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 9 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

### 4.7 `groupby-dense` - 5 symbols

#### `nf_group_count_only` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:183` (`#[no_mangle]`), origin: literal
- rust: `nf_group_count_only(keys: *const i32, n: usize, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, n i32, countsPtr i32, g i32) -> i32`
- what: dense keys-only occupancy pass (i64 counts)
- returns `i32` (number): 0 ok; -1 null; -2 out-of-range key
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: key outside [0,g)
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 3 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

#### `nf_group_mixed_sum_only` &nbsp;*(9 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:149` (`#[no_mangle]`), origin: literal
- rust: `nf_group_mixed_sum_only(keys: *const i32, i32_vals: *const i32, f64_vals: *const f64, n: usize, n_i32: usize, n_f64: usize, i32_sums: *mut i32, f64_sums: *mut f64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, i32ValsPtr i32, f64ValsPtr i32, n i32, nI32 i32, nF64 i32, i32SumsPtr i32, f64SumsPtr i32, g i32) -> i32`
- what: dense sum over n_i32 int32 + n_f64 float64 SoA columns, no counts lane
- returns `i32` (number): 0 ok; -1 null; -2 out-of-range key; -4 int32 overflow
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT: null pointer, or a value ptr null while its column count > 0
    - `-2` = BAD_RANGE: key outside [0,g)
    - `-4` = OVERFLOW: checked_add failed, never wraps
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `i32_vals` | i32 | in | `n_i32 * n` | input buffer of Int32Array, n_i32 * n lane(s) |
| 2 | `f64_vals` | i32 | in | `n_f64 * n` | input buffer of Float64Array, n_f64 * n lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `n_i32` | i32 | scalar | `-` | scalar usize |
| 5 | `n_f64` | i32 | scalar | `-` | scalar usize |
| 6 | `i32_sums` | i32 | out | `n_i32 * g` | output buffer of Int32Array, n_i32 * g lane(s) (overwritten) |
| 7 | `f64_sums` | i32 | out | `n_f64 * g` | output buffer of Float64Array, n_f64 * g lane(s) (overwritten) |
| 8 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

#### `nf_group_multi_sum_count` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:116` (`#[no_mangle]`), origin: literal
- rust: `nf_group_multi_sum_count(keys: *const i32, values: *const f64, n: usize, ncols: usize, sums: *mut f64, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, valuesPtr i32, n i32, ncols i32, sumsPtr i32, countsPtr i32, g i32) -> i32`
- what: SoA multi-column dense sum + shared count
- returns `i32` (number): 0 ok; -1 null; -2 out-of-range key
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT: null pointer
    - `-2` = BAD_RANGE: key outside [0,g)
- can trap:
    - keys[i] outside [0,g) returns -2 (fused_scatter_soa range-checks); it does NOT trap

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `values` | i32 | in | `ncols * n` | input buffer of Float64Array, ncols * n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `ncols` | i32 | scalar | `-` | SoA column count |
| 4 | `sums` | i32 | out | `ncols * g` | output buffer of Float64Array, ncols * g lane(s) (overwritten) |
| 5 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 6 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

#### `nf_group_owner_2i32_1f64` &nbsp;*(12 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:211` (`#[no_mangle]`), origin: literal
- rust: `nf_group_owner_2i32_1f64(keys: *const i32, a: *const i32, b: *const i32, c: *const f64, n: usize, lo: usize, hi: usize, s1: *mut i32, s2: *mut i32, s3: *mut f64, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, aPtr i32, bPtr i32, cPtr i32, n i32, lo i32, hi i32, s1Ptr i32, s2Ptr i32, s3Ptr i32, countsPtr i32, g i32) -> i32`
- what: owner-shard dense sum of 2xi32 + 1xf64 with counts; writes only lanes [lo,hi)
- returns `i32` (number): 0 ok; -1 null OR lo>hi OR hi>g; -2 out-of-range key; -4 overflow
- BigInt: no
- allocates: no
- result ownership: caller; only lanes [lo,hi) are written
- concurrency: DESIGNED FOR CONCURRENCY: disjoint [lo,hi) group ranges over shared state, no atomics, no merge. Rows are skipped, not partitioned.
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT: null pointer, OR lo > hi, OR hi > g (see HAZARD H7)
    - `-2` = BAD_RANGE: key outside [0,g)
    - `-4` = OVERFLOW: int32 checked_add failed
- can trap:
    - lo > hi or hi > g is reported as -1, the SAME code as a null pointer (HAZARD H7)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `a` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `b` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 3 | `c` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 4 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 5 | `lo` | i32 | scalar | `-` | first group written (inclusive); lo > hi or hi > g returns -1 |
| 6 | `hi` | i32 | scalar | `-` | group limit (exclusive); hi > g returns -1 |
| 7 | `s1` | i32 | out | `g` | output buffer of Int32Array, g lane(s) (overwritten) |
| 8 | `s2` | i32 | out | `g` | output buffer of Int32Array, g lane(s) (overwritten) |
| 9 | `s3` | i32 | out | `g` | output buffer of Float64Array, g lane(s) (overwritten) |
| 10 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 11 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

#### `nf_group_sum_count` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:84` (`#[no_mangle]`), origin: literal
- rust: `nf_group_sum_count(keys: *const i32, values: *const f64, n: usize, sums: *mut f64, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, valuesPtr i32, n i32, sumsPtr i32, countsPtr i32, g i32) -> i32`
- what: dense fused group-by: f64 sum + i64 count per dense int32 code
- returns `i32` (number): 0 ok; -1 NULL_OR_ABORT (any null pointer)
- BigInt: no
- allocates: no
- result ownership: caller owns every buffer
- concurrency: reentrant: no statics, no threads; concurrent calls on disjoint buffers are safe
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT: null pointer
- can trap:
    - keys[i] >= g or keys[i] < 0 -> safe-Rust index panic in dense_scatter -> WebAssembly RuntimeError unreachable (MEASURED); native = abort

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `values` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `sums` | i32 | out | `g` | output buffer of Float64Array, g lane(s) (overwritten) |
| 4 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 5 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

### 4.8 `groupby-dense research` - 3 symbols

#### `nf_group_variant_f64` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:538` (`#[no_mangle]`), origin: literal
- rust: `nf_group_variant_f64(keys: *const i32, values: *const f64, n: usize, variant: u32, sums: *mut f64, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, valuesPtr i32, n i32, variant i32, sumsPtr i32, countsPtr i32, g i32) -> i32`
- what: variant select 0=dense 1=checked 2=sorted-dense over f64
- returns `i32` (number): 0 ok; -1 null; -2 out-of-range; -3 unknown variant; -5 unsorted
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: out-of-range key
    - `-3` = BAD_VARIANT: unknown id, outputs untouched
    - `-5` = UNSORTED: V_SORTED got unsorted keys
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `values` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `variant` | i32 | scalar | `-` | 0 dense, 1 checked, 2 sorted-dense |
| 4 | `sums` | i32 | out | `g` | output buffer of Float64Array, g lane(s) (overwritten) |
| 5 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 6 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

#### `nf_group_variant_i32` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:595` (`#[no_mangle]`), origin: literal
- rust: `nf_group_variant_i32(keys: *const i32, ticks: *const i32, n: usize, variant: u32, sums: *mut i32, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, ticksPtr i32, n i32, variant i32, sumsPtr i32, countsPtr i32, g i32) -> i32`
- what: variant select over int32 ticks (overflow-demo lanes)
- returns `i32` (number): 0 ok; -1 null; -2; -3 (variant 1 and every unknown are unsupported here); -5 unsorted
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE
    - `-3` = BAD_VARIANT (also for variant==1)
    - `-5` = UNSORTED
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `ticks` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `variant` | i32 | scalar | `-` | 0 dense, 1 checked, 2 sorted-dense |
| 4 | `sums` | i32 | out | `g` | output buffer of Int32Array, g lane(s) (overwritten) |
| 5 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 6 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

#### `nf_group_variant_i64` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:568` (`#[no_mangle]`), origin: literal
- rust: `nf_group_variant_i64(keys: *const i32, ticks: *const i64, n: usize, variant: u32, sums: *mut i64, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, ticksPtr i32, n i32, variant i32, sumsPtr i32, countsPtr i32, g i32) -> i32`
- what: variant select over int64 physical ticks
- returns `i32` (number): 0 ok; -1 null; -2; -3; -4 overflow; -5 unsorted
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE
    - `-3` = BAD_VARIANT
    - `-4` = OVERFLOW
    - `-5` = UNSORTED
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `ticks` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `variant` | i32 | scalar | `-` | 0 dense, 1 checked, 2 sorted-dense |
| 4 | `sums` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 5 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 6 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

### 4.9 `join` - 7 symbols

#### `nf_join_build` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1079` (`#[no_mangle]`), origin: literal
- rust: `nf_join_build(right_keys: *const i32, s: usize, table_keys: *mut i32, table_pos: *mut i32, table_occ: *mut u8, cap: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(rightKeysPtr i32, s i32, tableKeysPtr i32, tablePosPtr i32, tableOccPtr i32, cap i32) -> i32`
- what: build the right hash table from unique int32 keys (H2O contract)
- returns `i32` (number): 0 ok; -1 null; -2 duplicate right key (lanes before it written); -3 bad geometry
- BigInt: no
- allocates: no
- result ownership: caller allocates cap lanes x3
- concurrency: single-threaded (no scope/spawn) - the only join symbol that works on wasm32
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = duplicate right key, caller must rebuild
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `right_keys` | i32 | in | `s` | input buffer of Int32Array, s lane(s) |
| 1 | `s` | i32 | scalar | `-` | right-row count (join) or shift period count (shift) - read the symbol row |
| 2 | `table_keys` | i32 | out | `cap` | output buffer of Int32Array, cap lane(s) (overwritten) |
| 3 | `table_pos` | i32 | out | `cap` | output buffer of Int32Array, cap lane(s) (overwritten) |
| 4 | `table_occ` | i32 | out | `cap` | output buffer of Uint8Array, cap lane(s) (overwritten) |
| 5 | `cap` | i32 | scalar | `-` | hash table capacity, power of two |

#### `nf_join_build_i64` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2759` (`#[no_mangle]`), origin: literal
- rust: `nf_join_build_i64(right_keys: *const i64, s: usize, table_keys: *mut i64, table_pos: *mut i32, table_occ: *mut u8, cap: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(rightKeysPtr i32, s i32, tableKeysPtr i32, tablePosPtr i32, tableOccPtr i32, cap i32) -> i32`
- what: build the hash table over int64 right keys (full range incl MIN/MAX)
- returns `i32` (number): 0 ok; -1 null; -2 duplicate right key; -3 bad geometry
- BigInt: no
- allocates: no
- result ownership: caller allocates cap lanes; tablePos is i32 here, tableKeys i64
- concurrency: single-threaded build
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = duplicate right key
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `right_keys` | i32 | in | `s` | input buffer of BigInt64Array, s lane(s) |
| 1 | `s` | i32 | scalar | `-` | right-row count (join) or shift period count (shift) - read the symbol row |
| 2 | `table_keys` | i32 | out | `cap` | output buffer of BigInt64Array, cap lane(s) (overwritten) |
| 3 | `table_pos` | i32 | out | `cap` | output buffer of Int32Array, cap lane(s) (overwritten) |
| 4 | `table_occ` | i32 | out | `cap` | output buffer of Uint8Array, cap lane(s) (overwritten) |
| 5 | `cap` | i32 | scalar | `-` | hash table capacity, power of two |

#### `nf_join_fused_inner_i32` &nbsp;*(13 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1260` (`#[no_mangle]`), origin: literal
- rust: `nf_join_fused_inner_i32(table_keys: *const i32, table_pos: *const i32, table_occ: *const u8, cap: usize, left_keys: *const i32, left_v1: *const i32, payload: *const i32, s: usize, n: usize, out_keys: *mut i32, out_v1: *mut i32, out_v2: *mut i32, nthreads: usize) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i64`
- js: `(tableKeysPtr i32, tablePosPtr i32, tableOccPtr i32, cap i32, leftKeysPtr i32, leftV1Ptr i32, payloadPtr i32, s i32, n i32, outKeysPtr i32, outV1Ptr i32, outV2Ptr i32, nthreads i32) -> bigint`
- what: fused INNER join: two-pass count + re-probe fill -> compact keys/v1/v2
- returns `i64` (bigint): m (>= 0) = compact hits; -2 NULL_STATE; -3 bad geometry
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller
- concurrency: MT internally
- return codes:
    - `0+` = m = compact hits
    - `-2` = NULL_STATE
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the MT path calls std::thread::scope/scope.spawn, unsupported on wasm32-unknown-unknown, and panic=abort

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `table_keys` | i32 | in | `cap` | input buffer of Int32Array, cap lane(s) |
| 1 | `table_pos` | i32 | in | `cap` | input buffer of Int32Array, cap lane(s) |
| 2 | `table_occ` | i32 | in | `cap` | input buffer of Uint8Array, cap lane(s) |
| 3 | `cap` | i32 | scalar | `-` | hash table capacity, power of two |
| 4 | `left_keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 5 | `left_v1` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 6 | `payload` | i32 | in | `s` | input buffer of Int32Array, s lane(s) |
| 7 | `s` | i32 | scalar | `-` | right-row count (join) or shift period count (shift) - read the symbol row |
| 8 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 9 | `out_keys` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 10 | `out_v1` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 11 | `out_v2` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 12 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_join_fused_left_i32` &nbsp;*(14 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1195` (`#[no_mangle]`), origin: literal
- rust: `nf_join_fused_left_i32(table_keys: *const i32, table_pos: *const i32, table_occ: *const u8, cap: usize, left_keys: *const i32, left_v1: *const i32, payload: *const i32, s: usize, n: usize, out_keys: *mut i32, out_v1: *mut i32, out_v2: *mut i32, valid_out: *mut u8, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(tableKeysPtr i32, tablePosPtr i32, tableOccPtr i32, cap i32, leftKeysPtr i32, leftV1Ptr i32, payloadPtr i32, s i32, n i32, outKeysPtr i32, outV1Ptr i32, outV2Ptr i32, validOutPtr i32, nthreads i32) -> i32`
- what: fused LEFT join: single-pass probe -> full-size keys/v1/v2 + u8 valid sidecar
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry (cap not pow2; s == 0 with n > 0)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: MT internally
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the MT path calls std::thread::scope/scope.spawn, unsupported on wasm32-unknown-unknown, and panic=abort

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `table_keys` | i32 | in | `cap` | input buffer of Int32Array, cap lane(s) |
| 1 | `table_pos` | i32 | in | `cap` | input buffer of Int32Array, cap lane(s) |
| 2 | `table_occ` | i32 | in | `cap` | input buffer of Uint8Array, cap lane(s) |
| 3 | `cap` | i32 | scalar | `-` | hash table capacity, power of two |
| 4 | `left_keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 5 | `left_v1` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 6 | `payload` | i32 | in | `s` | input buffer of Int32Array, s lane(s) |
| 7 | `s` | i32 | scalar | `-` | right-row count (join) or shift period count (shift) - read the symbol row |
| 8 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 9 | `out_keys` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 10 | `out_v1` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 11 | `out_v2` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 12 | `valid_out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |
| 13 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_join_gather_i32` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1159` (`#[no_mangle]`), origin: literal
- rust: `nf_join_gather_i32(payload: *const i32, s: usize, pos: *const i64, m: usize, out: *mut i32, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(payloadPtr i32, s i32, posPtr i32, m i32, outPtr i32, nthreads i32) -> i32`
- what: MT positional gather of one int32 payload column
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry (first OOB position; s == 0 with m > 0)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: MT internally
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry, partial write
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the MT path calls std::thread::scope/scope.spawn, unsupported on wasm32-unknown-unknown, and panic=abort

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `payload` | i32 | in | `s` | input buffer of Int32Array, s lane(s) |
| 1 | `s` | i32 | scalar | `-` | right-row count, i.e. len(payload); s == 0 with m > 0 is -3 |
| 2 | `pos` | i32 | in | `m` | input buffer of BigInt64Array, m lane(s) |
| 3 | `m` | i32 | scalar | `-` | number of positions to gather |
| 4 | `out` | i32 | out | `m` | output buffer of Int32Array, m lane(s) (overwritten) |
| 5 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_join_probe` &nbsp;*(9 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1113` (`#[no_mangle]`), origin: literal
- rust: `nf_join_probe(table_keys: *const i32, table_pos: *const i32, table_occ: *const u8, cap: usize, left_keys: *const i32, n: usize, pos_out: *mut i64, hit_out: *mut u8, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(tableKeysPtr i32, tablePosPtr i32, tableOccPtr i32, cap i32, leftKeysPtr i32, n i32, posOutPtr i32, hitOutPtr i32, nthreads i32) -> i32`
- what: MT hash probe of int32 left keys -> int64 match position + u8 hit flags
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry (cap not pow2)
- BigInt: no
- allocates: no (spawns host threads on native)
- result ownership: caller
- concurrency: MT internally; read-only table shared across scoped threads, disjoint output shards
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the MT path calls std::thread::scope/scope.spawn, unsupported on wasm32-unknown-unknown, and panic=abort

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `table_keys` | i32 | in | `cap` | input buffer of Int32Array, cap lane(s) |
| 1 | `table_pos` | i32 | in | `cap` | input buffer of Int32Array, cap lane(s) |
| 2 | `table_occ` | i32 | in | `cap` | input buffer of Uint8Array, cap lane(s) |
| 3 | `cap` | i32 | scalar | `-` | hash table capacity, power of two |
| 4 | `left_keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 5 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 6 | `pos_out` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 7 | `hit_out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |
| 8 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_join_probe_i64` &nbsp;*(9 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2794` (`#[no_mangle]`), origin: literal
- rust: `nf_join_probe_i64(table_keys: *const i64, table_pos: *const i32, table_occ: *const u8, cap: usize, left_keys: *const i64, n: usize, pos_out: *mut i64, hit_out: *mut u8, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(tableKeysPtr i32, tablePosPtr i32, tableOccPtr i32, cap i32, leftKeysPtr i32, n i32, posOutPtr i32, hitOutPtr i32, nthreads i32) -> i32`
- what: MT hash probe of int64 left keys against a built int64 table
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry (cap not pow2)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: MT internally
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: bad geometry
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - ALWAYS TRAPS on wasm32 (MEASURED): the MT path calls std::thread::scope/scope.spawn, unsupported on wasm32-unknown-unknown, and panic=abort

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `table_keys` | i32 | in | `cap` | input buffer of BigInt64Array, cap lane(s) |
| 1 | `table_pos` | i32 | in | `cap` | input buffer of Int32Array, cap lane(s) |
| 2 | `table_occ` | i32 | in | `cap` | input buffer of Uint8Array, cap lane(s) |
| 3 | `cap` | i32 | scalar | `-` | hash table capacity, power of two |
| 4 | `left_keys` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 5 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 6 | `pos_out` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 7 | `hit_out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |
| 8 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

### 4.10 `map` - 9 symbols

#### `nf_map_f32` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:955` (`#[no_mangle]`), origin: macro `map_ffi`
- rust: `nf_map_f32(a: *const f32, b: *const f32, n: usize, op: u32, out: *mut f32) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, bPtr i32, n i32, op i32, outPtr i32) -> i32   // lanes: Float32Array`
- what: f32 lanes, array-array; out is f32 (5 ops, no div/pow)
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `b` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |

#### `nf_map_f32_divpow` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:971` (`#[no_mangle]`), origin: macro `map_ffi`
- rust: `nf_map_f32_divpow(a: *const f32, b: *const f32, n: usize, op: u32, out: *mut f64) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, bPtr i32, n i32, op i32, outPtr i32) -> i32   // lanes: Float64Array`
- what: f32 div/pow with f64 OUTPUT (4 bytes/lane in, 8 out)
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `b` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |

#### `nf_map_f64` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:987` (`#[no_mangle]`), origin: macro `map_ffi`
- rust: `nf_map_f64(a: *const f64, b: *const f64, n: usize, op: u32, out: *mut f64) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, bPtr i32, n i32, op i32, outPtr i32) -> i32   // lanes: Float64Array`
- what: f64 lanes, array-array, all 7 ops
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 1 | `b` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |

#### `nf_map_fscalar_i32` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:947` (`#[no_mangle]`), origin: macro `map_scalar_ffi`
- rust: `nf_map_fscalar_i32(a: *const i32, n: usize, s: f64, op: u32, out: *mut i32) -> i32`
- wasm: `(i32, i32, f64, i32, i32) -> i32`
- js: `(aPtr i32, n i32, s f64, op i32, outPtr i32) -> i32   // lanes: Int32Array`
- what: i32 lanes, FLOAT scalar: s is f64; floor_div/mod rejected -> -2
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count |
| 2 | `s` | f64 | scalar | `-` | the scalar OPERAND, an f64 (never pre-truncated) |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_map_i32` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:931` (`#[no_mangle]`), origin: macro `map_ffi`
- rust: `nf_map_i32(a: *const i32, b: *const i32, n: usize, op: u32, out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, bPtr i32, n i32, op i32, outPtr i32) -> i32   // lanes: Int32Array`
- what: i32 lanes, array-array, all 7 ops
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `b` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_map_scalar_f32` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:963` (`#[no_mangle]`), origin: macro `map_scalar_ffi`
- rust: `nf_map_scalar_f32(a: *const f32, n: usize, s: f64, op: u32, out: *mut f32) -> i32`
- wasm: `(i32, i32, f64, i32, i32) -> i32`
- js: `(aPtr i32, n i32, s f64, op i32, outPtr i32) -> i32   // lanes: Float32Array`
- what: f32 lanes, f64 scalar demoted to f32 first
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count |
| 2 | `s` | f64 | scalar | `-` | the scalar OPERAND, an f64, demoted to the f32 lane |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |

#### `nf_map_scalar_f32_divpow` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:979` (`#[no_mangle]`), origin: macro `map_scalar_ffi`
- rust: `nf_map_scalar_f32_divpow(a: *const f32, n: usize, s: f64, op: u32, out: *mut f64) -> i32`
- wasm: `(i32, i32, f64, i32, i32) -> i32`
- js: `(aPtr i32, n i32, s f64, op i32, outPtr i32) -> i32   // lanes: Float64Array`
- what: f32 scalar div/pow with f64 OUTPUT
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count |
| 2 | `s` | f64 | scalar | `-` | the scalar OPERAND, an f64 |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |

#### `nf_map_scalar_f64` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:995` (`#[no_mangle]`), origin: macro `map_scalar_ffi`
- rust: `nf_map_scalar_f64(a: *const f64, n: usize, s: f64, op: u32, out: *mut f64) -> i32`
- wasm: `(i32, i32, f64, i32, i32) -> i32`
- js: `(aPtr i32, n i32, s f64, op i32, outPtr i32) -> i32   // lanes: Float64Array`
- what: f64 lanes, f64 scalar, all 7 ops
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count |
| 2 | `s` | f64 | scalar | `-` | the scalar OPERAND, an f64 |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |

#### `nf_map_scalar_i32` &nbsp;*(5 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:939` (`#[no_mangle]`), origin: macro `map_scalar_ffi`
- rust: `nf_map_scalar_i32(a: *const i32, n: usize, s: i32, op: u32, out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, n i32, s i32, op i32, outPtr i32) -> i32   // lanes: Int32Array`
- what: i32 lanes, INTEGER scalar: s is i32, NOT f64
- returns `i32` (number): 0 ok; -1 null; -2 op not covered by this symbol
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: op outside this symbol's coverage
- can trap:
    - a negative or > 2^31-1 n arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED on nf_mask_not, same borrow path)
    - op outside this symbol's coverage -> -2, never garbage

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count |
| 2 | `s` | i32 | scalar | `-` | the scalar OPERAND, an i32 (not a pointer, not a length) |
| 3 | `op` | i32 | scalar | `-` | map fn code: 0 add 1 sub 2 mul 3 div 4 pow 5 floor_div 6 mod |
| 4 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

### 4.11 `pack` - 3 symbols

#### `nf_pack_i32_direct` &nbsp;*(5 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:261` (`#[no_mangle]`), origin: literal
- rust: `nf_pack_i32_direct(k1: *const i32, k2: *const i32, m2: i32, n: usize, out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(k1Ptr i32, k2Ptr i32, m2 i32, n i32, outPtr i32) -> i32`
- what: out[i] = k1[i]*m2 + k2[i] (mixed-radix composite code)
- returns `i32` (number): 0 ok; -1 null; -2 negative input or int32 overflow
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: negative code or int32 overflow
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `k1` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `k2` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `m2` | i32 | scalar | `-` | mixed-radix multiplier |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_pack_sum_count_f64` &nbsp;*(8 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:315` (`#[no_mangle]`), origin: literal
- rust: `nf_pack_sum_count_f64(k1: *const i32, k2: *const i32, m2: i32, vals: *const f64, n: usize, sums: *mut f64, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(k1Ptr i32, k2Ptr i32, m2 i32, valsPtr i32, n i32, sumsPtr i32, countsPtr i32, g i32) -> i32`
- what: fused pack + float64 sum + i64 count
- returns `i32` (number): 0 ok; -1 null; -2 pack/key range (no overflow code; inf on extremes)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: pack/key range
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `k1` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `k2` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `m2` | i32 | scalar | `-` | mixed-radix multiplier |
| 3 | `vals` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 4 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 5 | `sums` | i32 | out | `g` | output buffer of Float64Array, g lane(s) (overwritten) |
| 6 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 7 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

#### `nf_pack_sum_count_i32` &nbsp;*(8 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:286` (`#[no_mangle]`), origin: literal
- rust: `nf_pack_sum_count_i32(k1: *const i32, k2: *const i32, m2: i32, vals: *const i32, n: usize, sums: *mut i32, counts: *mut i64, g: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(k1Ptr i32, k2Ptr i32, m2 i32, valsPtr i32, n i32, sumsPtr i32, countsPtr i32, g i32) -> i32`
- what: fused pack + int32 sum + i64 count, no packed buffer materialised
- returns `i32` (number): 0 ok; -1 null; -2 pack/key range; -4 i32 overflow
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: pack/key range
    - `-4` = OVERFLOW
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `k1` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `k2` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `m2` | i32 | scalar | `-` | mixed-radix multiplier |
| 3 | `vals` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 4 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 5 | `sums` | i32 | out | `g` | output buffer of Int32Array, g lane(s) (overwritten) |
| 6 | `counts` | i32 | out | `g` | output buffer of BigInt64Array, g lane(s) (overwritten) |
| 7 | `g` | i32 | scalar | `-` | group count, i.e. the dense width |

### 4.12 `pair-insert` - 1 symbol

#### `nf_pair_insert_i64` &nbsp;*(7 args, 0 wrapped/unwrapped &nbsp;**NOT IN THE SHIPPED .wasm**)*

- source: `src/lib.rs:2972` (`#[no_mangle]`), origin: literal
- rust: `nf_pair_insert_i64(keys: *const i64, vals: *const i64, n: usize, tkeys: *mut i64, tvals: *mut i64, used: *mut u8, cap: usize) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i64`
- js: `(keysPtr i32, valsPtr i32, n i32, tkeysPtr i32, tvalsPtr i32, usedPtr i32, cap i32) -> bigint`
- what: generic open-addressing insert of (i64 key, i64 value) pairs; duplicate key = last-write-wins
- returns `i64` (bigint): ng (>= 0) occupied slots; -2 null-state; -3 bad geometry
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller owns tkeys/tvals/used[0..cap]; used is WARM-zeroed then overwritten
- concurrency: single-threaded by contract (ST reference owns input order and last-write-wins)
- return codes:
    - `0+` = ng
    - `-2` = NULL_STATE
    - `-3` = PAIR_GEOM: cap not pow2, cap == 0, short table, or a full table
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - NOT PRESENT in the shipped ts/dist artefact (85 exports). A fresh build of this working tree exports it (86). See CENSUS.md section 2.

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `vals` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `tkeys` | i32 | out | `cap` | output buffer of BigInt64Array, cap lane(s) (overwritten) |
| 4 | `tvals` | i32 | out | `cap` | output buffer of BigInt64Array, cap lane(s) (overwritten) |
| 5 | `used` | i32 | out | `cap` | output buffer of Uint8Array, cap lane(s) (overwritten) |
| 6 | `cap` | i32 | scalar | `-` | table capacity, power of two, >= 1; tkeys/tvals/used each need cap lanes |

### 4.13 `pattern` - 1 symbol

#### `nf_pattern_encode` &nbsp;*(10 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:351` (`#[no_mangle]`), origin: literal
- rust: `nf_pattern_encode(data: *const u8, total: usize, offs: *const i32, n: usize, prefix: *const u8, prefix_len: usize, codes: *mut i32, valid: *mut u8, width_out: *mut i32, err_row_out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(dataPtr i32, total i32, offsPtr i32, n i32, prefixPtr i32, prefixLen i32, codesPtr i32, validPtr i32, widthOutPtr i32, errRowOutPtr i32) -> i32`
- what: prefix+int string encode -> int32 codes + u8 validity + detected width
- returns `i32` (number): 0 ok; -1 null; -2 int32 overflow (err_row = first bad row, rows before it written); -3 malformed offsets
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: int32 overflow, partial write
    - `-3` = MALFORMED: bad offsets
- can trap:
    - offs outside [0,total] or non-monotone -> -3, not a trap (MEASURED for nf_text_length, same checker)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `data` | i32 | in | `total` | input buffer of Uint8Array, total lane(s) |
| 1 | `total` | i32 | scalar | `-` | byte length of the concatenated row data buffer `data` |
| 2 | `offs` | i32 | in | `n + 1` | input buffer of Int32Array, n + 1 lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `prefix` | i32 | in | `prefix_len` | input buffer of Uint8Array, prefix_len lane(s) |
| 5 | `prefix_len` | i32 | scalar | `-` | byte length of the encode prefix |
| 6 | `codes` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 7 | `valid` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |
| 8 | `width_out` | i32 | out | `1 (scalar slot)` | OUTPUT SCALAR SLOT: one Int32Array lane |
| 9 | `err_row_out` | i32 | out | `1 (scalar slot)` | OUTPUT SCALAR SLOT: one Int32Array lane |

### 4.14 `rng` - 5 symbols

#### `nf_rng_fill_f64` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1555` (`#[no_mangle]`), origin: literal
- rust: `nf_rng_fill_f64(out: *mut f64, n: usize, seed: u64, stream: u64, offset: u64, lo: f64, hi: f64) -> i32`
- wasm: `(i32, i32, i64, i64, i64, f64, f64) -> i32`
- js: `(outPtr i32, n i32, seed bigint, stream bigint, offset bigint, lo f64, hi f64) -> i32`
- what: counter-based fill of float64 lanes in [lo, hi); 53-bit mantissa draws
- returns `i32` (number): 0 ok; -1 null; -2 non-finite bounds or lo >= hi
- BigInt: **required** - seed, stream, offset are u64 -> three BigInt arguments
- allocates: no
- result ownership: caller
- concurrency: stateless by counter offset; reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-finite or lo >= hi
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `seed` | i64 | scalar | `-` | counter seed |
| 3 | `stream` | i64 | scalar | `-` | counter stream id |
| 4 | `offset` | i64 | scalar | `-` | counter offset; lane i draws counter offset+i |
| 5 | `lo` | f64 | scalar | `-` | inclusive low bound; must be finite and < hi |
| 6 | `hi` | f64 | scalar | `-` | exclusive high bound; must be finite |

#### `nf_rng_fill_i32` &nbsp;*(8 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1521` (`#[no_mangle]`), origin: literal
- rust: `nf_rng_fill_i32(out: *mut i32, n: usize, seed: u64, stream: u64, offset: u64, lo: i32, hi: i32, mode: u32) -> i32`
- wasm: `(i32, i32, i64, i64, i64, i32, i32, i32) -> i32`
- js: `(outPtr i32, n i32, seed bigint, stream bigint, offset bigint, lo i32, hi i32, mode i32) -> i32`
- what: counter-based fill of int32 lanes in [lo, hi); chunkable bit-exact
- returns `i32` (number): 0 ok; -1 null; -2 bad range (hi <= lo); -3 bad geometry (mode != 0)
- BigInt: **required** - seed, stream, offset are u64 -> three BigInt arguments; a Number is rejected
- allocates: no
- result ownership: caller
- concurrency: stateless by counter offset; reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: hi <= lo
    - `-3` = MALFORMED: mode != 0
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `seed` | i64 | scalar | `-` | counter seed |
| 3 | `stream` | i64 | scalar | `-` | counter stream id |
| 4 | `offset` | i64 | scalar | `-` | counter offset; lane i draws counter offset+i |
| 5 | `lo` | i32 | scalar | `-` | inclusive low bound of the drawn range [lo, hi) |
| 6 | `hi` | i32 | scalar | `-` | exclusive high bound; hi <= lo is -2 |
| 7 | `mode` | i32 | scalar | `-` | RNG mode; only 0 exists, anything else is -3 |

#### `nf_rng_map_round` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1587` (`#[no_mangle]`), origin: literal
- rust: `nf_rng_map_round(x: *const f64, valid: *const u8, n: usize, ndigits: u32, out: *mut f64, out_valid: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(xPtr i32, validPtr i32, n i32, ndigits i32, outPtr i32, outValidPtr i32) -> i32`
- what: half-even round to ndigits (0..15) with a u8 validity sidecar in and out
- returns `i32` (number): 0 ok; -1 null; -2 ndigits > 15
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: ndigits > 15
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `x` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 1 | `valid` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `ndigits` | i32 | scalar | `-` | round digits, 0..15 |
| 4 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |
| 5 | `out_valid` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_rng_permutation` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1647` (`#[no_mangle]`), origin: literal
- rust: `nf_rng_permutation(n: usize, seed: u64, stream: u64, offset: u64, pool: *mut i32, out: *mut i32) -> i64`
- wasm: `(i32, i64, i64, i64, i32, i32) -> i64`
- js: `(n i32, seed bigint, stream bigint, offset bigint, poolPtr i32, outPtr i32) -> bigint`
- what: full Yates permutation of 0..n; same contract as nf_rng_sample_no_replace with k == n
- returns `i64` (bigint): n (>= 0); -2 NULL_STATE
- BigInt: **required** - seed, stream, offset are u64 -> three BigInt arguments; RETURN is BigInt
- allocates: no
- result ownership: caller allocates pool[n] and out[n]
- concurrency: single-threaded
- return codes:
    - `0+` = n
    - `-2` = NULL_STATE
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 1 | `seed` | i64 | scalar | `-` | counter seed |
| 2 | `stream` | i64 | scalar | `-` | counter stream id |
| 3 | `offset` | i64 | scalar | `-` | counter offset; lane i draws counter offset+i |
| 4 | `pool` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 5 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_rng_sample_no_replace` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1622` (`#[no_mangle]`), origin: literal
- rust: `nf_rng_sample_no_replace(n: usize, k: usize, seed: u64, stream: u64, offset: u64, pool: *mut i32, out: *mut i32) -> i64`
- wasm: `(i32, i32, i64, i64, i64, i32, i32) -> i64`
- js: `(n i32, k i32, seed bigint, stream bigint, offset bigint, poolPtr i32, outPtr i32) -> bigint`
- what: Fisher-Yates first-k sample; draws are sequential, chunkable=false
- returns `i64` (bigint): k (>= 0; 0 for k == 0); -2 null-state AND k > n (same code, see HAZARD H7); -3 n > i32::MAX
- BigInt: **required** - seed, stream, offset are u64 -> three BigInt arguments; RETURN is BigInt
- allocates: no (pool is caller scratch)
- result ownership: caller allocates pool[n] and out[k]
- concurrency: single-threaded by contract (one global sample, sequential draw stream)
- return codes:
    - `0+` = k
    - `-2` = NULL_STATE, or k > n (overloaded)
    - `-3` = MALFORMED: n > i32::MAX
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 1 | `k` | i32 | scalar | `-` | number of draws; k > n returns -2 |
| 2 | `seed` | i64 | scalar | `-` | counter seed |
| 3 | `stream` | i64 | scalar | `-` | counter stream id |
| 4 | `offset` | i64 | scalar | `-` | counter offset; lane i draws counter offset+i |
| 5 | `pool` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 6 | `out` | i32 | out | `k` | output buffer of Int32Array, k lane(s) (overwritten) |

### 4.15 `rng R-COMPAT` - 2 symbols

#### `nf_rng_compat_runif` &nbsp;*(5 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1672` (`#[no_mangle]`), origin: literal
- rust: `nf_rng_compat_runif(out: *mut f64, n: usize, seed: i32, lo: f64, hi: f64) -> i32`
- wasm: `(i32, i32, i32, f64, f64) -> i32`
- js: `(outPtr i32, n i32, seed i32, lo f64, hi f64) -> i32`
- what: R 4.3.x-compatible runif; strict sequential draws, chunkable=false
- returns `i32` (number): 0 ok; -1 null; -2 non-finite bounds or lo >= hi
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: single-threaded (one MT state per call)
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `seed` | i32 | scalar | `-` | counter seed |
| 3 | `lo` | f64 | scalar | `-` | inclusive low bound; must be finite and < hi |
| 4 | `hi` | f64 | scalar | `-` | exclusive high bound; must be finite |

#### `nf_rng_compat_sample` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1698` (`#[no_mangle]`), origin: literal
- rust: `nf_rng_compat_sample(n: i64, out: *mut i32, m: usize, seed: i32) -> i32`
- wasm: `(i64, i32, i32, i32) -> i32`
- js: `(n bigint, outPtr i32, m i32, seed i32) -> i32`
- what: R-compatible 0-based index draws (rejection chunks, strict sequential)
- returns `i32` (number): 0 ok; -1 null; -2 bad range (n <= 0)
- BigInt: **required** - n is i64 -> a BigInt argument (the only rng param that is signed i64)
- allocates: no
- result ownership: caller
- concurrency: single-threaded
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: n <= 0
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `n` | i64 | scalar | `-` | element count (rows / lanes) |
| 1 | `out` | i32 | out | `m` | output buffer of Int32Array, m lane(s) (overwritten) |
| 2 | `m` | i32 | scalar | `-` | element count (groups / draws / positions) |
| 3 | `seed` | i32 | scalar | `-` | counter seed |

### 4.16 `router` - 1 symbol

#### `nf_router_route` &nbsp;*(11 args, 0 wrapped/unwrapped)*

- source: `src/router.rs:95` (`#[no_mangle]`), origin: literal
- rust: `nf_router_route(n: usize, indptr: *const i64, m: usize, indices: *const i32, weights: *const i64, sources: *const i32, nsrc: usize, is_target: *const u8, out_dist: *mut i64, out_target: *mut i32, out_visited: *mut i64) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(n i32, indptrPtr i32, m i32, indicesPtr i32, weightsPtr i32, sourcesPtr i32, nsrc i32, isTargetPtr i32, outDistPtr i32, outTargetPtr i32, outVisitedPtr i32) -> i32`
- what: multi-source Dijkstra over flat CSR with early exit on a target set
- returns `i32` (number): 0 found; 1 ROUTER_UNREACHABLE (out_dist = INT64_MAX, out_target = -1); -1 null; -2 bad range
- BigInt: no
- allocates: YES - vec![INF; n] plus a BinaryHeap (router.rs:44-46)
- result ownership: kernel frees both; the caller owns the three 1-lane output buffers
- concurrency: reentrant (all state is call-local)
- return codes:
    - `0` = found
    - `1` = ROUTER_UNREACHABLE (a positive code, unique to this symbol)
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: n/m/nsrc == 0, indptr[n] != m, non-monotone indptr, index outside [0,n)
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - out_dist/out_target/out_visited are THREE SEPARATE 1-element output slots, not one array

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `n` | i32 | scalar | `-` | VERTEX COUNT, and it is argument 0 - before every pointer |
| 1 | `indptr` | i32 | in | `n + 1` | input buffer of BigInt64Array, n + 1 lane(s) |
| 2 | `m` | i32 | scalar | `-` | edge count; indptr[n] must equal m |
| 3 | `indices` | i32 | in | `m` | input buffer of Int32Array, m lane(s) |
| 4 | `weights` | i32 | in | `m` | input buffer of BigInt64Array, m lane(s) |
| 5 | `sources` | i32 | in | `nsrc` | input buffer of Int32Array, nsrc lane(s) |
| 6 | `nsrc` | i32 | scalar | `-` | number of source vertices; out-of-range entries are ignored, not an error |
| 7 | `is_target` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 8 | `out_dist` | i32 | out | `1` | output buffer of BigInt64Array, 1 lane(s) (overwritten) |
| 9 | `out_target` | i32 | out | `1` | output buffer of Int32Array, 1 lane(s) (overwritten) |
| 10 | `out_visited` | i32 | out | `1` | output buffer of BigInt64Array, 1 lane(s) (overwritten) |

### 4.17 `rowwise` - 3 symbols

#### `nf_rowwise_kway_time_argmin_gather` &nbsp;*(7 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2648` (`#[no_mangle]`), origin: literal
- rust: `nf_rowwise_kway_time_argmin_gather(t_ptrs: *const *const i32, d_ptrs: *const *const f32, k: usize, n: usize, t_best: *mut i32, d_best: *mut f32, m_best: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(tPtrsPtr i32, dPtrsPtr i32, k i32, n i32, tBestPtr i32, dBestPtr i32, mBestPtr i32) -> i32`
- what: canonical K-way rowwise MIN over int32 selector lanes + f32 payload gather; t_ptrs/d_ptrs are arrays OF OFFSETS
- returns `i32` (number): 0 ok; -1 null; -2 bad range (k == 0 or k > 256)
- BigInt: no
- allocates: no
- result ownership: caller allocates k offset lanes plus 3 x n outputs
- concurrency: reentrant, row-independent
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: k outside 1..=256
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - a null element INSIDE t_ptrs/d_ptrs returns -1 (checked per lane)
    - t_ptrs/d_ptrs are pointer arrays stored in linear memory as u32 offsets, not as inline lane data (HAZARD H5)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `t_ptrs` | i32 | in | `k` | pointer to an array of lane OFFSETS (pointer-to-pointer; one i32 in wasm) |
| 1 | `d_ptrs` | i32 | in | `k` | pointer to an array of lane OFFSETS (pointer-to-pointer; one i32 in wasm) |
| 2 | `k` | i32 | scalar | `-` | lane count, 1..=256; t_ptrs/d_ptrs have k lanes each |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `t_best` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 5 | `d_best` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |
| 6 | `m_best` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_rowwise_min4_argmin_gather` &nbsp;*(12 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2486` (`#[no_mangle]`), origin: literal
- rust: `nf_rowwise_min4_argmin_gather(t0: *const i32, t1: *const i32, t2: *const i32, t3: *const i32, d0: *const f32, d1: *const f32, d2: *const f32, d3: *const f32, n: usize, t_best: *mut i32, d_best: *mut f32, m_best: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(t0Ptr i32, t1Ptr i32, t2Ptr i32, t3Ptr i32, d0Ptr i32, d1Ptr i32, d2Ptr i32, d3Ptr i32, n i32, tBestPtr i32, dBestPtr i32, mBestPtr i32) -> i32`
- what: rowwise MIN over 4 int32 lanes keyed by 4 f32 lanes + argmin gather
- returns `i32` (number): 0 ok; -1 null
- BigInt: no
- allocates: no
- result ownership: caller allocates 3 x n output lanes
- concurrency: reentrant, row-independent
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `t0` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `t1` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `t2` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 3 | `t3` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 4 | `d0` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 5 | `d1` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 6 | `d2` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 7 | `d3` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 8 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 9 | `t_best` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 10 | `d_best` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |
| 11 | `m_best` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_rowwise_min4_time_argmin_gather` &nbsp;*(12 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2560` (`#[no_mangle]`), origin: literal
- rust: `nf_rowwise_min4_time_argmin_gather(t0: *const i32, t1: *const i32, t2: *const i32, t3: *const i32, d0: *const f32, d1: *const f32, d2: *const f32, d3: *const f32, n: usize, t_best: *mut i32, d_best: *mut f32, m_best: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(t0Ptr i32, t1Ptr i32, t2Ptr i32, t3Ptr i32, d0Ptr i32, d1Ptr i32, d2Ptr i32, d3Ptr i32, n i32, tBestPtr i32, dBestPtr i32, mBestPtr i32) -> i32   // pass 0 for d0..d3 and dBest on the time-only path`
- what: rowwise MIN over 4 int32 TIME lanes; d0..d3 all-null = time-only path, all-non-null = gather path
- returns `i32` (number): 0 ok; -1 null / contract violation (mixed d-lane nulls, or d_best null on the gather path)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant, row-independent
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT: null pointer OR d-lane contract violation
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - d0..d3 are an ALL-OR-NOTHING group; a mixed null pattern returns -1, it is not a trap
    - on the time-only path d_best must be NULL and is never dereferenced
- **non-obvious contract** (resolved from source, not ambiguous): The optional d-lane group is certain from the source (it tests d_all / d_none explicitly) but it is the only export where passing NULL is a legal, meaningful argument value.

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `t0` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `t1` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 2 | `t2` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 3 | `t3` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 4 | `d0` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 5 | `d1` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 6 | `d2` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 7 | `d3` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 8 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 9 | `t_best` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 10 | `d_best` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |
| 11 | `m_best` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

### 4.18 `segmented` - 3 symbols

#### `nf_segment_count` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1989` (`#[no_mangle]`), origin: literal
- rust: `nf_segment_count(bounds: *const u32, m: usize, n: usize, out: *mut u32) -> i32`
- wasm: `(i32, i32, i32, i32) -> i32`
- js: `(boundsPtr i32, m i32, n i32, outPtr i32) -> i32`
- what: u32 count lane over monotone bounds (exact diffs)
- returns `i32` (number): 0 ok; -1 null; -2 bad range; -3 bad geometry
- BigInt: no
- allocates: no
- result ownership: caller allocates m lanes
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-monotone bounds, bounds[m] != n, lane outside [0,n], n == 0 with m != 0
    - `-3` = MALFORMED: n > 4194240
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - n <= 4194240 (SEG_MAX) or -3 bad geometry; outside that the caller must chunk

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `bounds` | i32 | in | `m + 1` | input buffer of Uint32Array, m + 1 lane(s) |
| 1 | `m` | i32 | scalar | `-` | group count; bounds has m+1 lanes |
| 2 | `n` | i32 | scalar | `-` | total element count the bounds describe; bounds[m] must equal n |
| 3 | `out` | i32 | out | `m` | output buffer of Uint32Array, m lane(s) (overwritten) |

#### `nf_segment_reduce_f32` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2048` (`#[no_mangle]`), origin: macro `segment_reduce_ffi`
- rust: `nf_segment_reduce_f32(values: *const f32, n: usize, bounds: *const u32, m: usize, op: u32, out: *mut f32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(valuesPtr i32, n i32, boundsPtr i32, m i32, op i32, outPtr i32) -> i32   // lanes: Float32Array`
- what: bounds reduce over f32 lanes; sum IEEE-propagate, min/max NaN-propagate
- returns `i32` (number): 0 ok; -1 null; -2 bad range / empty group under min or max; -3 bad geometry
- BigInt: no
- allocates: no
- result ownership: caller allocates m lanes
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-monotone bounds, bounds[m] != n, lane outside [0,n], op == 1 (count) or unknown, min/max over an empty group
    - `-3` = MALFORMED: n > 4194240
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - n <= 4194240 (SEG_MAX) or -3 bad geometry; outside that the caller must chunk

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `values` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count; bounds[m] must equal n |
| 2 | `bounds` | i32 | in | `m + 1` | input buffer of Uint32Array, m + 1 lane(s) |
| 3 | `m` | i32 | scalar | `-` | group count; bounds has m+1 lanes |
| 4 | `op` | i32 | scalar | `-` | reduce code: 0 sum, 2 min, 3 max; 1 and anything else are -2 |
| 5 | `out` | i32 | out | `m` | output buffer of Float32Array, m lane(s) (overwritten) |

#### `nf_segment_reduce_i32` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:2055` (`#[no_mangle]`), origin: macro `segment_reduce_ffi`
- rust: `nf_segment_reduce_i32(values: *const i32, n: usize, bounds: *const u32, m: usize, op: u32, out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i32`
- js: `(valuesPtr i32, n i32, boundsPtr i32, m i32, op i32, outPtr i32) -> i32   // lanes: Int32Array`
- what: bounds reduce over i32 lanes; sum SATURATES, never wraps
- returns `i32` (number): 0 ok; -1 null; -2 bad range / empty group under min or max; -3 bad geometry
- BigInt: no
- allocates: no
- result ownership: caller allocates m lanes
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-monotone bounds, bounds[m] != n, lane outside [0,n], op == 1 (count) or unknown, min/max over an empty group
    - `-3` = MALFORMED: n > 4194240
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - n <= 4194240 (SEG_MAX) or -3 bad geometry; outside that the caller must chunk

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `values` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count; bounds[m] must equal n |
| 2 | `bounds` | i32 | in | `m + 1` | input buffer of Uint32Array, m + 1 lane(s) |
| 3 | `m` | i32 | scalar | `-` | group count; bounds has m+1 lanes |
| 4 | `op` | i32 | scalar | `-` | reduce code: 0 sum, 2 min, 3 max; 1 and anything else are -2 |
| 5 | `out` | i32 | out | `m` | output buffer of Int32Array, m lane(s) (overwritten) |

### 4.19 `select` - 9 symbols

#### `nf_mask_and` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1013` (`#[no_mangle]`), origin: literal
- rust: `nf_mask_and(a: *const u8, b: *const u8, n: usize, out: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, bPtr i32, n i32, outPtr i32) -> i32`
- what: elementwise mask AND, output normalised 0/1
- returns `i32` (number): 0 ok; -1 null
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 1 | `b` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_mask_not` &nbsp;*(3 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1049` (`#[no_mangle]`), origin: literal
- rust: `nf_mask_not(a: *const u8, n: usize, out: *mut u8) -> i32`
- wasm: `(i32, i32, i32) -> i32`
- js: `(aPtr i32, n i32, outPtr i32) -> i32`
- what: elementwise mask NOT, output normalised 0/1
- returns `i32` (number): 0 ok; -1 null
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_mask_or` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1031` (`#[no_mangle]`), origin: literal
- rust: `nf_mask_or(a: *const u8, b: *const u8, n: usize, out: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32) -> i32`
- js: `(aPtr i32, bPtr i32, n i32, outPtr i32) -> i32`
- what: elementwise mask OR, output normalised 0/1
- returns `i32` (number): 0 ok; -1 null
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `a` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 1 | `b` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_select_count` &nbsp;*(2 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:629` (`#[no_mangle]`), origin: literal
- rust: `nf_select_count(mask: *const u8, n: usize) -> i64`
- wasm: `(i32, i32) -> i64`
- js: `(maskPtr i32, n i32) -> bigint`
- what: count kept rows in a u8 mask (0 = drop)
- returns `i64` (bigint): m (>= 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0+` = m = rows kept
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `mask` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |

#### `nf_select_scatter_f32` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:685` (`#[no_mangle]`), origin: macro `select_scatter_ffi`
- rust: `nf_select_scatter_f32(src: *const f32, mask: *const u8, n: usize, out: *mut f32) -> i64`
- wasm: `(i32, i32, i32, i32) -> i64`
- js: `(srcPtr i32, maskPtr i32, n i32, outPtr i32) -> bigint   // lanes: Float32Array`
- what: boolean selection (filter) over f32 lanes; first m lanes of out, order preserved
- returns `i64` (bigint): m (>= 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller; out is n lanes, only m written
- concurrency: reentrant
- return codes:
    - `0+` = m = rows written
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `mask` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `out` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |

#### `nf_select_scatter_f64` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:690` (`#[no_mangle]`), origin: macro `select_scatter_ffi`
- rust: `nf_select_scatter_f64(src: *const f64, mask: *const u8, n: usize, out: *mut f64) -> i64`
- wasm: `(i32, i32, i32, i32) -> i64`
- js: `(srcPtr i32, maskPtr i32, n i32, outPtr i32) -> bigint   // lanes: Float64Array`
- what: boolean selection (filter) over f64 lanes; first m lanes of out, order preserved
- returns `i64` (bigint): m (>= 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller; out is n lanes, only m written
- concurrency: reentrant
- return codes:
    - `0+` = m = rows written
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 1 | `mask` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |

#### `nf_select_scatter_i32` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:675` (`#[no_mangle]`), origin: macro `select_scatter_ffi`
- rust: `nf_select_scatter_i32(src: *const i32, mask: *const u8, n: usize, out: *mut i32) -> i64`
- wasm: `(i32, i32, i32, i32) -> i64`
- js: `(srcPtr i32, maskPtr i32, n i32, outPtr i32) -> bigint   // lanes: Int32Array`
- what: boolean selection (filter) over i32 lanes; first m lanes of out, order preserved
- returns `i64` (bigint): m (>= 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller; out is n lanes, only m written
- concurrency: reentrant
- return codes:
    - `0+` = m = rows written
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `mask` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_select_scatter_i64` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:680` (`#[no_mangle]`), origin: macro `select_scatter_ffi`
- rust: `nf_select_scatter_i64(src: *const i64, mask: *const u8, n: usize, out: *mut i64) -> i64`
- wasm: `(i32, i32, i32, i32) -> i64`
- js: `(srcPtr i32, maskPtr i32, n i32, outPtr i32) -> bigint   // lanes: BigInt64Array`
- what: boolean selection (filter) over i64 lanes; first m lanes of out, order preserved
- returns `i64` (bigint): m (>= 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller; out is n lanes, only m written
- concurrency: reentrant
- return codes:
    - `0+` = m = rows written
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `mask` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `out` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |

#### `nf_select_scatter_u8` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:695` (`#[no_mangle]`), origin: macro `select_scatter_ffi`
- rust: `nf_select_scatter_u8(src: *const u8, mask: *const u8, n: usize, out: *mut u8) -> i64`
- wasm: `(i32, i32, i32, i32) -> i64`
- js: `(srcPtr i32, maskPtr i32, n i32, outPtr i32) -> bigint   // lanes: Uint8Array`
- what: boolean selection (filter) over u8 lanes; first m lanes of out, order preserved
- returns `i64` (bigint): m (>= 0); -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller; out is n lanes, only m written
- concurrency: reentrant
- return codes:
    - `0+` = m = rows written
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 1 | `mask` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

### 4.20 `shift` - 3 symbols

#### `nf_shift_f32` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:746` (`#[no_mangle]`), origin: macro `shift_ffi`
- rust: `nf_shift_f32(src: *const f32, n: usize, periods: usize, out: *mut f32) -> i32`
- wasm: `(i32, i32, i32, i32) -> i32`
- js: `(srcPtr i32, n i32, periods i32, outPtr i32) -> i32   // lanes: Float32Array`
- what: positional right-shift over f32 lanes; head min(periods,n) zero, tail bit-exact copy
- returns `i32` (number): 0 ok; -1 null (n == 0 returns 0 without dereferencing)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - periods >= n zero-fills rather than trapping

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Float32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `periods` | i32 | scalar | `-` | right-shift amount, >= 0; 0 copies, >= n zero-fills |
| 3 | `out` | i32 | out | `n` | output buffer of Float32Array, n lane(s) (overwritten) |

#### `nf_shift_f64` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:751` (`#[no_mangle]`), origin: macro `shift_ffi`
- rust: `nf_shift_f64(src: *const f64, n: usize, periods: usize, out: *mut f64) -> i32`
- wasm: `(i32, i32, i32, i32) -> i32`
- js: `(srcPtr i32, n i32, periods i32, outPtr i32) -> i32   // lanes: Float64Array`
- what: positional right-shift over f64 lanes; head min(periods,n) zero, tail bit-exact copy
- returns `i32` (number): 0 ok; -1 null (n == 0 returns 0 without dereferencing)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - periods >= n zero-fills rather than trapping

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `periods` | i32 | scalar | `-` | right-shift amount, >= 0; 0 copies, >= n zero-fills |
| 3 | `out` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |

#### `nf_shift_i32` &nbsp;*(4 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:741` (`#[no_mangle]`), origin: macro `shift_ffi`
- rust: `nf_shift_i32(src: *const i32, n: usize, periods: usize, out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32) -> i32`
- js: `(srcPtr i32, n i32, periods i32, outPtr i32) -> i32   // lanes: Int32Array`
- what: positional right-shift over i32 lanes; head min(periods,n) zero, tail bit-exact copy
- returns `i32` (number): 0 ok; -1 null (n == 0 returns 0 without dereferencing)
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - periods >= n zero-fills rather than trapping

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `src` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `periods` | i32 | scalar | `-` | right-shift amount, >= 0; 0 copies, >= n zero-fills |
| 3 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

### 4.21 `sort` - 2 symbols

#### `nf_sort_perm_i32` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1325` (`#[no_mangle]`), origin: literal
- rust: `nf_sort_perm_i32(keys: *const i32, n: usize, descending: u8, perm_out: *mut i32, tmp0: *mut u32, tmp1: *mut u32, tmp_p: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, n i32, descending i32, permOutPtr i32, tmp0Ptr i32, tmp1Ptr i32, tmpPPtr i32) -> i32`
- what: stable LSD radix argsort permutation of int32 keys (asc == np.argsort(kind='stable'))
- returns `i32` (number): 0 ok; -1 null; -3 bad geometry (n > i32::MAX)
- BigInt: no
- allocates: no
- result ownership: caller allocates 4 scratch buffers of n lanes
- concurrency: single-threaded by contract (same as the ST oracle)
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: n > i32::MAX
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `descending` | i32 | scalar | `-` | u8 flag; nonzero = descending stable |
| 3 | `perm_out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 4 | `tmp0` | i32 | out | `n` | output buffer of Uint32Array, n lane(s) (overwritten) |
| 5 | `tmp1` | i32 | out | `n` | output buffer of Uint32Array, n lane(s) (overwritten) |
| 6 | `tmp_p` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_sort_perm_i64` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1362` (`#[no_mangle]`), origin: literal
- rust: `nf_sort_perm_i64(keys: *const i64, n: usize, descending: u8, perm_out: *mut i32, tmp0: *mut u64, tmp1: *mut u64, tmp_p: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(keysPtr i32, n i32, descending i32, permOutPtr i32, tmp0Ptr i32, tmp1Ptr i32, tmpPPtr i32) -> i32`
- what: int64-key variant of nf_sort_perm_i32; positions stay int32, tmp0/tmp1 become u64
- returns `i32` (number): 0 ok; -1 null; -3 n > i32::MAX
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: single-threaded
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-3` = MALFORMED: n > i32::MAX
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `descending` | i32 | scalar | `-` | u8 flag; nonzero = descending stable |
| 3 | `perm_out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 4 | `tmp0` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 5 | `tmp1` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 6 | `tmp_p` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

### 4.22 `sorted-run` - 2 symbols

#### `nf_sorted_run_f64` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:424` (`#[no_mangle]`), origin: literal
- rust: `nf_sorted_run_f64(keys: *const i32, vals: *const f64, n: usize, ukeys: *mut i64, sums: *mut f64, counts: *mut i64) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i64`
- js: `(keysPtr i32, valsPtr i32, n i32, ukeysPtr i32, sumsPtr i32, countsPtr i32) -> bigint`
- what: float64 variant of nf_sorted_run_i64; ukeys/counts stay i64, sums are f64
- returns `i64` (bigint): ng (>= 0); -1 unsorted; -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller
- concurrency: reentrant
- return codes:
    - `0+` = ng
    - `-1` = UNSORTED_ABORT
    - `-2` = NULL_STATE
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `vals` | i32 | in | `n` | input buffer of Float64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `ukeys` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 4 | `sums` | i32 | out | `n` | output buffer of Float64Array, n lane(s) (overwritten) |
| 5 | `counts` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |

#### `nf_sorted_run_i64` &nbsp;*(6 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:398` (`#[no_mangle]`), origin: literal
- rust: `nf_sorted_run_i64(keys: *const i32, vals: *const i64, n: usize, ukeys: *mut i64, sums: *mut i64, counts: *mut i64) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32) -> i64`
- js: `(keysPtr i32, valsPtr i32, n i32, ukeysPtr i32, sumsPtr i32, countsPtr i32) -> bigint`
- what: run aggregation over nondecreasing int32 keys, int64 values -> compact (ukeys, sums, counts)
- returns `i64` (bigint): ng (>= 0, 0 for n == 0); -1 unsorted speculative abort; -2 NULL_STATE
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller allocates n lanes for ukeys/sums/counts
- concurrency: reentrant
- return codes:
    - `0+` = ng = number of distinct runs
    - `-1` = UNSORTED_ABORT: keys not nondecreasing
    - `-2` = NULL_STATE: null pointer
- can trap: no (only via the shared negative-length path, see H3)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `vals` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 2 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 3 | `ukeys` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 4 | `sums` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 5 | `counts` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |

### 4.23 `sssp` - 3 symbols

#### `nf_sssp_batch` &nbsp;*(9 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2916` (`#[no_mangle]`), origin: literal
- rust: `nf_sssp_batch(indptr: *const u32, np: usize, indices: *const u32, weights: *const u32, e: usize, sources: *const u32, k: usize, out: *mut u32, nthreads: usize) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(indptrPtr i32, np i32, indicesPtr i32, weightsPtr i32, e i32, sourcesPtr i32, k i32, outPtr i32, nthreads i32) -> i32`
- what: sources[k] -> out[k*V] row-major (out[r*V..(r+1)*V] holds dist from sources[r])
- returns `i32` (number): 0 ok; -1 null; -2 bad range; -3 bad geometry
- BigInt: no
- allocates: YES (BinaryHeap per source, plus Vec chunking)
- result ownership: caller owns out[0..k*V]
- concurrency: reentrant; the wasm build forces nthreads = 1 and loops the same shards inline (sssp.rs:225-231)
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: reserved/out-of-range source
    - `-3` = MALFORMED: indptr[np-1] != e, short out
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `indptr` | i32 | in | `np` | input buffer of Uint32Array, np lane(s) |
| 1 | `np` | i32 | scalar | `-` | indptr length in lanes; V = np-1 for a CSR graph |
| 2 | `indices` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 3 | `weights` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 4 | `e` | i32 | scalar | `-` | edge count |
| 5 | `sources` | i32 | in | `k` | input buffer of Uint32Array, k lane(s) |
| 6 | `k` | i32 | scalar | `-` | number of sources; out has k*V lanes where V = np-1 |
| 7 | `out` | i32 | out | `k.saturating_mul(v` | output buffer of Uint32Array, k.saturating_mul(v lane(s) (overwritten) |
| 8 | `nthreads` | i32 | scalar | `-` | thread hint clamped to [1,64]; 0 means 1 |

#### `nf_sssp_csr` &nbsp;*(7 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2849` (`#[no_mangle]`), origin: literal
- rust: `nf_sssp_csr(indptr: *const u32, np: usize, indices: *const u32, weights: *const u32, e: usize, source: u32, dist: *mut u32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(indptrPtr i32, np i32, indicesPtr i32, weightsPtr i32, e i32, source i32, distPtr i32) -> i32`
- what: canonical single-source shortest path over CSR + u32 weights; UINT32_MAX edges are skipped
- returns `i32` (number): 0 ok; -1 null; -2 bad range; -3 bad geometry
- BigInt: no
- allocates: YES inside the kernel: BinaryHeap with_capacity(min(V,1024)) per call (sssp.rs:91)
- result ownership: kernel frees the heap; caller owns dist[0..V]
- concurrency: reentrant; per call, no shared state
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: reserved/out-of-range index or source, non-monotone indptr, empty graph
    - `-3` = MALFORMED: lane outside [0,E], indptr[np-1] != e, short output
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `indptr` | i32 | in | `np` | input buffer of Uint32Array, np lane(s) |
| 1 | `np` | i32 | scalar | `-` | indptr length in lanes; V = np-1 for a CSR graph |
| 2 | `indices` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 3 | `weights` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 4 | `e` | i32 | scalar | `-` | edge count; len(indices) == len(weights) == e |
| 5 | `source` | i32 | scalar | `-` | start vertex, u32 in [0,V) where V = np-1 |
| 6 | `dist` | i32 | out | `v` | output buffer of Uint32Array, v lane(s) (overwritten) |

#### `nf_sssp_csr_pred` &nbsp;*(8 args, 1 wrapped/unwrapped)*

- source: `src/lib.rs:2874` (`#[no_mangle]`), origin: literal
- rust: `nf_sssp_csr_pred(indptr: *const u32, np: usize, indices: *const u32, weights: *const u32, e: usize, source: u32, dist: *mut u32, pred: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(indptrPtr i32, np i32, indicesPtr i32, weightsPtr i32, e i32, source i32, distPtr i32, predPtr i32) -> i32`
- what: nf_sssp_csr PLUS a predecessor OUTPUT array; pred[0..V] i32, -1 = root/unreached
- returns `i32` (number): 0 ok; -1 null; -2 bad range; -3 bad geometry (same as nf_sssp_csr)
- BigInt: no
- allocates: YES (same BinaryHeap as nf_sssp_csr)
- result ownership: caller owns dist[0..V] AND pred[0..V]
- concurrency: reentrant
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE
    - `-3` = MALFORMED
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - pred is an OUTPUT of V i32 lanes (4*V bytes). Passing a buffer shorter than that is an out-of-bounds WRITE with no trap (MEASURED - see the shipped-wrapper defect in CENSUS.md)
- **non-obvious contract** (resolved from source, not ambiguous): NOT ambiguous in the source. The shipped TS wrapper and abi.ts both claim pred is a per-edge PREDICATE BYTE INPUT; the Rust body is `*mut i32` borrowed for `np-1` lanes and FILLED with SSSP_INF before the search. See CENSUS.md section 3.

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `indptr` | i32 | in | `np` | input buffer of Uint32Array, np lane(s) |
| 1 | `np` | i32 | scalar | `-` | indptr length in lanes; V = np-1 for a CSR graph |
| 2 | `indices` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 3 | `weights` | i32 | in | `e` | input buffer of Uint32Array, e lane(s) |
| 4 | `e` | i32 | scalar | `-` | edge count; len(indices) == len(weights) == e |
| 5 | `source` | i32 | scalar | `-` | start vertex, u32 in [0,V) where V = np-1 |
| 6 | `dist` | i32 | out | `v` | output buffer of Uint32Array, v lane(s) (overwritten) |
| 7 | `pred` | i32 | out | `v` | output buffer of Int32Array, v lane(s) (overwritten) |

### 4.24 `text` - 5 symbols

#### `nf_text_contains` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1771` (`#[no_mangle]`), origin: literal
- rust: `nf_text_contains(data: *const u8, total: usize, offs: *const i32, n: usize, needle: *const u8, needle_len: usize, out: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(dataPtr i32, total i32, offsPtr i32, n i32, needlePtr i32, needleLen i32, outPtr i32) -> i32`
- what: UTF-8 substring hit per row (u8 0/1)
- returns `i32` (number): 0 ok; -1 null; -2 non-UTF-8 needle (even when n == 0); -3 malformed offsets / invalid UTF-8 row
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant, row-independent, chunkable
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-UTF-8 needle
    - `-3` = MALFORMED: bad offsets / invalid UTF-8 row
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - length is in CODE POINTS, not bytes
    - non-monotone or out-of-range offs -> -3 MALFORMED, not a trap (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `data` | i32 | in | `total` | input buffer of Uint8Array, total lane(s) |
| 1 | `total` | i32 | scalar | `-` | CAPACITY in lanes; a capacity, NOT a checksum |
| 2 | `offs` | i32 | in | `n + 1` | input buffer of Int32Array, n + 1 lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `needle` | i32 | in | `needle_len` | input buffer of Uint8Array, needle_len lane(s) |
| 5 | `needle_len` | i32 | scalar | `-` | needle length in BYTES (not code points) |
| 6 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_text_endswith` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1844` (`#[no_mangle]`), origin: literal
- rust: `nf_text_endswith(data: *const u8, total: usize, offs: *const i32, n: usize, needle: *const u8, needle_len: usize, out: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(dataPtr i32, total i32, offsPtr i32, n i32, needlePtr i32, needleLen i32, outPtr i32) -> i32`
- what: anchored suffix hit per row (u8 0/1)
- returns `i32` (number): 0 ok; -1 null; -2 non-UTF-8 needle (even when n == 0); -3 malformed offsets / invalid UTF-8 row
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant, row-independent, chunkable
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-UTF-8 needle
    - `-3` = MALFORMED: bad offsets / invalid UTF-8 row
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - length is in CODE POINTS, not bytes
    - non-monotone or out-of-range offs -> -3 MALFORMED, not a trap (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `data` | i32 | in | `total` | input buffer of Uint8Array, total lane(s) |
| 1 | `total` | i32 | scalar | `-` | CAPACITY in lanes; a capacity, NOT a checksum |
| 2 | `offs` | i32 | in | `n + 1` | input buffer of Int32Array, n + 1 lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `needle` | i32 | in | `needle_len` | input buffer of Uint8Array, needle_len lane(s) |
| 5 | `needle_len` | i32 | scalar | `-` | needle length in BYTES |
| 6 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_text_equals` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1941` (`#[no_mangle]`), origin: literal
- rust: `nf_text_equals(data: *const u8, total: usize, offs: *const i32, n: usize, needle: *const u8, needle_len: usize, out: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(dataPtr i32, total i32, offsPtr i32, n i32, needlePtr i32, needleLen i32, outPtr i32) -> i32`
- what: byte-exact full-row equality (u8 0/1)
- returns `i32` (number): 0 ok; -1 null; -2 non-UTF-8 needle (even when n == 0); -3 malformed offsets / invalid UTF-8 row
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant, row-independent, chunkable
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-UTF-8 needle
    - `-3` = MALFORMED: bad offsets / invalid UTF-8 row
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - length is in CODE POINTS, not bytes
    - non-monotone or out-of-range offs -> -3 MALFORMED, not a trap (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `data` | i32 | in | `total` | input buffer of Uint8Array, total lane(s) |
| 1 | `total` | i32 | scalar | `-` | CAPACITY in lanes; a capacity, NOT a checksum |
| 2 | `offs` | i32 | in | `n + 1` | input buffer of Int32Array, n + 1 lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `needle` | i32 | in | `needle_len` | input buffer of Uint8Array, needle_len lane(s) |
| 5 | `needle_len` | i32 | scalar | `-` | needle length in BYTES |
| 6 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

#### `nf_text_length` &nbsp;*(5 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1738` (`#[no_mangle]`), origin: literal
- rust: `nf_text_length(data: *const u8, total: usize, offs: *const i32, n: usize, out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32) -> i32`
- js: `(dataPtr i32, total i32, offsPtr i32, n i32, outPtr i32) -> i32`
- what: code-point length per row
- returns `i32` (number): 0 ok; -1 null; -2 non-UTF-8 needle (even when n == 0); -3 malformed offsets / invalid UTF-8 row
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant, row-independent, chunkable
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-UTF-8 needle
    - `-3` = MALFORMED: bad offsets / invalid UTF-8 row
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - length is in CODE POINTS, not bytes
    - non-monotone or out-of-range offs -> -3 MALFORMED, not a trap (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `data` | i32 | in | `total` | input buffer of Uint8Array, total lane(s) |
| 1 | `total` | i32 | scalar | `-` | CAPACITY in lanes; a capacity, NOT a checksum |
| 2 | `offs` | i32 | in | `n + 1` | input buffer of Int32Array, n + 1 lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_text_startswith` &nbsp;*(7 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1811` (`#[no_mangle]`), origin: literal
- rust: `nf_text_startswith(data: *const u8, total: usize, offs: *const i32, n: usize, needle: *const u8, needle_len: usize, out: *mut u8) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(dataPtr i32, total i32, offsPtr i32, n i32, needlePtr i32, needleLen i32, outPtr i32) -> i32`
- what: anchored prefix hit per row (u8 0/1)
- returns `i32` (number): 0 ok; -1 null; -2 non-UTF-8 needle (even when n == 0); -3 malformed offsets / invalid UTF-8 row
- BigInt: no
- allocates: no
- result ownership: caller
- concurrency: reentrant, row-independent, chunkable
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT
    - `-2` = BAD_RANGE: non-UTF-8 needle
    - `-3` = MALFORMED: bad offsets / invalid UTF-8 row
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
    - length is in CODE POINTS, not bytes
    - non-monotone or out-of-range offs -> -3 MALFORMED, not a trap (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `data` | i32 | in | `total` | input buffer of Uint8Array, total lane(s) |
| 1 | `total` | i32 | scalar | `-` | CAPACITY in lanes; a capacity, NOT a checksum |
| 2 | `offs` | i32 | in | `n + 1` | input buffer of Int32Array, n + 1 lane(s) |
| 3 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 4 | `needle` | i32 | in | `needle_len` | input buffer of Uint8Array, needle_len lane(s) |
| 5 | `needle_len` | i32 | scalar | `-` | needle length in BYTES |
| 6 | `out` | i32 | out | `n` | output buffer of Uint8Array, n lane(s) (overwritten) |

### 4.25 `unique` - 3 symbols

#### `nf_unique_dict_utf8` &nbsp;*(10 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1890` (`#[no_mangle]`), origin: literal
- rust: `nf_unique_dict_utf8(data: *const u8, data_len: usize, offs: *const i32, offs_len: usize, valid: *const u8, n: usize, codes_out: *mut i32, uniq_data_out: *mut u8, uniq_offs_out: *mut i32, ng_out: *mut i32) -> i32`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32`
- js: `(dataPtr i32, dataLen i32, offsPtr i32, offsLen i32, validPtr i32, n i32, codesOutPtr i32, uniqDataOutPtr i32, uniqOffsOutPtr i32, ngOutPtr i32) -> i32`
- what: UTF-8 dictionary dedup over Arrow-like buffers -> codes + sorted unique bytes + offsets
- returns `i32` (number): 0 ok; -1 null pointer, offs_len != n+1, n > i32::MAX, data_len > i32::MAX, or a bad-offset failure from the kernel
- BigInt: no
- allocates: YES - HashMap<u32, Vec<(usize,usize,i32)>> plus several Vec (unique.rs:159-235). Not in linear memory.
- result ownership: kernel owns its temporaries and frees them; the caller owns uniq_data_out/uniq_offs_out
- concurrency: reentrant (all state is call-local)
- return codes:
    - `0` = ok
    - `-1` = NULL_OR_ABORT (also reused for every geometry rejection, see HAZARD H7)
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)
- **non-obvious contract** (resolved from source, not ambiguous): valid may be NULL meaning 'all rows valid' - the only export with an OPTIONAL pointer. The Rust body tests valid.is_null() explicitly, so this is certain from the source, but a wrapper must pass 0 rather than a real buffer when there is no validity sidecar.

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `data` | i32 | in | `data_len` | input buffer of Uint8Array, data_len lane(s) |
| 1 | `data_len` | i32 | scalar | `-` | byte length of the concatenated row data |
| 2 | `offs` | i32 | in | `offs_len` | input buffer of Int32Array, offs_len lane(s) |
| 3 | `offs_len` | i32 | scalar | `-` | offsets length in lanes; must equal n+1 |
| 4 | `valid` | i32 | in | `n` | input buffer of Uint8Array, n lane(s) |
| 5 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 6 | `codes_out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 7 | `uniq_data_out` | i32 | out | `data_len` | output buffer of Uint8Array, data_len lane(s) (overwritten) |
| 8 | `uniq_offs_out` | i32 | out | `n + 1` | output buffer of Int32Array, n + 1 lane(s) (overwritten) |
| 9 | `ng_out` | i32 | scalar | `-` | output buffer of Int32Array, unspecified lane(s) (overwritten) |

#### `nf_unique_inverse_i32` &nbsp;*(8 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1418` (`#[no_mangle]`), origin: literal
- rust: `nf_unique_inverse_i32(keys: *const i32, n: usize, uniq_out: *mut i32, inv_out: *mut i32, perm: *mut i32, tmp0: *mut u32, tmp1: *mut u32, tmp_p: *mut i32) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32) -> i64`
- js: `(keysPtr i32, n i32, uniqOutPtr i32, invOutPtr i32, permPtr i32, tmp0Ptr i32, tmp1Ptr i32, tmpPPtr i32) -> bigint`
- what: sorted-order unique + inverse codes of int32 keys (uniq[inv] == keys)
- returns `i64` (bigint): ng (>= 0); -2 NULL_STATE; -3 n > i32::MAX
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller allocates 5 scratch buffers of n lanes
- concurrency: single-threaded
- return codes:
    - `0+` = ng = distinct values
    - `-2` = NULL_STATE
    - `-3` = MALFORMED: n > i32::MAX
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of Int32Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `uniq_out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 3 | `inv_out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 4 | `perm` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 5 | `tmp0` | i32 | out | `n` | output buffer of Uint32Array, n lane(s) (overwritten) |
| 6 | `tmp1` | i32 | out | `n` | output buffer of Uint32Array, n lane(s) (overwritten) |
| 7 | `tmp_p` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

#### `nf_unique_inverse_i64` &nbsp;*(8 args, 0 wrapped/unwrapped)*

- source: `src/lib.rs:1462` (`#[no_mangle]`), origin: literal
- rust: `nf_unique_inverse_i64(keys: *const i64, n: usize, uniq_out: *mut i64, inv_out: *mut i32, perm: *mut i32, tmp0: *mut u64, tmp1: *mut u64, tmp_p: *mut i32) -> i64`
- wasm: `(i32, i32, i32, i32, i32, i32, i32, i32) -> i64`
- js: `(keysPtr i32, n i32, uniqOutPtr i32, invOutPtr i32, permPtr i32, tmp0Ptr i32, tmp1Ptr i32, tmpPPtr i32) -> bigint`
- what: int64-key variant of nf_unique_inverse_i32 (codes stay int32)
- returns `i64` (bigint): ng (>= 0); -2 NULL_STATE; -3 n > i32::MAX
- BigInt: **required** - RETURN is BigInt
- allocates: no
- result ownership: caller
- concurrency: single-threaded
- return codes:
    - `0+` = ng
    - `-2` = NULL_STATE
    - `-3` = MALFORMED: n > i32::MAX
- can trap:
    - a negative or > 2^31-1 length arrives as a huge usize -> slice::from_raw_parts -> TRAP (MEASURED)

| # | name | wasm | dir | len | role |
|---|---|---|---|---|---|
| 0 | `keys` | i32 | in | `n` | input buffer of BigInt64Array, n lane(s) |
| 1 | `n` | i32 | scalar | `-` | element count (rows / lanes) |
| 2 | `uniq_out` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 3 | `inv_out` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 4 | `perm` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |
| 5 | `tmp0` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 6 | `tmp1` | i32 | out | `n` | output buffer of BigInt64Array, n lane(s) (overwritten) |
| 7 | `tmp_p` | i32 | out | `n` | output buffer of Int32Array, n lane(s) (overwritten) |

---

## 5. Aggregates

### Symbols that need a JS `BigInt` somewhere

21 of 86 in a fresh build; 20 of 85 in the shipped artefact (the difference is
`nf_pair_insert_i64`). `errors.ts` measures that a JS `Number` is **rejected** where the ABI wants
an `i64`.

| kind | count | symbols |
|---|---|---|
| `i64` **return** only | 14 | `nf_carry_build_f64`, `nf_carry_build_i64`, `nf_cost_intern`, `nf_join_fused_inner_i32`, `nf_pair_insert_i64`*, `nf_select_count`, `nf_select_scatter_f32`, `_f64`, `_i32`, `_i64`, `_u8`, `nf_sorted_run_f64`, `nf_sorted_run_i64`, `nf_unique_inverse_i32`, `nf_unique_inverse_i64` |
| `i64` **parameter** only | 4 | `nf_ghash_fp_i64` (`pmask`), `nf_rng_compat_sample` (`n`), `nf_rng_fill_i32` and `nf_rng_fill_f64` (`seed`, `stream`, `offset`) |
| both | 2 | `nf_rng_permutation`, `nf_rng_sample_no_replace` |

\* not in the shipped artefact.

`seed`, `stream` and `offset` are `u64` counters, so they arrive as `bigint` even though their
values are small. A wrapper that passes them as `Number` fails.

### Symbols that allocate

Six. All others are caller-buffer-in / caller-buffer-out with no allocation on the kernel path.

| symbol | allocates | who owns the result |
|---|---|---|
| `nf_cost_intern` | `HashMap<Vec<u32>, u32>` with capacity `n`, plus one `Vec` per distinct row (`cost.rs:250-262`) | kernel frees its temporaries; caller owns `ids[0..n]`, `uniq[0..total]` |
| `nf_router_route` | `vec![INF; n]` + a `BinaryHeap` (`router.rs:44-46`) | kernel frees both; caller owns three 1-lane output buffers |
| `nf_unique_dict_utf8` | `HashMap<u32, Vec<(usize,usize,i32)>>` + several `Vec` (`unique.rs:159-235`) | kernel frees its temporaries; caller owns `uniq_data_out`, `uniq_offs_out` |
| `nf_sssp_csr`, `nf_sssp_csr_pred` | `BinaryHeap::with_capacity(min(V,1024))` per call (`sssp.rs:91`) | kernel frees it; caller owns `dist[0..V]` (+ `pred[0..V]`) |
| `nf_sssp_batch` | one `BinaryHeap` per source, plus `Vec` chunking | kernel frees them; caller owns `out[0..k*V]` |

No symbol allocates into the caller's linear memory, and nothing allocates in a `static`. A
`grep` for `static` / `static mut` / `Atomic*` / `OnceLock` / `thread_local` across the crate
returns **nothing**, so every symbol is reentrant with respect to caller buffers. The one
deliberate exception is `nf_group_owner_2i32_1f64`, whose contract is *disjoint* `[lo, hi)` group
ranges over shared state with no atomics and no merge - it exists to be called concurrently.

### `nthreads` and `nf_sssp_batch`

`nthreads` is a **hint, not part of any contract**: clamped to `[1, 64]`, `0` means `1`. It is the
last argument of 12 symbols. On `wasm32` `nf_sssp_batch` ignores it and loops the same shards
inline (`sssp.rs:225-231`), so it is the only multi-source symbol that works there. Every other
`nthreads`-taking symbol is in the trap list below.

---

## 6. ABI hazards found, and deliberately NOT fixed

A list of hazards is the deliverable; a patch is not. Nothing below was changed. All of it is
recorded in `census.json` -> `_meta.hazards` with locations.

### H1 - CRITICAL - `nf_sssp_csr_pred`: an output shipped as an input

The 8th argument is written as `4 * (np-1)` bytes of i32 predecessor data. The shipped wrapper hands
it a buffer sized for a per-edge predicate byte array, so the write lands past the buffer and over
the `dist` output that follows it.

- where: `src/lib.rs:2874` (the `#[no_mangle]`), `2875-2904` (the body); `src/sssp.rs:79-90, 110-112`
- evidence: measured on the shipped artefact - `rc = 0`, `dist = [0xFFFFFFFF x3]` instead of `[0, 2, 7]`
- consequence: silently wrong distances with a success code. No trap, so `errors.ts` cannot catch it.

### H2 - HIGH - the shipped artefact is one symbol behind the source

Section 2, in full. `nf_pair_insert_i64` is declared ungated and missing from the artefact; the
export and i64 counts are both stale; `build.mjs`'s own guard turns into a build failure on a clean
clone.

### H3 - HIGH - there is no bounds check anywhere at the FFI boundary

`core::buffers::borrow` / `borrow_mut` are raw `slice::from_raw_parts{,_mut}` (`buffers.rs:16-37`).
Every length, capacity and range is caller-enforced; the kernels validate almost none of them.

- evidence: measured - `nf_mask_not(pA, -1, pOut)` traps with `RuntimeError: unreachable`, because a
  negative `i32` becomes a ~4.29e9-element slice. `errors.ts` documents the same class as
  "63 of the 85 kernels".
- consequence: a wrong length is undefined behaviour natively, and on `wasm32` either a trap or -
  if the address stays inside linear memory - **silent corruption with a success code**. There is
  no middle path, which is why argument order in this census is not a style question.

### H4 - HIGH - 11 exports are dead on the `wasm32` surface: they always trap

`join.rs`, `join_i64.rs` and `groupby/hash_grouped.rs` call `std::thread::scope` + `scope.spawn`
with **no `#[cfg(target_arch = "wasm32")]` fallback**. Only `sssp.rs` has one.
`wasm32-unknown-unknown` cannot spawn, and `Cargo.toml` sets `panic = "abort"`.

| trapped on `wasm32` | body |
|---|---|
| `nf_join_probe`, `nf_join_gather_i32`, `nf_join_fused_left_i32`, `nf_join_fused_inner_i32` | `join.rs:182, 233, 339, 496` |
| `nf_join_probe_i64` | `join_i64.rs:161` |
| `nf_ghash_fp_i64`, `nf_ghash_count_i64`, `nf_ghash_scatter_i64`, `nf_ghash_pins_i64`, `nf_ghash_occ_count`, `nf_ghash_occ_fill` | `hash_grouped.rs:121, 179, 287, 444, 524, 621` |

- measured traps (`RuntimeError: unreachable`) on `nf_join_probe`, `nf_join_gather_i32`,
  `nf_join_fused_left_i32`, `nf_ghash_fp_i64`, `nf_ghash_occ_count`; the other six have the
  identical call shape. `nf_join_build` (single-threaded) and `nf_sssp_batch` (wasm-gated) both
  return `0`.
- consequence: these 11 are reachable through `callRaw` and will trap on every non-empty call.
  Nothing in the source or the wasm comments says so, and the comments say the opposite
  ("no wasm export", "Native-only"). They are perfectly usable natively.

### H5 - MEDIUM - a pointer-to-pointer crosses the ABI as a single `i32`

`nf_rowwise_kway_time_argmin_gather` takes `*const *const i32` and `*const *const f32`. In wasm
these are single `i32`s: the **address** of an offset array. A wrapper that writes lane data there,
or that passes the offset array itself, is wrong in a way no type check will catch. It is the only
exported pointer-to-pointer, and no length accompanies it.

### H6 - MEDIUM - three symbols derive their row count from a data value

`nf_ghash_pins_i64` takes `n = starts[p]`; `nf_ghash_occ_count` takes `m = toff[p]`;
`nf_ghash_occ_fill` takes `m = toff[p]` and `nocc = ostart[p]`. **There is no `n` or `m` parameter
at all** (`lib.rs:2401, 2431, 2463-2464`). Reading these as `(buffer, count)` pairs - which is what
the names suggest, and what every other `*_i64` symbol actually is - is precisely the convention
error that left 69 kernels unwrapped.

### H7 - MEDIUM - `-1` and `-2` are overloaded: they do not mean "you passed null"

| symbol | `-1` / `-2` also means |
|---|---|
| `nf_group_owner_2i32_1f64` | `-1` for `lo > hi` or `hi > g` (`lib.rs:227-238`) |
| `nf_unique_dict_utf8` | `-1` for `offs_len != n+1`, `n > i32::MAX`, `data_len > i32::MAX`, and a bad-offset failure from the kernel (`lib.rs:1903-1935`) |
| `nf_rng_sample_no_replace` | `-2` for `k > n` |

A wrapper that renders `-1` as "null pointer" states a falsehood about a range error.
`errors.ts` resolves codes through a per-symbol table precisely to avoid this, so the table has to
carry the second meaning - an incomplete table is worse than a bare number.

### H8 - MEDIUM - argument order follows no pointer-then-length convention

Concrete counter-examples, all in the census with their borrow-lengths:

- `nf_pack_sum_count_i32` / `_f64`: the scalar `m2` comes **before** `n`, so the reading "pointer,
  then its length" puts a multiplier where a row count belongs
- `nf_select_scatter_*`: `(src, mask, n, out)` - two inputs precede `n`, the output is last
- `nf_segment_reduce_*`: `(values, n, bounds, m, op, out)`
- `nf_pattern_encode`: interleaved `(ptr, len)` pairs, ending with two 1-lane scalar-out pointers
- `nf_router_route`: the vertex count `n` is **argument 0**, before every pointer
- worst offenders by non-adjacent pointer count: `nf_rowwise_min4_argmin_gather` and
  `nf_rowwise_min4_time_argmin_gather` (10 of 12), `nf_join_fused_left_i32` (9 of 13),
  `nf_ghash_pins_i64` and `nf_group_owner_2i32_1f64` (8 each)

Across the whole surface: **346 pointer parameters, 35 adjacent (10%), 86 of 86 symbols with at
least one non-adjacent pointer.**

### H9 - LOW - source comments contradict the build for fifteen symbols

"no wasm export", "Native-only" and "no WASM export (native CPU research surface)" appear above
`nf_join_build` / `_probe` / `_gather_i32` / `_fused_left_i32` / `_fused_inner_i32`,
`nf_join_build_i64` / `_probe_i64`, `nf_sort_perm_i32` / `_i64`, `nf_unique_inverse_i32` / `_i64`,
`nf_unique_dict_utf8`, `nf_segment_count` / `_reduce_f32` / `_reduce_i32`,
`nf_adjacency_slice` / `_gather`, `nf_cost_travel_batch` / `_intern`, all six `nf_ghash_*`,
`nf_rowwise_min4_argmin_gather` / `_min4_time_argmin_gather` /
`_kway_time_argmin_gather`, `nf_bounded_select_2i32`, `nf_sssp_csr` / `_csr_pred` / `_batch` - all
of which **are** exported by the shipped artefact.

A reader deciding what is reachable from those comments reaches the opposite of the truth for 15
symbols, and the one place a reader would look to confirm `nf_pair_insert_i64`'s status is the
comment that is accidentally right.

---

## 7. Ambiguity: what is left, and what it would cost

**Nothing is ambiguous.** All 86 symbols are resolved with certainty, and the resolution is not a
judgement call:

1. Pointer-vs-length is read from the `borrow()` / `borrow_mut()` call in that symbol's own body.
   That is the literal length the kernel will use, not an inference.
2. Argument **order** and **value type** are then confirmed positionally against the compiler's own
   `FuncType` for every argument of every symbol - 86/86, 0 mismatches.
3. Role, direction, trap behaviour, allocation, ownership and concurrency come from the bodies, the
   doc comments and `core/errors.rs`.

Three symbols carry a contract that is **non-obvious but resolved**, and are flagged in the census
under `ambiguity.resolved-from-source` rather than left as guesses:

| symbol | the surprise | resolved from |
|---|---|---|
| `nf_sssp_csr_pred` | argument 7 is an i32 **output** of `V` lanes, not a per-edge input | `lib.rs:2885-2903`, `sssp.rs:79-90`; reproduced against the artefact |
| `nf_unique_dict_utf8` | `valid` may legitimately be **NULL**, meaning "all rows valid" - the only export with an optional pointer | `lib.rs:1923-1927` tests `valid.is_null()` explicitly |
| `nf_rowwise_min4_time_argmin_gather` | `d0..d3` are **all-or-nothing**: all four non-null is the gather path, all four null is the time-only path, and `d_best` must be null on that second path. Mixed nulls return `-1` | `lib.rs:2588-2622` tests `d_all` / `d_none` explicitly |

### What resolving them would have cost, had they stayed open

The cheap answer is that nothing would have shipped. Each of the three would have produced a
plausible-looking wrapper and a wrong result:

- `nf_sssp_csr_pred` guessed as an input **did** ship, and reproduces as silently wrong `dist` with
  `rc = 0` (H1). That is the measured cost of one unresolved symbol.
- `nf_unique_dict_utf8` guessed as "a required `n`-lane input" would reject every call made without a
  validity sidecar - which is every call the doc says is legal - or would read a null buffer.
- `nf_rowwise_min4_time_argmin_gather` guessed as "4 required payload pointers" would trap on the
  time-only path instead of returning `0`.

Settling any future ambiguity needs one of, in increasing order of cost:

1. **the borrow length in the body** - free, and already done for all 86
2. **a `FuncType` read** - free for order and types; `ts/wasm-info.mjs` already does this and
   `build.mjs` already checks arity and return type per symbol
3. **one direct call against the artefact** - the method used for all six claims in this census that
   the source alone could not settle (H1, H4, `-2` on `nf_cost_travel_batch`, `-3` on
   `nf_adjacency_gather`, the 86-vs-85 count, the stale-artefact hash). This costs no timing and
   no benchmark: it is a handful of tiny buffers and a return code.

There is no remaining item that needs a benchmark, a profiler or a judgement call.

---

## 8. Reproducing this census

Nothing here is generated at build time and nothing depends on it. To re-derive:

1. **enumerate** - `grep -n '#\[no_mangle\]' numfast-native/src/lib.rs numfast-native/src/router.rs`,
   then resolve each of the six `macro_rules!` templates against its invocations. `lib.rs` reports 69
   `#[no_mangle]` lines: 63 literal functions + 6 templates.
2. **read the lengths** - in each `extern "C"` body, the second argument of every
   `borrow(...)` / `borrow_mut(...)`. That is the length table.
3. **confirm against the compiler** - build for wasm and read the export section:
   `node numfast-native/ts/wasm-info.mjs <artefact>.wasm` prints the export count and the i64 list;
   `describe()` returns every `FuncType`. Compare param-by-param, positionally.
4. **settle behaviour by direct call** - only for claims the source leaves open, and only with tiny
   inputs so the shared box is not disturbed. No timing was taken for this census.
