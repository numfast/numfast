// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// PER-SYMBOL ABI TABLE -- the thing the WASM signature cannot express.
//
// A `FuncType` says "seven i32 in, one i32 out". It does not say which i32 is
// a pointer, which is a length, which is an enum, or what a non-zero return
// code MEANS. Acceptance criterion W4 was BLOCKED precisely because that
// mapping exists only in prose inside the Rust source. This file is that
// mapping, for the surface this package actually wraps, and
// `test/abi.test.mjs` checks every `wasm` field against the bytes of the
// shipped .wasm -- so the table cannot drift away from the artefact silently.
//
// SCOPE, STATED: 17 of 85 kernels. The other 68 are listed by name in
// `UNWRAPPED_SYMBOLS` (kernels.ts) and are NOT reachable from this package
// except through `callRaw`, which is explicitly marked unvalidated.
//
// SOURCES, per row: the `#[no_mangle] extern "C"` signature in
// numfast-native/src/lib.rs, plus the doc comment on the same symbol, plus
// numfast-native/src/core/errors.rs for the code meanings. Nothing here is
// inferred from the .wasm; the .wasm is only used to CHECK it.

/** How a scalar argument is passed. */
export type ScalarKind =
  | "u32"      // length, count, enum, or plain integer
  | "i32"      // plain integer (never a length)
  | "f64"      // float scalar, widened to f64 by the caller
  | "i64";     // BigInt on the JS side; a Number is rejected

export interface AbiEntry {
  /** The `nf_*` export name. */
  readonly symbol: string;
  /** Parameters in ABI order, tagged. Verified against `FuncType.params`. */
  readonly wasm: readonly string[];
  /** Kernel result type. Verified against `FuncType.results`. */
  readonly returns: string;
  /** Non-zero return codes and what they mean FOR THIS SYMBOL.
   *  A code absent from this list is not "unknown": errors.ts says so. */
  readonly codes: Readonly<Record<number, string>>;
  /** One line a caller needs and the signature does not give. */
  readonly note: string;
}

/** Frozen by `numfast-native/src/core/errors.rs`: "do not renumber". */
export const RC_OK = 0;
export const RC_NULL_OR_ABORT = -1;
export const RC_BAD_RANGE = -2;
export const RC_MALFORMED = -3;

const PTR_NULL = "null pointer";
const NULL_RANGE = "null pointer or out-of-range argument";

export const ABI: Readonly<Record<string, AbiEntry>> = {
  // ---- graph kernels (the pre-existing bridge surface) -------------------
  nf_sssp_csr: {
    symbol: "nf_sssp_csr",
    wasm: ["indptr", "np", "indices", "weights", "e", "source", "dist"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "source out of range, or indptr not monotonic" },
    note: "CSR + weights, one source. dist is u32 with Q_INF (0xffffffff) for unreachable.",
  },
  nf_sssp_csr_pred: {
    symbol: "nf_sssp_csr_pred",
    wasm: ["indptr", "np", "indices", "weights", "e", "source", "dist", "pred"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "source out of range, or indptr not monotonic" },
    note: "nf_sssp_csr plus a per-edge predicate byte; pred[i]==0 skips edge i.",
  },
  nf_sssp_batch: {
    symbol: "nf_sssp_batch",
    wasm: ["indptr", "np", "indices", "weights", "e", "sources", "k", "out", "nthreads"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "k == 0, or a source out of range" },
    note: "sources[k] -> out[k*np] row-major. The wasm build forces nthreads=1; it is a hint, not a contract.",
  },
  nf_cost_travel_batch: {
    symbol: "nf_cost_travel_batch",
    wasm: ["dist", "speed", "k", "n", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL },
    note: "out[i] = (dist[i]*k[i] + speed[i]/2) / speed[i]; speed[i]==0 yields Q_INF, never a division fault.",
  },
  nf_cost_intern: {
    symbol: "nf_cost_intern",
    wasm: ["vecs", "n", "width", "ids", "uniq", "total"],
    returns: "i64",
    codes: {},
    note: "Row-major vecs[n*width] -> ids[n] + uniq[ng*width]. Returns the group count ng (an i64, so a BigInt in JS), NOT a status code.",
  },
  nf_rowwise_kway_time_argmin_gather: {
    symbol: "nf_rowwise_kway_time_argmin_gather",
    wasm: ["tPtrs", "dPtrs", "k", "n", "tBest", "dBest", "mBest"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "k outside 1..=256" },
    note: "k lanes of (i32 time selector, f32 payload); tPtrs/dPtrs are u32 arrays OF OFFSETS, not lane data.",
  },
  nf_adjacency_slice: {
    symbol: "nf_adjacency_slice",
    wasm: ["indptr", "np", "indices", "e", "query", "k", "begins", "ends"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "a query id has no node" },
    note: "query[k] -> the half-open [begins, ends) row slice in `indices`.",
  },
  nf_adjacency_gather: {
    symbol: "nf_adjacency_gather",
    wasm: ["indices", "e", "begins", "ends", "k", "out", "total"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "sum(ends-begins) != total" },
    note: "Flattens the k slices into out[total]. total is a CAPACITY, and must equal the sum.",
  },

  // ---- elementwise map (numfast-native/src/series/map.rs) -----------------
  nf_map_i32: {
    symbol: "nf_map_i32",
    wasm: ["a", "b", "n", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code outside 0..=6 for this symbol" },
    note: "op: 0 add, 1 sub, 2 mul, 3 div, 4 pow, 5 floor_div, 6 mod. Integer lanes are bit-exact; div/pow round-trip through f64.",
  },
  nf_map_scalar_i32: {
    symbol: "nf_map_scalar_i32",
    wasm: ["a", "n", "s", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code outside 0..=6 for this symbol" },
    note: "Integer scalar `s`, kept in-lane: add/sub/mul wrap mod 2^32.",
  },
  nf_map_fscalar_i32: {
    symbol: "nf_map_fscalar_i32",
    wasm: ["a", "n", "s", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code outside 0..=4 for this symbol" },
    note: "Float scalar `s` (f64, never pre-truncated). floor_div/mod are REJECTED here: NumPy widens them to f64, so use nf_map_scalar_f64.",
  },
  nf_map_f32: {
    symbol: "nf_map_f32",
    wasm: ["a", "b", "n", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code is div or pow for this symbol" },
    note: "f32 add/sub/mul/floor_div/mod. div and pow widen to f64 output and live in nf_map_f32_divpow.",
  },
  nf_map_f32_divpow: {
    symbol: "nf_map_f32_divpow",
    wasm: ["a", "b", "n", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code is not div or pow for this symbol" },
    note: "f32 div/pow with f64 output (the CPU astype(f64) round-trip).",
  },
  nf_map_scalar_f32: {
    symbol: "nf_map_scalar_f32",
    wasm: ["a", "n", "s", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code is div or pow for this symbol" },
    note: "f64 scalar demoted to f32 first (NumPy value-based casting).",
  },
  nf_map_scalar_f32_divpow: {
    symbol: "nf_map_scalar_f32_divpow",
    wasm: ["a", "n", "s", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code is not div or pow for this symbol" },
    note: "f32 scalar div/pow with f64 output.",
  },
  nf_map_f64: {
    symbol: "nf_map_f64",
    wasm: ["a", "b", "n", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code outside 0..=6 for this symbol" },
    note: "All seven ops on f64. div/pow/floor_div/mod follow the CPython divmod algorithm, which is what NumPy implements.",
  },
  nf_map_scalar_f64: {
    symbol: "nf_map_scalar_f64",
    wasm: ["a", "n", "s", "op", "out"],
    returns: "i32",
    codes: { [RC_NULL_OR_ABORT]: PTR_NULL, [RC_BAD_RANGE]: "op code outside 0..=6 for this symbol" },
    note: "All seven ops, f64 scalar. This is the symbol an i32 lane with a FLOAT scalar must use for floor_div/mod.",
  },
};

/** symbol -> { code -> message }, for errors.ts. Codes with no entry are
 *  reported as "not in this package's table", never as a bare number. */
export const RC_TABLE: Readonly<Record<string, Readonly<Record<number, string>>>> =
  Object.freeze(Object.fromEntries(
    Object.values(ABI).map((e) => [e.symbol, Object.freeze({ ...e.codes })])));

/** The map op codes, as frozen in `numfast-native/src/series/map.rs`. */
export const MAP_OP = Object.freeze({
  add: 0, sub: 1, mul: 2, div: 3, pow: 4, floorDiv: 5, mod: 6,
} as const);

export type MapOpName = keyof typeof MAP_OP;