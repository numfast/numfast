// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// TYPED KERNELS -- the whole wrapped surface of this package.
//
// One function per kernel, each of which:
//   * validates every length BEFORE the call (prevention, not catch);
//   * allocates its own buffers above the module's static data;
//   * resolves a non-zero return code through the per-symbol table in abi.ts;
//   * converts a `WebAssembly.RuntimeError` into `NumFastTrap`;
//   * returns a COPY of the output, never a view, because on any failure the
//     kernel's contract is "outputs past the abort point are caller-owned
//     garbage" and a view would let a caller read that garbage as a result.
//
// COVERAGE IS A CONSTANT, NOT A README PARAGRAPH. `WRAPPED` and
// `TOTAL_EXPORTS` are exported, so `WRAPPED.length` is checkable from a test.

import type { Bridge, WasmExports } from "./bridge.js";
import {
  NumFastError,
  callGuarded, callTrapped, checkLengths, NumFastArgumentError, requireBigInt, _setRcTable,
} from "./errors.js";
import { MAP_OP, RC_TABLE, type MapOpName } from "./abi.js";

_setRcTable(RC_TABLE);

/** Sentinel the kernels write for "unreachable" / "no speed". */
export const Q_INF = 0xffffffff;

/** Every kernel this package wraps, by export name. */
export const WRAPPED = Object.freeze([
  "nf_sssp_csr", "nf_sssp_csr_pred", "nf_sssp_batch",
  "nf_cost_travel_batch", "nf_cost_intern",
  "nf_rowwise_kway_time_argmin_gather",
  "nf_adjacency_slice", "nf_adjacency_gather",
  "nf_map_i32", "nf_map_scalar_i32", "nf_map_fscalar_i32",
  "nf_map_f32", "nf_map_f32_divpow", "nf_map_scalar_f32", "nf_map_scalar_f32_divpow",
  "nf_map_f64", "nf_map_scalar_f64",
] as const);

export type WrappedSymbol = (typeof WRAPPED)[number];

/** Function exports in the .wasm this package is built against. Asserted
 *  against the artefact by test/abi.test.mjs: if a kernel is added the test
 *  fails here rather than this number quietly going stale. */
export const TOTAL_EXPORTS = 85;

/** Call one kernel by name, unvalidated: no length checking, no BigInt
 *  coercion, only the shared return-code table and trap conversion. Present
 *  so the 68 unwrapped kernels are REACHABLE -- with the symbol named at the
 *  call site -- instead of hidden, and so widening coverage is one line. */
export function callRaw(
  bridge: Bridge,
  symbol: string,
  ...args: readonly (number | bigint)[]
): number | bigint {
  const fn = (bridge.ex as unknown as Record<string, unknown>)[symbol];
  if (typeof fn !== "function") {
    throw new NumFastArgumentError(symbol,
      `not an export of this build. It has ${TOTAL_EXPORTS} function exports; ` +
      `the ${WRAPPED.length} with a documented ABI have typed wrappers.`);
  }
  const rc = callGuarded<number | bigint>(
    symbol, fn as (...a: never[]) => unknown, ...(args as number[]));
  return rc;
}

/** The raw exports, for a caller who wants offsets themselves. */
export function exportsOf(bridge: Bridge): WasmExports { return bridge.ex; }

// ---------------------------------------------------------------------------
// graph kernels
// ---------------------------------------------------------------------------

/** CSR + weights, one source -> u32 distances, Q_INF where unreachable. */
export function ssspCsr(
  bridge: Bridge,
  indptr: Uint32Array, indices: Uint32Array, weights: Uint32Array, source: number,
): Uint32Array {
  const sym = "nf_sssp_csr";
  checkLengths(sym, indptr.length, indices.length, weights.length);
  const np = indptr.length, v = np - 1;
  if (source < 0 || source >= v) {
    throw new NumFastArgumentError(sym, `source ${source} outside 0..${v - 1}`);
  }
  bridge.reset();
  const pInd = bridge.put(indptr), pIx = bridge.put(indices), pW = bridge.put(weights);
  const pOut = bridge.alloc(v * 4, 8);
  bridge.ensure(pOut + v * 4);
  callGuarded<number>(sym, bridge.ex.nf_sssp_csr, pInd, np, pIx, pW, indices.length, source, pOut);
  bridge.assertGuard();
  return bridge.u32(pOut, v).slice();
}

/** CSR + per-edge predicate byte (`pred[i] === 0` skips edge i), one source. */
export function ssspCsrPred(
  bridge: Bridge,
  indptr: Uint32Array, indices: Uint32Array, weights: Uint32Array,
  source: number, pred: Uint8Array,
): Uint32Array {
  const sym = "nf_sssp_csr_pred";
  checkLengths(sym, indptr.length, indices.length, weights.length, pred.length);
  const np = indptr.length, v = np - 1;
  if (source < 0 || source >= v) {
    throw new NumFastArgumentError(sym, `source ${source} outside 0..${v - 1}`);
  }
  bridge.reset();
  const pInd = bridge.put(indptr), pIx = bridge.put(indices);
  const pW = bridge.put(weights), pP = bridge.put(pred);
  const pOut = bridge.alloc(v * 4, 8);
  bridge.ensure(pOut + v * 4);
  callGuarded<number>(sym, bridge.ex.nf_sssp_csr_pred,
    pInd, np, pIx, pW, indices.length, source, pOut, pP);
  bridge.assertGuard();
  return bridge.u32(pOut, v).slice();
}

/** sources[k] -> out[k*np] row-major. The wasm build forces `nthreads = 1`;
 *  it is a hint there, not part of the contract. */
export function ssspBatch(
  bridge: Bridge,
  indptr: Uint32Array, indices: Uint32Array, weights: Uint32Array, sources: Uint32Array,
): Uint32Array {
  const sym = "nf_sssp_batch";
  checkLengths(sym, indptr.length, indices.length, weights.length, sources.length);
  const v = indptr.length - 1, k = sources.length;
  bridge.reset();
  const pInd = bridge.put(indptr), pIx = bridge.put(indices);
  const pW = bridge.put(weights), pSrc = bridge.put(sources);
  const pOut = bridge.alloc(k * v * 4, 8);
  bridge.ensure(pOut + k * v * 4);
  callGuarded<number>(sym, bridge.ex.nf_sssp_batch,
    pInd, indptr.length, pIx, pW, indices.length, pSrc, k, pOut, 1);
  bridge.assertGuard();
  return bridge.u32(pOut, k * v).slice();
}

/** out[i] = (dist[i]*k[i] + speed[i]/2) / speed[i]; `speed[i] === 0` -> Q_INF. */
export function costTravelBatch(
  bridge: Bridge, dist: Uint32Array, speed: Uint32Array, k: Uint16Array,
): Uint32Array {
  const sym = "nf_cost_travel_batch";
  checkLengths(sym, dist.length, speed.length, k.length);
  if (speed.length !== dist.length || k.length !== dist.length) {
    throw new NumFastArgumentError(sym,
      `lane lengths differ: dist=${dist.length} speed=${speed.length} k=${k.length}`);
  }
  const n = dist.length;
  bridge.reset();
  const pD = bridge.put(dist), pS = bridge.put(speed), pK = bridge.put(k);
  const pOut = bridge.alloc(n * 4, 8);
  bridge.ensure(pOut + n * 4);
  callGuarded<number>(sym, bridge.ex.nf_cost_travel_batch, pD, pS, pK, n, pOut);
  bridge.assertGuard();
  return bridge.u32(pOut, n).slice();
}

/** Row-major `vecs[n*width]` -> per-row group ids + the interned rows.
 *
 *  `ng` is a COUNT, not a status code -- but it crosses an i64 parameter, so
 *  it arrives as a `bigint`. This is the only wrapped kernel on the BigInt
 *  side; 20 of the 85 exports do, and 19 of them are unwrapped here. */
export function costIntern(
  bridge: Bridge, vecs: Uint32Array, n: number, width: number,
): { ng: number; ids: Uint32Array; uniq: Uint32Array } {
  const sym = "nf_cost_intern";
  checkLengths(sym, n, width);
  if (n * width !== vecs.length) {
    throw new NumFastArgumentError(sym, `n*width = ${n * width} but ${vecs.length} lanes were given`);
  }
  bridge.reset();
  const pV = bridge.put(vecs);
  const pIds = bridge.alloc(n * 4, 8);
  const pUniq = bridge.alloc(n * width * 4, 8);
  bridge.ensure(pUniq + n * width * 4);
  // callTrapped, NOT callGuarded: this kernel answers with the group COUNT,
  // so a successful call returns 3 and the return-code check would read that
  // as a failure. Only the trap channel applies here.
  const ngBig = callTrapped<bigint>(sym, bridge.ex.nf_cost_intern,
    pV, n, width, pIds, pUniq, n * width);
  // The ABI declares the return as i64, so the engine hands back a BigInt.
  // Narrow it explicitly rather than letting it become a Number implicitly.
  requireBigInt(sym, "the returned group count", ngBig);
  const ng = Number(ngBig);
  if (!Number.isSafeInteger(ng) || ng < 0) {
    // A negative count is this kernel's failure channel. It is not one of the
    // frozen codes in errors.rs, so it gets its own message rather than being
    // reported as an undocumented code number.
    throw new NumFastError(sym, ng,
      "negative group count; the kernel reports failure for this symbol by " +
      "returning a negative value, not one of the frozen return codes");
  }
  bridge.assertGuard();
  return { ng, ids: bridge.u32(pIds, n).slice(), uniq: bridge.u32(pUniq, ng * width).slice() };
}

/** k lanes of (i32 time selector, f32 payload) -> best time, min payload,
 *  argmin mask. In the raw ABI `tPtrs`/`dPtrs` are arrays OF OFFSETS. */
export function rowwiseKwayTimeArgminGather(
  bridge: Bridge, tLanes: Int32Array[], dLanes: Float32Array[],
): { tBest: Int32Array; dBest: Float32Array; mBest: Uint8Array } {
  const sym = "nf_rowwise_kway_time_argmin_gather";
  const k = tLanes.length, n = k > 0 ? (tLanes[0] as Int32Array).length : 0;
  if (k < 1 || k > 256) throw new NumFastArgumentError(sym, `k=${k}, must be 1..=256`);
  if (dLanes.length !== k) {
    throw new NumFastArgumentError(sym, `${k} selector lanes but ${dLanes.length} payload lanes`);
  }
  for (let i = 0; i < k; i++) {
    const t = tLanes[i] as Int32Array, d = dLanes[i] as Float32Array;
    if (t.length !== n || d.length !== n) {
      throw new NumFastArgumentError(sym, `lane ${i} has ${t.length}/${d.length} elements, expected ${n}`);
    }
  }
  checkLengths(sym, k, n);
  bridge.reset();
  const tOffs = tLanes.map((t) => bridge.put(t));
  const dOffs = dLanes.map((d) => bridge.put(d));
  const pT = bridge.alloc(k * 4, 8), pD = bridge.alloc(k * 4, 8);
  const pTB = bridge.alloc(n * 4, 8), pDB = bridge.alloc(n * 4, 8), pMB = bridge.alloc(n, 8);
  bridge.ensure(pMB + n);
  const u8 = bridge.u8;
  const tw = new Uint32Array(u8.buffer as ArrayBuffer, pT, k);
  const dw = new Uint32Array(u8.buffer as ArrayBuffer, pD, k);
  for (let i = 0; i < k; i++) { tw[i] = tOffs[i] as number; dw[i] = dOffs[i] as number; }
  callGuarded<number>(sym, bridge.ex.nf_rowwise_kway_time_argmin_gather, pT, pD, k, n, pTB, pDB, pMB);
  bridge.assertGuard();
  return {
    tBest: bridge.i32(pTB, n).slice(),
    dBest: bridge.f32(pDB, n).slice(),
    mBest: bridge.u8v(pMB, n).slice(),
  };
}

/** query[k] -> the half-open [begins, ends) row slice in `indices`. */
export function adjacencySlice(
  bridge: Bridge, indptr: Uint32Array, indices: Uint32Array, query: Uint32Array,
): { begins: Uint32Array; ends: Uint32Array } {
  const sym = "nf_adjacency_slice";
  checkLengths(sym, indptr.length, indices.length, query.length);
  const k = query.length;
  bridge.reset();
  const pInd = bridge.put(indptr), pIx = bridge.put(indices), pQ = bridge.put(query);
  const pB = bridge.alloc(k * 4, 8), pE = bridge.alloc(k * 4, 8);
  bridge.ensure(pE + k * 4);
  callGuarded<number>(sym, bridge.ex.nf_adjacency_slice,
    pInd, indptr.length, pIx, indices.length, pQ, k, pB, pE);
  bridge.assertGuard();
  return { begins: bridge.u32(pB, k).slice(), ends: bridge.u32(pE, k).slice() };
}

/** Flatten k slices into out[total]; `total` must equal sum(ends - begins). */
export function adjacencyGather(
  bridge: Bridge, indices: Uint32Array, begins: Uint32Array, ends: Uint32Array, total: number,
): Uint32Array {
  const sym = "nf_adjacency_gather";
  checkLengths(sym, indices.length, begins.length, ends.length, total);
  if (ends.length !== begins.length) {
    throw new NumFastArgumentError(sym, `begins has ${begins.length} entries, ends has ${ends.length}`);
  }
  bridge.reset();
  const pIx = bridge.put(indices), pB = bridge.put(begins), pE = bridge.put(ends);
  const pOut = bridge.alloc(total * 4, 8);
  bridge.ensure(pOut + total * 4);
  callGuarded<number>(sym, bridge.ex.nf_adjacency_gather,
    pIx, indices.length, pB, pE, begins.length, pOut, total);
  bridge.assertGuard();
  return bridge.u32(pOut, total).slice();
}

// ---------------------------------------------------------------------------
// elementwise map -- numfast-native/src/series/map.rs
// ---------------------------------------------------------------------------

type Lane = Int32Array | Float32Array | Float64Array;
type VecFn5 = (a: number, b: number, n: number, op: number, out: number) => number;
type VecFn5s = (a: number, n: number, s: number, op: number, out: number) => number;

function opCode(sym: string, op: MapOpName): number {
  const c = MAP_OP[op];
  if (c === undefined) {
    throw new NumFastArgumentError(sym, `unknown op "${op}"; codes are frozen in series/map.rs`);
  }
  return c;
}

function copyOut(bridge: Bridge, off: number, n: number, bytes: number): Uint8Array {
  return new Uint8Array(bridge.mem.buffer as ArrayBuffer, off, n * bytes).slice();
}

function mapArray(
  sym: string, fn: VecFn5, bridge: Bridge,
  a: Lane, b: Lane, op: MapOpName, outBytes: 4 | 8,
): Int32Array | Float32Array | Float64Array {
  checkLengths(sym, a.length, b.length);
  if (a.length !== b.length) {
    throw new NumFastArgumentError(sym, `lane lengths differ: ${a.length} vs ${b.length}`);
  }
  const n = a.length;
  bridge.reset();
  const pA = bridge.put(a), pB = bridge.put(b);
  const pOut = bridge.alloc(n * outBytes, 8);
  bridge.ensure(pOut + n * outBytes);
  callGuarded<number>(sym, fn as (...x: never[]) => unknown, pA, pB, n, opCode(sym, op), pOut);
  bridge.assertGuard();
  const raw = copyOut(bridge, pOut, n, outBytes);
  if (outBytes === 8) return new Float64Array(raw.buffer);
  return a instanceof Int32Array && b instanceof Int32Array
    ? new Int32Array(raw.buffer) : new Float32Array(raw.buffer);
}

function mapScalar(
  sym: string, fn: VecFn5s, bridge: Bridge,
  a: Lane, s: number, op: MapOpName, outBytes: 4 | 8,
): Int32Array | Float32Array | Float64Array {
  checkLengths(sym, a.length);
  if (typeof s !== "number" || !Number.isFinite(s)) {
    throw new NumFastArgumentError(sym, `scalar ${String(s)} is not a finite f64`);
  }
  const n = a.length;
  bridge.reset();
  const pA = bridge.put(a);
  const pOut = bridge.alloc(n * outBytes, 8);
  bridge.ensure(pOut + n * outBytes);
  callGuarded<number>(sym, fn as (...x: never[]) => unknown, pA, n, s, opCode(sym, op), pOut);
  bridge.assertGuard();
  const raw = copyOut(bridge, pOut, n, outBytes);
  if (outBytes === 8) return new Float64Array(raw.buffer);
  return a instanceof Int32Array ? new Int32Array(raw.buffer) : new Float32Array(raw.buffer);
}

/** i32 lanes, array-array. Integer lanes are bit-exact; `div`/`pow` round-trip
 *  through f64 and `rint`, so a fractional exponent is a rounding boundary,
 *  not an exact result. */
export function mapI32(bridge: Bridge, a: Int32Array, b: Int32Array, op: MapOpName): Int32Array {
  return mapArray("nf_map_i32", bridge.ex.nf_map_i32, bridge, a, b, op, 4) as Int32Array;
}

/** i32 lanes, INTEGER scalar: add/sub/mul wrap mod 2^32. */
export function mapScalarI32(bridge: Bridge, a: Int32Array, s: number, op: MapOpName): Int32Array {
  return mapScalar("nf_map_scalar_i32", bridge.ex.nf_map_scalar_i32, bridge, a, s, op, 4) as Int32Array;
}

/** i32 lanes, FLOAT scalar. `floorDiv`/`mod` are rejected by this symbol:
 *  NumPy widens them to f64 -- use `mapFscalarI32Widened`. */
export function mapFscalarI32(bridge: Bridge, a: Int32Array, s: number, op: MapOpName): Int32Array {
  return mapScalar("nf_map_fscalar_i32", bridge.ex.nf_map_fscalar_i32, bridge, a, s, op, 4) as Int32Array;
}

/** f32 lanes, array-array, f32 out: add/sub/mul/floorDiv/mod. */
export function mapF32(bridge: Bridge, a: Float32Array, b: Float32Array, op: MapOpName): Float32Array {
  return mapArray("nf_map_f32", bridge.ex.nf_map_f32, bridge, a, b, op, 4) as Float32Array;
}

/** f32 div/pow, **f64** output: the CPU contract is an `astype(f64)`
 *  round-trip, so an f32 result here would be silently wrong. */
export function mapF32Divpow(bridge: Bridge, a: Float32Array, b: Float32Array, op: "div" | "pow"): Float64Array {
  return mapArray("nf_map_f32_divpow", bridge.ex.nf_map_f32_divpow, bridge, a, b, op, 8) as Float64Array;
}

/** f32 lanes, f64 scalar demoted to f32 first (NumPy value-based casting). */
export function mapScalarF32(bridge: Bridge, a: Float32Array, s: number, op: MapOpName): Float32Array {
  return mapScalar("nf_map_scalar_f32", bridge.ex.nf_map_scalar_f32, bridge, a, s, op, 4) as Float32Array;
}

/** f32 scalar div/pow, f64 output. */
export function mapScalarF32Divpow(bridge: Bridge, a: Float32Array, s: number, op: "div" | "pow"): Float64Array {
  return mapScalar("nf_map_scalar_f32_divpow", bridge.ex.nf_map_scalar_f32_divpow, bridge, a, s, op, 8) as Float64Array;
}

/** f64 lanes, array-array, all seven ops. */
export function mapF64(bridge: Bridge, a: Float64Array, b: Float64Array, op: MapOpName): Float64Array {
  return mapArray("nf_map_f64", bridge.ex.nf_map_f64, bridge, a, b, op, 8) as Float64Array;
}

/** f64 lanes, f64 scalar, all seven ops. Also the symbol an i32 lane with a
 *  float scalar must use for `floorDiv`/`mod`. */
export function mapScalarF64(bridge: Bridge, a: Float64Array, s: number, op: MapOpName): Float64Array {
  return mapScalar("nf_map_scalar_f64", bridge.ex.nf_map_scalar_f64, bridge, a, s, op, 8) as Float64Array;
}

/** i32 lane + float scalar, `floorDiv`/`mod`: NumPy promotes to f64, so the
 *  i32->f64 widening is exact and this is the contract-matched call. */
export function mapFscalarI32Widened(
  bridge: Bridge, a: Int32Array, s: number, op: "floorDiv" | "mod",
): Float64Array {
  return mapScalarF64(bridge, Float64Array.from(a), s, op);
}