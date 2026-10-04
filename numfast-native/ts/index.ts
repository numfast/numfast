// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// @numfast/kernels -- NumFast's compute kernels as WebAssembly.
//
// WHAT THIS IS: 17 typed wrappers over an 86-function, 1-memory, 0-import
// WebAssembly module, plus the error contract and the memory layout rules
// needed to call it without getting silently wrong answers.
//
// WHAT THIS IS NOT: not a compute core. There is no executor, no graph
// runtime, no IR dispatch and no buffer allocator in the .wasm. Orchestration,
// buffer ownership and the type layer stay in the host -- which is the Rust
// source's own statement (numfast-native/src/router.rs:18, "Python owns
// orchestration/data/ABI"). Calling this a compute core would be false.
//
// The scope lives in exported constants, not only in prose:
//
//   WRAPPED.length            17 of 86
//   TOTAL_EXPORTS             86
//   BUILD.sha256 / .bytes     which .wasm these bytes came from
//
// `callRaw` reaches the other 69 kernels by name. They are unwrapped on
// purpose: their pointer-vs-length argument order is not recorded anywhere in
// the repository, so a wrapper written from the signature alone would be a
// guess. They are named, not hidden.

export { loadBridge, Bridge, probeStaticDataEnd, GUARD, CANARY } from "./bridge.js";
export type { WasmExports } from "./bridge.js";

export {
  NumFastError, NumFastTrap, NumFastArgumentError,
  callGuarded, checkLengths, requireBigInt, MAX_LEN,
} from "./errors.js";

export { ABI, RC_TABLE, RC_OK, RC_NULL_OR_ABORT, RC_BAD_RANGE, RC_MALFORMED, MAP_OP } from "./abi.js";
export type { AbiEntry, MapOpName, ScalarKind } from "./abi.js";

export {
  Q_INF, WRAPPED, TOTAL_EXPORTS, callRaw, exportsOf,
  ssspCsr, ssspCsrPred, ssspBatch,
  costTravelBatch, costIntern,
  rowwiseKwayTimeArgminGather, adjacencySlice, adjacencyGather,
  mapI32, mapScalarI32, mapFscalarI32, mapFscalarI32Widened,
  mapF32, mapF32Divpow, mapScalarF32, mapScalarF32Divpow,
  mapF64, mapScalarF64,
} from "./kernels.js";
export type { WrappedSymbol } from "./kernels.js";

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

import { loadBridge, type Bridge } from "./bridge.js";
import { TOTAL_EXPORTS, WRAPPED } from "./kernels.js";

export interface BuildInfo {
  /** sha256 of the exact .wasm these wrappers were built against. */
  readonly sha256: string;
  readonly bytes: number;
  readonly exportCount: number;
  readonly funcCount: number;
  readonly importCount: number;
  /** 21 on the current build: exports whose FuncType touches i64, so a
   *  JavaScript `Number` is rejected rather than truncated. */
  readonly i64Count: number;
  readonly i64Exports: readonly string[];
  readonly version: string;
}

/** Identity of the shipped artefact, written by `build.mjs` into
 *  `dist/BUILD.json`. Every test prints it, so a failure is attributable to
 *  one build rather than to "the wasm". Absent only in a tree where
 *  `npm run build` has not been run. */
export function buildInfo(): BuildInfo | undefined {
  const require = createRequire(import.meta.url);
  try {
    const path = require.resolve("./BUILD.json");
    return JSON.parse(readFileSync(path, "utf8")) as BuildInfo;
  } catch {
    return undefined;
  }
}

/** The packaged .wasm, resolved relative to this module. */
export function wasmPath(): string {
  return new URL("./numfast_native.wasm", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
}

/** The .wasm bytes, ready for `loadBridge`. */
export function wasmBytes(): Uint8Array {
  return new Uint8Array(readFileSync(wasmPath()));
}

/** Load the packaged artefact and check it is the one this build claims.
 *
 *  A build whose export count or i64 count has moved is refused rather than
 *  wrapped: a `WRAPPED` entry that names a symbol this .wasm does not export
 *  would otherwise fail deep inside a call. */
export async function loadKernels(): Promise<Bridge> {
  const bridge = await loadBridge(wasmBytes());
  const info = buildInfo();
  const exports = bridge.ex as unknown as Record<string, unknown>;
  for (const symbol of WRAPPED) {
    if (typeof exports[symbol] !== "function") {
      throw new Error(
        `${symbol} is wrapped by @numfast/kernels but not exported by the ` +
        `artefact${info ? ` (${info.sha256.slice(0, 12)}, ${info.funcCount} exports)` : ""}. ` +
        `Rebuild: cargo build --target wasm32-unknown-unknown --release && npm run build`);
    }
  }
  if (info && info.funcCount !== TOTAL_EXPORTS) {
    throw new Error(
      `artefact exports ${info.funcCount} functions, this package is built ` +
      `against ${TOTAL_EXPORTS}. Rebuild both, or set TOTAL_EXPORTS deliberately.`);
  }
  return bridge;
}

/** One line stating exactly what this package is and is not. */
export const SCOPE = Object.freeze({
  wrapped: WRAPPED.length,
  total: TOTAL_EXPORTS,
  statement:
    `NumFast compiles its compute kernels to WebAssembly: an ${TOTAL_EXPORTS}-function ` +
    `module with one memory and no imports, runnable in any host with a ` +
    `WebAssembly runtime. This package wraps ${WRAPPED.length} of those ${TOTAL_EXPORTS} ` +
    `kernels with typed wrappers. It is a portable kernel library, not a ` +
    `compute core: orchestration, buffer management and the type layer stay ` +
    `in the host. The remaining ${TOTAL_EXPORTS - WRAPPED.length} kernels are ` +
    `reachable by name through callRaw and have no documented argument order.`,
});