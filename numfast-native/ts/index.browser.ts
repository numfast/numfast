// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE BROWSER ENTRY. Selected by the `browser` condition in package.json's
// exports map, so `@numfast/kernels` resolves here for a bundler that honours
// it and to index.ts for Node. Both entries export the SAME names with the SAME
// types; the only difference is where the .wasm bytes come from.
//
// Why a second file rather than one clever file: `import { readFileSync } from
// "node:fs"` is a STATIC edge. A bundler resolves it whether or not the call is
// ever reached, so guarding the call site at runtime does not help -- the bundle
// fails to build at all. Measured against this tree with esbuild 0.28.2
// `--bundle --platform=browser`: three unresolved specifiers, `node:module`,
// `node:fs` and `node:crypto`, all three from dist/index.js. The only way the
// bare entry bundles for a browser is for the browser's module graph to contain
// no `node:` specifier at all, which means a second entry.
//
// WHY `loadKernels` TAKES BYTES HERE. The Node entry reads its own .wasm off
// disk at load, because a Node process has a synchronous filesystem and a
// portable path from `import.meta.url`. A browser has neither: there is no disk
// and no sync read, and a bundler cannot put a file in the filesystem. The bytes
// have to be handed in by whoever fetched them.
//
// That is a LOADER affordance, not an API change, and deliberately so:
//   * `loadKernels(bytes?: Uint8Array)` -- one OPTIONAL parameter. Called with no
//     argument it behaves exactly as the Node entry does, so existing Node
//     callers are untouched, and the return type is unchanged.
//   * Every other export keeps its name and its signature, and the kernel
//     surface -- `Bridge`, `loadBridge`, the `WRAPPED` wrappers, `callRaw`, the
//     error contract, the ABI tables -- is re-exported from the same modules
//     index.ts re-exports, not reimplemented.
//   * `BuildInfo` and `SCOPE` come from `identity.ts`, shared with index.ts, so
//     the two entries cannot drift apart on a generated sentence.
//
// The browser has no package.json to read and no node:crypto to hash it with, so
// `buildInfo()` returns undefined -- which its return type has always allowed,
// and which `loadKernels` already handles (the build-time self-check is skipped,
// not failed).

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

export type { BuildInfo } from "./identity.js";
export { SCOPE } from "./identity.js";

import { loadBridge, type Bridge } from "./bridge.js";
import { WRAPPED, TOTAL_EXPORTS } from "./kernels.js";
import type { BuildInfo } from "./identity.js";

/** No package.json to read and no node:crypto to hash it with, so: nothing.
 *  Absent in this build by construction, not by failure -- `build.mjs` writes
 *  BUILD.json, and a bundler has no filesystem to read it from afterwards. */
export function buildInfo(): BuildInfo | undefined {
  return undefined;
}

/** The .wasm's URL, which IS derivable here -- `import.meta.url` is universal.
 *  It is returned rather than fetched: only a bundler can turn this into
 *  something loadable, and a bundler is the one that already knows the asset
 *  name. Copy `numfast_native.wasm` next to the bundle, or point your bundler's
 *  asset rule at `@numfast/kernels/wasm`. */
export function wasmPath(): string {
  return new URL("./numfast_native.wasm", import.meta.url).href;
}

/** There is no synchronous filesystem in a browser, and pretending otherwise by
 *  returning an empty array would hand `loadBridge` a module that fails to
 *  instantiate. The error names the two ways out. */
export function wasmBytes(): Uint8Array {
  throw new Error(
    "wasmBytes() reads the .wasm from disk and there is no disk in a browser. " +
    "Either pass the bytes to loadKernels(bytes) -- fetch(wasmPath()) or import " +
    "the './wasm' export -- or use the Node entry, which does this for you.",
  );
}

/** Identical check to the Node entry's, on the bytes it is given.
 *
 *  A build whose export count or i64 count has moved is refused rather than
 *  wrapped: a `WRAPPED` entry naming a symbol this .wasm does not export would
 *  otherwise fail deep inside a call. */
export async function loadKernels(bytes?: Uint8Array): Promise<Bridge> {
  const source = bytes ?? requireBytes();
  const bridge = await loadBridge(source);
  const info = buildInfo();
  const exports = bridge.ex as unknown as Record<string, unknown>;
  for (const symbol of WRAPPED) {
    if (typeof exports[symbol] !== "function") {
      throw new Error(
        `${symbol} is wrapped by @numfast/kernels but not exported by the ` +
        `artefact${info ? ` (${info.sha256.slice(0, 12)}, ${info.funcCount} exports)` : ""}. ` +
        `Rebuild: cargo build --target wasm32-unknown-unknown --release && npm run build`,
      );
    }
  }
  if (info && info.funcCount !== TOTAL_EXPORTS) {
    throw new Error(
      `artefact exports ${info.funcCount} functions, this package is built ` +
      `against ${TOTAL_EXPORTS}. Rebuild both, or set TOTAL_EXPORTS deliberately.`,
    );
  }
  return bridge;
}

function requireBytes(): Uint8Array {
  try {
    return wasmBytes();
  } catch (e) {
    throw new Error(
      "loadKernels() needs the .wasm bytes in this build: " + (e as Error).message,
    );
  }
}