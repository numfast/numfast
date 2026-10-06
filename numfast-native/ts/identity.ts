// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// WHAT THIS FILE IS: the two facts about the package that are the SAME on every
// host -- the shape of `dist/BUILD.json`, and the sentence that says what the
// package is and is not.
//
// WHY IT IS ITS OWN FILE. Both facts are quoted by `index.ts` (Node) and
// `index.browser.ts` (browser). `index.ts` also carries the host-specific half:
// `createRequire`, `readFileSync`, the .wasm read. A second entry that re-declared
// `BuildInfo` and the sentence would be a second copy of a contract that is
// generated from `TOTAL_EXPORTS` and `WRAPPED.length`, and the readme test
// (`test/readme.test.mjs`) compares the SENTENCE verbatim against the README. Two
// copies of that sentence is one more thing to keep in step, so the generated
// part is built once here and both entries import it.
//
// It imports nothing. A module with no imports is host-independent by
// construction, which is the property that makes it safe for the browser entry
// to depend on.

import { TOTAL_EXPORTS, WRAPPED } from "./kernels.js";

/** The contents of `dist/BUILD.json`, written by `build.mjs`. Every field is
 *  required because the file is generated, not authored: a field is either
 *  present or the build did not run. */
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

/**
 * One line stating exactly what this package is and is not.
 *
 * Generated from the two counters the README quotes, so the three numbers that
 * move together (86 / 17 / 69) cannot disagree with each other here. The README
 * reproduces this word for word and `test/readme.test.mjs` fails when it drifts.
 *
 * `index.browser.ts` returns this same object: the scope is a property of the
 * package, not of the host.
 */
export const SCOPE = Object.freeze({
  wrapped: WRAPPED.length,
  total: TOTAL_EXPORTS,
  statement:
    `NumFast compiles its compute kernels to WebAssembly: an ${TOTAL_EXPORTS}-function ` +
    `module with one memory and no imports, so the module itself instantiates in any ` +
    `host with a WebAssembly runtime. This package wraps ${WRAPPED.length} of those ` +
    `${TOTAL_EXPORTS} kernels with typed wrappers and is a portable kernel library, ` +
    `not a compute core: orchestration, buffer management and the type layer stay in ` +
    `the host. Its own entry point is Node-only (it reads the packaged .wasm with ` +
    `node:fs); the browser-facing module is dist/bridge.js, which takes the bytes ` +
    `you give it. The remaining ${TOTAL_EXPORTS - WRAPPED.length} kernels are ` +
    `reachable by name through callRaw and have no documented argument order.`,
});