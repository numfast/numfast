// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE ARTEFACT GUARD, shared by every test in this directory.
//
// Why this file exists, in one paragraph: four parity scripts in
// `numfast-native/tools/` used to run a TRACKED 56-function .wasm that sits in
// git beside the current 85-function build, 29 symbols behind, and reported
// green. A test that cannot name the bytes it validated is not a test, it is a
// print statement. So every test here refuses to run against anything but the
// build `dist/BUILD.json` describes, and says which bytes that is.
//
// WHAT IT REFUSES, LOUDLY:
//   * no .wasm, or no BUILD.json   -- `npm run build` has not been run
//   * an export count that has moved -- a kernel was added or removed
//   * a sha256 that does not match BUILD.json -- a different build is on disk
//
// It never falls back to another .wasm and never skips. There is no code path
// out of here that returns a bridge it has not verified.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadBridge } from "../dist/bridge.js";
import { describe as describeWasm } from "../wasm-info.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const TS = join(HERE, "..");
const DIST = join(TS, "dist");
const WASM = join(DIST, "numfast_native.wasm");
const BUILD_JSON = join(DIST, "BUILD.json");

/** 86 = 85 Func + 1 Memory. Measured on this build; the README and the
 *  package description both quote it, so it is checked here. */
export const EXPECTED_EXPORT_COUNT = 86;
export const EXPECTED_FUNC_COUNT = 85;

function abort(msg) {
  throw new Error(
    `PARITY ABORTED: ${msg}\n` +
    `  Build the artefact and the package first:\n` +
    `    cargo build --manifest-path numfast-native/Cargo.toml --target wasm32-unknown-unknown --release\n` +
    `    cd numfast-native/ts && npm install && npm run build\n` +
    `  These tests do not fall back to another .wasm and do not skip. A parity\n` +
    `  result against an unnamed artefact is worse than no result.`);
}

/** Validated bytes + manifest for the artefact under test. */
export function artefact() {
  let build;
  try {
    build = JSON.parse(readFileSync(BUILD_JSON, "utf8"));
  } catch {
    abort(`no dist/BUILD.json (looked in ${BUILD_JSON}). It records which .wasm these wrappers were built against; without it a failure is not attributable to a build.`);
  }
  let bytes;
  try {
    bytes = readFileSync(WASM);
  } catch {
    abort(`no dist/numfast_native.wasm (looked in ${WASM}).`);
  }
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== build.sha256) {
    abort(`dist/numfast_native.wasm is ${bytes.length} bytes sha256 ${sha256},\n` +
      `  but dist/BUILD.json records ${build.bytes} bytes sha256 ${build.sha256}.\n` +
      `  Something replaced the artefact after the build. Rebuild.`);
  }
  const info = describeWasm(bytes);
  if (info.exportCount !== EXPECTED_EXPORT_COUNT || info.funcCount !== EXPECTED_FUNC_COUNT) {
    abort(`the artefact exports ${info.exportCount} (${info.funcCount} functions); ` +
      `${EXPECTED_EXPORT_COUNT} (${EXPECTED_FUNC_COUNT}) is what this suite is written against. ` +
      `If a kernel was legitimately added, move the constants in build.mjs, kernels.ts ` +
      `and the README together -- do not delete this check. That check is the only reason ` +
      `a 56-function build survived in git unnoticed.`);
  }
  return { bytes, build, info };
}

/** An instantiated, verified bridge plus a one-line provenance banner. */
export async function bridgeFor(label) {
  const { bytes, build, info } = artefact();
  const bridge = await loadBridge(bytes);
  process.stdout.write(
    `[${label}] wasm ${build.bytes} B sha256 ${build.sha256} | ` +
    `${info.exportCount} exports = ${info.funcCount} functions + ${info.memoryExports.length} memory | ` +
    `${info.importCount} imports | ${info.i64Count} i64 | version ${build.version}\n`);
  return bridge;
}

/**
 * Highest byte of the module's own initialised data, as the bridge measured it.
 * `base` must sit above it: that region holds the float constants `powf` reads,
 * and a caller buffer over it produces plausible, wrong numbers with rc=0.
 */
export function layoutFacts(bridge) {
  return { staticDataEnd: bridge.staticDataEnd, base: bridge.base };
}