// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE EXPECTED SURFACE, DERIVED -- NOT TYPED IN.
//
// Why this file exists. `build.mjs` used to carry `EXPECTED_FUNCS = 85`, a
// number copied from the README. `nf_pair_insert_i64` was then added to
// `numfast-native/src/lib.rs`, the number was not moved, and the guard started
// failing on every CLEAN CLONE: the release .wasm is gitignored, so a clone
// rebuilds it from source and got 86, while the guard demanded 85. The package
// could not be rebuilt by anyone who had not read the README. A hand-kept
// integer in a build script is a number only the author can update, and the
// author was not the person cloning.
//
// The authority here is `numfast-native/abi/census.json`, generated FROM the
// `#[no_mangle]` bodies and the six `macro_rules!` templates in
// `numfast-native/src/{lib.rs,router.rs}`. That provenance is the whole point:
// a manifest derived from a .wasm would check the artefact against itself,
// which is the circular check this replaces. Each of its 86 entries carries the
// FuncType the compiler emitted for the source, so the surface can be compared
// by NAME and by SIGNATURE -- a missing export is named, not counted -- and the
// i64 boundary set falls out of the same table instead of being a second
// hand-kept number.
//
// Reading the census is reading a committed manifest, so a clone can do it: no
// toolchain, no cargo, no network.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

/** The source-derived manifest. Committed; never generated at build time. */
export const CENSUS_PATH = join(HERE, "..", "abi", "census.json");

let cached = null;

export function census() {
  if (cached) return cached;
  let raw;
  try {
    raw = JSON.parse(readFileSync(CENSUS_PATH, "utf8"));
  } catch (e) {
    throw new Error(
      `cannot read the source-derived ABI census at ${CENSUS_PATH}: ${e.message}\n` +
      `  It is the authority for which exports a build must have. Without it there\n` +
      `  is nothing to check the artefact against, and a hand-typed count is what\n` +
      `  this file exists to replace.`);
  }
  const symbols = raw.symbols ?? {};
  if (Object.keys(symbols).length === 0) {
    throw new Error(`${CENSUS_PATH} has no symbols; it is not a usable authority`);
  }
  const missingSig = Object.keys(symbols).filter((s) => !symbols[s].wasm_signature);
  if (missingSig.length > 0) {
    throw new Error(`${CENSUS_PATH}: ${missingSig.length} symbol(s) carry no ` +
      `wasm_signature (${missingSig.slice(0, 5).join(", ")}), so the signature ` +
      `cross-check cannot run`);
  }
  cached = raw;
  return cached;
}

/** Every `#[no_mangle] pub extern "C"` the source declares, sorted. */
export function expectedFuncExports() {
  return Object.keys(census().symbols).sort();
}

/** Symbols whose FuncType touches i64, so a JS `Number` is refused. */
export function expectedI64Exports() {
  const { symbols } = census();
  return Object.keys(symbols).filter((s) => {
    const sig = symbols[s].wasm_signature;
    return sig.params.includes("i64") || sig.results.includes("i64");
  }).sort();
}

/** symbol -> { params, results }, straight from the source. */
export function expectedSignatures() {
  const out = {};
  for (const [s, v] of Object.entries(census().symbols)) {
    out[s] = { params: v.wasm_signature.params, results: v.wasm_signature.results };
  }
  return out;
}

/**
 * Which names the source declares and the artefact does not export, and the
 * reverse. Sets, not counts: a count cannot say WHICH kernel went missing,
 * which is the difference between a five-second fix and an afternoon.
 */
export function surfaceDrift(info) {
  const want = expectedFuncExports();
  const have = new Set(info.funcExports);
  const w = new Set(want);
  return {
    missing: want.filter((s) => !have.has(s)),
    extra: info.funcExports.filter((s) => !w.has(s)).sort(),
  };
}

/** One paragraph naming what drifted, or "" when the surface matches. */
export function surfaceDriftReport(info) {
  const { missing, extra } = surfaceDrift(info);
  if (missing.length === 0 && extra.length === 0) return "";
  const parts = [];
  if (missing.length > 0) {
    parts.push(`the source declares ${missing.length} export(s) this artefact does ` +
      `not have: ${missing.join(", ")}`);
  }
  if (extra.length > 0) {
    parts.push(`the artefact exports ${extra.length} symbol(s) the source does not ` +
      `declare: ${extra.join(", ")}`);
  }
  return `${parts.join("; ")}.\n` +
    `  The .wasm is a gitignored build product: rebuild it with\n` +
    `    cargo build --manifest-path numfast-native/Cargo.toml --target wasm32-unknown-unknown --release\n` +
    `  If the source really is ahead of abi/census.json, regenerate the census -- do\n` +
    `  not relax this check.\n`;
}

/** FuncType differences between the source's table and the artefact's bytes. */
export function signatureDrift(info) {
  const want = expectedSignatures();
  const bad = [];
  for (const [s, sig] of Object.entries(want)) {
    const got = info.signatures[s];
    if (!got) continue; // absence is surfaceDrift's job, not this one's
    const p = JSON.stringify(sig.params) === JSON.stringify(got.params);
    const r = JSON.stringify(sig.results) === JSON.stringify(got.results);
    if (!p || !r) bad.push(`${s}: census [${sig.params}] -> [${sig.results}], ` +
      `artefact [${got.params}] -> [${got.results}]`);
  }
  return bad;
}