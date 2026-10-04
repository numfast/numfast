// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// BUILD: tsc -> dist/, copy the release .wasm, record what was built.
//
// Three things this exists to answer, none of which a committed `dist/` can:
//
//   * WHICH ARTEFACT? `dist/BUILD.json` carries the sha256, the byte count,
//     the export/import counts and the i64 export list of the .wasm that was
//     copied. A parity failure is then attributable to one build.
//   * IS IT THE RIGHT SURFACE? Every export name is checked against
//     `abi/census.json`, which is generated from the Rust source, and every
//     signature against the FuncType recorded there. A kernel added without a
//     rebuild of the wrapper table fails the build by NAME; nothing has to be
//     re-typed for the build to know the surface moved.
//   * IS THE DOCUMENTED ABI TABLE TRUE? Every `wasm:` field in abi.ts is
//     checked against the bytes of this .wasm. The table can no longer drift
//     away from the artefact without failing the build.
//
// The .wasm is NOT committed: `numfast-native/.gitignore` excludes `target/`,
// and that is correct -- it is a build product, and a committed copy is
// exactly how a 56-function build survived in git 29 symbols behind the
// current one. Reproduce it with the documented cargo line below.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { describe } from "./wasm-info.mjs";
import {
  expectedFuncExports, expectedI64Exports, surfaceDriftReport, signatureDrift,
} from "./abi-surface.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE = join(HERE, "..");
const DIST = join(HERE, "dist");
const WASM_SRC = join(NATIVE, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm");

export const CARGO_CMD =
  "cargo build --manifest-path numfast-native/Cargo.toml --target wasm32-unknown-unknown --release";

/** The engine's version. The Python package owns the version number
 *  (pyproject.toml `[project].version`, mirrored to `numfast.__version__`);
 *  `Cargo.toml` still says 0.1.0 and is stale. Read from pyproject so the
 *  package version cannot silently fork from the engine's. */
function engineVersion() {
  const pyproject = join(NATIVE, "..", "pyproject.toml");
  if (!existsSync(pyproject)) throw new Error(`cannot find ${pyproject}`);
  const m = /^version\s*=\s*"([^"]+)"/m.exec(readFileSync(pyproject, "utf8"));
  if (!m) throw new Error(`no [project].version in ${pyproject}`);
  const v = m[1];
  const pkg = JSON.parse(readFileSync(join(HERE, "package.json"), "utf8"));
  if (pkg.version !== v) {
    throw new Error(
      `package.json version ${pkg.version} != engine version ${v} (pyproject.toml). ` +
      `A kernel package that reports a different version from the engine cannot ` +
      `be traced to a commit.`);
  }
  return v;
}

function fail(msg) {
  process.stderr.write(`BUILD FAILED: ${msg}\n`);
  process.exit(1);
}

const checkOnly = process.argv.includes("--check-only");

if (!existsSync(WASM_SRC)) {
  fail(
    `no .wasm at\n  ${relative(process.cwd(), WASM_SRC)}\n` +
    `The release build is a build product and is gitignored, so a clean clone has none.\n` +
    `Produce one with:\n  ${CARGO_CMD}`);
}

const wasm = readFileSync(WASM_SRC);
const info = describe(wasm);

// IS IT THE RIGHT SURFACE? Not by a hand-typed count -- by NAME, against the
// source-derived census in abi/census.json. A count cannot say which kernel is
// missing, and a count is what let a clean clone fail: EXPECTED_FUNCS stayed 85
// after nf_pair_insert_i64 was added, and the .wasm is gitignored, so every
// clone rebuilt 86 and the guard demanded 85.
const drift = surfaceDriftReport(info);
if (drift) {
  fail(`the artefact does not match the source.\n  ${drift}`);
}
// Signatures too: the census carries the FuncType the compiler emitted for
// each source body, so a wrapper built against a moved argument is caught here
// rather than by a caller.
const sigDrift = signatureDrift(info);
if (sigDrift.length > 0) {
  fail(`${sigDrift.length} export(s) have a different FuncType than the source declares:\n  ` +
    sigDrift.join("\n  "));
}
if (info.importCount !== 0) {
  fail(`the artefact imports ${info.importCount} symbol(s); the "no imports" claim in the ` +
    `README and package description is what makes this module portable, and it is now false.`);
}
const wantI64 = expectedI64Exports();
const driftI64 = [
  ...wantI64.filter((s) => !info.i64Exports.includes(s)),
  ...info.i64Exports.filter((s) => !wantI64.includes(s)),
];
if (driftI64.length > 0) {
  fail(`the i64 boundary moved: ${wantI64.length} exports declare i64 in the source, ` +
    `this artefact has ${info.i64Count}. Not matching: ${driftI64.join(", ")}.\n` +
    `  The BigInt rule in the README is derived from this set; if the set is right, ` +
    `update the numbers the README quotes and move on.`);
}

// --- cross-check the documented ABI table against the bytes -----------------
// Imported from the TypeScript source, not from a copy: a table that is
// checked from somewhere other than the shipped one checks nothing.
const { ABI } = await import("./abi.ts");
for (const [symbol, entry] of Object.entries(ABI)) {
    const sig = info.signatures[symbol];
  if (!sig) fail(`abi.ts documents ${symbol}, which this artefact does not export`);
  const want = entry.wasm.length;
  if (sig.params.length !== want) {
    fail(`abi.ts says ${symbol} takes ${want} parameters; the artefact says ${sig.params.length}`);
  }
  if (sig.results.length !== 1 || sig.results[0] !== entry.returns) {
    fail(`abi.ts says ${symbol} returns ${entry.returns}; the artefact says ` +
      `${sig.results.join(",") || "void"}`);
  }
}

const sha256 = createHash("sha256").update(wasm).digest("hex");
const manifest = {
  sha256,
  bytes: info.bytes,
  exportCount: info.exportCount,
  funcCount: info.funcCount,
  importCount: info.importCount,
  i64Count: info.i64Count,
  i64Exports: info.i64Exports,
  initialMemoryPages: info.initialMemoryPages,
  /** The wasm global `__stack_pointer` initialises to (1048576 = 0x100000). */
  stackPointer: info.stackPointer,
  /** Upper bound from the DATA SECTION. Understates the real footprint: the
   *  runtime probe in bridge.ts is authoritative for buffer placement. */
  dataSectionEnd: info.staticDataEnd,
  version: engineVersion(),
  builtFrom: process.env.GITHUB_SHA ?? gitSha() ?? "unknown",
  cargoCommand: CARGO_CMD,
};

if (checkOnly) {
  process.stdout.write(JSON.stringify(manifest, null, 2) + "\n");
  process.exit(0);
}

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { cwd: NATIVE, encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

mkdirSync(DIST, { recursive: true });
// tsc is resolved from node_modules and run under the SAME node binary, not
// through npx: npx on Windows re-enters npm and fails here, and a build that
// cannot run its own compiler on the developer's platform is not a build.
const tscJs = createRequire(import.meta.url).resolve("typescript/bin/tsc");
execFileSync(process.execPath, [tscJs, "-p", join(HERE, "tsconfig.json")], { stdio: "inherit" });
copyFileSync(WASM_SRC, join(DIST, "numfast_native.wasm"));
writeFileSync(join(DIST, "BUILD.json"), JSON.stringify(manifest, null, 2) + "\n");

process.stdout.write(
  `@numfast/kernels ${manifest.version}\n` +
  `  wasm      ${manifest.bytes} bytes, sha256 ${manifest.sha256}\n` +
  `  surface   ${manifest.exportCount} exports = ${manifest.funcCount} functions + 1 memory, ` +
  `${manifest.importCount} imports\n` +
  `  bigint    ${manifest.i64Count} of ${manifest.funcCount} exports cross the i64 boundary\n` +
  `  builtFrom ${manifest.builtFrom}\n` +
  `  dist      ${relative(process.cwd(), DIST)}\n`);