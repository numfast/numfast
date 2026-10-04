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
//   * IS IT THE RIGHT SURFACE? The export count is checked against the
//     constant the package and the README quote. A kernel added without a
//     rebuild of the wrapper table is a build failure, not a silent change
//     to a published number.
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

/** The 85 the README, the package and kernels.ts all state. */
const EXPECTED_FUNCS = 85;
const EXPECTED_I64 = 20;

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

if (info.funcCount !== EXPECTED_FUNCS) {
  fail(`the artefact exports ${info.funcCount} functions, this package is built against ` +
    `${EXPECTED_FUNCS}. Either a kernel was added or removed, or this is a stale build. ` +
    `If the change is intended, update EXPECTED_FUNCS in build.mjs, TOTAL_EXPORTS in ` +
    `kernels.ts and the README together -- do not delete the check.`);
}
if (info.importCount !== 0) {
  fail(`the artefact imports ${info.importCount} symbol(s); the "no imports" claim in the ` +
    `README and package description is what makes this module portable, and it is now false.`);
}
if (info.i64Count !== EXPECTED_I64) {
  fail(`the artefact has ${info.i64Count} i64 exports, the docs say ${EXPECTED_I64}. ` +
    `Update the docs and the BigInt section of the README together.`);
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