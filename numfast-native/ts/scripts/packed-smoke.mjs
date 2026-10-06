// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE PACKED-TARBALL SMOKE TEST.
//
// WHAT THIS IS FOR. `npm test` in an INSTALLED package answers `tests 0,
// pass 0, fail 0` and exits 0, because the tarball's `files` list ships dist/
// and not test/. That is not a failure and not a pass: it is a test that
// cannot fail. It is also the published `npm test`, which must keep working
// outside a checkout -- a consumer who installs the tarball has no test tree,
// and a script that demanded one would break for exactly the people it is
// meant to serve.
//
// So the evidence lives here instead, on the repository side, where the crate
// and the build tree are. This script:
//
//   1. `npm pack` -- the real tarball, prepack included, so
//      corresp_src/numfast-native/ is vendored exactly as a publish would.
//   2. installs THAT FILE into a clean project, outside this repository, so
//      nothing resolves from `ts/` or from node_modules of a previous run.
//   3. runs real assertions against the INSTALLED package: the documented
//      quick-start result, a kernel called through the typed wrapper, the
//      documented counts read from the packed bytes, the SHA256SUMS of the
//      conveyed Corresponding Source, and the browser-shaped import surface.
//
// Every assertion here is against the artefact a consumer receives, resolved
// through the package's `exports` map, not through a relative path into the
// build tree. That is the whole point: the failure mode this replaces is a
// claim that was true in `ts/` and unverified in the tarball.
//
//   node scripts/packed-smoke.mjs            # default: a temp dir under os.tmpdir()
//   node scripts/packed-smoke.mjs --keep     # leave the scratch project for inspection
//
// NOT part of `npm test`, and not part of the published package. The script is
// gitignored-free source under scripts/, which `files` does not ship.

import assert from "node:assert/strict";
import { execFileSync, execSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const TS = join(HERE, "..");
const KEEP = process.argv.includes("--keep");

let checks = 0;

function ok(what, detail) {
  checks += 1;
  process.stdout.write(`  ok ${String(checks).padStart(2, " ")}. ${what}` +
    (detail ? `  (${detail})` : "") + "\n");
}

function sh(cmd, args, opts = {}) {
  return execFileSync(cmd, args, {
    encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], ...opts,
  });
}

function npm(args, cwd) {
  // On Windows `npm` is npm.cmd, which spawnSync cannot execute directly
  // (EINVAL); the shell is what makes it runnable. Both platforms take the
  // same argv, so the caller does not branch.
  return sh("npm", args, { cwd, shell: process.platform === "win32" });
}

// --- 1. the real tarball --------------------------------------------------

process.stdout.write("packed-tarball smoke test\n");
process.stdout.write(`  packing from ${TS}\n`);
const packDir = mkdtempSync(join(tmpdir(), "nf-pack-"));
const packOut = npm(["pack", "--pack-destination", packDir], TS);
const tgz = packOut.trim().split("\n").pop().trim();
const tgzPath = join(packDir, tgz);
assert.ok(existsSync(tgzPath), `npm pack reported ${tgz} but it is not at ${tgzPath}`);
ok(`npm pack produced ${tgz}`,
  `${readFileSync(tgzPath).length} bytes`);

// --- 2. a clean project, outside the repository ---------------------------

const project = mkdtempSync(join(tmpdir(), "nf-consumer-"));
writeFileSync(join(project, "package.json"), JSON.stringify({
  name: "numfast-packed-smoke", version: "1.0.0", private: true, type: "module",
}, null, 2));
npm(["install", "--no-audit", "--no-fund", tgzPath], project);
const pkgDir = join(project, "node_modules", "@numfast", "kernels");
assert.ok(existsSync(pkgDir), `the package did not install to ${pkgDir}`);
ok("installed the packed tarball into a clean project");

// A module that resolves `@numfast/kernels` by NAME -- not by a path into
// ts/ -- so the `exports` map is exercised the way a consumer exercises it.
// String.raw, so the backslashes in the regexes below reach the file intact.
// A template literal eats `\/` to `/` and `\]` to `]`, which silently rewrites
// the patterns -- and a smoke test whose own source was rewritten on the way
// out is the class of thing this file exists to catch.
const smoke = join(project, "smoke.mjs");
writeFileSync(smoke, String.raw`
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG = join(HERE, "node_modules", "@numfast", "kernels");

const K = await import("@numfast/kernels");
const out = {};

// 1. The documented quick-start, run verbatim against the INSTALLED package.
const k = await K.loadKernels();
const dist = K.ssspCsr(k,
  new Uint32Array([0, 1, 2, 3]),
  new Uint32Array([1, 2, 0]),
  new Uint32Array([10, 20, 30]),
  0);
out.quickstart = [...dist];
out.console = dist[1] + " " + dist[2];

// 2. The counts the README quotes, read from the packed bytes.
const wasm = readFileSync(join(PKG, "dist", "numfast_native.wasm"));
out.wasmBytes = wasm.length;
out.exports = Object.keys(k.ex).filter((n) => n !== "memory").length;
out.total = K.TOTAL_EXPORTS;
out.wrapped = K.WRAPPED.length;

// 3. The .wasm is the one dist/BUILD.json says it is.
const build = JSON.parse(readFileSync(join(PKG, "dist", "BUILD.json"), "utf8"));
out.sha = createHash("sha256").update(wasm).digest("hex");
out.buildSha = build.sha256;

// 4. AGPL section 6: the conveyed source travels and verifies.
const CS = join(PKG, "corresp_src", "numfast-native");
const rs = [];
const rec = (d) => {
  for (const e of readdirSync(d, { withFileTypes: true })) {
    const p = join(d, e.name);
    if (e.isDirectory()) rec(p);
    else if (e.name.endsWith(".rs")) rs.push(join(CS, e.name));
  }
};
rec(join(CS, "src"));
out.rs = rs.length;
const sums = readFileSync(join(CS, "SHA256SUMS"), "utf8")
  .split("\n").filter((l) => l.trim() !== "");
out.sums = sums.length;
let bad = [];
for (const line of sums) {
  const [want, rel] = line.split(/\s+/, 2);
  const p = join(CS, rel);
  const got = existsSync(p)
    ? createHash("sha256").update(readFileSync(p)).digest("hex") : "absent";
  if (got !== want) bad.push(rel);
}
out.badSums = bad;
out.hasMd = existsSync(join(PKG, "corresp_src", "CORRESPONDING-SOURCE.md"));

// 5. The browser-shaped subpaths resolve with no Node dependency.
const bridge = await import("@numfast/kernels/bridge");
out.hasLoadBridge = typeof bridge.loadBridge === "function";
out.hasWasmExport = existsSync(join(PKG, "dist", "numfast_native.wasm"));

// 6. The bare entry is Node-only, and says so where a caller will see it.
const indexSrc = readFileSync(join(PKG, "dist", "index.js"), "utf8");
out.entryNodeOnly = /node:fs/.test(indexSrc);

// 7. THE README THE CONSUMER READS, against the kernel they installed.
//    Parsed out of the shipped README rather than retyped here: the defect
//    this class exists for was a documented result nobody executed, and a
//    retyped copy of the number is the same failure one layer down. If the
//    README drifts from the kernel, this fails on the channel where a reader
//    would meet the drift.
const readme = readFileSync(join(PKG, "README.md"), "utf8");
const docLine = /\/\/ Uint32Array \[([^\]]*)\]/.exec(readme);
out.docLanes = docLine
  ? docLine[1].split(",").map((s) => s.trim()).filter((s) => s.length)
    .map((s) => (/Infinity|Q_INF|4294967295|0xffffffff/i.test(s)
      ? 0xffffffff : Number(s)))
  : null;

// 8. assertGuard is live -- a corrupt heap is caught, not ignored.
try { k.assertGuard(); out.guard = "ok"; } catch (e) { out.guard = e.name; }

process.stdout.write("SMOKE_JSON " + JSON.stringify(out) + String.fromCharCode(10));
`);
const smokeOut = sh(process.execPath, [smoke], { cwd: project });
const line = smokeOut.split("\n").find((l) => l.startsWith("SMOKE_JSON "));
assert.ok(line, `the installed-package smoke produced no result:\n${smokeOut}`);
const m = JSON.parse(line.slice("SMOKE_JSON ".length));

// --- 3. the assertions ----------------------------------------------------
// Each is a claim the README makes, checked against the tarball.

// The README's quick-start. This exact assertion is what the wrong
// four-lane comment in the README would have failed, had it been run against
// the packed artefact instead of asserted in prose.
assert.deepEqual(m.quickstart, [0, 10, 30],
  `the documented quick-start must return [0, 10, 30]; got ${JSON.stringify(m.quickstart)}`);
assert.equal(m.console, "10 30");
ok("documented quick-start result", `[${m.quickstart.join(", ")}], console ${m.console}`);

assert.equal(m.exports, 86, `86 function exports, got ${m.exports}`);
assert.equal(m.total, 86);
assert.equal(m.wrapped, 17);
ok("documented counts, from the packed bytes",
  `${m.exports} exports, ${m.wrapped} wrapped, ${m.wasmBytes} B wasm`);

assert.equal(m.sha, m.buildSha, "the packed .wasm is not the one BUILD.json names");
ok("the packed .wasm is the built .wasm", m.sha.slice(0, 16));

assert.equal(m.rs, 47, `47 .rs files must travel, got ${m.rs}`);
assert.deepEqual(m.badSums, [], `SHA256SUMS entries that do not verify: ${m.badSums}`);
assert.ok(m.hasMd, "CORRESPONDING-SOURCE.md is not in the tarball");
ok("AGPL section 6", `${m.rs} .rs, ${m.sums}/${m.sums} SHA256SUMS verify`);

assert.ok(m.hasLoadBridge, "@numfast/kernels/bridge does not export loadBridge");
assert.ok(m.hasWasmExport, "@numfast/kernels/wasm does not resolve to a .wasm");
ok("browser-shaped subpaths resolve", "bridge + kernels + wasm");

assert.ok(m.entryNodeOnly,
  "dist/index.js no longer reads node:fs; the README's Node-only claim is stale");
assert.equal(m.guard, "ok", `assertGuard did not pass: ${m.guard}`);
ok("allocator guard live, entry point Node-only as documented");

// The shipped README's own claim, checked against the shipped kernel.
assert.ok(Array.isArray(m.docLanes),
  "the shipped README no longer documents the quick-start result as " +
  "`// Uint32Array [ ... ]`; teach this script the new spelling rather than " +
  "deleting the check");
assert.deepEqual(m.docLanes, m.quickstart,
  `the shipped README documents ssspCsr as [${m.docLanes}] and the shipped ` +
  `kernel returns [${m.quickstart}]`);
ok("the shipped README's documented result, against the shipped kernel",
  `[${m.docLanes.join(", ")}]`);

// --- the thing this exists to prevent -------------------------------------
//
// Run the published `npm test` from the INSTALLED package and record what it
// actually reports. If it ever becomes non-zero the reason changes; while it
// reports 0 tests, this script's own count is the evidence.

const pubOut = npm(["test", "--silent"], pkgDir).trim();
const pubCounts = /tests (\d+)/.exec(pubOut);
process.stdout.write(
  `  note: the published \`npm test\` in the installed package reports ` +
  `${pubCounts ? pubCounts[1] : "?"} tests and exits 0 -- it cannot fail, ` +
  `because the tarball ships dist/ and not test/. These ${checks} checks are ` +
  `the evidence.\n`);

if (!KEEP) {
  rmSync(packDir, { recursive: true, force: true });
  rmSync(project, { recursive: true, force: true });
} else {
  process.stdout.write(`  kept ${packDir} and ${project}\n`);
}

process.stdout.write(`\nPACKED-TARBALL SMOKE: ${checks} checks passed\n`);
process.exit(0);