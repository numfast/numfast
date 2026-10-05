// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// CORRESPONDING SOURCE for the conveyed .wasm (AGPL-3.0 section 6).
//
// This is the mirror of `setup.py::build_py._vendor_corresponding_source`, and
// it exists for the same reason. This tarball conveys `dist/numfast_native.wasm`
// -- object code -- and shipped zero `.rs` files, so section 6 was not discharged
// on the npm channel while it was discharged by construction on the Python one.
// The Python side already solved this; a second approach would be a second thing
// to keep true, so this copies its shape exactly:
//
//   * the same explicit list of what `cargo build` needs and nothing else,
//   * the same refusal: a missing file is a hard failure, never a short list,
//   * SHA256SUMS written from the BYTES THAT SHIPPED, so the claim is checkable
//     with `sha256sum -c` rather than asserted,
//   * the repository's CORRESPONDING-SOURCE.md alongside the source.
//
// WHY prepack AND NOT RELOCATION. The alternative is committing a second copy of
// the 47 `.rs` files under `ts/`. Then the `.wasm` is built from
// `numfast-native/src/` while the tarball ships `ts/corresp_src/.../src/`, and
// SHA256SUMS -- regenerated at pack time -- would happily hash the STALE copy and
// pass. The obligation would then be satisfied in appearance only, which is the
// failure this whole mechanism exists to make impossible. Copying at pack time
// keeps one source of truth: the bytes shipped are the bytes the .wasm was built
// from, by construction.
//
// `node corresp-src.mjs`         vendor, for `npm pack` / `npm publish` (prepack)
// `node corresp-src.mjs --check` verify an existing copy against the crate
//
// The generated directory is a build output and is gitignored, exactly as
// `dist/` is.

import { createHash } from "node:crypto";
import {
  cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, relative } from "node:path";import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const CRATE = join(HERE, "..");
const REPO = join(CRATE, "..");
const OUT = join(HERE, "corresp_src");
const DEST = join(OUT, "numfast-native");

/** Everything `cargo build` needs to reproduce the shipped `.wasm`, and nothing
 *  else. Identical to CORRESPONDING_SOURCE in setup.py, and for the same
 *  reasons: REUSE.md is cited by src/lib.rs, `.cargo/config.toml` and the two
 *  `tools/nf-link.*` are the linker configuration, Cargo.lock pins the (empty)
 *  dependency set. Anything else in the crate is development material, and
 *  shipping it would claim it is build input when it is not. */
const CORRESPONDING_SOURCE = [
  "Cargo.toml",
  "Cargo.lock",
  "REUSE.md",
  ".cargo/config.toml",
  "tools/nf-link.bat",
  "tools/nf-link.py",
];

/** All files under DEST except SHA256SUMS itself, sorted. */
function shippedFiles(root) {
  const out = [];
  const rec = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, entry.name);
      if (entry.isDirectory()) rec(p);
      else if (entry.name !== "SHA256SUMS") out.push(p);
    }
  };
  rec(root);
  out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return out;
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** Path of `p` relative to DEST, with forward slashes. SHA256SUMS is read by
 *  `sha256sum -c` on any platform, so the separators in it cannot be this
 *  machine's. Same shape as setup.py's `relative_to(dest).as_posix()`. */
function relPosix(p) {
  return relative(DEST, p).split(/[\\/]/).join("/");
}

function fail(msg) {
  process.stderr.write(`CORRESPONDING SOURCE: ${msg}\n`);
  process.exit(1);
}

function requireCrate() {
  const missing = CORRESPONDING_SOURCE.filter((rel) => !existsSync(join(CRATE, rel)));
  if (!existsSync(join(CRATE, "src"))) missing.unshift("src/");
  const md = join(REPO, "CORRESPONDING-SOURCE.md");
  if (!existsSync(md)) missing.unshift("../CORRESPONDING-SOURCE.md");
  if (missing.length > 0) {
    fail(
      `incomplete, missing from ${CRATE}: ${missing.join(", ")}.\n` +
      `  This tarball conveys dist/numfast_native.wasm; without the crate source\n` +
      `  AGPL-3.0 section 6 is not satisfied. Run this from a repository checkout,\n` +
      `  not from an unpacked tarball.`);
  }
}

function vendor() {
  requireCrate();
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(DEST, { recursive: true });

  cpSync(join(CRATE, "src"), join(DEST, "src"), {
    recursive: true,
    filter: (p) => !p.endsWith("__pycache__") && !p.endsWith(".pyc"),
  });
  for (const rel of CORRESPONDING_SOURCE) {
    const target = join(DEST, rel);
    mkdirSync(dirname(target), { recursive: true });
    cpSync(join(CRATE, rel), target);
  }
  cpSync(join(REPO, "CORRESPONDING-SOURCE.md"), join(OUT, "CORRESPONDING-SOURCE.md"));

  // Written from the bytes that shipped, in the same format setup.py writes, so
  // `sha256sum -c SHA256SUMS` in the unpacked tarball is the check.
  const files = shippedFiles(DEST);
  const lines = files.map((p) => `${sha256(p)}  ${relPosix(p)}`);
  writeFileSync(join(DEST, "SHA256SUMS"), lines.join("\n") + "\n", "utf8");

  const rs = files.filter((p) => p.endsWith(".rs")).length;
  process.stdout.write(
    `corresponding source -> ${relative(process.cwd(), DEST)}\n` +
    `  ${files.length} files (${rs} .rs + ${files.length - rs} build inputs) + SHA256SUMS\n` +
    `  CORRESPONDING-SOURCE.md at ${relative(process.cwd(), OUT)}\n`);
  return files.length;
}

/** The shipped copy must be byte-identical to the crate, and its SHA256SUMS
 *  must verify. Either failure means the tarball is not carrying the source of
 *  the .wasm, which is the whole claim. */
function check() {
  requireCrate();
  if (!existsSync(join(DEST, "SHA256SUMS"))) {
    fail(`no ${relative(process.cwd(), DEST)}/SHA256SUMS. Run \`npm run prepack\` ` +
         `(or node corresp-src.mjs) to vendor it.`);
  }
  const sums = readFileSync(join(DEST, "SHA256SUMS"), "utf8")
    .split("\n").filter((l) => l.trim() !== "");
  const listed = new Set(sums.map((l) => l.split(/\s+/).slice(1).join(" ")));
  const onDisk = new Set(shippedFiles(DEST).map((p) => relPosix(p)));

  const problems = [];
  let bad = 0;
  for (const line of sums) {
    const [want, rel] = line.split(/\s+/, 2);
    const p = join(DEST, rel);
    if (!existsSync(p)) { problems.push(`listed but absent: ${rel}`); continue; }
    const got = sha256(p);
    if (got !== want) { problems.push(`sha256 differs: ${rel}`); bad += 1; }
  }
  for (const rel of onDisk) {
    if (!listed.has(rel)) problems.push(`shipped but unlisted in SHA256SUMS: ${rel}`);
  }

  // And the decisive direction: every shipped byte equals the crate's byte.
  let drift = 0;
  const rs = [];
  for (const p of shippedFiles(DEST)) {
    const rel = relPosix(p);
    if (rel.endsWith(".rs")) rs.push(rel);
    // src/<x>.rs and the six named build inputs all sit at the same relative
    // path under the crate root, which is what makes a plain path compare the
    // decisive check rather than a re-implementation of the copy.
    const upstream = join(CRATE, rel);
    if (!existsSync(upstream)) { problems.push(`no crate file for ${rel}`); drift += 1; continue; }
    if (!readFileSync(p).equals(readFileSync(upstream))) {
      problems.push(`differs from the crate: ${rel}`); drift += 1;
    }
  }
  const mdShipped = join(OUT, "CORRESPONDING-SOURCE.md");
  if (!existsSync(mdShipped) ||
      !readFileSync(mdShipped).equals(readFileSync(join(REPO, "CORRESPONDING-SOURCE.md")))) {
    problems.push("CORRESPONDING-SOURCE.md is absent or differs from the repository's");
  }

  if (problems.length > 0) {
    fail(`${problems.length} problem(s):\n  ` + problems.join("\n  "));
  }
  process.stdout.write(
    `corresponding source OK -> ${relative(process.cwd(), DEST)}\n` +
    `  ${sums.length}/${sums.length} SHA256SUMS verify, ` +
    `${rs.length} .rs byte-identical to numfast-native/\n`);
  return sums.length;
}

const checkOnly = process.argv.includes("--check");
const n = checkOnly ? check() : vendor();
process.exit(0);
