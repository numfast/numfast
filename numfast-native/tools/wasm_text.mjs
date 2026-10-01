// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Usage:
//   node wasm_text.mjs <wasm> <data.bin> <offs.i32> <n> <mode> [needle] [outdir]
// data.bin: concatenated row bytes (UTF-8); offs.i32: n+1 int32 LE offsets.
// mode=len-parity: 1 call nf_text_length, writes lens.i32, prints {rc,ms}.
// mode=contains-parity|startswith-parity|endswith-parity|equals-parity:
//   1 call nf_text_contains|nf_text_startswith|nf_text_endswith|nf_text_equals,
//   writes hits.u8, prints {rc,ms}.
// mode=len-bench / contains-bench|startswith-bench|endswith-bench|equals-bench:
//   5 warm + 10 measured, prints bench stats.
import { readFileSync, writeFileSync } from "node:fs";
import { BASE, alloc, assertGuard, assertLayout, assertN, bench, blit, blitBytes, dataView, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, dataPath, offsPath, nS, mode, ...rest] = process.argv.slice(2);
const n = Number(nS);
assertN(n);
const data = new Uint8Array(readFileSync(dataPath));
const offs = fileView(offsPath, Int32Array);
if (offs.length !== n + 1) throw new Error("offs size mismatch");

const FN_BY_MODE = {
  len: "nf_text_length",
  contains: "nf_text_contains",
  startswith: "nf_text_startswith",
  endswith: "nf_text_endswith",
  equals: "nf_text_equals",
};
const prefix = mode.split("-")[0];
const sym = FN_BY_MODE[prefix];
if (!sym) throw new Error("mode must be len|contains|startswith|endswith|equals + -parity|-bench");
const isLen = prefix === "len";
const isParity = mode.endsWith("parity");
let needle = new Uint8Array(0);
let outdir = rest[0];
if (!isLen) {
  needle = new TextEncoder().encode(rest[0] ?? "");
  outdir = rest[1];
}

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports[sym];
if (!fn) throw new Error(sym + " not exported");

let cur = BASE;
const [dataOff, c1] = alloc(cur, data.length, 4);
const [offsOff, c2] = alloc(c1, (n + 1) * 4, 4);
const [ndlOff, c3] = alloc(c2, needle.length, 4);
const outBytes = isLen ? n * 4 : n;
const [outOff, need] = alloc(c3, outBytes, 4);
assertLayout([[dataOff, data.length], [offsOff, (n + 1) * 4], [ndlOff, needle.length], [outOff, outBytes]], "text");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blitBytes(u8, data, dataOff);
blit(u8, offs, offsOff);
blitBytes(u8, needle, ndlOff);

// Thin wrapper: memory offsets + 1 call, no compute.
function call() {
  const t0 = performance.now();
  const rc = isLen
    ? fn(dataOff, data.length, offsOff, n, outOff)
    : fn(dataOff, data.length, offsOff, n, ndlOff, needle.length, outOff);
  return { rc, ms: performance.now() - t0 };
}

if (isParity) {
  const { rc, ms } = call();
  assertGuard(mem, "text");
  if (rc !== 0) throw new Error("rc=" + rc);
  writeFileSync(`${outdir}/${isLen ? "lens.i32" : "hits.u8"}`,
    Buffer.from(mem.buffer.slice(outOff, outOff + outBytes)));
  console.log(JSON.stringify({ rc, ms }));
} else if (mode.endsWith("bench")) {
  const r = bench(call);
  assertGuard(mem, "text");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be <op>-parity|<op>-bench, op=len|contains|startswith|endswith|equals");
}
