// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for nf_pattern_encode — parity + bench, no deps.
// Usage:
//   node wasm_pattern.mjs <wasm> <data.bin> <offs.i32> <prefix> <n> <mode> [outdir]
// data.bin: concatenated row bytes; offs.i32: n+1 int32 LE offsets.
// mode=parity: 1 call, writes codes.i32 + valid.u8, prints {rc,ms,width,err_row}.
// mode=bench:  5 warm + 10 measured calls, prints {warm,runs,median_ms,...}.
import { readFileSync, writeFileSync } from "node:fs";
import { BASE, alloc, assertGuard, assertLayout, assertN, bench, blit, blitBytes, dataView, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, dataPath, offsPath, prefixS, nS, mode, outdir] = process.argv.slice(2);
const n = Number(nS);
assertN(n);
const prefix = new TextEncoder().encode(prefixS);

const data = new Uint8Array(readFileSync(dataPath));
const offs = fileView(offsPath, Int32Array);
if (offs.length !== n + 1) throw new Error("offs size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports.nf_pattern_encode;
if (!fn) throw new Error("nf_pattern_encode not exported");

let cur = BASE;
const [dataOff, c1] = alloc(cur, data.length, 4);
const [offsOff, c2] = alloc(c1, (n + 1) * 4, 4);
const [pfxOff, c3] = alloc(c2, prefix.length, 4);
const [codesOff, c4] = alloc(c3, n * 4, 4);
const [validOff, c5] = alloc(c4, n, 4);
const [widthOff, c6] = alloc(c5, 4, 4);
const [errOff, need] = alloc(c6, 4, 4);
assertLayout([[dataOff, data.length], [offsOff, (n + 1) * 4], [pfxOff, prefix.length], [codesOff, n * 4], [validOff, n], [widthOff, 4], [errOff, 4]], "pattern");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blitBytes(u8, data, dataOff);
blit(u8, offs, offsOff);
blitBytes(u8, prefix, pfxOff);

// Thin wrapper: memory offsets + 1 call, no compute.
function call() {
  const t0 = performance.now();
  const rc = fn(dataOff, data.length, offsOff, n, pfxOff, prefix.length, codesOff, validOff, widthOff, errOff);
  const dv = dataView(mem);
  return { rc, ms: performance.now() - t0, width: dv.getInt32(widthOff, true), err_row: dv.getInt32(errOff, true) };
}

if (mode === "parity") {
  const { rc, ms, width, err_row } = call();
  assertGuard(mem, "pattern");
  if (rc !== 0) throw new Error("rc=" + rc);
  writeFileSync(`${outdir}/pat_codes.i32`, Buffer.from(mem.buffer.slice(codesOff, codesOff + n * 4)));
  writeFileSync(`${outdir}/pat_valid.u8`, Buffer.from(mem.buffer.slice(validOff, validOff + n)));
  console.log(JSON.stringify({ rc, ms, width, err_row }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "pattern");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
