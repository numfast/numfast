// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for nf_pack_i32_direct — parity + bench, no deps.
// Usage:
//   node wasm_pack.mjs <wasm> <k1.i32> <k2.i32> <m2> <n> <mode> [outdir]
import { writeFileSync } from "node:fs";
import { BASE, alloc, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, k1Path, k2Path, m2S, nS, mode, outdir] = process.argv.slice(2);
const m2 = Number(m2S), n = Number(nS);
assertN(n); assertN(m2, "m2");

const k1 = fileView(k1Path, Int32Array);
const k2 = fileView(k2Path, Int32Array);
if (k1.length !== n || k2.length !== n) throw new Error("vector size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports.nf_pack_i32_direct;
if (!fn) throw new Error("nf_pack_i32_direct not exported");

let cur = BASE;
const [k1Off, c1] = alloc(cur, n * 4, 1);
const [k2Off, c2] = alloc(c1, n * 4, 1);
const [outOff, need] = alloc(c2, n * 4, 8);
assertLayout([[k1Off, n * 4], [k2Off, n * 4], [outOff, n * 4]], "pack");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, k1, k1Off);
blit(u8, k2, k2Off);

// Thin wrapper: memory offsets + 1 call, no compute.
function call() {
  const t0 = performance.now();
  const rc = fn(k1Off, k2Off, m2, n, outOff);
  return { rc, ms: performance.now() - t0 };
}

if (mode === "parity") {
  const { rc, ms } = call();
  assertGuard(mem, "pack");
  if (rc !== 0) throw new Error("rc=" + rc);
  writeFileSync(`${outdir}/pack_out.i32`, Buffer.from(mem.buffer.slice(outOff, outOff + n * 4)));
  console.log(JSON.stringify({ rc, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "pack");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
