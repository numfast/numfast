// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for numfast_native.wasm — parity + bench, no deps.
// Usage:
//   node wasm_run.mjs <wasm> <keys.i32> <values.f64> <n> <g> <mode> [outdir]
// mode=parity: 1 call, writes sums.f64/counts.i64 to outdir, prints {"rc":..,"ms":..}
// mode=bench:  5 warm + 10 measured calls, prints {"warm":5,"runs":[...],"median_ms":..}
import { writeFileSync } from "node:fs";
import { BASE, alloc, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, keysPath, valsPath, nS, gS, mode, outdir] = process.argv.slice(2);
const n = Number(nS), g = Number(gS);
assertN(n, "n"); assertN(g, "g");

const keys = fileView(keysPath, Int32Array);
const vals = fileView(valsPath, Float64Array);
if (keys.length !== n || vals.length !== n) throw new Error("vector size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports.nf_group_sum_count;

// Layout with 8-byte alignment, BASE=0x101000 (above WASM stack top + guard).
let cur = BASE;
const [keysOff, c1] = alloc(cur, n * 4, 8);
const [valsOff, c2] = alloc(c1, n * 8, 8);
const [sumsOff, c3] = alloc(c2, g * 8, 8);
const [countsOff, need] = alloc(c3, g * 8, 8);
assertLayout([[keysOff, n * 4], [valsOff, n * 8], [sumsOff, g * 8], [countsOff, g * 8]], "run");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, keys, keysOff);
blit(u8, vals, valsOff);

// Thin wrapper: memory offsets + 1 call, no compute.
function call() {
  const t0 = performance.now();
  const rc = fn(keysOff, valsOff, n, sumsOff, countsOff, g);
  return { rc, ms: performance.now() - t0 };
}

if (mode === "parity") {
  const { rc, ms } = call();
  assertGuard(mem, "run");
  if (rc !== 0) throw new Error("rc=" + rc);
  writeFileSync(`${outdir}/sums.f64`, Buffer.from(mem.buffer.slice(sumsOff, sumsOff + g * 8)));
  writeFileSync(`${outdir}/counts.i64`, Buffer.from(mem.buffer.slice(countsOff, countsOff + g * 8)));
  console.log(JSON.stringify({ rc, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "run");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
