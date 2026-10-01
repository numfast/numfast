// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for nf_group_multi_sum_count — parity + bench, no deps.
// Usage:
//   node wasm_multi.mjs <wasm> <keys.i32> <valsFlat.f64> <n> <ncols> <g> <mode> [outdir]
// valsFlat: SoA flat, column c at [c*n..(c+1)*n] float64 LE.
// mode=parity: 1 call, writes sums.f64 (ncols*g) + counts.i64 (g) to outdir.
// mode=bench:  5 warm + 10 measured calls.
import { writeFileSync } from "node:fs";
import { BASE, alloc, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, keysPath, valsPath, nS, ncS, gS, mode, outdir] = process.argv.slice(2);
const n = Number(nS), ncols = Number(ncS), g = Number(gS);
assertN(n, "n"); assertN(ncols, "ncols"); assertN(g, "g");

const keys = fileView(keysPath, Int32Array);
const vals = fileView(valsPath, Float64Array);
if (keys.length !== n) throw new Error("keys size mismatch");
if (vals.length !== ncols * n) throw new Error("vals size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports.nf_group_multi_sum_count;
if (!fn) throw new Error("nf_group_multi_sum_count not exported");

let cur = BASE;
const [keysOff, c1] = alloc(cur, n * 4, 8);
const [valsOff, c2] = alloc(c1, ncols * n * 8, 8);
const [sumsOff, c3] = alloc(c2, ncols * g * 8, 8);
const [countsOff, need] = alloc(c3, g * 8, 8);
assertLayout([[keysOff, n * 4], [valsOff, ncols * n * 8], [sumsOff, ncols * g * 8], [countsOff, g * 8]], "multi");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, keys, keysOff);
blit(u8, vals, valsOff);

// Thin wrapper: memory offsets + 1 call, no compute.
function call() {
  const t0 = performance.now();
  const rc = fn(keysOff, valsOff, n, ncols, sumsOff, countsOff, g);
  return { rc, ms: performance.now() - t0 };
}

if (mode === "parity") {
  const { rc, ms } = call();
  assertGuard(mem, "multi");
  if (rc !== 0) throw new Error("rc=" + rc);
  writeFileSync(`${outdir}/multi_sums.f64`, Buffer.from(mem.buffer.slice(sumsOff, sumsOff + ncols * g * 8)));
  writeFileSync(`${outdir}/multi_counts.i64`, Buffer.from(mem.buffer.slice(countsOff, countsOff + g * 8)));
  console.log(JSON.stringify({ rc, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "multi");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
