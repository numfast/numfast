// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for nf_sorted_run_i64/f64 — parity + bench, no deps.
// Usage:
//   node wasm_sorted.mjs <wasm> <keys.i32> <vals.(i64|f64)> <int64:0|1> <n> <mode> [outdir]
// vals file: int64 LE if <int64>=1 else float64 LE.
// mode=parity: 1 call, writes ukeys.i64 + sums.(i64|f64) + counts.i64,
//   prints {ng,ms} (ng may be -1 abort / -2 null).
// mode=bench: 5 warm + 10 measured calls.
import { writeFileSync } from "node:fs";
import { BASE, alloc, asNumber, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, keysPath, valsPath, isI64S, nS, mode, outdir] = process.argv.slice(2);
const isI64 = isI64S === "1", n = Number(nS);
assertN(n);

const keys = fileView(keysPath, Int32Array);
const vals = isI64 ? fileView(valsPath, BigInt64Array) : fileView(valsPath, Float64Array);
if (keys.length !== n || vals.length !== n) throw new Error("vector size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = isI64 ? instance.exports.nf_sorted_run_i64 : instance.exports.nf_sorted_run_f64;
if (!fn) throw new Error("sorted_run export missing");

let cur = BASE;
const [keysOff, c1] = alloc(cur, n * 4, 8);
const [valsOff, c2] = alloc(c1, n * 8, 8);
const [ukOff, c3] = alloc(c2, n * 8, 8);
const [sumsOff, c4] = alloc(c3, n * 8, 8);
const [cntOff, need] = alloc(c4, n * 8, 8);
assertLayout([[keysOff, n * 4], [valsOff, n * 8], [ukOff, n * 8], [sumsOff, n * 8], [cntOff, n * 8]], "sorted");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, keys, keysOff);
blit(u8, vals, valsOff);

// Thin wrapper: memory offsets + 1 call, no compute (i64 ret via BigInt).
function call() {
  const t0 = performance.now();
  const ng = asNumber(fn(keysOff, valsOff, n, ukOff, sumsOff, cntOff));
  return { ng, ms: performance.now() - t0 };
}

const suf = isI64 ? "i64" : "f64";
if (mode === "parity") {
  const { ng, ms } = call();
  assertGuard(mem, "sorted");
  if (ng >= 0) {
    writeFileSync(`${outdir}/sort_uk_${suf}.i64`, Buffer.from(mem.buffer.slice(ukOff, ukOff + ng * 8)));
    writeFileSync(`${outdir}/sort_sums_${suf}.${suf}`, Buffer.from(mem.buffer.slice(sumsOff, sumsOff + ng * 8)));
    writeFileSync(`${outdir}/sort_counts_${suf}.i64`, Buffer.from(mem.buffer.slice(cntOff, cntOff + ng * 8)));
  }
  console.log(JSON.stringify({ ng, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "sorted");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
