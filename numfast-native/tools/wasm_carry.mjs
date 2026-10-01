// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for nf_carry_build_i64/f64 — parity + bench, no deps.
// Usage:
//   node wasm_carry.mjs <wasm> <counts.i64> <sums.(i64|f64)> <int64:0|1> <m> <mode> [outdir]
import { writeFileSync } from "node:fs";
import { BASE, alloc, asNumber, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, cntPath, sumsPath, isI64S, mS, mode, outdir] = process.argv.slice(2);
const isI64 = isI64S === "1", m = Number(mS);
assertN(m, "m");

const cnt = fileView(cntPath, BigInt64Array);
const sms = isI64 ? fileView(sumsPath, BigInt64Array) : fileView(sumsPath, Float64Array);
if (cnt.length !== m || sms.length !== m) throw new Error("vector size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = isI64 ? instance.exports.nf_carry_build_i64 : instance.exports.nf_carry_build_f64;
if (!fn) throw new Error("carry export missing");

let cur = BASE;
const [cntOff, c1] = alloc(cur, m * 8, 8);
const [smsOff, c2] = alloc(c1, m * 8, 8);
const [ukOff, c3] = alloc(c2, m * 8, 8);
const [coOff, c4] = alloc(c3, m * 8, 8);
const [soOff, need] = alloc(c4, m * 8, 8);
assertLayout([[cntOff, m * 8], [smsOff, m * 8], [ukOff, m * 8], [coOff, m * 8], [soOff, m * 8]], "carry");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, cnt, cntOff);
blit(u8, sms, smsOff);

// Thin wrapper: memory offsets + 1 call, no compute (i64 ret via BigInt).
function call() {
  const t0 = performance.now();
  const ng = asNumber(fn(cntOff, smsOff, m, ukOff, coOff, soOff));
  return { ng, ms: performance.now() - t0 };
}

const suf = isI64 ? "i64" : "f64";
if (mode === "parity") {
  const { ng, ms } = call();
  assertGuard(mem, "carry");
  if (ng >= 0) {
    writeFileSync(`${outdir}/carry_uk_${suf}.i64`, Buffer.from(mem.buffer.slice(ukOff, ukOff + ng * 8)));
    writeFileSync(`${outdir}/carry_counts_${suf}.i64`, Buffer.from(mem.buffer.slice(coOff, coOff + ng * 8)));
    writeFileSync(`${outdir}/carry_sums_${suf}.${suf}`, Buffer.from(mem.buffer.slice(soOff, soOff + ng * 8)));
  }
  console.log(JSON.stringify({ ng, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "carry");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
