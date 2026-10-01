// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for shift exports — parity + bench, no deps.
// Only real WASM exports, thin wrapper (memory offsets + 1 call, no compute):
//   nf_shift_i32, nf_shift_f32, nf_shift_f64.
// Usage:
//   node wasm_shift.mjs <wasm> <srcPath> <kind> <n> <periods> <mode> [outdir]
//   kind: i32|f32|f64
//   mode=parity: 1 call, writes shift_out.<kind> to outdir, prints {"rc":..,"ms":..}
//   mode=bench:  5 warm + 10 measured calls, prints {"warm":5,"runs":[...],"median_ms":..}
import { writeFileSync } from "node:fs";
import { BASE, alloc, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, srcPath, kind, nS, pS, mode, outdir] = process.argv.slice(2);
const n = Number(nS), periods = Number(pS);
assertN(n);

const LANES = { i32: [Int32Array, 4], f32: [Float32Array, 4], f64: [Float64Array, 8] };
const spec = LANES[kind];
if (!spec) throw new Error("kind must be i32|f32|f64");
const [T, B] = spec;

const src = fileView(srcPath, T);
if (src.length !== n) throw new Error("vector size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports["nf_shift_" + kind];
if (!fn) throw new Error("nf_shift_" + kind + " not exported");

// Layout with alignment, BASE=0x101000 (above WASM stack top + guard).
const [srcOff, c1] = alloc(BASE, n * B, B);
const [outOff, need] = alloc(c1, n * B, B);
assertLayout([[srcOff, n * B], [outOff, n * B]], "shift");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, src, srcOff);

// Thin wrapper: memory offsets + 1 call, no compute.
function call() {
  const t0 = performance.now();
  const rc = fn(srcOff, n, periods, outOff);
  return { rc, ms: performance.now() - t0 };
}

if (mode === "parity") {
  const { rc, ms } = call();
  assertGuard(mem, "shift");
  if (rc !== 0) throw new Error("rc=" + rc);
  writeFileSync(`${outdir}/shift_out.${kind}`, Buffer.from(mem.buffer.slice(outOff, outOff + n * B)));
  console.log(JSON.stringify({ rc, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "shift");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
