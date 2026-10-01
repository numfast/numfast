// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for map exports — parity + bench, no deps.
// Only real WASM exports, thin wrapper (memory offsets + 1 call, no compute).
// Op codes (numfast-native/src/series/map.rs): 0 add, 1 sub, 2 mul,
// 3 div, 4 pow, 5 floor_div, 6 mod.
// Symbol routing mirrors src/Drivers/CPU/_lib/native_cpu.py map_scatter:
//   i32: arr -> nf_map_i32; int scalar -> nf_map_scalar_i32 (s:i32);
//        float scalar -> nf_map_fscalar_i32 (s:f64)
//   f32: arr -> nf_map_f32 (div/pow -> nf_map_f32_divpow, f64 out);
//        scalar likewise (scalar always f64, demoted in-lane for add/sub/mul)
//   f64: arr -> nf_map_f64; scalar -> nf_map_scalar_f64 (s:f64)
// Usage:
//   node wasm_map.mjs <wasm> <aPath> <bPath|-> <kind> <op> <n> <scalarKind> <scalarVal> <mode> [outdir]
//   kind: i32|f32|f64; op: 0..6
//   array form:  <bPath>=file scalarKind=arr      (scalarVal ignored)
//   scalar form: <bPath>=-    scalarKind=i32|f64  (s value = scalarVal)
//   mode=parity: 1 call, writes map_out.<suf> to outdir, prints {"rc":..,"ms":..}
//   mode=bench:  5 warm + 10 measured calls, prints {"warm":5,"runs":[...],"median_ms":..}
import { writeFileSync } from "node:fs";
import { BASE, alloc, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, aPath, bPath, kind, opS, nS, scalarKind, scalarValS, mode, outdir] = process.argv.slice(2);
const n = Number(nS), op = Number(opS);
assertN(n);

const LANES = { i32: [Int32Array, 4], f32: [Float32Array, 4], f64: [Float64Array, 8] };
const spec = LANES[kind];
if (!spec) throw new Error("kind must be i32|f32|f64");
if (!(op >= 0 && op <= 6)) throw new Error("op must be 0..6");
const [T, B] = spec;

const a = fileView(aPath, T);
if (a.length !== n) throw new Error("vector size mismatch");

// Widening mirror of native map_scatter: i32 lane + float scalar
// floor_div/mod promote to float64 (NumPy rule) — exact i32->f64,
// then the f64 scalar lane.
let effKind = kind, effA = a, effB = B;
if (kind === "i32" && scalarKind === "f64" && (op === 5 || op === 6)) {
  effKind = "f64";
  effA = Float64Array.from(a);
  effB = 8;
}

const isArr = scalarKind === "arr";
let b = null;
if (isArr) {
  b = fileView(bPath, LANES[effKind][0]);
  if (b.length !== n) throw new Error("vector size mismatch");
}

// Route to the real export (same routing as native map_scatter).
let fnName, outKind = effKind, scalarI32 = false;
if (effKind === "i32") {
  if (isArr) fnName = "nf_map_i32";
  else if (scalarKind === "i32") { fnName = "nf_map_scalar_i32"; scalarI32 = true; }
  else if (scalarKind === "f64") fnName = "nf_map_fscalar_i32";
  else throw new Error("i32 scalarKind must be arr|i32|f64");
} else if (effKind === "f32") {
  const divpow = (op === 3 || op === 4);
  if (divpow) outKind = "f64";
  if (isArr) fnName = divpow ? "nf_map_f32_divpow" : "nf_map_f32";
  else {
    if (scalarKind !== "f64") throw new Error("f32 scalarKind must be arr|f64");
    fnName = divpow ? "nf_map_scalar_f32_divpow" : "nf_map_scalar_f32";
  }
} else {
  if (scalarKind !== "arr" && scalarKind !== "f64") throw new Error("f64 scalarKind must be arr|f64");
  fnName = isArr ? "nf_map_f64" : "nf_map_scalar_f64";
}
const [OT, OB] = LANES[outKind];

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports[fnName];
if (!fn) throw new Error(fnName + " not exported");

// Layout with alignment, BASE=0x101000 (above WASM stack top + guard).
const [aOff, c1] = alloc(BASE, n * effB, effB);
let bOff = 0, c2 = c1, outOff = 0, need = 0;
if (isArr) {
  [bOff, c2] = alloc(c1, n * effB, effB);
  [outOff, need] = alloc(c2, n * OB, OB);
} else {
  [outOff, need] = alloc(c1, n * OB, OB);
}
assertLayout(isArr ? [[aOff, n * effB], [bOff, n * effB], [outOff, n * OB]] : [[aOff, n * effB], [outOff, n * OB]], "map");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, effA, aOff);
if (isArr) blit(u8, b, bOff);

const sVal = Number(scalarValS);
// Thin wrapper: memory offsets + 1 call, no compute.
function call() {
  const t0 = performance.now();
  const rc = isArr ? fn(aOff, bOff, n, op, outOff)
    : scalarI32 ? fn(aOff, n, (sVal | 0), op, outOff)
    : fn(aOff, n, sVal, op, outOff);
  return { rc, ms: performance.now() - t0 };
}

if (mode === "parity") {
  const { rc, ms } = call();
  assertGuard(mem, "map");
  if (rc !== 0) throw new Error("rc=" + rc);
  writeFileSync(`${outdir}/map_out.${outKind}`, Buffer.from(mem.buffer.slice(outOff, outOff + n * OB)));
  console.log(JSON.stringify({ rc, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "map");
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
