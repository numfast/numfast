// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for select/mask exports — parity + bench, no deps.
// Only real WASM exports, thin wrappers (memory offsets + 1 call, no compute):
//   nf_select_count, nf_select_scatter_{i32,i64,f32,f64,u8}, nf_mask_{and,or,not}.
// Usage:
//   node wasm_select.mjs <wasm> <op> <kind> <aPath> <bPath|-> <n> <mode> [outdir]
//   op: count | scatter | and | or | not
//   kind: i32|i64|f32|f64|u8 for scatter, "-" otherwise
//   count:   aPath=mask.u8; prints {m,ms}
//   scatter: aPath=src.<kind>, bPath=mask.u8; writes sel_out.<kind>, prints {m,ms}
//   and/or:  aPath,bPath=u8 lanes; writes mask_out.u8, prints {rc,ms}
//   not:     aPath=u8 lanes; writes mask_out.u8, prints {rc,ms}
// mode=parity: 1 call + outputs; mode=bench: 5 warm + 10 measured calls.
import { writeFileSync } from "node:fs";
import { BASE, alloc, asNumber, assertGuard, assertLayout, assertN, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, op, kind, aPath, bPath, nS, mode, outdir] = process.argv.slice(2);
const n = Number(nS);
assertN(n);

const LANES = { i32: [Int32Array, 4], i64: [BigInt64Array, 8], f32: [Float32Array, 4], f64: [Float64Array, 8], u8: [Uint8Array, 1] };

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const E = instance.exports;
function needFn(name) {
  const fn = E[name];
  if (!fn) throw new Error(name + " not exported");
  return fn;
}

let call, onParity;
if (op === "count") {
  const mask = fileView(aPath, Uint8Array);
  if (mask.length !== n) throw new Error("mask size mismatch");
  const fn = needFn("nf_select_count");
  const [maskOff, need] = alloc(BASE, n, 1);
  assertLayout([[maskOff, n]], "select-count");
  let u8c = ensureMem(mem, need);
  u8c = guardFill(mem);
  blit(u8c, mask, maskOff);
  // Thin wrapper: memory offsets + 1 call, no compute (i64 ret via BigInt).
  call = () => {
    const t0 = performance.now();
    const m = asNumber(fn(maskOff, n));
    return { ng: m, ms: performance.now() - t0 };
  };
  onParity = ({ ng, ms }) => console.log(JSON.stringify({ m: ng, ms }));
} else if (op === "scatter") {
  const spec = LANES[kind];
  if (!spec) throw new Error("kind must be i32|i64|f32|f64|u8");
  const [T, B] = spec;
  const src = fileView(aPath, T);
  const mask = fileView(bPath, Uint8Array);
  if (src.length !== n || mask.length !== n) throw new Error("vector size mismatch");
  const fn = needFn("nf_select_scatter_" + kind);
  const al = Math.min(B, 8);
  const [srcOff, c1] = alloc(BASE, n * B, al);
  const [maskOff, c2] = alloc(c1, n, al);
  const [outOff, need] = alloc(c2, n * B, al);
  assertLayout([[srcOff, n * B], [maskOff, n], [outOff, n * B]], "select-scatter");
  let u8s = ensureMem(mem, need);
  u8s = guardFill(mem);
  blit(u8s, src, srcOff);
  blit(u8s, mask, maskOff);
  // Thin wrapper: memory offsets + 1 call, no compute (i64 ret via BigInt).
  call = () => {
    const t0 = performance.now();
    const m = asNumber(fn(srcOff, maskOff, n, outOff));
    return { ng: m, ms: performance.now() - t0 };
  };
  onParity = ({ ng, ms }) => {
    if (ng >= 0) writeFileSync(`${outdir}/sel_out.${kind}`, Buffer.from(mem.buffer.slice(outOff, outOff + ng * B)));
    console.log(JSON.stringify({ m: ng, ms }));
  };
} else if (op === "and" || op === "or" || op === "not") {
  const a = fileView(aPath, Uint8Array);
  if (a.length !== n) throw new Error("a size mismatch");
  const fn = needFn(op === "not" ? "nf_mask_not" : "nf_mask_" + op);
  const [aOff, c1] = alloc(BASE, n, 1);
  let bOff = 0, c2 = c1, b = null;
  if (op !== "not") {
    b = fileView(bPath, Uint8Array);
    if (b.length !== n) throw new Error("b size mismatch");
    [bOff, c2] = alloc(c1, n, 1);
  }
  const [outOff, need] = alloc(c2, n, 1);
  assertLayout(b ? [[aOff, n], [bOff, n], [outOff, n]] : [[aOff, n], [outOff, n]], "select-mask");
  let u8m = ensureMem(mem, need);
  u8m = guardFill(mem);
  blit(u8m, a, aOff);
  if (b) blit(u8m, b, bOff);
  // Thin wrapper: memory offsets + 1 call, no compute.
  call = () => {
    const t0 = performance.now();
    const rc = op === "not" ? fn(aOff, n, outOff) : fn(aOff, bOff, n, outOff);
    return { rc, ms: performance.now() - t0 };
  };
  onParity = ({ rc, ms }) => {
    if (rc !== 0) throw new Error("rc=" + rc);
    writeFileSync(`${outdir}/mask_out.u8`, Buffer.from(mem.buffer.slice(outOff, outOff + n)));
    console.log(JSON.stringify({ rc, ms }));
  };
} else {
  throw new Error("op must be count|scatter|and|or|not");
}

if (mode === "parity") {
  const r0 = call();
  assertGuard(mem, "select-" + op);
  onParity(r0);
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem, "select-" + op);
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
