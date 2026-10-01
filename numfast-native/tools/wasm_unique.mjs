// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node driver for unique exports — parity + bench, no deps.
// Only real WASM exports, thin wrapper (memory offsets + 1 call, no compute):
//   nf_unique_inverse_i32, nf_unique_inverse_i64.
// Contract (frozen): sorted-order uniq + inv = sorted-position codes
// (NOT first-appearance); inv lanes int32; ng = distinct count (i64 ret
// surfaces as BigInt on wasm32). Caller-owned scratch (perm/tmp0/tmp1/
// tmp_p) lives in linear memory, same as the native ctypes lane.
// Usage:
//   node wasm_unique.mjs <wasm> <srcPath> <kind> <n> <mode> [outdir]
//   kind: i32|i64 (keys dtype; tmp lanes u32 for i32, u64 for i64)
//   mode=parity: 1 call, writes unique_out.<kind> (ng lanes) + unique_inv.i32
//     (n lanes) to outdir, prints {"ng":..,"ms":..}
//   mode=bench:  5 warm + 10 measured calls, prints {"warm":5,"runs":[...],"median_ms":..}
import { writeFileSync } from "node:fs";
import { alloc, asNumber, assertGuard, assertLayout, assertN, BASE, bench, blit, ensureMem, fileView, guardFill, loadWasm } from "./wasm_mem.mjs";

const [wasmPath, srcPath, kind, nS, mode, outdir] = process.argv.slice(2);
const n = Number(nS);
assertN(n);

const KEYS = { i32: [Int32Array, 4], i64: [BigInt64Array, 8] };
const spec = KEYS[kind];
if (!spec) throw new Error("kind must be i32|i64");
const [TK, BK] = spec;
const BTF = kind === "i32" ? 4 : 8; // tmp0/tmp1 lane bytes (u32 vs u64)

const src = fileView(srcPath, TK);
if (src.length !== n) throw new Error("vector size mismatch");

const instance = await loadWasm(wasmPath);
const mem = instance.exports.memory;
const fn = instance.exports["nf_unique_inverse_" + kind];
if (!fn) throw new Error("nf_unique_inverse_" + kind + " not exported");

// Layout with alignment, BASE=0x101000 (above WASM stack top + guard).
// n==0: all sizes 0, offsets stay non-null; Rust returns 0 w/o deref.
const [keysOff, c1] = alloc(BASE, n * BK, BK);
const [uniqOff, c2] = alloc(c1, n * BK, BK);
const [invOff, c3] = alloc(c2, n * 4, 4);
const [permOff, c4] = alloc(c3, n * 4, 4);
const [tmp0Off, c5] = alloc(c4, n * BTF, BTF);
const [tmp1Off, c6] = alloc(c5, n * BTF, BTF);
const [tmpPOff, need] = alloc(c6, n * 4, 4);
assertLayout([[keysOff, n * BK], [uniqOff, n * BK], [invOff, n * 4], [permOff, n * 4], [tmp0Off, n * BTF], [tmp1Off, n * BTF], [tmpPOff, n * 4]], "unique");

let u8 = ensureMem(mem, need);
u8 = guardFill(mem);
blit(u8, src, keysOff);

// Thin wrapper: memory offsets + 1 call, no compute (i64 ret via BigInt).
function call() {
  const t0 = performance.now();
  const ng = asNumber(fn(keysOff, n, uniqOff, invOff, permOff, tmp0Off, tmp1Off, tmpPOff));
  return { ng, ms: performance.now() - t0 };
}

if (mode === "parity") {
  const { ng, ms } = call();
  assertGuard(mem);
  if (ng < 0) throw new Error("ng=" + ng);
  writeFileSync(`${outdir}/unique_out.${kind}`, Buffer.from(mem.buffer.slice(uniqOff, uniqOff + ng * BK)));
  writeFileSync(`${outdir}/unique_inv.i32`, Buffer.from(mem.buffer.slice(invOff, invOff + n * 4)));
  console.log(JSON.stringify({ ng, ms }));
} else if (mode === "bench") {
  const r = bench(call);
  assertGuard(mem);
  console.log(JSON.stringify(r));
} else {
  throw new Error("mode must be parity|bench");
}
