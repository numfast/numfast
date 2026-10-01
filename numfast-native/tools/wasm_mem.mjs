// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Shared linear-memory helper for Node WASM drivers. No compute, no kernels:
// base/align/grow/blit + BigInt wrapper for i64 returns + bench loop.
// Imported by wasm_*.mjs; keeps per-driver code to memory offsets + 1 call.
import { readFileSync } from "node:fs";

// WASM stack grows DOWN from 0x100000 (measured: scatter_pass `pos`
// histogram occupies [0xFFC00, 0x100000); old low-BASE buffers collided once
// n*4+16 crossed 1MiB — silent +182 groups at 1M). Buffers start 4K
// ABOVE the stack top; a canary in between fails LOUD if a toolchain
// rebuild ever moves the stack top upward into our buffers.
export const STACK_TOP = 0x100000;
export const GUARD = 0x1000;
export const CANARY = 0x9e3779b9;
export const BASE = STACK_TOP + GUARD; // 0x101000, offset 0 stays NULL in Rust FFI
export const PAGE = 65536;

export const alignUp = (off, a) => (off + a - 1) & ~(a - 1);

// Sequential bump allocator: alloc(BASE, 40, 8) -> [BASE, BASE+40].
// Returns [off, next]; caller threads `next` through layout.
export function alloc(cur, bytes, align = 8) {
  const off = alignUp(cur, align);
  return [off, off + bytes];
}

// Harness-only length gate: n must be a safe int in [0, 2**31).
// N>=2**31 is a harness assertion only — kernels/ABI/IR never change.
export function assertN(n, label = "n") {
  if (!Number.isInteger(n) || n < 0 || n >= 2147483648)
    throw new Error(`wasm harness: ${label}=${n} out of [0,2**31)`);
}

// Harness-only layout gate: every [off, off+bytes) must sit at/above BASE
// (never inside [STACK_TOP,BASE) guard) and ranges must not overlap.
// ranges: [[off, bytes], ...]. Zero-byte ranges are allowed (n==0 lanes).
export function assertLayout(ranges, label = "") {
  const tag = label ? " [" + label + "]" : "";
  const rs = ranges.map(([off, bytes]) => [off, off + bytes]);
  for (const [s, e] of rs) {
    if (s < BASE) throw new Error("wasm layout below BASE" + tag + " off=" + s);
    if (e < s) throw new Error("wasm layout negative range" + tag);
    if (s < STACK_TOP + GUARD) throw new Error("wasm layout inside guard" + tag);
  }
  const sorted = [...rs].sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < sorted.length; i++)
    if (sorted[i][0] < sorted[i - 1][1])
      throw new Error("wasm buffers overlap" + tag + " [" + sorted[i - 1] + ") vs [" + sorted[i] + ")");
}

export function fileView(p, T) {
  const b = readFileSync(p);
  return new T(b.buffer, b.byteOffset, b.byteLength / T.BYTES_PER_ELEMENT);
}

export async function loadWasm(wasmPath, imports = {}) {
  const bytes = readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  return instance;
}

// Grow memory if `need` exceeds current size; returns a FRESH u8 view
// (old views detach on grow, so always re-fetch after this call).
export function ensureMem(mem, need) {
  const have = mem.buffer.byteLength;
  if (need > have) mem.grow(Math.ceil((need - have) / PAGE));
  return new Uint8Array(mem.buffer);
}

// Canary in [STACK_TOP, BASE): downward frames never touch it; an upward
// stack-top drift (toolchain layout change) eats it -> loud failure.
// Call guardFill(mem) right after ensureMem (returns fresh u8 view),
// then guardOk(mem)/assertGuard(mem) after the kernel call.
export function guardFill(mem) {
  const g = new Uint32Array(mem.buffer, STACK_TOP, GUARD / 4);
  g.fill(CANARY);
  return new Uint8Array(mem.buffer);
}

export function guardOk(mem) {
  const g = new Uint32Array(mem.buffer, STACK_TOP, GUARD / 4);
  for (let i = 0; i < g.length; i++) if (g[i] !== CANARY) return false;
  return true;
}

export function assertGuard(mem, label = "") {
  if (!guardOk(mem)) throw new Error("wasm stack layout drifted into buffers" + (label ? " [" + label + "]" : ""));
}

export function blit(u8, typed, off) {
  u8.set(new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength), off);
}

export function blitBytes(u8, bytes, off) {
  u8.set(bytes, off);
}

// wasm32 i64 returns surface as BigInt; all our ng/m results fit in Number.
export const asNumber = (v) => Number(v);

export function dataView(mem) {
  return new DataView(mem.buffer);
}

// Shared bench shape: 5 warm + 10 measured, sorted runs + median.
// call() returns {ms, ...}; a present rc/ng field is validated like drivers did.
export function bench(call, warm = 5, runs = 10) {
  const check = (r) => {
    if ("rc" in r && r.rc !== 0) throw new Error("rc=" + r.rc);
    if ("ng" in r && r.ng < 0) throw new Error("ng=" + r.ng);
  };
  for (let i = 0; i < warm; i++) check(call());
  const times = [];
  for (let i = 0; i < runs; i++) { const r = call(); check(r); times.push(r.ms); }
  times.sort((a, b) => a - b);
  return { warm, runs: times, median_ms: times[runs >> 1] };
}
