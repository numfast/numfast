// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Minimal deterministic table generator over nfs_table.mjs.
//
// Output feeds DIRECTLY into encodeNfsTable: {col:{values,dtype}}.
// No CSV, no disk writes, no kernel duplication (no delta/dict/pack here).
// PRNG: mulberry32 (32-bit, fully deterministic for a given uint32 seed).
// Browser-safe: no node: imports (directly or transitively).
import { encodeNfsTable, SUPPORTED_DTYPES } from "./nfs_table.mjs";

// Re-exported so callers wire generator -> encoder without new imports.
export { encodeNfsTable, SUPPORTED_DTYPES };

const UINT32 = 4294967296;

// seed must be an explicit uint32 int. No default — seed is mandatory.
function spawn(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
    throw new Error("nfs_random: seed must be uint32 int (mandatory)");
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / UINT32; // [0,1)
  };
}

function checkNrows(nrows) {
  if (!Number.isInteger(nrows) || nrows < 0)
    throw new Error("nfs_random: nrows must be non-negative int");
}

function checkRange(dtype, lo, hi) {
  if (typeof lo !== "number" || typeof hi !== "number" || !(hi > lo))
    throw new Error(`nfs_random: column dtype '${dtype}' needs lo<hi numbers`);
}

// Defaults per dtype (only encodeNfsTable lanes).
const DEFAULTS = {
  i32: { lo: 0, hi: 100 },
  f32: { lo: 0, hi: 100 },
  f64: { lo: 0, hi: 100 },
  u8: { lo: 0, hi: 256 },
};

const CTORS = {
  i32: Int32Array,
  f32: Float32Array,
  f64: Float64Array,
  u8: Uint8Array,
};

// Thin namespace entry: {seed, nrows, cols:[{name,dtype,lo?,hi?}]} -> table.
// Preserves cols array order. One PRNG stream, columns filled in order.
export function randomTable({ seed, nrows, cols } = {}) {
  if (seed === undefined) throw new Error("nfs_random: seed is mandatory");
  checkNrows(nrows);
  if (!Array.isArray(cols) || cols.length === 0) {
    if (nrows === 0 && Array.isArray(cols) && cols.length === 0) return {};
    throw new Error("nfs_random: cols must be non-empty array");
  }
  const seen = new Set();
  const rand = spawn(seed);
  const out = {};
  for (const c of cols) {
    if (c === null || typeof c !== "object" || Array.isArray(c))
      throw new Error("nfs_random: each col must be {name,dtype,lo?,hi?}");
    const { name, dtype } = c;
    if (typeof name !== "string" || name.length === 0)
      throw new Error("nfs_random: each col needs non-empty string name");
    if (seen.has(name)) throw new Error(`nfs_random: duplicate column '${name}'`);
    seen.add(name);
    if (!CTORS[dtype])
      throw new Error(`nfs_random: unsupported dtype '${dtype}' (scope: ${SUPPORTED_DTYPES.join(",")})`);
    const d = DEFAULTS[dtype];
    const lo = c.lo === undefined ? d.lo : c.lo;
    const hi = c.hi === undefined ? d.hi : c.hi;
    checkRange(dtype, lo, hi);
    const T = CTORS[dtype];
    const values = new T(nrows);
    const span = hi - lo;
    if (dtype === "i32" || dtype === "u8") {
      const hiI = Math.floor(hi);
      for (let i = 0; i < nrows; i++) values[i] = lo + Math.floor(rand() * (hiI - lo));
    } else {
      for (let i = 0; i < nrows; i++) values[i] = lo + rand() * span;
    }
    if (dtype === "u8") for (let i = 0; i < nrows; i++) values[i] &= 0xff;
    out[name] = { values, dtype };
  }
  return out;
}

// H2O-style fixture: id1/i32 + id2/i32 + v1..vN/f64. nv in 1..3.
export function randomH2O({ seed, nrows, k1 = 8, k2 = 8, nv = 1 } = {}) {
  if (seed === undefined) throw new Error("nfs_random: seed is mandatory");
  checkNrows(nrows);
  for (const [k, v] of [["k1", k1], ["k2", k2]]) {
    if (!Number.isInteger(v) || v <= 0)
      throw new Error(`nfs_random: ${k} must be positive int`);
  }
  if (!Number.isInteger(nv) || nv < 1 || nv > 3)
    throw new Error("nfs_random: nv must be int 1..3");
  const cols = [
    { name: "id1", dtype: "i32", lo: 0, hi: k1 },
    { name: "id2", dtype: "i32", lo: 0, hi: k2 },
  ];
  for (let i = 1; i <= nv; i++) cols.push({ name: "v" + i, dtype: "f64", lo: 0, hi: 100 });
  return randomTable({ seed, nrows, cols });
}

export const random = { table: randomTable, h2o: randomH2O };
export const nf = { random };
