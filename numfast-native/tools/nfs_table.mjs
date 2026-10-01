// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Minimal multi-column in-memory table over existing mechanisms.
//
// Input:  { id1: {values, dtype}, id2: {values, dtype}, v1: {values, dtype}, ... }
// Output: MemTable { nrows, ncols, names, columns, layout } — all in-memory.
//
// Mechanisms reused (same semantics, pure JS — no .wasm binary needed):
// - f32 delta encode: exact mirror of develop/browser-demo encodeNfs
//   (fixed-point x1e4 -> int32 deltas). Parity-tested, never modified.
// - i32 dictionary: sorted-order uniq+codes, mirrors
//   unique::unique_inverse_from_perm (uniq ascending, inv = sorted position,
//   uniq[inv] == keys). NOT first-appearance order.
// - packPair: mirrors core::numeric::pack_elem (a*m2+b, negative/overflow reject).
// - layout: bump-alloc offsets mirror wasm_mem.mjs (BASE=0x101000, alloc/alignUp).
//
// Browser-safe: NO node:fs import here (directly or transitively). BASE/
// alignUp/alloc are 3-line local mirrors of wasm_mem.mjs for exactly this
// reason — importing wasm_mem.mjs would drag node:fs into a browser bundle.
// This is NOT a full NFS format layer: no headers/sections/serialization/
// random-access. Do NOT call MemTable "NFS".

// Mirrors wasm_mem.mjs BASE = STACK_TOP(0x100000)+GUARD(0x1000) (local copy, see above).
const BASE = 0x101000;
const alignUp = (off, a) => (off + a - 1) & ~(a - 1);
// Sequential bump allocator, same contract as wasm_mem.mjs alloc().
function alloc(cur, bytes, align = 8) {
  const off = alignUp(cur, align);
  return [off, off + bytes];
}

// Minimal dtype scope: only lanes already covered by existing kernels/helpers.
// IN scope:  i32 (pack/pack_fused/unique/select), f32 (select/scatter),
//            f64 (groupby/select/sorted/carry), u8 (mask/select lanes).
// OUT scope: i64 (kernels exist but no JS single-col encoder covers it),
//   strings/utf8 (pattern kernel takes raw bytes+offs, not a column dtype),
//   f16/bool/nested — rejected with a controlled error.
const LANES = {
  i32: [Int32Array, 4],
  f32: [Float32Array, 4],
  f64: [Float64Array, 8],
  u8: [Uint8Array, 1],
};
export const SUPPORTED_DTYPES = Object.keys(LANES);

// Zero-copy when input already is the right TypedArray (subarray view shares
// the buffer); otherwise one honest copy via the TypedArray constructor.
// Throws a controlled error on unsupported dtype / missing / bad values.
export function coerceColumn(values, dtype) {
  const spec = LANES[dtype];
  if (!spec) throw new Error(`nfs_table: unsupported dtype '${dtype}' (scope: ${SUPPORTED_DTYPES.join(",")})`);
  if (values === null || values === undefined)
    throw new Error(`nfs_table: column missing values (dtype ${dtype})`);
  const [T] = spec;
  if (values instanceof T) return { data: values.subarray(0), shared: true };
  try {
    return { data: new T(values), shared: false };
  } catch (e) {
    throw new Error(`nfs_table: bad values for dtype ${dtype}: ${e.message}`);
  }
}

// Exact mirror of browser-demo encodeNfs (dataset.ts / app.js).
// n=0: {empty deltas, base 0} (single-col encoder has no empty contract).
export function encodeFloat32Deltas(close) {
  const n = close.length;
  if (n === 0) return { deltas: new Int32Array(0), base: 0 };
  const deltas = new Int32Array(n);
  const base = Math.round(close[0] * 10000);
  let prev = base;
  for (let i = 0; i < n; i++) {
    const fx = Math.round(close[i] * 10000);
    deltas[i] = i === 0 ? 0 : fx - prev;
    prev = fx;
  }
  return { deltas, base };
}

// Sorted-order dictionary, mirrors unique_inverse_from_perm_i32 semantics.
export function dictionaryEncodeI32(keys) {
  const n = keys.length;
  const uniqSorted = [...new Set(keys)].sort((a, b) => a - b);
  const pos = new Map(uniqSorted.map((v, i) => [v, i]));
  const uniq = new Int32Array(uniqSorted);
  const codes = new Int32Array(n);
  for (let i = 0; i < n; i++) codes[i] = pos.get(keys[i]);
  return { uniq, codes };
}

// Mirrors core::numeric::pack_elem: out[i] = k1[i]*m2 + k2[i].
// Negative input or int32 overflow -> controlled error (never wraps).
export function packPair(k1, k2, m2) {
  if (k1.length !== k2.length) throw new Error("nfs_table: packPair length mismatch");
  if (!Number.isInteger(m2) || m2 <= 0) throw new Error("nfs_table: packPair m2 must be positive int");
  const n = k1.length;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const a = k1[i], b = k2[i];
    if (a < 0 || b < 0) throw new Error(`nfs_table: packPair negative input at row ${i}`);
    const v = a * m2 + b; // exact: operands in int32 range, product < 2^53
    if (v > 2147483647) throw new Error(`nfs_table: packPair int32 overflow at row ${i}`);
    out[i] = v;
  }
  return out;
}

// Main entry: object of {values, dtype} columns -> MemTable (in-memory only).
// Preserves Object.keys insertion order in `names`. All columns must share
// one length; {} -> empty table (nrows 0, ncols 0). Controlled errors only.
export function encodeNfsTable(input) {
  if (input === null || typeof input !== "object" || Array.isArray(input))
    throw new Error("nfs_table: input must be {name:{values,dtype}}");
  const names = Object.keys(input);
  if (names.length === 0) return { nrows: 0, ncols: 0, names: [], columns: [], layout: [] };
  const staged = [];
  let nrows = -1;
  for (const name of names) {
    const spec = input[name];
    if (spec === null || typeof spec !== "object" || Array.isArray(spec))
      throw new Error(`nfs_table: column '${name}' must be {values,dtype}`);
    if (typeof spec.dtype !== "string")
      throw new Error(`nfs_table: column '${name}' missing explicit dtype`);
    const { data, shared } = coerceColumn(spec.values, spec.dtype);
    if (nrows < 0) nrows = data.length;
    else if (data.length !== nrows)
      throw new Error(`nfs_table: length mismatch in '${name}' (got ${data.length}, want ${nrows})`);
    staged.push({ name, dtype: spec.dtype, data, shared });
  }
  const columns = staged.map((c) => {
    if (c.dtype === "f32") {
      const { deltas, base } = encodeFloat32Deltas(c.data);
      return { ...c, encoding: { kind: "delta-f32", base, deltas } };
    }
    if (c.dtype === "i32") {
      const { uniq, codes } = dictionaryEncodeI32(c.data);
      return { ...c, encoding: { kind: "dict-i32", uniq, codes } };
    }
    if (c.dtype === "f64") return { ...c, encoding: { kind: "raw-f64" } };
    return { ...c, encoding: { kind: "raw-u8" } }; // u8
  });
  let cur = BASE;
  const layout = [];
  for (const c of columns) {
    const B = LANES[c.dtype][1];
    const [off, next] = alloc(cur, c.data.length * B, 8);
    layout.push({ name: c.name, off, bytes: c.data.length * B });
    cur = next;
  }
  return { nrows, ncols: names.length, names, columns, layout };
}
