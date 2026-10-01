// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node smoke for nfs_table.mjs (no deps beyond node builtins).
// Usage: node tools/test_nfs_table.mjs  (exit 0 = PASS, nonzero = FAIL)
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import {
  SUPPORTED_DTYPES,
  coerceColumn,
  dictionaryEncodeI32,
  encodeFloat32Deltas,
  encodeNfsTable,
  packPair,
} from "./nfs_table.mjs";

let failures = 0;
function t(name, fn) {
  try {
    fn();
    console.log("PASS " + name);
  } catch (e) {
    failures++;
    console.log("FAIL " + name + " :: " + (e && e.message ? e.message : e));
  }
}
// Reference single-column encoder: verbatim copy of browser-demo encodeNfs.
function refEncodeNfs(close) {
  const n = close.length;
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
const eq = (a, b) =>
  assert(a.length === b.length && a.every((v, i) => v === b[i]), `arrays differ`);

// 1. Single f32 column: regression + parity with single-column encoder.
t("1col-f32-regression", () => {
  const close = new Float32Array([100, 100.5, 99.25, 101.75]);
  const mt = encodeNfsTable({ close: { values: close, dtype: "f32" } });
  assert(mt.nrows === 4 && mt.ncols === 1 && mt.names[0] === "close");
  const ref = refEncodeNfs(close);
  const enc = mt.columns[0].encoding;
  assert(enc.kind === "delta-f32" && enc.base === ref.base);
  eq([...enc.deltas], [...ref.deltas]);
});

// 2. Three columns of different types; order preserved; lengths checked.
t("3col-mixed-order", () => {
  const mt = encodeNfsTable({
    a: { values: new Int32Array([3, 1, 3, 2]), dtype: "i32" },
    b: { values: new Float32Array([1.5, 2.5, 3.5, 4.5]), dtype: "f32" },
    c: { values: new Float64Array([10, 20, 30, 40]), dtype: "f64" },
  });
  assert.deepEqual(mt.names, ["a", "b", "c"]);
  assert(mt.nrows === 4 && mt.ncols === 3);
  assert(mt.columns.map((c) => c.dtype).join(",") === "i32,f32,f64");
  assert(mt.columns[0].encoding.kind === "dict-i32");
  assert(mt.columns[1].encoding.kind === "delta-f32");
  assert(mt.columns[2].encoding.kind === "raw-f64");
  // layout offsets strictly increasing, 8-byte aligned, BASE=0x101000 start
  assert(mt.layout[0].off === 0x101000 && mt.layout[0].off % 8 === 0);
  for (let i = 1; i < mt.layout.length; i++) assert(mt.layout[i].off > mt.layout[i - 1].off);
});

// 3. H2O-like id1/id2/v1: pack composite + dictionary roundtrip.
t("h2o-id1-id2-v1", () => {
  const id1 = new Int32Array([0, 1, 0, 2, 1]);
  const id2 = new Int32Array([1, 0, 1, 2, 0]);
  const v1 = new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5]);
  const mt = encodeNfsTable({
    id1: { values: id1, dtype: "i32" },
    id2: { values: id2, dtype: "i32" },
    v1: { values: v1, dtype: "f64" },
  });
  assert.deepEqual(mt.names, ["id1", "id2", "v1"]);
  const m2 = Math.max(...id2) + 1;
  const packed = packPair(id1, id2, m2);
  assert([...packed].join(",") === "1,3,1,8,3"); // k1*3+k2
  for (const col of mt.columns.filter((c) => c.dtype === "i32")) {
    const { uniq, codes } = col.encoding;
    for (let i = 0; i < mt.nrows; i++) assert(uniq[codes[i]] === col.data[i]);
  }
  // sorted-order codes: uniq ascending
  const u = mt.columns[0].encoding.uniq;
  for (let i = 1; i < u.length; i++) assert(u[i - 1] < u[i]);
});

// 4. Empty table: {} and zero-length columns.
t("empty-table", () => {
  const e0 = encodeNfsTable({});
  assert(e0.nrows === 0 && e0.ncols === 0 && e0.names.length === 0);
  const e1 = encodeNfsTable({
    a: { values: new Int32Array(0), dtype: "i32" },
    b: { values: new Float32Array(0), dtype: "f32" },
  });
  assert(e1.nrows === 0 && e1.ncols === 2);
});

// 5. Mismatched lengths -> controlled error.
t("length-mismatch-error", () => {
  assert.throws(
    () =>
      encodeNfsTable({
        a: { values: new Int32Array([1, 2, 3]), dtype: "i32" },
        b: { values: new Float64Array([1, 2]), dtype: "f64" },
      }),
    /length mismatch/,
  );
});

// 6. Controlled errors: dtype/format. i64 out of scope by design.
t("controlled-errors", () => {
  assert.throws(() => encodeNfsTable({ x: { values: [1], dtype: "i64" } }), /unsupported dtype/);
  assert.throws(() => encodeNfsTable({ x: { values: [1] } }), /missing explicit dtype/);
  assert.throws(() => encodeNfsTable({ x: { dtype: "i32" } }), /missing values/);
  assert.throws(() => encodeNfsTable([]), /must be \{name/);
  assert.throws(() => packPair(new Int32Array([0, -1]), new Int32Array([0, 0]), 4), /negative/);
  assert.throws(
    () => packPair(new Int32Array([2147483647]), new Int32Array([1]), 2),
    /overflow/,
  );
});

// 7. Zero-copy when typed already; honest copy otherwise.
t("zero-copy-vs-copy", () => {
  const src = new Int32Array([5, 6, 7]);
  const v = coerceColumn(src, "i32");
  assert(v.shared === true && v.data.buffer === src.buffer);
  const c = coerceColumn([5, 6, 7], "i32");
  assert(c.shared === false && c.data instanceof Int32Array);
  eq([...c.data], [5, 6, 7]);
});

// 8. Core has no Node-fs dependency (browser-safe).
t("core-no-fs", () => {
  const src = readFileSync(new URL("./nfs_table.mjs", import.meta.url), "utf8");
  assert(!src.includes('from "node:') && !src.includes("from 'node:"), "core imports node builtin");
  assert(!src.includes("readFileSync") && !src.includes("writeFileSync"), "core touches fs");
  assert(!src.includes("encodeNfs(close)"), "core must not redefine single-col encoder");
});

// 9. u8 lane + dictionary unit parity (sorted-order, uniq[inv]==keys).
t("u8-and-dict-unit", () => {
  const mt = encodeNfsTable({ m: { values: new Uint8Array([1, 0, 1]), dtype: "u8" } });
  assert(mt.columns[0].encoding.kind === "raw-u8");
  const { uniq, codes } = dictionaryEncodeI32(new Int32Array([30, 10, 30, 20]));
  assert([...uniq].join(",") === "10,20,30");
  assert([...codes].join(",") === "2,0,2,1");
  void encodeFloat32Deltas;
});

console.log(failures === 0 ? "SMOKE-ALL-PASS" : "SMOKE-FAILURES=" + failures);
process.exit(failures === 0 ? 0 : 1);
