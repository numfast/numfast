// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Node smoke for nfs_random.mjs (no deps beyond node builtins).
// Usage: node tools/test_nfs_random.mjs  (exit 0 = PASS, nonzero = FAIL)
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import {
  encodeNfsTable,
  nf,
  random,
  randomH2O,
  randomTable,
} from "./nfs_random.mjs";

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
const bytes = (arr) => Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString("hex");
const sameBytes = (a, b) => bytes(a) === bytes(b);

// 1. Same seed -> bit-identical (raw bytes, all dtypes).
t("same-seed-bit-identical", () => {
  const cols = [
    { name: "a", dtype: "i32" },
    { name: "b", dtype: "f32" },
    { name: "c", dtype: "f64" },
    { name: "m", dtype: "u8" },
  ];
  const t1 = randomTable({ seed: 42, nrows: 64, cols });
  const t2 = randomTable({ seed: 42, nrows: 64, cols });
  for (const c of cols) assert(sameBytes(t1[c.name].values, t2[c.name].values), c.name);
  const h1 = randomH2O({ seed: 7, nrows: 32 });
  const h2 = randomH2O({ seed: 7, nrows: 32 });
  for (const k of Object.keys(h1)) assert(sameBytes(h1[k].values, h2[k].values), k);
});

// 2. Different seed -> changes.
t("diff-seed-changes", () => {
  const cols = [
    { name: "a", dtype: "i32" },
    { name: "b", dtype: "f32" },
    { name: "c", dtype: "f64" },
  ];
  const t1 = randomTable({ seed: 1, nrows: 64, cols });
  const t2 = randomTable({ seed: 2, nrows: 64, cols });
  const anyDiff = cols.some((c) => !sameBytes(t1[c.name].values, t2[c.name].values));
  assert(anyDiff, "different seeds must change output");
});

// 3. nrows honored on every column.
t("nrows", () => {
  for (const n of [0, 1, 17]) {
    const tb = randomTable({
      seed: 42, nrows: n,
      cols: [{ name: "x", dtype: "i32" }, { name: "y", dtype: "f64" }],
    });
    assert(tb.x.values.length === n && tb.y.values.length === n);
  }
  const h = randomH2O({ seed: 42, nrows: 13, nv: 3 });
  for (const k of ["id1", "id2", "v1", "v2", "v3"]) assert(h[k].values.length === 13, k);
});

// 4. Multiple dtypes -> correct TypedArrays.
t("multi-dtype", () => {
  const tb = randomTable({
    seed: 42, nrows: 8,
    cols: [
      { name: "a", dtype: "i32" }, { name: "b", dtype: "f32" },
      { name: "c", dtype: "f64" }, { name: "m", dtype: "u8" },
    ],
  });
  assert(tb.a.values instanceof Int32Array && tb.a.dtype === "i32");
  assert(tb.b.values instanceof Float32Array && tb.b.dtype === "f32");
  assert(tb.c.values instanceof Float64Array && tb.c.dtype === "f64");
  assert(tb.m.values instanceof Uint8Array && tb.m.dtype === "u8");
});

// 5. Generator -> encodeNfsTable directly (incl. H2O id1/id2/v1+v2+v3).
t("gen-to-encode", () => {
  const tb = randomTable({
    seed: 42, nrows: 16,
    cols: [{ name: "a", dtype: "i32" }, { name: "b", dtype: "f32" }, { name: "m", dtype: "u8" }],
  });
  const mt = encodeNfsTable(tb);
  assert(mt.nrows === 16 && mt.ncols === 3);
  assert.deepEqual(mt.names, ["a", "b", "m"]);
  const h = randomH2O({ seed: 11, nrows: 10, nv: 3 });
  const mh = encodeNfsTable(h);
  assert.deepEqual(mh.names, ["id1", "id2", "v1", "v2", "v3"]);
  assert(mh.columns[0].encoding.kind === "dict-i32");
  assert(mh.columns[2].encoding.kind === "raw-f64");
  const h1 = randomH2O({ seed: 11, nrows: 10, nv: 1 });
  assert.deepEqual(Object.keys(h1), ["id1", "id2", "v1"]);
});

// 6. Column order preserved.
t("column-order", () => {
  const tb = randomTable({
    seed: 5, nrows: 4,
    cols: [
      { name: "z", dtype: "f64" }, { name: "a", dtype: "i32" }, { name: "m", dtype: "u8" },
    ],
  });
  assert.deepEqual(Object.keys(tb), ["z", "a", "m"]);
  assert.deepEqual(encodeNfsTable(tb).names, ["z", "a", "m"]);
});

// 7. Empty (nrows 0) -> zero-length columns, encodes to nrows 0.
t("empty", () => {
  const tb = randomTable({ seed: 42, nrows: 0, cols: [{ name: "a", dtype: "i32" }] });
  assert(tb.a.values.length === 0);
  const mt = encodeNfsTable(tb);
  assert(mt.nrows === 0 && mt.ncols === 1);
  const h = randomH2O({ seed: 42, nrows: 0 });
  assert(h.id1.values.length === 0 && encodeNfsTable(h).nrows === 0);
});

// 8. Deterministic repeat (3x) + nf.random namespace present.
t("repeat-and-namespace", () => {
  const mk = () => randomTable({ seed: 99, nrows: 24, cols: [{ name: "v", dtype: "f64" }] });
  const a = mk(), b = mk(), c = mk();
  assert(sameBytes(a.v.values, b.v.values) && sameBytes(b.v.values, c.v.values));
  assert(nf.random.table === randomTable && nf.random.h2o === randomH2O);
  assert(random.table === randomTable && random.h2o === randomH2O);
});

// 9. Controlled errors.
t("controlled-errors", () => {
  assert.throws(() => randomTable({ nrows: 4, cols: [{ name: "a", dtype: "i32" }] }), /seed.*mandatory/);
  assert.throws(() => randomTable({ seed: 1.5, nrows: 4, cols: [{ name: "a", dtype: "i32" }] }), /uint32/);
  assert.throws(() => randomTable({ seed: 42, nrows: -1, cols: [{ name: "a", dtype: "i32" }] }), /nrows/);
  assert.throws(() => randomTable({ seed: 42, nrows: 4, cols: [] }), /cols/);
  assert.throws(() => randomTable({ seed: 42, nrows: 4, cols: [{ name: "a", dtype: "i64" }] }), /unsupported dtype/);
  assert.throws(
    () => randomTable({ seed: 42, nrows: 4, cols: [{ name: "a", dtype: "i32" }, { name: "a", dtype: "i32" }] }),
    /duplicate/,
  );
  assert.throws(() => randomH2O({ nrows: 4 }), /seed.*mandatory/);
  assert.throws(() => randomH2O({ seed: 1, nrows: 4, nv: 4 }), /nv/);
  assert.throws(() => randomTable({ seed: 42, nrows: 4, cols: [{ name: "a", dtype: "i32", lo: 5, hi: 5 }] }), /lo<hi/);
});

// 10. No kernel duplication / no IO / no CSV: generator reuses encoder.
t("no-kernel-dup", () => {
  const src = readFileSync(new URL("./nfs_random.mjs", import.meta.url), "utf8");
  assert(src.includes("from \"./nfs_table.mjs\""), "must reuse nfs_table.mjs");
  assert(src.includes("encodeNfsTable"), "must wire to encodeNfsTable");
  for (const frag of ["deltas[i]", "uniqSorted", "pack_elem", "a * m2 + b", "Math.round(close", "10000"])
    assert(!src.includes(frag), "kernel duplication: " + frag);
  assert(!src.includes('from "node:') && !src.includes("from 'node:"), "core imports node builtin");
  for (const frag of ["writeFileSync", "appendFile", "createWriteStream", ".csv", "readFileSync"])
    assert(!src.includes(frag), "IO/CSV forbidden: " + frag);
});

console.log(failures === 0 ? "SMOKE-ALL-PASS" : "SMOKE-FAILURES=" + failures);
process.exit(failures === 0 ? 0 : 1);
