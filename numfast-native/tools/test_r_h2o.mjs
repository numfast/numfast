// Parity: full H2O (r_h2o.mjs) vs scalar oracle + vs r_ref_fixture.json + encode linkage.
// Usage: node tools/test_r_h2o.mjs (exit 0 = PASS, nonzero = FAIL).
// Oracles imported read-only; r_rng.mjs / nfs_table.mjs never modified.
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { makeR, rRound } from "./r_rng.mjs";
import { RBlockSource } from "./r_block.mjs";
import { RColumnSink } from "./r_colsink.mjs";
import {
  H2O_COLUMNS,
  blockH2O,
  encodeNfsTable,
  fmtId,
  formatH2OIds,
  h2oDns,
  scalarH2O,
} from "./r_h2o.mjs";

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
const eqTable = (a, b) => {
  assert.deepEqual(Object.keys(a), H2O_COLUMNS, "col order");
  assert.deepEqual(Object.keys(b), H2O_COLUMNS, "col order");
  for (const k of H2O_COLUMNS) {
    const va = a[k].values, vb = b[k].values;
    assert(a[k].dtype === b[k].dtype, k + " dtype");
    assert(va.length === vb.length, k + " len");
    for (let i = 0; i < va.length; i++) {
      if (k === "v3") assert(Object.is(va[i], vb[i]), `${k}[${i}] ${va[i]}!==${vb[i]}`);
      else assert(va[i] === vb[i], `${k}[${i}] ${va[i]}!==${vb[i]}`);
    }
  }
};
const fx = JSON.parse(readFileSync(new URL("./r_ref_fixture.json", import.meta.url), "utf8"));

// 1. Full N=100 K=10 seed 108: block === scalar (bit-by-bit, all 9 cols)
//    + formatted heads vs fixture (display contract of groupby-datagen.R).
t("h2o-N100-fixture", () => {
  const B = blockH2O({ seed: 108, N: 100, K: 10 });
  const S = scalarH2O({ seed: 108, N: 100, K: 10 });
  eqTable(B, S);
  const F = formatH2OIds(B);
  assert.deepEqual(F.id1.slice(0, 10), fx.datagen_mini_heads.id1, "id1 heads");
  assert.deepEqual(F.id2.slice(0, 10), fx.datagen_mini_heads.id2, "id2 heads");
  assert.deepEqual(F.id3.slice(0, 10), fx.datagen_mini_heads.id3, "id3 heads");
  for (const [k, fk] of [["id4", "id4"], ["id5", "id5"], ["id6", "id6"], ["v1", "v1_head"], ["v2", "v2_head"]]) {
    const head = fx.datagen_mini_heads[fk] ?? fx.datagen_mini[fk];
    assert.deepEqual(Array.from(B[k].values.slice(0, 10)), head, k + " heads");
  }
  fx.datagen_mini.v3_head6.forEach((s, i) => assert(Number(s) === B.v3.values[i], `v3 ${i}`));
  assert.deepEqual(Array.from(B.v3.values.slice(0, 10), (v) => v.toFixed(6)), fx.datagen_mini.v3_head6, "v3 fmt6");
  // v3u exact channel: raw runif stream (17-sig-digit strings) via same stream order.
  const sink = new RColumnSink(new RBlockSource().init(108));
  for (const dn of h2oDns(100, 10)) sink.sampleReplaceBlock(dn, 100);
  const raw = sink.runifBlock(100, 0, 100);
  fx.datagen_mini.v3u_head17.slice(0, 5).forEach((s, i) => assert(Number(s) === raw[i], `v3u ${i}`));
  assert(Object.is(sink.rRoundBlock(raw, 6)[0], B.v3.values[0]), "v3u->v3 stream");
});

// 2. Several N/K/seeds: block === scalar bit-by-bit (incl. N=0, N<K, 1-chunk/2-chunk dns).
t("h2o-multi-NK-seeds", () => {
  const cases = [
    [108, 100, 10], [42, 1000, 7], [7, 1, 1], [5, 0, 10], [42, 5, 10],
    [12345, 20000, 100], [99, 1000, 65536], [1, 5000, 3], [108, 10000, 10],
  ];
  for (const [seed, N, K] of cases) eqTable(blockH2O({ seed, N, K }), scalarH2O({ seed, N, K }));
});

// 3. Independent oracle cross-check: manual makeR stream (no r_h2o helpers)
//    === blockH2O, for one non-fixture case (guards against shared-helper bias).
t("h2o-oracle-crosscheck", () => {
  const seed = 777, N = 1000, K = 13;
  const R = makeR(seed);
  const sub = Math.max(1, Math.floor(N / K));
  const want = [
    Int32Array.from(R.sampleReplace(K, N)), Int32Array.from(R.sampleReplace(K, N)),
    Int32Array.from(R.sampleReplace(sub, N)), Int32Array.from(R.sampleReplace(K, N)),
    Int32Array.from(R.sampleReplace(K, N)), Int32Array.from(R.sampleReplace(sub, N)),
    Int32Array.from(R.sampleReplace(5, N)), Int32Array.from(R.sampleReplace(15, N)),
  ];
  const wantV3 = Float64Array.from(R.runif(N, 0, 100), (x) => rRound(x, 6));
  const B = blockH2O({ seed, N, K });
  for (let c = 0; c < 8; c++) assert.deepEqual(Array.from(B[H2O_COLUMNS[c]].values), Array.from(want[c]), H2O_COLUMNS[c]);
  assert.deepEqual(Array.from(B.v3.values), Array.from(wantV3), "v3");
});

// 4. encodeNfsTable linkage (encode untouched): fixed order, dtypes, zero-copy, N=0.
t("h2o-encode-link", () => {
  const B = blockH2O({ seed: 108, N: 100, K: 10 });
  assert.deepEqual(Object.keys(B), H2O_COLUMNS, "table order");
  for (const k of H2O_COLUMNS.slice(0, 8)) assert(B[k].values instanceof Int32Array && B[k].dtype === "i32", k);
  assert(B.v3.values instanceof Float64Array && B.v3.dtype === "f64", "v3");
  const mt = encodeNfsTable(B);
  assert(mt.nrows === 100 && mt.ncols === 9, "shape");
  assert.deepEqual(mt.names, H2O_COLUMNS, "encode order");
  for (const c of mt.columns.slice(0, 8)) assert(c.encoding.kind === "dict-i32", c.name);
  assert(mt.columns[8].encoding.kind === "raw-f64", "v3 encoding");
  assert(mt.columns[0].shared === true, "zero-copy coerce");
  // scalar table encodes to identical bytes (same draws -> same MemTable).
  const ms = encodeNfsTable(scalarH2O({ seed: 108, N: 100, K: 10 }));
  assert.deepEqual(ms.names, mt.names);
  for (let c = 0; c < 9; c++) {
    const a = mt.columns[c].data, b = ms.columns[c].data;
    assert(a.length === b.length, "enc len " + c);
    for (let i = 0; i < a.length; i++) assert(Object.is(a[i], b[i]), `enc ${c}[${i}]`);
  }
  const e0 = encodeNfsTable(blockH2O({ seed: 1, N: 0, K: 10 }));
  assert(e0.nrows === 0 && e0.ncols === 9 && e0.names.length === 9, "empty encodes");
  assert(fmtId(10, 3) === "id010" && fmtId(10, 10) === "id0000000010", "fmt contract");
});

// 5. Guards + hygiene: arg validation, browser-safe, oracles untouched.
t("h2o-guards-hygiene", () => {
  assert.throws(() => blockH2O({ N: 10 }), /mandatory/);
  assert.throws(() => scalarH2O({ N: 10 }), /mandatory/);
  assert.throws(() => blockH2O({ seed: 1.5, N: 10 }), /int/);
  assert.throws(() => blockH2O({ seed: 1, N: -1 }), /N/);
  assert.throws(() => blockH2O({ seed: 1, N: 10, K: 0 }), /K/);
  assert.deepEqual(h2oDns(5, 10), [10, 10, 1, 10, 10, 1, 5, 15], "N<K clamp");
  assert.deepEqual(h2oDns(100, 10), [10, 10, 10, 10, 10, 10, 5, 15], "fixture dns");
  const src = readFileSync(new URL("./r_h2o.mjs", import.meta.url), "utf8");
  assert(!src.includes('from "node:') && !src.includes("from 'node:"), "browser-safe");
  const oracle = readFileSync(new URL("./r_rng.mjs", import.meta.url), "utf8");
  assert(oracle.includes("sampleKind"), "oracle untouched (still has sampleKind)");
  const enc = readFileSync(new URL("./nfs_table.mjs", import.meta.url), "utf8");
  assert(enc.includes("raw-f64") && enc.includes("dict-i32"), "encode untouched");
});

console.log(failures === 0 ? "PARITY-ALL-PASS" : "PARITY-FAILURES=" + failures);
process.exit(failures === 0 ? 0 : 1);
