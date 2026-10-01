// Full exact-R H2O generator over RBlockSource + RColumnSink.
//
// Oracle: tools/r_rng.mjs (FROZEN — imported read-only, never modified).
// Draw stream (exact groupby-datagen.R order after set.seed(seed), nas=0 sort=0):
//   id1=sample(K,N) -> id2=sample(K,N) -> id3=sample(N//K,N) -> id4=sample(K,N) ->
//   id5=sample(K,N) -> id6=sample(N//K,N) -> v1=sample(5,N) -> v2=sample(15,N) ->
//   v3u=runif(N,0,100) -> v3=round(v3u,6).
// Column order fixed: id1,id2,id3,id4,id5,id6,v1,v2,v3.
// Dtypes: id1..v2=i32 (Int32Array, 1-based like oracle), v3=f64 (Float64Array).
// Output feeds DIRECTLY into encodeNfsTable: {col:{values,dtype}} (encode untouched).
// String ids ("id%03d"/"id%010d") are display-only: see formatH2OIds (numeric path
// has no strings). Pure ESM, no imports beyond relative modules, no node: APIs.
import { makeR, rRound } from "./r_rng.mjs";
import { RBlockSource } from "./r_block.mjs";
import { RColumnSink } from "./r_colsink.mjs";
import { encodeNfsTable, SUPPORTED_DTYPES } from "./nfs_table.mjs";

// Re-exported so callers wire generator -> encoder without new imports.
export { encodeNfsTable, SUPPORTED_DTYPES };

export const H2O_COLUMNS = ["id1", "id2", "id3", "id4", "id5", "id6", "v1", "v2", "v3"];

function checkArgs(seed, N, K) {
  if (seed === undefined) throw new Error("r_h2o: seed is mandatory");
  if (!Number.isInteger(seed)) throw new Error("r_h2o: seed must be int (coerced via |0 like oracle)");
  if (!Number.isInteger(N) || N < 0) throw new Error("r_h2o: N must be non-negative int");
  if (!Number.isInteger(K) || K <= 0) throw new Error("r_h2o: K must be positive int");
}

// Population sizes per column in stream order. id3/id6 draw from N//K
// (groupby-datagen.R); clamp to >=1 so N<K stays a valid sample() call.
export function h2oDns(N, K) {
  const sub = Math.max(1, Math.floor(N / K));
  return [K, K, sub, K, K, sub, 5, 15];
}

function assemble(cols8, v3) {
  const table = {};
  for (let c = 0; c < 8; c++) table[H2O_COLUMNS[c]] = { values: cols8[c], dtype: "i32" };
  table.v3 = { values: v3, dtype: "f64" };
  return table;
}

// Scalar reference: oracle draws (plain arrays) + conversion to TypedArrays.
// Exists for parity/bench baseline only; production path is blockH2O.
export function scalarH2O({ seed, N = 100, K = 10 } = {}) {
  checkArgs(seed, N, K);
  const R = makeR(seed);
  const cols8 = h2oDns(N, K).map((dn) => Int32Array.from(R.sampleReplace(dn, N)));
  const v3 = Float64Array.from(R.runif(N, 0, 100), (x) => rRound(x, 6));
  return assemble(cols8, v3);
}

// Production path: one src/sink pair, stream consumed in column order.
export function blockH2O({ seed, N = 100, K = 10 } = {}) {
  checkArgs(seed, N, K);
  const sink = new RColumnSink(new RBlockSource().init(seed));
  const cols8 = h2oDns(N, K).map((dn) => sink.sampleReplaceBlock(dn, N));
  const v3 = sink.rRoundBlock(sink.runifBlock(N, 0, 100), 6);
  return assemble(cols8, v3);
}

// Display-only formatting (NOT part of encode path): R rendering "id%03d"
// for id1/id2, "id%010d" for id3. Full-string arrays; time separately on bench.
export function fmtId(v, w) {
  return "id" + String(v).padStart(w, "0");
}

export function formatH2OIds(table) {
  const map = (vals, w) => Array.from(vals, (v) => fmtId(v, w));
  return { id1: map(table.id1.values, 3), id2: map(table.id2.values, 3), id3: map(table.id3.values, 10) };
}

export const random = { h2o: blockH2O, h2oScalar: scalarH2O };
export const nf = { random };
