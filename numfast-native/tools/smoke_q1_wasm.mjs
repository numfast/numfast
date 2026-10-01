// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// smoke_q1_wasm: minimal H2O Q1 groupby (sum v1 by id1) over the REAL wasm.
// Data: canonical R-compatible stream (r_rng.mjs, seed 108).
// Encode: encodeNfsTable (nfs_table.mjs). Kernel: nf_group_sum_count (single
// export = full Q1; no chain, no new kernel). Reference: independent JS loop.
// Usage: node tools/smoke_q1_wasm.mjs [n] [k]  (defaults 10000 100)
// Exit 0 = PASS (exact parity), nonzero = FAIL. Last stdout line = JSON.
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { makeR, rRound } from "./r_rng.mjs";
import { encodeNfsTable } from "./nfs_table.mjs";
import { BASE, alloc, blit, ensureMem, loadWasm } from "./wasm_mem.mjs";

const ROOT = join(fileURLToPath(import.meta.url), "..");
const SEED = 108;
const N = Number(process.argv[2] ?? 10000);
const K = Number(process.argv[3] ?? 100);
if (!Number.isInteger(N) || N < 1000 || N > 10000) throw new Error(`N must be int 1000..10000 (got ${N})`);
if (!Number.isInteger(K) || K <= 0 || K > N) throw new Error(`K must be int 1..N (got ${K})`);

const t = (f) => { const t0 = performance.now(); const r = f(); return [r, performance.now() - t0]; };
const fail = (msg, extra = {}) => {
  console.log(JSON.stringify({ pass: false, fail: msg, seed: SEED, n: N, k: K, ...extra }));
  process.exit(1);
};

// ---- Stage 1: datagen (R stream; id-strings skipped: nfs_table scope=i32/f64/u8, no utf8 lane) ----
const [gen, tGen] = t(() => {
  const R = makeR(SEED); // rejection = R>=4.x default; verified vs R 4.3.2 seed 108
  const draw1 = R.sampleReplace(K, N); // 1-based, R do_sample replace path
  const keys = new Int32Array(N);
  for (let i = 0; i < N; i++) keys[i] = draw1[i] - 1; // -> dense 0..K-1
  const raw = R.runif(N, 0, 100);
  const vals = new Float64Array(N);
  for (let i = 0; i < N; i++) vals[i] = rRound(raw[i], 2); // R round(), half-to-even
  return { keys, vals };
});

// ---- Stage 2: encodeNfsTable ----
const [tab, tEnc] = t(() => encodeNfsTable({
  id1: { values: gen.keys, dtype: "i32" },
  v1: { values: gen.vals, dtype: "f64" },
}));
// encoding checks
if (tab.nrows !== N || tab.ncols !== 2 || tab.names.join(",") !== "id1,v1") fail("encoding-shape", { tab: { nrows: tab.nrows, ncols: tab.ncols, names: tab.names } });
const cId = tab.columns[0], cV = tab.columns[1];
if (cId.encoding.kind !== "dict-i32" || cV.encoding.kind !== "raw-f64") fail("encoding-kind", { id: cId.encoding.kind, v: cV.encoding.kind });
for (let i = 0; i < N; i++) { const k = cId.data[i]; if (!Number.isInteger(k) || k < 0 || k >= K) fail("encoding-key-range", { i, k }); }
if (cId.encoding.uniq.length !== K) fail("encoding-ngroups", { uniq: cId.encoding.uniq.length, want: K });
for (const v of cId.encoding.uniq) { if (!Number.isInteger(v)) fail("encoding-uniq-int", {}); }
for (let i = 1; i < cId.encoding.uniq.length; i++) if (cId.encoding.uniq[i - 1] >= cId.encoding.uniq[i]) fail("encoding-uniq-sorted", {});
for (let i = 0; i < N; i++) if (!Number.isFinite(cV.data[i])) fail("encoding-nonnan", { i });

// ---- Stage 3: independent JS reference (row-order accumulate, same as dense_scatter) ----
const [ref, tRef] = t(() => {
  const sums = new Float64Array(K), counts = new Float64Array(K);
  for (let i = 0; i < N; i++) { sums[gen.keys[i]] += gen.vals[i]; counts[gen.keys[i]]++; }
  return { sums, counts };
});

// ---- Stage 4: load real wasm + layout (mirrors wasm_run.mjs) ----
const [inst, tLoad] = await (async () => { const t0 = performance.now(); const w = await loadWasm(join(ROOT, "numfast_native.wasm")); return [w, performance.now() - t0]; })();
const fn = inst.exports.nf_group_sum_count;
if (typeof fn !== "function") fail("kernel-missing-export", {});
const mem = inst.exports.memory;
let cur = BASE;
const [keysOff, c1] = alloc(cur, N * 4, 8);
const [valsOff, c2] = alloc(c1, N * 8, 8);
const [sumsOff, c3] = alloc(c2, K * 8, 8);
const [countsOff, need] = alloc(c3, K * 8, 8);
const [u8, tBlitPrep] = t(() => ensureMem(mem, need));
const t0b = performance.now();
blit(u8, cId.data, keysOff);
blit(u8, cV.data, valsOff);
const tBlit = performance.now() - t0b + tBlitPrep;

// ---- Stage 5: kernel call (Q1 = this single export) ----
let rc, tKernel;
try {
  const t0 = performance.now();
  rc = fn(keysOff, valsOff, N, sumsOff, countsOff, K);
  tKernel = performance.now() - t0;
} catch (e) { fail("kernel-trap", { error: String(e).slice(0, 200) }); }
if (rc !== 0) fail("kernel-rc", { rc });
const gotSums = new Float64Array(mem.buffer.slice(sumsOff, sumsOff + K * 8));
const gotCounts = new BigInt64Array(mem.buffer.slice(countsOff, countsOff + K * 8));

// ---- Stage 6: verify ----
const t0v = performance.now();
let maxAbsDiff = 0, countMismatch = 0, nanCount = 0, nonFiniteSum = 0;
let sumCounts = 0n;
for (let g = 0; g < K; g++) {
  const s = gotSums[g], c = gotCounts[g];
  if (Number.isNaN(s)) nanCount++;
  if (!Number.isFinite(s)) nonFiniteSum++;
  if (c < 0n) fail("count-negative", { g, c: String(c) });
  sumCounts += c;
  if (c !== BigInt(Math.round(ref.counts[g]))) countMismatch++;
  const d = Math.abs(s - ref.sums[g]);
  if (d > maxAbsDiff) maxAbsDiff = d;
  if (!(d === 0)) fail("parity-sum", { g, got: s, want: ref.sums[g], diff: d });
}
if (countMismatch !== 0) fail("parity-counts", { countMismatch });
if (nanCount !== 0 || nonFiniteSum !== 0) fail("nan-overflow", { nanCount, nonFiniteSum });
if (sumCounts !== BigInt(N)) fail("counts-total", { sum: String(sumCounts), want: N });
if (need > mem.buffer.byteLength) fail("memory-overrun", { need, have: mem.buffer.byteLength });
const tVerify = performance.now() - t0v;

console.log(`Q1 seed=${SEED} N=${N} K=${K} groups=${cId.encoding.uniq.length} rc=${rc} maxAbsDiff=${maxAbsDiff} countsOK sum=${sumCounts}`);
console.log(`timings ms: gen=${tGen.toFixed(2)} encode=${tEnc.toFixed(2)} ref=${tRef.toFixed(2)} load=${tLoad.toFixed(2)} blit=${tBlit.toFixed(2)} kernel=${tKernel.toFixed(2)} verify=${tVerify.toFixed(2)}`);
console.log(JSON.stringify({
  pass: true, export: "nf_group_sum_count", seed: SEED, n: N, k: K, ngroups: K, rc,
  layout: { keysOff, valsOff, sumsOff, countsOff, need, base: BASE },
  inputs: { keys: "id1 int32 dense [0,K)", values: "v1 float64 round(runif(0,100),2)", idStrings: "skipped: nfs_table has no utf8 lane (fixated)" },
  maxAbsDiff, countMismatch, nanCount, sumCounts: String(sumCounts),
  timings_ms: { gen: tGen, encode: tEnc, ref: tRef, load: tLoad, blit: tBlit, kernel: tKernel, verify: tVerify },
}));
