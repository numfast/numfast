// Bench: full H2O generation scalar oracle vs block (RNG / alloc+construct / format).
// Usage: node tools/bench_r_h2o.mjs <N> [K] [seed]  (one N per process: isolates OOM risk)
// Fixed seed default 42. Prints stage ms + rows/s. No files, no encode changes.
import { makeR, rRound } from "./r_rng.mjs";
import { RBlockSource } from "./r_block.mjs";
import { RColumnSink } from "./r_colsink.mjs";
import { H2O_COLUMNS, blockH2O, formatH2OIds, h2oDns, scalarH2O } from "./r_h2o.mjs";

const N = Number(process.argv[2]);
const K = Number(process.argv[3] ?? 10);
const SEED = Number(process.argv[4] ?? 42);
if (!Number.isInteger(N) || N < 0) throw new Error("bench: N must be non-negative int");
const now = () => performance.now();
const cksum = (tb) => H2O_COLUMNS.map((k) => {
  const v = tb[k].values;
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  return s;
});

// Warmup (discarded): twists, JIT, sink refill paths.
{
  const w = blockH2O({ seed: SEED, N: 10000, K });
  scalarH2O({ seed: SEED, N: 10000, K });
  if (w.v3.values.length !== 10000) throw new Error("warmup shape");
}

// --- scalar staged (same expressions as scalarH2O) ---
let sRng, sAlloc, sTab;
{
  const t0 = now();
  const R = makeR(SEED);
  const dns = h2oDns(N, K);
  const plain8 = dns.map((dn) => R.sampleReplace(dn, N));
  const plainV3u = R.runif(N, 0, 100);
  const plainV3 = plainV3u.map((x) => rRound(x, 6));
  const t1 = now();
  const cols8 = plain8.map((a) => Int32Array.from(a));
  const v3 = Float64Array.from(plainV3);
  const tb = {};
  for (let c = 0; c < 8; c++) tb[H2O_COLUMNS[c]] = { values: cols8[c], dtype: "i32" };
  tb.v3 = { values: v3, dtype: "f64" };
  const t2 = now();
  sRng = t1 - t0; sAlloc = t2 - t1; sTab = tb;
}
// --- block staged (same calls as blockH2O) ---
let bRng, bAlloc, bTab;
{
  const t0 = now();
  const sink = new RColumnSink(new RBlockSource().init(SEED));
  const cols8 = h2oDns(N, K).map((dn) => sink.sampleReplaceBlock(dn, N));
  const v3 = sink.rRoundBlock(sink.runifBlock(N, 0, 100), 6);
  const t1 = now();
  const tb = {};
  for (let c = 0; c < 8; c++) tb[H2O_COLUMNS[c]] = { values: cols8[c], dtype: "i32" };
  tb.v3 = { values: v3, dtype: "f64" };
  const t2 = now();
  bRng = t1 - t0; bAlloc = t2 - t1; bTab = tb;
}
// Staged-vs-library checksum (exact, incl. f64 sums): proves bench timed the same stream.
{
  const refS = cksum(scalarH2O({ seed: SEED, N, K }));
  const gotS = cksum(sTab);
  const refB = cksum(blockH2O({ seed: SEED, N, K }));
  const gotB = cksum(bTab);
  for (let i = 0; i < 9; i++) {
    if (!Object.is(refS[i], gotS[i])) throw new Error(`scalar stage diverged col ${i}`);
    if (!Object.is(refB[i], gotB[i])) throw new Error(`block stage diverged col ${i}`);
    if (!Object.is(refS[i], refB[i])) throw new Error(`scalar vs block diverged col ${i}`);
  }
}
// Formatting stage (display-only strings, NOT encode path): only timed for N<=100K.
let sFmt = NaN, bFmt = NaN;
if (N <= 100000) {
  let t0 = now();
  const f = formatH2OIds(sTab);
  sFmt = now() - t0;
  if (f.id1.length !== N || f.id3.length !== N) throw new Error("fmt shape");
  t0 = now();
  formatH2OIds(bTab);
  bFmt = now() - t0;
}
const sTot = sRng + sAlloc, bTot = bRng + bAlloc;
const rs = (n, ms) => Math.round(n / (ms / 1000));
console.log(`N=${N} K=${K} seed=${SEED} dns=[${h2oDns(N, K).join(",")}]`);
console.log(`scalar rng=${sRng.toFixed(1)}ms alloc=${sAlloc.toFixed(1)}ms total=${sTot.toFixed(1)}ms rows/s=${rs(N, sTot)}` +
  (Number.isNaN(sFmt) ? " fmt=SKIP" : ` fmt=${sFmt.toFixed(1)}ms`));
console.log(`block  rng=${bRng.toFixed(1)}ms alloc=${bAlloc.toFixed(1)}ms total=${bTot.toFixed(1)}ms rows/s=${rs(N, bTot)}` +
  (Number.isNaN(bFmt) ? " fmt=SKIP" : ` fmt=${bFmt.toFixed(1)}ms`));
console.log(`speedup_rng=${(sRng / bRng).toFixed(2)}x speedup_total=${(sTot / bTot).toFixed(2)}x`);
