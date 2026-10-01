// Parity: block exact-R vs scalar oracle r_rng.mjs (exact, bit-to-bit, NOT distribution).
// Usage: node tools/test_r_block.mjs (exit 0 = PASS, nonzero = FAIL).
// Oracle is imported read-only; never modified.
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { makeR, rRound } from "./r_rng.mjs";
import { RBlockSource } from "./r_block.mjs";
import { RColumnSink } from "./r_colsink.mjs";

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
const eqI32 = (a, b) => {
  assert(a instanceof Int32Array && b.length === a.length, "len");
  for (let i = 0; i < a.length; i++) assert(a[i] === b[i], `i=${i} ${a[i]}!==${b[i]}`);
};
const eqF64 = (a, b) => {
  assert(a instanceof Float64Array && b.length === a.length, "len");
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(b[i])) assert(Number.isNaN(a[i]), `NaN i=${i}`);
    else assert(Object.is(a[i], b[i]) || a[i] === b[i], `i=${i} ${a[i]}!==${b[i]}`);
  }
};
const blk = (seed) => {
  const src = new RBlockSource().init(seed);
  return { src, sink: new RColumnSink(src) };
};

// 1. sampleReplace exact across dn (rejection-heavy, edges) x seeds.
t("sample-exact", () => {
  const cases = [
    [108, 100, 10], [108, 10, 100], [42, 10, 1000], [42, 5, 500], [42, 15, 500],
    [7, 1, 50], [7, 2, 200], [7, 3, 300], [12345, 16, 200], [12345, 17, 300],
    [99, 65535, 200], [99, 65536, 200], [99, 100000, 200], [5, 7, 0],
    [1, 2147483647, 50], [0, 2, 100], [2147483647, 100, 1000], [-5, 10, 100],
  ];
  for (const [seed, dn, k] of cases) {
    const want = makeR(seed).sampleReplace(dn, k);
    const got = blk(seed).sink.sampleReplaceBlock(dn, k);
    assert(got instanceof Int32Array && got.length === k, `shape ${seed}/${dn}/${k}`);
    for (let i = 0; i < k; i++) assert(got[i] === want[i], `${seed}/${dn}/${k} i=${i}`);
  }
});

// 2. Continued stream across columns: one src/sink pair vs one scalar (order incl. rejection).
t("sample-continued-stream", () => {
  for (const seed of [108, 42]) {
    const R = makeR(seed);
    const { sink } = blk(seed);
    for (const [dn, k] of [[10, 100], [100, 50], [3, 200], [17, 150], [65536, 60]]) {
      eqI32(sink.sampleReplaceBlock(dn, k), R.sampleReplace(dn, k));
    }
  }
});

// 3. nextU32Block / nextUnifBlock === sequential scalar draws (incl. twist boundary 624+).
t("source-blocks", () => {
  for (const seed of [108, 42]) {
    const R = makeR(seed);
    const src = new RBlockSource().init(seed);
    for (const n of [0, 1, 623, 624, 625, 1300]) {
      const want = new Uint32Array(n);
      for (let i = 0; i < n; i++) want[i] = R.genrandInt32();
      const got = src.nextU32Block(n);
      assert(got instanceof Uint32Array, "u32 type");
      for (let i = 0; i < n; i++) assert(got[i] === want[i], `u32 ${seed} n=${n} i=${i}`);
    }
    const R2 = makeR(seed);
    const src2 = new RBlockSource().init(seed);
    for (const n of [1, 700, 2000]) {
      const want = new Float64Array(n);
      for (let i = 0; i < n; i++) want[i] = R2.unif();
      eqF64(src2.nextUnifBlock(n), want);
    }
  }
});

// 4. runif exact (incl. continued after sample draws, n=0, negative range).
t("runif-exact", () => {
  const cases = [[108, 10, 0, 100], [42, 1000, 0, 1], [7, 0, 0, 1], [5, 100, -5, 5], [1, 5000, 0, 100]];
  for (const [seed, n, lo, hi] of cases) {
    eqF64(blk(seed).sink.runifBlock(n, lo, hi), makeR(seed).runif(n, lo, hi));
  }
  const R = makeR(108);
  R.sampleReplace(100, 10);
  const want = R.runif(10, 0, 100);
  const { sink } = blk(108);
  sink.sampleReplaceBlock(100, 10);
  eqF64(sink.runifBlock(10, 0, 100), want);
});

// 5. rRound exact (ties to even, -0, non-finite, NaN, digits 0..6).
t("rround-exact", () => {
  const vals = [2.5, 3.5, -2.5, -3.5, 0.5, -0.5, 1.5, 0, -0, 1.005, 2.675, -1.005,
    123.456, -123.456, 76.47780342958868, 1e15 + 0.5, Infinity, -Infinity, NaN, 0.125, 0.375];
  const { sink } = blk(42);
  for (const d of [0, 1, 2, 3, 6]) {
    const want = vals.map((x) => rRound(x, d));
    const got = sink.rRoundBlock(new Float64Array(vals), d);
    assert(got instanceof Float64Array, "f64 type");
    for (let i = 0; i < vals.length; i++) {
      if (Number.isNaN(want[i])) assert(Number.isNaN(got[i]), `NaN d=${d}`);
      else assert(Object.is(got[i], want[i]), `d=${d} i=${i} ${got[i]}!==${want[i]}`);
    }
  }
  assert(Object.is(sink.rRoundBlock(new Float64Array([-0]), 0)[0], -0), "-0 kept");
});

// 6. Chunk premise: (u>>>16) === scalar floor(fixup(u*2^-32)*65536) incl. u32 edges.
t("chunk-formula-edges", () => {
  const I2 = 2.3283064365386963e-10;
  const fix = (x) => (x <= 0 ? 0.5 * I2 : 1 - x <= 0 ? 1 - 0.5 * I2 : x);
  for (const u of [0, 1, 2, 255, 65535, 65536, 131071, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff, 123456789, 987654321]) {
    assert((u >>> 16) === Math.floor(fix(u * I2) * 65536), `u=${u}`);
  }
});

// 7. Full datagen N=100 seed 108: block vs scalar full arrays + vs r_ref_fixture.json heads.
t("datagen-N100-fixture", () => {
  const DNS = [10, 10, 10, 10, 10, 10, 5, 15];
  const runCols = (R) => {
    const cols = DNS.map((dn) => R.sampleReplace(dn, 100));
    const raw = R.runif(100, 0, 100);
    const v3 = raw.map((x) => rRound(x, 6));
    return { cols, raw, v3 };
  };
  const S = runCols(makeR(108));
  const { sink } = blk(108);
  const Bcols = DNS.map((dn) => sink.sampleReplaceBlock(dn, 100));
  const Braw = sink.runifBlock(100, 0, 100);
  const Bv3 = sink.rRoundBlock(Braw, 6);
  for (let c = 0; c < DNS.length; c++) eqI32(Bcols[c], S.cols[c]);
  eqF64(Braw, S.raw);
  eqF64(Bv3, S.v3);
  const fx = JSON.parse(readFileSync(new URL("./r_ref_fixture.json", import.meta.url), "utf8"));
  const pad = (v, w) => "id" + String(v).padStart(w, "0");
  assert.deepEqual(Array.from(Bcols[0].slice(0, 10), (v) => pad(v, 3)), fx.datagen_mini_heads.id1);
  assert.deepEqual(Array.from(Bcols[1].slice(0, 10), (v) => pad(v, 3)), fx.datagen_mini_heads.id2);
  assert.deepEqual(Array.from(Bcols[2].slice(0, 10), (v) => pad(v, 10)), fx.datagen_mini_heads.id3);
  assert.deepEqual(Array.from(Bcols[3].slice(0, 10)), fx.datagen_mini_heads.id4);
  assert.deepEqual(Array.from(Bcols[4].slice(0, 10)), fx.datagen_mini_heads.id5);
  assert.deepEqual(Array.from(Bcols[5].slice(0, 10)), fx.datagen_mini_heads.id6);
  assert.deepEqual(Array.from(Bcols[6].slice(0, 10)), fx.datagen_mini.v1_head);
  assert.deepEqual(Array.from(Bcols[7].slice(0, 10)), fx.datagen_mini.v2_head);
  fx.datagen_mini.v3_head6.forEach((s, i) => assert(Number(s) === Bv3[i], `v3 ${i}`));
  fx.datagen_mini.v3u_head17.slice(0, 5).forEach((s, i) => assert(Number(s) === Braw[i], `v3u ${i}`));
  // probe1 anchor: fresh seed 108 -> sample(100,10,TRUE) then runif(10,max=100)
  const Rp = makeR(108);
  assert.deepEqual(Rp.sampleReplace(100, 10), fx.probe1.sample100_10_replace);
  const { sink: ps } = blk(108);
  assert.deepEqual(Array.from(ps.sampleReplaceBlock(100, 10)), fx.probe1.sample100_10_replace);
  eqF64(ps.runifBlock(10, 0, 100), Rp.runif(10, 0, 100));
  // probe1 continued runif vs fixture strings (Number() parse of 17-digit repr).
  const Rq = makeR(108);
  Rq.sampleReplace(100, 10);
  const { sink: qs } = blk(108);
  qs.sampleReplaceBlock(100, 10);
  const qRaw = qs.runifBlock(10, 0, 100);
  eqF64(qRaw, Rq.runif(10, 0, 100));
  fx.probe1.runif10_max100_continued_str.forEach((s, i) => assert(Number(s) === qRaw[i], `probe1 runif ${i}`));
});

// 8. Several N/K/seeds, Q1 shape (sample K,N + runif + round2): full-stream exact.
t("multi-NK-seeds", () => {
  for (const [seed, N, K] of [[108, 1000, 100], [42, 5000, 7], [1, 100, 10], [12345, 20000, 1000]]) {
    const R = makeR(seed);
    const wKeys = R.sampleReplace(K, N);
    const wRaw = R.runif(N, 0, 100);
    const wVals = wRaw.map((x) => rRound(x, 2));
    const { sink } = blk(seed);
    eqI32(sink.sampleReplaceBlock(K, N), wKeys);
    const gRaw = sink.runifBlock(N, 0, 100);
    eqF64(gRaw, wRaw);
    eqF64(sink.rRoundBlock(gRaw, 2), wVals);
  }
});

// 9. Guards: init once, init-first, bad args. New files stay browser-safe + oracle untouched.
t("guards-and-hygiene", () => {
  assert.throws(() => new RBlockSource().init(1).init(1), /once/);
  assert.throws(() => new RBlockSource().nextU32Block(1), /init\(seed\) first/);
  assert.throws(() => new RBlockSource().nextUnifBlock(1), /init\(seed\) first/);
  const { sink } = blk(1);
  assert.throws(() => sink.sampleReplaceBlock(0, 5), /dn/);
  assert.throws(() => sink.sampleReplaceBlock(10, -1), /k/);
  assert.throws(() => sink.runifBlock(-1), /n/);
  assert.throws(() => new RColumnSink({}), /nextU32Block/);
  for (const f of ["./r_block.mjs", "./r_colsink.mjs"]) {
    const src = readFileSync(new URL(f, import.meta.url), "utf8");
    assert(!src.includes("sampleKind"), f + " must drop dead sampleKind branch");
    assert(!src.includes('from "node:') && !src.includes("from 'node:"), f + " browser-safe");
    assert(!src.includes("String(") && !src.includes("padStart"), f + " no strings in numeric path");
  }
  const oracle = readFileSync(new URL("./r_rng.mjs", import.meta.url), "utf8");
  assert(oracle.includes("sampleKind"), "oracle must be untouched (still has sampleKind)");
});

console.log(failures === 0 ? "PARITY-ALL-PASS" : "PARITY-FAILURES=" + failures);
process.exit(failures === 0 ? 0 : 1);
