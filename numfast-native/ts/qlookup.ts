// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Q-lookup bench: quantized pair artifacts (dist u32 + speed u32 + k u16)
// through nf_cost_travel_batch. Seed 42. JSON last line.
// Usage: node ts/qlookup.ts --gen [n] | node ts/qlookup.ts [n]
//
// The .bin artefacts are NOT tracked in git any more: this file regenerates
// them with --gen and the browser demo generates its own input in-page. Ten
// megabytes of committed binary that one function reproduces is not a
// dependency, it is history.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
// The high-level wrapper lives in kernels.ts (the shipped package surface).
// This driver imports it from dist/, not from the .ts source: kernels.ts uses
// NodeNext `./x.js` specifiers, which `tsc` resolves and Node's type stripping
// does not. So running this file means running it against the BUILT package,
// which is also the honest thing for a benchmark to measure.
//   cd numfast-native/ts && npm run build && node qlookup.ts [n]
import { loadBridge } from "./bridge.ts";
import { costTravelBatch } from "./dist/kernels.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)));
const ART = join(ROOT, "artifacts");
const N = Number(process.argv[3] ?? process.argv[2] ?? 1_000_000);
const INF = 0xffffffff;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

function gen(n: number): { dist: Uint32Array; speed: Uint32Array; k: Uint16Array } {
  const r = lcg(42);
  const dist = new Uint32Array(n), speed = new Uint32Array(n), k = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    dist[i] = 1 + Math.floor(r() * 50_000);
    speed[i] = r() < 0.02 ? 0 : 500 + Math.floor(r() * 29_500); // 2% zero->INF
    k[i] = 800 + Math.floor(r() * 401);
  }
  return { dist, speed, k };
}

function ref(dist: Uint32Array, speed: Uint32Array, k: Uint16Array): Uint32Array {
  const out = new Uint32Array(dist.length);
  for (let i = 0; i < dist.length; i++) {
    const s = speed[i], kk = k[i];
    if (s === 0 || s === INF || dist[i] === INF || kk === 0) { out[i] = INF; continue; }
    let t = Math.floor((dist[i] * kk + (s >> 1)) / s);
    if (dist[i] > 0 && t < 1) t = 1;
    out[i] = t >= INF ? INF : t;
  }
  return out;
}

if (process.argv[2] === "--gen") {
  const { dist, speed, k } = gen(N);
  await mkdir(ART, { recursive: true });
  await writeFile(join(ART, "dist.bin"), Buffer.from(dist.buffer, dist.byteOffset, dist.byteLength));
  await writeFile(join(ART, "speed.bin"), Buffer.from(speed.buffer, speed.byteOffset, speed.byteLength));
  await writeFile(join(ART, "k.bin"), Buffer.from(k.buffer, k.byteOffset, k.byteLength));
  console.log(JSON.stringify({ gen: true, n: N, mb: (dist.byteLength + speed.byteLength + k.byteLength) / 1048576 }));
  process.exit(0);
}

const wasm = await readFile(join(ROOT, "..", "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm"));
const dist = new Uint32Array((await readFile(join(ART, "dist.bin"))).buffer);
const speed = new Uint32Array((await readFile(join(ART, "speed.bin"))).buffer);
const kk = new Uint16Array((await readFile(join(ART, "k.bin"))).buffer);
const n = Math.min(N, dist.length);
const d = dist.subarray(0, n), s = speed.subarray(0, n), kv = kk.subarray(0, n);

// Parity on 10k head sample
const bridge = await loadBridge(wasm);
const got = costTravelBatch(bridge, d.subarray(0, 10_000), s.subarray(0, 10_000), kv.subarray(0, 10_000));
const want = ref(d.subarray(0, 10_000), s.subarray(0, 10_000), kv.subarray(0, 10_000));
let bad = 0;
for (let i = 0; i < 10_000; i++) if (got[i] !== want[i]) bad++;
if (bad > 0) { console.log(JSON.stringify({ pass: false, parityBad: bad })); process.exit(1); }

// Bench: persistent layout, kernel-only timing
bridge.reset();
const dOff = bridge.put(d), sOff = bridge.put(s), kOff = bridge.put(kv);
const outOff = bridge.alloc(n * 4, 8);
bridge.ensure(outOff + n * 4);
const call = (): number => {
  const t0 = performance.now();
  const rc = bridge.costTravelInto(dOff, sOff, kOff, n, outOff);
  const ms = performance.now() - t0;
  if (rc !== 0) throw new Error(`rc=${rc}`);
  return ms;
};
for (let i = 0; i < 5; i++) call();
const times: number[] = [];
for (let i = 0; i < 10; i++) times.push(call());
times.sort((a, b) => a - b);
const med = times[5];
console.log(JSON.stringify({
  pass: true, n, parityBad: 0, medianMs: +med.toFixed(3),
  lookupPerS: Math.round(n / (med / 1000)),
  // No native comparison: the old `nativePerS: 171_000` here was a hardcoded
  // number with nothing measured behind it, and this box is shared, so any
  // ratio printed next to it would be a claim nobody checked.
  wasm: "target/wasm32-unknown-unknown/release/numfast_native.wasm",
}));
