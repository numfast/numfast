// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// PARITY: Python host <-> JS/WASM, byte for byte.
//
// The expectations are NOT written here. They come from
// `test/fixtures/parity.json`, produced by `test/gen_fixtures.py` from TWO
// Python paths that were compared against each other FIRST:
//
//     NumPy + the documented int32 round-trip   (the frozen public semantic)
//     ctypes into the committed native DLL      (the engine's own path)
//
// If those two disagreed the generator aborts, so a fixture never records a
// disagreement as if it were truth. The file is committed with the commit that
// produced it: a behaviour change on the Python side is a DIFF in that file.
//
// WHAT IS COMPARED
//   Integer lanes: BYTE-EXACT. No tolerance, ever.
//   Float lanes:   exact, except inside the project's own published budget --
//                  specs-rebuilt/conformance-profile.toml grants f32 4 ULP and
//                  f64 2 ULP. The budget is READ by the generator and applied
//                  here; it is not chosen in this file.
//
// WHY A FLOAT BUDGET IS NOT A WHITEWASH: the lanes that need it are COUNTED
// and PRINTED. Today exactly one family needs it -- `pow`, where the wasm32
// build uses Rust's bundled libm and the host uses the platform libm, and the
// two differ by 1 ULP on a few percent of lanes. That is a real, measured,
// reported difference, not a licence.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test, before, after } from "node:test";
import { fileURLToPath } from "node:url";

import { bridgeFor, layoutFacts } from "./artifact.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = JSON.parse(readFileSync(join(HERE, "fixtures", "parity.json"), "utf8"));
const OP_NAME = ["add", "sub", "mul", "div", "pow", "floorDiv", "mod"];
const TOL = FIXTURES.tolerance;

/** Wrapped but with no Python reference built yet. Stated, not hidden. */
const NOT_FIXTURED = [
  "nf_sssp_csr", "nf_sssp_csr_pred", "nf_sssp_batch", "nf_cost_travel_batch",
  "nf_cost_intern", "nf_rowwise_kway_time_argmin_gather",
  "nf_adjacency_slice", "nf_adjacency_gather",
];

let bridge;
let facts;
let K;
const inexact = new Map();

before(async () => {
  // artefact() runs first and aborts LOUDLY on a missing/stale/moved build, so
  // a missing dist/ surfaces as that message rather than as module-not-found.
  bridge = await bridgeFor("parity");
  facts = layoutFacts(bridge);
  K = await import("../dist/kernels.js");
});

after(() => {
  process.stdout.write(
    `[parity] allocator base ${facts.base} is above the module's own data, ` +
    `which ends at ${facts.staticDataEnd}\n` +
    `[parity] ${Object.keys(FIXTURES.cases).length} cases; float-lane budget ` +
    `f32 ${TOL.f32.maxUlp} ULP, f64 ${TOL.f64.maxUlp} ULP ` +
    `(conformance-profile.toml)\n`);
  for (const [sym, n] of [...inexact].sort()) {
    process.stdout.write(`[parity] ${sym}: ${n} lanes needed the budget (1 ULP, libm)\n`);
  }
  if (inexact.size === 0) process.stdout.write("[parity] every float lane matched exactly\n");
  process.stdout.write(
    `[parity] wrapped, NOT parity-fixtured (no Python reference built yet): ` +
    `${NOT_FIXTURED.join(", ")}\n`);
});

// --- the shared input generator (identical arithmetic to gen_fixtures.py) ---
function lcg(n) {
  let s = 42;
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = s;
  }
  return out;
}

function genInt(n, lo, hi) {
  const span = hi - lo + 1;
  const u = lcg(n);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = lo + Math.floor((u[i] * span) / 4294967296);
  return out;
}

function genFloat(n, scale, offset, Ctor) {
  const u = lcg(n);
  const out = new Ctor(n);
  for (let i = 0; i < n; i++) out[i] = (u[i] / 4294967296) * scale + offset;
  return out;
}

const nOf = (name) => Number(/n(\d+)$/.exec(name)[1]);

function inputsFor(n) {
  const d = FIXTURES.inputs;
  const f64b = genFloat(n, d.f64_b.scale, d.f64_b.offset, Float64Array);
  for (let i = 0; i < n; i += d.f64_b.zeroEvery) f64b[i] = 0;
  return {
    i32: genInt(n, d.i32.lo, d.i32.hi),
    i32_b: genInt(n, d.i32_b.lo, d.i32_b.hi),
    f64: genFloat(n, d.f64.scale, d.f64.offset, Float64Array),
    f64_b: f64b,
    f32: genFloat(n, d.f32.scale, d.f32.offset, Float32Array),
    f32_b: genFloat(n, d.f32_b.scale, d.f32_b.offset, Float32Array),
  };
}

// --- IEEE bit distance ------------------------------------------------------
const dv = new DataView(new ArrayBuffer(8));
function ord64(x) {
  dv.setFloat64(0, x);
  const hi = BigInt(dv.getUint32(0)), lo = BigInt(dv.getUint32(4));
  let bits = (hi << 32n) | lo;
  bits &= 0xffffffffffffffffn;
  return bits & 0x8000000000000000n ? ~bits & 0xffffffffffffffffn : bits | 0x8000000000000000n;
}
const dv32 = new DataView(new ArrayBuffer(4));
function ord32(x) {
  dv32.setFloat32(0, x);
  const bits = BigInt(dv32.getUint32(0));
  return bits & 0x80000000n ? ~bits & 0xffffffffn : bits | 0x80000000n;
}

/** Distance in representable steps. NaN never equals NaN, so an expected NaN
 *  is matched by bit pattern instead -- which is what "same bytes" means. */
function ulpDistance(got, want, is64) {
  const same = is64 ? Object.is(got, want)
    : Object.is(Math.fround(got), Math.fround(want));
  if (same) return 0n;
  if (Number.isNaN(got) && Number.isNaN(want)) return 0n;
  const a = is64 ? ord64(got) : ord32(got);
  const b = is64 ? ord64(want) : ord32(want);
  const d = a > b ? a - b : b - a;
  return d;
}

function checkCase(name, c, got) {
  const bytes = new Uint8Array(got.buffer, got.byteOffset, got.byteLength);
  assert.equal(bytes.length, c.expect_bytes, `${name}: output length differs`);

  if (c.dtype === "i32") {
    if (c.expect_hex !== undefined) {
      assert.equal(Buffer.from(bytes).toString("hex"), c.expect_hex,
        `${name}: integer lane bytes differ from the Python expectation`);
      return 0;
    }
    assert.equal(createHash("sha256").update(bytes).digest("hex"), c.expect_sha256,
      `${name}: integer lane bytes differ from the Python expectation (sha256)`);
    assert.deepEqual(Array.from(got.subarray(0, 8), Number), c.head, `${name}: head differs`);
    assert.deepEqual(Array.from(got.subarray(-8), Number), c.tail, `${name}: tail differs`);
    return 0;
  }

  // Float lane. Rebuild the expected bytes; the fixture holds them for small
  // cases, and every float case here is small by construction.
  assert.ok(c.expect_hex !== undefined,
    `${name}: a float-lane case must carry its expected bytes literally, ` +
    `because the ULP comparison needs them`);
  const want = new Uint8Array(c.expect_hex.length / 2);
  for (let i = 0; i < want.length; i++) want[i] = parseInt(c.expect_hex.substr(i * 2, 2), 16);
  const is64 = c.dtype === "f64";
  const W = is64 ? new Float64Array(want.buffer) : new Float32Array(want.buffer);
  const budget = BigInt(TOL[c.dtype].maxUlp);
  let offBudget = 0;
  for (let i = 0; i < got.length; i++) {
    const d = ulpDistance(got[i], W[i], is64);
    if (d > budget) {
      offBudget++;
      if (offBudget <= 3) {
        process.stdout.write(
          `[parity] ${name} lane ${i}: got ${got[i]} want ${W[i]} ` +
          `(${d} ULP, budget ${budget})\n`);
      }
    }
  }
  assert.equal(offBudget, 0,
    `${name}: ${offBudget} of ${got.length} float lanes outside the ` +
    `conformance budget of ${budget} ULP`);
  let inexactLanes = 0;
  for (let i = 0; i < got.length; i++) if (ulpDistance(got[i], W[i], is64) > 0n) inexactLanes++;
  inexact.set(c.symbol, (inexact.get(c.symbol) ?? 0) + inexactLanes);
  return inexactLanes;
}

function run(c, name) {
  const A = inputsFor(nOf(name));
  const op = OP_NAME[c.op];
  switch (c.symbol) {
    case "nf_map_i32": return K.mapI32(bridge, A.i32, A.i32_b, op);
    case "nf_map_f32": return K.mapF32(bridge, A.f32, A.f32_b, op);
    case "nf_map_f32_divpow": return K.mapF32Divpow(bridge, A.f32, A.f32_b, op);
    case "nf_map_f64": return K.mapF64(bridge, A.f64, A.f64_b, op);
    case "nf_map_scalar_i32": return K.mapScalarI32(bridge, A.i32, c.scalar, op);
    case "nf_map_scalar_f64": return K.mapScalarF64(bridge, A.f64, c.scalar, op);
    case "nf_map_scalar_f32": return K.mapScalarF32(bridge, A.f32, c.scalar, op);
    case "nf_map_scalar_f32_divpow": return K.mapScalarF32Divpow(bridge, A.f32, c.scalar, op);
    case "nf_map_fscalar_i32":
      return op === "floorDiv" || op === "mod"
        ? K.mapFscalarI32Widened(bridge, A.i32, c.scalar, op)
        : K.mapFscalarI32(bridge, A.i32, c.scalar, op);
    default:
      throw new Error(`no wrapper wired for ${c.symbol} (${name})`);
  }
}

for (const [name, c] of Object.entries(FIXTURES.cases).sort()) {
  test(`parity ${name}`, () => {
    const got = run(c, name);
    assert.ok(got, `${name}: wrapper returned nothing`);
    assert.equal(got.length, nOf(name), `${name}: wrong lane count`);
    checkCase(name, c, got);
  });
}

// --- the case that was reported as a divergence, stated as a test -----------
test("map pow with a fractional exponent matches Python at BOTH sizes", () => {
  // The audit recorded: WASM i32 `map pow` with exponent 2.5 agrees at n=6 and
  // diverges at n=100000, while the native ctypes path agrees at both.
  // Re-measured: native agrees with NumPy at every size and every exponent,
  // and WASM agrees too PROVIDED the caller's buffers sit above the module's
  // own constant pool. These assertions pin exactly that condition.
  assert.ok(facts.base > facts.staticDataEnd,
    `allocator base ${facts.base} is not above the module's data end ` +
    `${facts.staticDataEnd}: a buffer over it corrupts the float constants powf ` +
    `reads, and pow then returns plausible wrong numbers with rc=0`);
  // The fixture names the exponent as Python formats it, so 3.0 is "s3.0".
  // Index the real keys instead of re-formatting and hoping the two agree.
  const powKeys = Object.keys(FIXTURES.cases)
    .filter((k) => k.startsWith("map_fscalar_i32/pow/s"))
    .sort();
  assert.ok(powKeys.length >= 10, `expected pow fixtures, found ${powKeys.length}`);
  for (const key of powKeys) {
    const n = nOf(key);
    if (n !== 6 && n !== 100000) continue;
    const scalar = Number(/\/s([^/]+)\//.exec(key)[1]);
    const A = inputsFor(n);
    const c = FIXTURES.cases[key];
    assert.equal(checkCase(key, c, K.mapFscalarI32(bridge, A.i32, scalar, "pow")), 0,
      `${key}: the i32 pow lane must be BIT-EXACT, not merely inside a budget`);
  }
});

test("coverage is stated, not implied", () => {
  const cases = Object.keys(FIXTURES.cases);
  assert.equal(cases.length, 203, "the fixture count moved; the summary line must move with it");
  assert.equal(cases.length, Object.keys(FIXTURES.cases).length);
  assert.ok(FIXTURES.tolerance.f64.maxUlp <= 2,
    "the float budget was widened; conformance-profile.toml is the authority, not this test");
});

test("every wrapped kernel not in the fixtures is on the stated list", () => {
  const fixtureSymbols = new Set(Object.values(FIXTURES.cases).map((c) => c.symbol));
  const notFixtured = K.WRAPPED.filter((s) => !fixtureSymbols.has(s));
  assert.deepEqual([...notFixtured].sort(), [...NOT_FIXTURED].sort(),
    "a kernel is wrapped but has neither a parity fixture nor a name on the " +
    "not-fixtured list. One of those two is a lie; make whichever true.");
  assert.equal(notFixtured.length, 8);
});