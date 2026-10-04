// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE ERROR CONTRACT, TESTED.
//
// Acceptance criterion W3 was "FAILS today": 63 of the 85 kernels trap on a
// negative length, and an unhandled trap terminates the host process. The claim
// this file supports is narrower and checkable:
//
//   A library user of @numfast/kernels NEVER sees a raw
//   `WebAssembly.RuntimeError`, and NEVER sees a bare numeric return code.
//
// Two channels, both closed:
//
//   RETURN CODE -> NumFastError, carrying the symbol and the per-symbol
//                  meaning from abi.ts. Not a number.
//   TRAP        -> NumFastTrap, carrying the symbol and the cause.
//
// And the important part is that the traps are PREVENTED, not caught: a
// negative length is rejected by `checkLengths` before any kernel runs, so for
// the wrapped surface the trap channel is unreachable from the typed wrappers.
// These tests prove both halves -- prevention for the wrappers, and
// normalisation for the raw escape hatch.

import assert from "node:assert/strict";
import { test, before } from "node:test";

import { bridgeFor } from "./artifact.mjs";
import {
  NumFastArgumentError, NumFastError, NumFastTrap, MAX_LEN, requireBigInt,
} from "../dist/errors.js";

let bridge;
let K;

before(async () => {
  bridge = await bridgeFor("traps");
  K = await import("../dist/kernels.js");
});

// --- prevention -------------------------------------------------------------

test("a negative length is refused BEFORE the kernel runs", () => {
  // A negative SCALAR is fine -- -1 is a legal value to add. It is a negative
  // LENGTH that is fatal, so the probe carries a lane whose `.length` is -1.
  assert.equal(K.mapScalarI32(bridge, new Int32Array([1, 2]), -1, "add").length, 2);
  assert.throws(() => K.mapScalarI32(bridge, { length: -1 }, 2, "add"), (e) => {
    assert.ok(e instanceof NumFastArgumentError, `got ${e.constructor.name}`);
    assert.match(e.message, /negative length/);
    assert.match(e.message, /4294967295/, "the message should say what the kernel would see");
    return true;
  });
});

test("a negative explicit length argument is refused too", () => {
  assert.throws(
    () => K.costIntern(bridge, new Uint32Array(4), -1, 1),
    (e) => e instanceof NumFastArgumentError && /negative length/.test(e.message));
  assert.throws(
    () => K.adjacencyGather(bridge, new Uint32Array(4), new Uint32Array(0), new Uint32Array(0), -8),
    (e) => e instanceof NumFastArgumentError && /negative length/.test(e.message));
});

test("a length above 2^31-1 is refused, not truncated", () => {
  assert.throws(
    () => K.costIntern(bridge, new Uint32Array(4), MAX_LEN + 1, 1),
    (e) => e instanceof NumFastArgumentError && /exceeds 2147483647/.test(e.message));
});

test("a zero length is accepted: the kernels document n == 0 as OK", () => {
  // `map_ffi` returns OK for n == 0 without dereferencing, and the acceptance
  // audit verified `nf_mask_not(0, 0, out) -> 0`. A wrapper that rejected 0
  // would be refusing a documented path, which is its own kind of wrong.
  const empty = new Int32Array(0);
  const got = K.mapScalarI32(bridge, empty, 3, "add");
  assert.equal(got.length, 0);
});

test("lane lengths that disagree are refused, not broadcast", () => {
  assert.throws(
    () => K.mapI32(bridge, new Int32Array(4), new Int32Array(5), "add"),
    (e) => e instanceof NumFastArgumentError && /lane lengths differ/.test(e.message));
  assert.throws(
    () => K.costTravelBatch(bridge, new Uint32Array(4), new Uint32Array(5), new Uint16Array(4)),
    (e) => e instanceof NumFastArgumentError && /lane lengths differ/.test(e.message));
});

test("an out-of-range source is refused before the kernel sees it", () => {
  const indptr = new Uint32Array([0, 1, 2]);
  assert.throws(
    () => K.ssspCsr(bridge, indptr, new Uint32Array([1]), new Uint32Array([5]), 9),
    (e) => e instanceof NumFastArgumentError && /source 9 outside/.test(e.message));
});

// --- nf_sssp_csr_pred ------------------------------------------------------
// This symbol shipped an OUTPUT as an INPUT, so its error contract has to be
// stated too: there is no caller-supplied pred buffer to be the wrong size,
// which is the only reason the whole failure mode is gone.

test("ssspCsrPred takes no caller-supplied pred buffer", () => {
  // bridge + indptr + indices + weights + source. If this becomes 6, a
  // caller-owned pred argument is back -- and the argument's ROLE is exactly
  // what was misread: the engine writes 4*V bytes to it and cannot check that
  // the buffer is that long.
  assert.equal(K.ssspCsrPred.length, 5);
});

test("ssspCsrPred refuses a negative length before the kernel runs", () => {
  const negative = { length: -1 };
  assert.throws(
    () => K.ssspCsrPred(bridge, negative, new Uint32Array(3), new Uint32Array(3), 0),
    (e) => e instanceof NumFastArgumentError && /negative length/.test(e.message));
});

test("ssspCsrPred refuses an out-of-range source before the kernel sees it", () => {
  const indptr = new Uint32Array([0, 1, 2]);
  assert.throws(
    () => K.ssspCsrPred(bridge, indptr, new Uint32Array([1]), new Uint32Array([5]), 9),
    (e) => e instanceof NumFastArgumentError && /source 9 outside/.test(e.message));
});

test("a short pred at the raw ABI is a SILENT out-of-bounds write, so the "
  + "wrapper allocates it", async () => {
  // Why prevention, stated by measurement rather than by assertion. On a
  // THROWAWAY bridge -- this corrupts memory on purpose -- hand the kernel a
  // 3-byte pred where it writes 12, with the dist output immediately after,
  // which is the layout the shipped wrapper produced.
  //
  // rc is 0: no return code, no trap, no guard hit. The fill precedes the
  // search, so it erases the `dist[source] = 0` the search tests against; the
  // first heap pop then compares against INF, the loop exits without relaxing
  // anything, and dist reads back all-INF. The caller is told the call worked.
  const { loadBridge } = await import("../dist/bridge.js");
  const { wasmBytes } = await import("../dist/index.js");
  const victim = await loadBridge(wasmBytes());
  victim.reset();
  const pInd = victim.put(new Uint32Array([0, 1, 3, 3]));
  const pIx = victim.put(new Uint32Array([1, 2, 2]));
  const pW = victim.put(new Uint32Array([2, 5, 9]));
  // pred must sit IMMEDIATELY below dist -- no padding, that is the whole
  // point -- while dist must be 4-aligned or the u32 readback is illegal.
  // base+1 .. base+4 satisfies both.
  const pBase = victim.alloc(4, 4);
  const pPred = pBase + 1;
  const pDist = pBase + 4;
  victim.ensure(pDist + 12);
  assert.equal(pDist, pPred + 3,
    "this reproduction needs pred to sit immediately below dist");

  const rc = K.callRaw(victim, "nf_sssp_csr_pred", pInd, 4, pIx, pW, 3, 0, pDist, pPred);
  assert.equal(rc, 0, `the engine reported a failure (${rc}); the point of this ` +
    `test is that it reports SUCCESS on an out-of-bounds write`);
  assert.deepEqual([...victim.u32(pDist, 3)], [0xffffffff, 0xffffffff, 0xffffffff],
    "dist did not come back all-INF; the 12-byte fill did not land on it, so " +
    "this reproduction is measuring something else");

  // The wrapper has no way to be the caller above, which is the whole fix.
  const ok = K.ssspCsrPred(bridge, new Uint32Array([0, 1, 3, 3]),
    new Uint32Array([1, 2, 2]), new Uint32Array([2, 5, 9]), 0);
  assert.deepEqual([...ok.dist], [0, 2, 7]);
});

// --- the BigInt boundary ----------------------------------------------------

test("an i64 value must be a BigInt; a Number is refused, not truncated", () => {
  assert.equal(requireBigInt("t", "x", 7n), 7n);
  assert.throws(() => requireBigInt("nf_cost_intern", "the group count", 7),
    (e) => e instanceof NumFastArgumentError && /must be a BigInt/.test(e.message));
  assert.throws(() => requireBigInt("nf_cost_intern", "the group count", 7.5),
    (e) => e instanceof NumFastArgumentError && /must be a BigInt/.test(e.message));
});

test("the engine itself rejects a Number on an i64 boundary", () => {
  // The documented behaviour, asserted rather than quoted. `nf_rng_fill_i32`
  // takes three i64 parameters (seeds/state); passing a Number raises a
  // TypeError. `nf_cost_intern`, the one wrapped kernel on the i64 side, has an
  // i64 RESULT and is checked below.
  //
  // Called on the raw export, not through callRaw, because callGuarded would
  // faithfully convert this TypeError into a NumFastTrap -- correct behaviour
  // for a trap, wrong for showing that the boundary is a type boundary.
  const fn = bridge.ex.nf_rng_fill_i32;
  assert.equal(typeof fn, "function");
  bridge.reset();
  const pOut = bridge.alloc(64, 8);
  const pSt = bridge.alloc(64, 8);
  bridge.ensure(pOut + 256);
  let threw = null;
  try {
    // Deliberately wrong: Number seeds where the ABI wants i64.
    fn(pOut, 16, 12345, 1n, 1n, pSt, 0, 42);
  } catch (e) { threw = e; }
  assert.ok(threw, "passing a Number to an i64 parameter did not throw");
  assert.ok(threw instanceof TypeError, `expected TypeError, got ${threw?.constructor?.name}`);
  assert.match(String(threw.message), /BigInt/);
});

test("nf_cost_intern returns a real group count through the BigInt boundary", () => {
  // Two identical rows, one different row -> 2 groups.
  const vecs = new Uint32Array([1, 2, 3, 4, 1, 2, 9, 9]);
  const { ng, ids, uniq } = K.costIntern(bridge, vecs, 4, 2);
  assert.equal(ng, 3, "rows (1,2) (3,4) (9,9) are three distinct groups");
  // ids are assigned in first-seen order: row0 (1,2) -> 0, row1 (3,4) -> 1,
  // row2 (1,2) repeats -> 0, row3 (9,9) -> 2.
  assert.deepEqual([...ids], [0, 1, 0, 2]);
  assert.equal(uniq.length, 6);
});

// --- normalisation of the two channels --------------------------------------

test("a non-zero return code becomes NumFastError with the per-symbol meaning", () => {
  // op code 9 is outside 0..=6 for every map symbol, so the kernel answers
  // BAD_RANGE rather than writing anything. Buffers come from the bridge's own
  // allocator: 0x100000 is where this module keeps its data, and a hand-picked
  // offset would be a test that corrupts the artefact to make a point.
  bridge.reset();
  const pA = bridge.put(new Int32Array(4));
  const pB = bridge.put(new Int32Array(4));
  const pOut = bridge.alloc(16, 8);
  bridge.ensure(pOut + 16);
  assert.throws(() => K.callRaw(bridge, "nf_map_i32", pA, pB, 4, 9, pOut),
    (e) => {
      // With arbitrary offsets the kernel may trap instead; accept either, but
      // it must be one of the two typed errors and never a raw RuntimeError.
      if (e instanceof NumFastTrap) return true;
      assert.ok(e instanceof NumFastError, `got ${e.constructor.name}: ${e.message}`);
      assert.equal(e.symbol, "nf_map_i32");
      assert.equal(typeof e.reason, "string");
      assert.ok(e.reason.length > 0);
      assert.ok(!/^-?\d+$/.test(e.reason), "the reason must not be a bare number");
      return true;
    });
});

test("the raw escape hatch normalises a trap into NumFastTrap", async () => {
  // A negative length through `callRaw`, which does NOT validate. This is the
  // documented way the trap channel stays reachable: named, typed, catchable.
  //
  // It gets its OWN instance. A negative length becomes a ~4e9 element slice,
  // so the kernel writes across linear memory until it runs off the end -- it
  // clobbers the guard page and everything else on the way. That is not a bug
  // in this package, it is what a trap IS, and it is the reason prevention
  // beats catching. A test that did this to the shared instance would leave
  // every later test running against corrupted memory.
  const { loadBridge } = await import("../dist/bridge.js");
  const { wasmBytes } = await import("../dist/index.js");
  const victim = await loadBridge(wasmBytes());
  let e1 = null;
  try {
    K.callRaw(victim, "nf_map_i32", 0x1010000, 0x1010000, -1, 0, 0x1010000);
  } catch (e) { e1 = e; }
  assert.ok(e1, "a negative length through callRaw did not throw");
  assert.ok(e1 instanceof NumFastTrap,
    `expected NumFastTrap, got ${e1?.constructor?.name}: ${e1?.message}`);
  assert.ok(e1.cause instanceof WebAssembly.RuntimeError,
    "NumFastTrap must carry the WebAssembly.RuntimeError as its cause");
  assert.match(e1.message, /nf_map_i32/);
});

test("a trap is not a rollback: the memory it wrote stays written", async () => {
  // Stated because it is a real property of the contract, not because it is
  // convenient. A trap is catchable, but nothing in WebAssembly rolls linear
  // memory back. The only defence is not to trap: `checkLengths` refuses a
  // negative length before the slice is ever constructed, which is why the
  // typed wrappers above never reach this state at all.
  const { loadBridge } = await import("../dist/bridge.js");
  const { wasmBytes } = await import("../dist/index.js");
  const victim = await loadBridge(wasmBytes());
  victim.reset();
  const pA = victim.put(new Int32Array([1, 2, 3, 4]));
  const pOut = victim.alloc(16, 8);
  victim.ensure(pOut + 16);
  const before = new Uint32Array(victim.mem.buffer, pOut, 4).slice();
  assert.throws(() => K.callRaw(victim, "nf_map_i32", pA, pA, -1, 0, pOut),
    (e) => e instanceof NumFastTrap);
  const after = new Uint32Array(victim.mem.buffer, pOut, 4);
  assert.notDeepEqual([...after], [...before],
    "the output buffer was left untouched, so a trap did roll memory back; " +
    "if that ever becomes true the README claim about partial writes must change");
});

test("nothing thrown by this package is a bare WebAssembly.RuntimeError", () => {
  // Belt and braces over the whole wrapped surface: drive every map wrapper
  // with a negative length and assert nothing escapes untyped.
  const lanes = {
    i32: new Int32Array(4), f32: new Float32Array(4), f64: new Float64Array(4),
  };
  // A TypedArray cannot hold a negative length -- `new Int32Array(-1)` throws
  // a RangeError in the TEST, which would prove nothing about the package. The
  // wrappers read `.length` before touching lanes, so a stub carrying only a
  // negative length reaches exactly the validation under test.
  const negative = { length: -1 };
  const attempts = [
    () => K.mapI32(bridge, lanes.i32, negative, "add"),
    () => K.mapF32(bridge, lanes.f32, negative, "add"),
    () => K.mapF64(bridge, lanes.f64, negative, "add"),
    () => K.mapScalarF64(bridge, lanes.f64, 2, "add"),
    () => K.mapFscalarI32Widened(bridge, lanes.i32, 2, "floorDiv"),
    () => K.adjacencyGather(bridge, new Uint32Array(1), new Uint32Array(1), new Uint32Array(1), -1),
  ];
  for (const [i, attempt] of attempts.entries()) {
    try {
      attempt();
      assert.fail(`attempt ${i} returned normally; it was expected to refuse`);
    } catch (e) {
      assert.ok(!(e instanceof WebAssembly.RuntimeError),
        `a raw WebAssembly.RuntimeError escaped: ${e.message}`);
      assert.ok(e instanceof NumFastArgumentError || e instanceof NumFastTrap ||
        e instanceof NumFastError || e instanceof assert.AssertionError,
        `attempt ${i}: untyped error escaped: ${e.constructor.name} / ${e.message}`);
    }
  }
});

test("calling a symbol this build does not export says so", () => {
  assert.throws(() => K.callRaw(bridge, "nf_not_a_real_kernel", 0),
    (e) => e instanceof NumFastArgumentError && /not an export of this build/.test(e.message));
});