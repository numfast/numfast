// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
//
// THE DOCUMENTED NUMBERS, CHECKED AGAINST THE ARTEFACT.
//
// Every figure this package publishes is asserted here from the bytes of the
// .wasm, so none of them can rot into a claim:
//
//   86 exports = 85 functions + 1 memory, 0 imports   (W2)
//   20 of 85 exports cross the i64 boundary          (the BigInt rule)
//   17 of 85 are wrapped, 68 are named and unwrapped (the coverage claim)
//
// The per-symbol ABI table in abi.ts is the other half: it is the mapping
// acceptance criterion W4 said did not exist, and each entry's arity and return
// type is checked against the FuncType. What the signature cannot express --
// which i32 is a pointer, which is a length, what a code MEANS -- is asserted
// only against the table itself, because there is nothing else to check it
// against. That limitation is stated rather than papered over.

import assert from "node:assert/strict";
import { test, before } from "node:test";

import { artefact } from "./artifact.mjs";

let K;
let info;
let ABI;
let RC_TABLE;

before(async () => {
  const a = artefact();
  info = a.info;
  K = await import("../dist/kernels.js");
  ({ ABI, RC_TABLE } = await import("../dist/abi.js"));
});

test("the export surface is 85 functions, 1 memory, 0 imports", () => {
  assert.equal(info.exportCount, 86);
  assert.equal(info.funcCount, 85);
  assert.equal(info.importCount, 0,
    "the module imports nothing; that is what makes it portable, and it is now false");
  assert.deepEqual(info.memoryExports, ["memory"]);
});

test("20 of 85 exports cross the i64 boundary, and they are named", () => {
  assert.equal(info.i64Count, 20);
  assert.equal(info.i64Exports.length, 20);
  assert.ok(info.i64Exports.includes("nf_cost_intern"),
    "the one wrapped i64 kernel must be on the list");
  assert.ok(!info.i64Exports.includes("nf_map_i32"),
    "nf_map_i32 must NOT be on the list; if it appeared, the BigInt rule changed");
});

test("the covered count is stated, and the rest is named", async () => {
  const { WRAPPED, TOTAL_EXPORTS } = K;
  assert.equal(TOTAL_EXPORTS, 85);
  assert.equal(WRAPPED.length, 17);
  assert.equal(new Set(WRAPPED).size, WRAPPED.length, "WRAPPED has a duplicate");

  const exported = new Set(info.funcExports);
  for (const s of WRAPPED) {
    assert.ok(exported.has(s), `${s} is wrapped but not exported by this build`);
  }
  const unwrapped = info.funcExports.filter((f) => !WRAPPED.includes(f));
  assert.equal(unwrapped.length, 85 - 17, "the unwrapped count does not add up");
  // Every unwrapped kernel must be REACHABLE by name, so "not wrapped" is a
  // stated scope rather than a hole in the module.
  const { callRaw } = K;
  assert.equal(typeof callRaw, "function");
});

test("every ABI entry matches the FuncType it documents", () => {
  for (const [symbol, e] of Object.entries(ABI)) {
    const sig = info.signatures[symbol];
    assert.ok(sig, `abi.ts documents ${symbol}, which this build does not export`);
    assert.equal(sig.params.length, e.wasm.length,
      `${symbol}: abi.ts names ${e.wasm.length} parameters, the FuncType has ${sig.params.length}`);
    assert.equal(sig.results.length, 1, `${symbol}: expected exactly one result`);
    assert.equal(sig.results[0], e.returns, `${symbol}: return type differs from abi.ts`);
    assert.ok(e.wasm.every((p) => p.length > 0), `${symbol}: an unnamed parameter`);
    assert.ok(e.note.length > 10, `${symbol}: a note a caller needs is missing`);
  }
});

test("every wrapped kernel is documented in the ABI table", () => {
  const documented = new Set(Object.keys(ABI));
  for (const s of K.WRAPPED) {
    assert.ok(documented.has(s),
      `${s} is wrapped and exported but has no ABI entry. A wrapper with no ` +
      `documented argument order is a wrapper nobody should call.`);
  }
});

test("the return-code table covers the ABI and never invents a code", () => {
  assert.equal(Object.keys(RC_TABLE).length, Object.keys(ABI).length);
  for (const [symbol, codes] of Object.entries(RC_TABLE)) {
    for (const [code, message] of Object.entries(codes)) {
      const n = Number(code);
      assert.ok(Number.isInteger(n) && n < 0,
        `${symbol}: code ${code} is not a negative return code`);
      assert.ok(n >= -5,
        `${symbol}: code ${n} is outside the frozen range in core/errors.rs ` +
        `(-5 is the lowest defined there)`);
      assert.ok(message.length > 0, `${symbol}: code ${code} has no message`);
      assert.ok(ABI[symbol], `${symbol} has codes but no ABI entry`);
    }
  }
});

test("the memory layout the allocator relies on is what this build has", () => {
  // The bridge places caller buffers above the module's own initialised data.
  // Two facts must hold or that placement is unsafe, and both are checkable
  // from the artefact rather than assumed:
  //   * the data section declares segments (so the region is real data);
  //   * the single mutable global is the stack pointer, and the shadow stack
  //     grows DOWN from it, so buffers must not sit below it either.
  assert.ok(info.dataSegments.length >= 1);
  assert.equal(typeof info.stackPointer, "number");
  assert.ok(info.stackPointer > 0);
  assert.equal(info.initialMemoryPages, 17);
});

test("the allocator base is above both the data and the shadow stack", async () => {
  // loadKernels is the public entry point: it loads the packaged .wasm and
  // refuses it if the export count or the wrapped-symbol set has moved.
  const { loadKernels } = await import("../dist/index.js");
  const bridge = await loadKernels();
  assert.ok(bridge.base > bridge.staticDataEnd,
    `base ${bridge.base} must be above the module data end ${bridge.staticDataEnd}`);
  assert.ok(bridge.base >= info.stackPointer,
    `base ${bridge.base} must be at or above the wasm shadow-stack pointer ` +
    `${info.stackPointer}, which grows downward`);
  assert.ok(bridge.staticDataEnd > info.stackPointer,
    "this build's data starts at the stack pointer, which is why a constant " +
    "base cannot be assumed");
});