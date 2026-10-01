// Copyright (c) 2026 NumFast
// SPDX-License-Identifier: AGPL-3.0-only
// Minimal Node smoke tests for the WASM JS layer (no deps, no toolchain).
// - Unit: wasm_mem.mjs (align/alloc/ensure/blit/BigInt/DataView/bench).
// - Wiring: all 7 drivers run parity (+1 bench) against a mock .wasm that
//   exports the REAL symbol names with stub bodies (return 0). This proves
//   the thin path (layout + blit + call + outputs) without claiming kernel
//   correctness — that stays with tools/parity*.py once a real .wasm exists.
// Usage: node tools/smoke_wasm.mjs  (exit 0 = PASS, nonzero = FAIL)
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { BASE, PAGE, alignUp, alloc, asNumber, bench, blit, blitBytes, dataView, ensureMem, fileView, loadWasm } from "./wasm_mem.mjs";

const ROOT = join(fileURLToPath(import.meta.url), "..");
const TMP = join(ROOT, ".smoke-tmp");
let failures = 0;
function ok(cond, name) {
  console.log((cond ? "PASS" : "FAIL") + " " + name);
  if (!cond) failures++;
}

// ---------- Phase A: helper units ----------
ok(alignUp(16, 8) === 16 && alignUp(17, 8) === 24 && alignUp(17, 4) === 20, "alignUp");
// wasm_run layout for n=8,g=4 must equal the hand-computed offsets from BASE.
{
  let c = BASE;
  const [k, c1] = alloc(c, 8 * 4, 8);
  const [v, c2] = alloc(c1, 8 * 8, 8);
  const [s, c3] = alloc(c2, 4 * 8, 8);
  const [t, need] = alloc(c3, 4 * 8, 8);
  ok(k === BASE && v === BASE + 32 && s === BASE + 96 && t === BASE + 128 && need === BASE + 160, "alloc-chain==wasm_run-layout");
}
{
  const mem = new WebAssembly.Memory({ initial: 1 });
  const u8 = ensureMem(mem, 100000);
  ok(mem.buffer.byteLength >= 100000 && u8.byteLength === mem.buffer.byteLength, "ensureMem-grows");
  blit(u8, new Int32Array([1, 2, 3]), 16);
  const back = new Int32Array(mem.buffer.slice(16, 28));
  ok(back[0] === 1 && back[2] === 3, "blit-roundtrip");
  blitBytes(u8, new Uint8Array([9, 9]), 32);
  ok(u8[32] === 9 && u8[33] === 9, "blitBytes");
}
ok(asNumber(123n) === 123, "asNumber-BigInt");
{
  const mem = new WebAssembly.Memory({ initial: 1 });
  blitBytes(new Uint8Array(mem.buffer), new Uint8Array([0x78, 0x56, 0x34, 0x12]), 16);
  ok(dataView(mem).getInt32(16, true) === 0x12345678, "dataView-i32le");
}
{
  let n = 0;
  const r = bench(() => ({ rc: 0, ms: 1, ng: 0 }));
  ok(r.warm === 5 && r.runs.length === 10 && typeof r.median_ms === "number" && ++n === 1, "bench-shape");
  let threw = 0;
  try { bench(() => ({ rc: 1, ms: 1 })); } catch { threw++; }
  try { bench(() => ({ ng: -1, ms: 1 })); } catch { threw++; }
  ok(threw === 2, "bench-validates-rc/ng");
}
ok(PAGE === 65536 && BASE === 0x101000, "consts");
ok(typeof fileView === "function" && typeof loadWasm === "function", "imports-present");

// ---------- Phase B: mock wasm with REAL export names, stub bodies ----------
function leb(n) {
  const out = [];
  do { let b = n & 0x7f; n >>>= 7; out.push(n ? b | 0x80 : b); } while (n);
  return out;
}
function str(s) {
  const b = [...Buffer.from(s)];
  return [...leb(b.length), ...b];
}
function sec(id, payload) {
  return [id, ...leb(payload.length), ...payload];
}
// types: params(i32 x k) -> ret(0x7f=i32, 0x7e=i64)
function ftype(k, ret) {
  return [0x60, ...leb(k), ...new Array(k).fill(0x7f), 0x01, ret];
}
const I32 = 0x7f, I64 = 0x7e;
const types = [ftype(6, I32), ftype(7, I32), ftype(5, I32), ftype(10, I32), ftype(6, I64),
  ftype(2, I64), ftype(4, I64), ftype(4, I32), ftype(3, I32)];
const funcs = [ // [name, typeidx, i64ret?]
  ["nf_group_sum_count", 0], ["nf_group_multi_sum_count", 1], ["nf_pack_i32_direct", 2],
  ["nf_pattern_encode", 3], ["nf_sorted_run_i64", 4, 1], ["nf_sorted_run_f64", 4, 1],
  ["nf_carry_build_i64", 4, 1], ["nf_carry_build_f64", 4, 1], ["nf_select_count", 5, 1],
  ["nf_select_scatter_i32", 6, 1], ["nf_select_scatter_i64", 6, 1], ["nf_select_scatter_f32", 6, 1],
  ["nf_select_scatter_f64", 6, 1], ["nf_select_scatter_u8", 6, 1],
  ["nf_mask_and", 7], ["nf_mask_or", 7], ["nf_mask_not", 8],
];
{
  const typePayload = [...leb(types.length), ...types.flatMap((t) => t)];
  const funcPayload = [...leb(funcs.length), ...funcs.map((f) => f[1])];
  const memPayload = [0x01, 0x00, 0x01];
  const expPayload = [...leb(funcs.length + 1), ...str("memory"), 0x02, 0x00,
    ...funcs.flatMap((f, i) => [...str(f[0]), 0x00, ...leb(i)])];
  const codePayload = [...leb(funcs.length), ...funcs.flatMap((f) => {
    const body = f[2] ? [0x00, 0x42, 0x00, 0x0b] : [0x00, 0x41, 0x00, 0x0b];
    return [...leb(body.length), ...body];
  })];
  const wasm = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...sec(1, typePayload), ...sec(3, funcPayload), ...sec(5, memPayload),
    ...sec(7, expPayload), ...sec(10, codePayload)]);
  rmSync(TMP, { recursive: true, force: true });
  mkdirSync(TMP, { recursive: true });
  writeFileSync(join(TMP, "mock.wasm"), wasm);
  // sanity: mock loads and stub returns 0n/0
  const inst = await loadWasm(join(TMP, "mock.wasm"));
  ok(typeof inst.exports.nf_mask_not === "function" && inst.exports.nf_mask_not(0, 0, 0) === 0, "mock-loads");
  ok(inst.exports.nf_select_count(0, 0) === 0n, "mock-i64-stub");
}

// ---------- Phase C: drivers against mock ----------
function w(p, arr) { // typed array -> file
  writeFileSync(p, Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
}
{
  const n = 8, g = 4, m = 4, ncols = 2;
  w(join(TMP, "keys.i32"), new Int32Array([0, 1, 0, 2, 1, 0, 3, 2]));
  w(join(TMP, "vals.f64"), new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]));
  w(join(TMP, "flat.f64"), new Float64Array(16).map((_, i) => i + 1));
  w(join(TMP, "k1.i32"), new Int32Array([3, 1, 2, 0, 1, 2, 3, 0]));
  w(join(TMP, "k2.i32"), new Int32Array([0, 1, 1, 2, 0, 3, 2, 1]));
  w(join(TMP, "skeys.i32"), new Int32Array([0, 0, 1, 1, 1, 2, 3, 3]));
  w(join(TMP, "svals.i64"), new BigInt64Array([1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n]));
  w(join(TMP, "cnt.i64"), new BigInt64Array([2n, 0n, 3n, 1n]));
  w(join(TMP, "ssum.i64"), new BigInt64Array([10n, 0n, 30n, 40n]));
  w(join(TMP, "mask.u8"), new Uint8Array([1, 0, 1, 1, 0, 0, 1, 0]));
  w(join(TMP, "a.u8"), new Uint8Array([1, 1, 0, 0, 1, 0, 1, 0]));
  w(join(TMP, "b.u8"), new Uint8Array([1, 0, 1, 0, 1, 1, 0, 0]));
  const data = Buffer.from("r0;r1;r2;r3;r4;r5;r6;r7;");
  writeFileSync(join(TMP, "data.bin"), data);
  w(join(TMP, "offs.i32"), new Int32Array([0, 3, 6, 9, 12, 15, 18, 21, 24]));
}
function run(name, args, expectKeys, expectFiles) {
  const script = join(ROOT, name);
  let out;
  try {
    out = execFileSync(process.execPath, [script, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  } catch (e) {
    console.log("FAIL " + name + " " + args[2] + " exit: " + (e.stderr || e.message).slice(0, 300));
    failures++;
    return;
  }
  let j;
  try { j = JSON.parse(out.trim().split("\n").pop()); }
  catch { console.log("FAIL " + name + " bad-json: " + out.slice(0, 200)); failures++; return; }
  const keysOk = expectKeys.every((k) => k in j);
  let filesOk = true;
  for (const [f, size] of expectFiles) {
    const p = join(TMP, f);
    if (!existsSync(p) || readFileSync(p).length !== size) { filesOk = false; break; }
  }
  ok(keysOk && filesOk, name + " " + args.slice(2, 5).join(" ") + " -> " + JSON.stringify(j).slice(0, 80));
}
const W = join(TMP, "mock.wasm"), O = TMP;
run("wasm_run.mjs", [W, join(TMP, "keys.i32"), join(TMP, "vals.f64"), "8", "4", "parity", O],
  ["rc", "ms"], [["sums.f64", 32], ["counts.i64", 32]]);
run("wasm_multi.mjs", [W, join(TMP, "keys.i32"), join(TMP, "flat.f64"), "8", "2", "4", "parity", O],
  ["rc", "ms"], [["multi_sums.f64", 64], ["multi_counts.i64", 32]]);
run("wasm_pack.mjs", [W, join(TMP, "k1.i32"), join(TMP, "k2.i32"), "4", "8", "parity", O],
  ["rc", "ms"], [["pack_out.i32", 32]]);
run("wasm_pattern.mjs", [W, join(TMP, "data.bin"), join(TMP, "offs.i32"), "P", "8", "parity", O],
  ["rc", "ms", "width", "err_row"], [["pat_codes.i32", 32], ["pat_valid.u8", 8]]);
run("wasm_sorted.mjs", [W, join(TMP, "skeys.i32"), join(TMP, "svals.i64"), "1", "8", "parity", O],
  ["ng", "ms"], [["sort_uk_i64.i64", 0], ["sort_sums_i64.i64", 0], ["sort_counts_i64.i64", 0]]);
run("wasm_sorted.mjs", [W, join(TMP, "skeys.i32"), join(TMP, "vals.f64"), "0", "8", "parity", O],
  ["ng", "ms"], []);
run("wasm_carry.mjs", [W, join(TMP, "cnt.i64"), join(TMP, "ssum.i64"), "1", "4", "parity", O],
  ["ng", "ms"], [["carry_uk_i64.i64", 0], ["carry_counts_i64.i64", 0], ["carry_sums_i64.i64", 0]]);
run("wasm_select.mjs", [W, "count", "-", join(TMP, "mask.u8"), "-", "8", "parity", O], ["m", "ms"], []);
run("wasm_select.mjs", [W, "scatter", "i32", join(TMP, "k1.i32"), join(TMP, "mask.u8"), "8", "parity", O],
  ["m", "ms"], [["sel_out.i32", 0]]);
run("wasm_select.mjs", [W, "and", "-", join(TMP, "a.u8"), join(TMP, "b.u8"), "8", "parity", O],
  ["rc", "ms"], [["mask_out.u8", 8]]);
run("wasm_select.mjs", [W, "not", "-", join(TMP, "a.u8"), "-", "8", "parity", O],
  ["rc", "ms"], [["mask_out.u8", 8]]);
run("wasm_run.mjs", [W, join(TMP, "keys.i32"), join(TMP, "vals.f64"), "8", "4", "bench"], ["warm", "runs", "median_ms"], []);

rmSync(TMP, { recursive: true, force: true });
console.log(failures === 0 ? "SMOKE-ALL-PASS" : "SMOKE-FAILURES=" + failures);
process.exit(failures === 0 ? 0 : 1);
