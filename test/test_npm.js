/**
 * test_npm.js — npm test runner (J5/J6 fix).
 * Real execution against cross-language golden vectors:
 *   1. index.js public surface (12 legacy symbols + parity API)
 *   2. ExecutionPlan structure
 *   3. RandomKernel CPU vs Python golden vectors (test/golden_random.json, seed=42)
 *   4. Creation golden tests (zeros/ones/full/arange/linspace/index/tile/repeat,
 *      vs formulas; i32 WRAP semantics)
 *   5. Series operators + comparisons (MapBinary op0-3, Map 6/7, Compare)
 *   6. filter/topk smoke (Compare->Gather, Sort+Gather)
 *   7. math parity vs Math.* + stats via Reduce two-pass
 */
"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

let PASS = 0, FAIL = 0;
function check(name, ok, detail) {
  if (ok) { PASS++; console.log("  PASS  " + name + (detail ? ": " + detail : "")); }
  else { FAIL++; console.log("  FAIL  " + name + (detail ? ": " + detail : "")); }
}
function eqF32(a, b, tol) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(Number(a[i]) - Number(b[i]));
    if (!(d <= (tol !== undefined ? tol : 0))) return false;
  }
  return true;
}

// CHECK 1: index.js exports — 12 legacy symbols + public parity surface
console.log("CHECK 1: index.js exports");
const nf = require("../index.js");
const LEGACY = ["compile", "execute", "register_kernel", "Runtime", "CpuDriver",
  "register_all", "register_compute_kernels", "ExecutionPlan", "ExecutionContext",
  "BufferView", "InputSlot", "OutputSlot"];
const PUBLIC = ["zeros", "ones", "full", "arange", "linspace", "index", "tile",
  "repeat", "Series", "topk", "sin", "cos", "exp", "log", "sqrt", "abs",
  "square", "neg", "mean", "min", "max", "total", "count", "var", "std",
  "deviceInfo"];
check("legacy 12 symbols preserved",
  LEGACY.every(k => typeof nf[k] === "function"));
check("public parity surface present (" + PUBLIC.length + " symbols)",
  PUBLIC.every(k => typeof nf[k] === "function" || typeof nf[k] === "object"));
check("no unexpected export keys",
  LEGACY.concat(PUBLIC).every(k => k in nf) &&
  Object.keys(nf).length === LEGACY.length + PUBLIC.length,
  "total=" + Object.keys(nf).length);

// CHECK 2: ExecutionPlan structure
console.log("CHECK 2: ExecutionPlan structure");
const { ExecutionPlan, ExecutionContext } = require("../src/core/Runtime/_lib/mod_iface");
const plan = new ExecutionPlan({
  inputs: [{ name: "data", dtype: "float" }],
  outputs: [{ dtype: "float", template: "out_{n}" }],
  workspace: [],
  uniforms: { period: 3 },
});
plan.allocateOutputs(4);
plan.bindInputs([new Float64Array(4)]);
check("inputs/outputs/workspace/uniforms + allocate/bind",
  plan.inputs.length === 1 && plan.inputs[0].name === "data" && plan.inputs[0].dtype === "float" &&
  plan.outputs.length === 1 && plan.outputs[0].dtype === "float" &&
  plan.outputs[0].template === "out_{n}" && Array.isArray(plan.workspace) &&
  typeof plan.uniforms === "object" && typeof plan.outputs[0].view.read(0) === "number");

// CHECK 3: RandomKernel CPU vs golden vectors from Python cpu.py
console.log("CHECK 3: RandomKernel golden vectors (Python oracle)");
const golden = JSON.parse(fs.readFileSync(path.join(__dirname, "golden_random.json"), "utf-8"));
const registry = new Map(require("../src/math/Compute/Compute")
  .register_compute_kernels().map(k => [k[0], k]));
const rk = registry.get("RandomKernel");
const MODE_ID = { uniform: 0, integers: 1, normal: 2 };

function runRandom(c) {
  const pl = new ExecutionPlan(rk[1]({ n: c.n, mode: MODE_ID[c.mode], seed: golden.seed, p0: c.p0, p1: c.p1 }));
  pl.allocateOutputs(c.n);
  const ctx = new ExecutionContext(pl, []);
  rk[2](ctx);
  return pl.outputs[0].view._data;
}

for (const c of golden.cases) {
  const out = runRandom(c);
  let ok, detail;
  if (c.mode === "normal") {
    ok = true;
    for (let i = 0; i < c.first10.length; i++) {
      const diff = Math.abs(out[i] - c.first10[i]);
      const tol = Math.max(golden.tol_normal_abs, golden.tol_normal_rel * Math.abs(c.first10[i]));
      if (!(diff <= tol)) { ok = false; detail = "i=" + i + " js=" + out[i] + " py=" + c.first10[i]; break; }
    }
  } else {
    // bit-exact: sha256 of full f32 byte stream + bitwise first10
    const bytes = Buffer.from(new Float32Array(out).buffer);
    const bitsOk = crypto.createHash("sha256").update(bytes).digest("hex") === c.hash_full_hex;
    const headOk = c.first10.every((v, i) => Math.fround(out[i]) === v);
    ok = bitsOk && headOk;
    detail = bitsOk ? "sha256(f32 bits) == py" : "sha256 mismatch";
    if (!headOk) detail += " first10 bit mismatch";
  }
  check(c.n + "x " + c.mode, ok, detail);
}

// CHECK 4: IndexKernel registration + creation golden tests (vs formula)
console.log("CHECK 4: creation API");
check("IndexKernel registered", registry.has("IndexKernel"));
check("IndexKernelI32 registered", registry.has("IndexKernelI32"));

check("zeros(5)", eqF32(nf.zeros(5).data, [0, 0, 0, 0, 0]) &&
  nf.zeros(5).data instanceof Float32Array && nf.zeros(5).length === 5);
check("zeros([3],'int32')", eqF32(nf.zeros([3], "int32").data, [0, 0, 0]) &&
  nf.zeros([3], "int32").data instanceof Int32Array);
check("ones(4)", eqF32(nf.ones(4).data, [1, 1, 1, 1]));
check("full(3, 2.5)", eqF32(nf.full(3, 2.5).data, [2.5, 2.5, 2.5]));
check("arange(0,10)", eqF32(nf.arange(0, 10).data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
check("arange() default start=0 step=1", eqF32(nf.arange(5).data, [0, 1, 2, 3, 4]));
check("arange(2,10,3)", eqF32(nf.arange(2, 10, 3).data, [2, 5, 8]));
check("arange(5,1,-1)", eqF32(nf.arange(5, 1, -1).data, [5, 4, 3, 2]));
check("linspace(0,1,5) endpoint",
  eqF32(nf.linspace(0, 1, 5).data, [0, 0.25, 0.5, 0.75, 1]));
{
  const l = nf.linspace(0, 1, 5, false).data; // endpoint=false: step=(stop-start)/num
  check("linspace(0,1,5,endpoint=false)",
    eqF32(l, [0, 0.2, 0.4, 0.6, 0.8], 1e-6),
    Array.from(l).map(v => v.toFixed(3)).join(","));
}
check("index(5) default int32 exact", eqF32(nf.index(5).data, [0, 1, 2, 3, 4]) &&
  nf.index(5).data instanceof Int32Array);
check("index(4,'float32')", eqF32(nf.index(4, "float32").data, [0, 1, 2, 3]) &&
  nf.index(4, "float32").data instanceof Float32Array);
{ // int32 arange WRAP (two's complement), owner-documented contract.
  // Direct kernel call: creation API caps n at MAX_ELEMENTS by design.
  const ik = registry.get("IndexKernelI32");
  const pl = new ExecutionPlan(ik[1](
    { n: 4, mode: 1, p0: 2147483640, p1: 8, dtype: "int32" }));
  pl.allocateOutputs(4);
  ik[2](new ExecutionContext(pl, []));
  const got = Array.from(pl.outputs[0].view._data);
  check("i32 arange WRAP at 2^31",
    JSON.stringify(got) === JSON.stringify([2147483640, -2147483648, -2147483640, -2147483632]),
    JSON.stringify(got));
}
check("tile([1,2,3], 7)", eqF32(nf.tile([1, 2, 3], 7).data, [1, 2, 3, 1, 2, 3, 1]));
check("repeat([1,2,3], 2)", eqF32(nf.repeat([1, 2, 3], 2).data, [1, 1, 2, 2, 3, 3]));
check("repeat([1,2], 3, n=10)", // WGSL contract: out[i]=pat[min(i/rep,k-1)]
  eqF32(nf.repeat([1, 2], 3, 10).data, [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]));
check("creation n=0 -> empty", nf.zeros(0).length === 0 &&
  nf.arange(3, 3).length === 0 && nf.tile([1], 0).length === 0);
{
  let threw = false;
  try { nf.zeros(5000000); } catch (e) { threw = /chunking/.test(e.message); }
  check("n > 4194240 -> chunking error v1", threw);
}

// CHECK 5: Series operators + comparisons
console.log("CHECK 5: Series operators");
const s5 = nf.Series.from([1, 2, 3, 4]);
check("Series.from coerces to Float32Array", s5.data instanceof Float32Array);
check("len()/length", s5.len() === 4 && s5.length === 4);
check("compute() returns materialized Series",
  s5.compute() instanceof nf.Series && s5.compute().len() === 4);
check("add scalar", eqF32(s5.add(10).data, [11, 12, 13, 14]));
check("sub series", eqF32(s5.sub(nf.Series.from([4, 3, 2, 1])).data, [-3, -1, 1, 3]));
check("mul scalar", eqF32(s5.mul(3).data, [3, 6, 9, 12]));
check("div scalar", eqF32(s5.div(2).data, [0.5, 1, 1.5, 2]));
check("mod scalar (C-trunc)", eqF32(
  nf.Series.from([5, -5, 5.5]).mod(3).data, [2, -2, 2.5]));
check("mod by zero -> 0", eqF32(nf.Series.from([1, 2]).mod(0).data, [0, 0]));
check("pow scalar", eqF32(s5.pow(2).data, [1, 4, 9, 16]));
check("pow series", eqF32(s5.pow(nf.Series.from([2, 2, 2, 2])).data, [1, 4, 9, 16]));
check("square/neg (Map 7/6)",
  eqF32(s5.square().data, [1, 4, 9, 16]) && eqF32(s5.neg().data, [-1, -2, -3, -4]));
{
  const m = s5.gt(2);
  check("gt scalar -> Uint8Array mask", m.data instanceof Uint8Array &&
    Array.from(m.data).join(",") === "0,0,1,1" && m.isMask === true);
  const ops = {
    lt: [1, 0, 0, 0], le: [1, 1, 0, 0], gt: [0, 0, 1, 1],
    ge: [0, 1, 1, 1], eq: [0, 1, 0, 0], ne: [1, 0, 1, 1],
  };
  let allOk = true;
  for (const [name, want] of Object.entries(ops)) {
    const r = Array.from(s5[name](2).data).join(",");
    if (r !== want.join(",")) { allOk = false; break; }
  }
  check("lt/le/gt/ge/eq/ne masks vs scalar 2", allOk);
  check("comparison with series operand",
    Array.from(s5.lt(nf.Series.from([2, 2, 2, 2])).data).join(",") === "1,0,0,0");
}

// CHECK 6: filter/topk smoke
console.log("CHECK 6: filter/topk");
{
  const s = nf.Series.from([5, 1, 4, 2, 3]);
  const f = s.filter(s.gt(2)); // Compare->Gather composition
  check("filter(mask) values", eqF32(f.data, [5, 4, 3]));
  const t = nf.topk(s, 3);     // Sort+Gather, descending
  check("topk(3) descending", eqF32(t.data, [5, 4, 3]));
  check("topk(k>n) clamps", nf.topk(s, 99).len() === 5 &&
    eqF32(nf.topk(s, 99).data, [5, 4, 3, 2, 1]));
  check("topk on padded size 6", eqF32(nf.Series.from([9, 7]).topk(2).data, [9, 7]));
  check("filter empty mask", s.filter(nf.Series.from([0, 0, 0, 0, 0])).len() === 0);
  let threw = false;
  try { s.filter(nf.Series.from([1, 0])); } catch (e) { threw = true; }
  check("filter mask length mismatch throws", threw);
}

// CHECK 7: math parity vs Math.* + stats
console.log("CHECK 7: math + stats");
{
  const xs = [0.1, 0.5, 1, 2, 3.7];
  const f32x = Float32Array.from(xs); // Map reads f32-domain inputs
  const cases = [
    ["sin", Math.sin], ["cos", Math.cos], ["exp", Math.exp],
    ["sqrt", Math.sqrt], ["log", Math.log], ["abs", Math.abs],
  ];
  let allOk = true; let badName = "";
  for (const [name, ref] of cases) {
    const out = nf[name](nf.Series.from(xs)).toArray();
    for (let i = 0; i < xs.length; i++) {
      const want = Math.fround(ref(f32x[i]));
      if (!(Math.abs(out[i] - want) <= 1e-6 * Math.max(1, Math.abs(want)))) {
        allOk = false; badName = name + "[" + i + "]"; break;
      }
    }
    if (!allOk) break;
  }
  check("sin/cos/exp/sqrt/log/abs parity vs Math.*", allOk, badName);
  check("square/neg namespace",
    eqF32(nf.square(xs).data,
      Array.from(f32x, v => Math.fround(v * v))) &&
    eqF32(nf.neg(xs).data, Array.from(f32x, v => -v)));
  check("tan NOT in math ns (Python Map has no tan)", typeof nf.tan === "undefined");

  const st = nf.Series.from([1, 2, 3, 4, 5]);
  check("total", nf.total(st) === 15);
  check("count", nf.count(st) === 5);
  check("mean", Math.abs(nf.mean(st) - 3) < 1e-6);
  check("min/max", nf.min(st) === 1 && nf.max(st) === 5);
  check("var (two-pass, ddof=0)", Math.abs(nf.var(st) - 2) < 1e-6);
  check("std", Math.abs(nf.std(st) - Math.sqrt(2)) < 1e-6);

  const di = nf.deviceInfo();
  check("deviceInfo()", di.backend === "cpu" && di.version === "1.0.0-alpha.2" &&
    di.kernels === 33, JSON.stringify(di));
}

// Example output (deliverable): zeros -> Series compute chain
console.log("\nEXAMPLE:");
const zex = nf.zeros(5);
console.log("  nf.zeros(5)          -> data=[" + Array.from(zex.data) +
  "] length=" + zex.length + " dtype=" + zex.dtype);
const sex = nf.Series.from(nf.arange(1, 6).data);
const rex = sex.mul(sex).add(1);           // single-op chain, eager compute
console.log("  arange(1,6).mul(self).add(1) -> [" + rex.toArray().join(", ") + "]");
console.log("  topk(arange(1,6), 3) -> [" + nf.topk(sex, 3).toArray().join(", ") + "]");

console.log("\nPARITY " + (FAIL === 0 ? "PASS" : "FAIL") +
  ": tests=" + (PASS + FAIL) + " passed=" + PASS + " failed=" + FAIL +
  " | uniform/integers bits Python==JS: " + (FAIL === 0 ? "YES" : "NO"));
process.exit(FAIL === 0 ? 0 : 1);
