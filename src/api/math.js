/**
 * math.js — math namespace + stats (JS_PARITY_SPEC §2: math / stats rows).
 *
 * Element-wise functions run through Map cpu func-codes:
 *   0=sin 1=cos 2=exp 3=sqrt 4=log 5=abs 6=neg 7=square.
 * PARITY NOTE: tan is NOT provided — verified against Python
 * Compute/_lib/map/cpu.py: _FUNCS holds exactly codes 0..7, no tan.
 *
 * stats via Reduce cpu (op sum/min/max):
 *   total/minimum/maximum -> single Reduce pass;
 *   mean -> total/n; var/std -> two-pass: mean, then MapBinary(sub scalar) ->
 *   Map(square) -> Reduce(sum), divided by n (population variance, ddof=0).
 *   count(x) -> x.length.
 */

"use strict";

const { runKernel } = require("./_runner");
const { Series } = require("./series");

const _MAP_CODE = {
  sin: 0, cos: 1, exp: 2, sqrt: 3, log: 4, abs: 5, neg: 6, square: 7,
};

function _f32(x) {
  const raw = x instanceof Series ? x.data : x;
  return raw instanceof Float32Array ? raw : Float32Array.from(raw);
}

/** element-wise Map over Series|TypedArray|Array -> Series */
function map1(name, x) {
  const code = _MAP_CODE[name];
  if (code === undefined) throw new Error(`math: unknown function ${name}`);
  const a = _f32(x);
  const out = runKernel("Map", { func: code }, [a], a.length);
  return new Series(Float32Array.from(out));
}

function sin(x) { return map1("sin", x); }
function cos(x) { return map1("cos", x); }
function exp(x) { return map1("exp", x); }
function log(x) { return map1("log", x); }
function sqrt(x) { return map1("sqrt", x); }
function abs(x) { return map1("abs", x); }
function square(x) { return map1("square", x); }
function neg(x) { return map1("neg", x); }

// ---- stats ----

function count(x) {
  return _f32(x).length;
}

function total(x) {
  const a = _f32(x);
  if (a.length === 0) return 0;
  const out = runKernel("Reduce", { op: "sum" }, [a], 1);
  return Number(out[0]);
}

function minimum(x) {
  const a = _f32(x);
  if (a.length === 0) throw new Error("stats: minimum of empty input");
  const out = runKernel("Reduce", { op: "min" }, [a], 1);
  return Number(out[0]);
}

function maximum(x) {
  const a = _f32(x);
  if (a.length === 0) throw new Error("stats: maximum of empty input");
  const out = runKernel("Reduce", { op: "max" }, [a], 1);
  return Number(out[0]);
}

function min(x) { return minimum(x); }
function max(x) { return maximum(x); }

function mean(x) {
  const a = _f32(x);
  if (a.length === 0) throw new Error("stats: mean of empty input");
  return total(a) / a.length;
}

/** population variance (ddof=0), two passes through cpu kernels */
function var_(x) {
  const a = _f32(x);
  const n = a.length;
  if (n === 0) throw new Error("stats: var of empty input");
  const m = total(a) / n;
  const centered = runKernel("MapBinary",
    { op: 1, use_scalar_b: 1, scalar_b: m }, [a], n);       // x - mean
  const squared = runKernel("Map",
    { func: 7 }, [Float32Array.from(centered)], n);         // (x-mean)^2
  return Number(runKernel("Reduce", { op: "sum" }, [squared], 1)[0]) / n;
}
const variance = var_;

function std(x) {
  return Math.sqrt(var_(x));
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    sin, cos, exp, log, sqrt, abs, square, neg,
    mean, min, max, minimum, maximum, total, count,
    var: var_, variance, std,
  };
}
