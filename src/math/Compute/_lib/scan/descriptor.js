"use strict";

var BLOCK = 64;
var OP_NAMES = { 0: "sum", 1: "mul", 2: "max", 3: "min" };
var OP_CODES = { sum: 0.0, mul: 1.0, max: 2.0, min: 3.0 };

function resolve_op(params) {
  params = params || {};
  var op = params.op === undefined ? "sum" : params.op;
  var name;
  var code;
  if (typeof op === "number") {
    name = OP_NAMES[op] || "sum";
    code = op;
  } else {
    name = OP_CODES[op] !== undefined ? op : "sum";
    code = OP_CODES[name] !== undefined ? OP_CODES[name] : 0.0;
  }
  return { name: name, code: code };
}

function describe_local(params) {
  var code = resolve_op(params).code;
  function sizes(input_sizes) {
    var n = input_sizes[0];
    var num_blocks = Math.max(1, Math.ceil(n / BLOCK));
    return [n, num_blocks];
  }
  return {
    inputs: [{ name: "data", dtype: "float" }],
    outputs: [
      { dtype: "float", template: "scan_local" },
      { dtype: "float", template: "block_sum" },
    ],
    workspace: [],
    uniforms: { op: code },
    output_size_fn: sizes,
  };
}

function describe_totals(params) {
  var code = resolve_op(params).code;
  function sizes(input_sizes) {
    return [input_sizes[0]];
  }
  return {
    inputs: [{ name: "block_sums", dtype: "float" }],
    outputs: [{ dtype: "float", template: "block_prefix" }],
    workspace: [],
    uniforms: { op: code },
    output_size_fn: sizes,
  };
}

function describe_final(params) {
  var resolved = resolve_op(params);
  var template = resolved.name === "sum" ? "scan" : "scan_" + resolved.name;
  function sizes(input_sizes) {
    return [input_sizes[0]];
  }
  return {
    inputs: [
      { name: "data", dtype: "float" },
      { name: "local", dtype: "float" },
      { name: "prefix", dtype: "float" },
    ],
    outputs: [{ dtype: "float", template: template }],
    workspace: [],
    uniforms: { op: resolved.code },
    output_size_fn: sizes,
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { describe_local, describe_totals, describe_final };
}
