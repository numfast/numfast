"use strict";

var OPS = {
  sum: function (a, b) { return a + b; },
  mul: function (a, b) { return a * b; },
  max: function (a, b) { return Math.max(a, b); },
  min: function (a, b) { return Math.min(a, b); },
};
var CODES = { 0: "sum", 1: "mul", 2: "max", 3: "min" };
var NEUTRAL = { sum: 0.0, mul: 1.0, max: -3.4028235e38, min: 3.4028235e38 };

function op_name(ctx) {
  var op = ctx.uniforms.op === undefined ? "sum" : ctx.uniforms.op;
  if (typeof op !== "string") {
    op = CODES[op];
    if (op === undefined) { throw new Error("Unknown Scan op code: " + ctx.uniforms.op); }
  }
  if (!OPS[op]) { throw new Error("Unknown Scan op: " + op); }
  return op;
}

function cpu_local(ctx) {
  const src = ctx.inputs[0].view;
  const dst = ctx.outputs[0].view;
  const sums = ctx.outputs[1].view;
  const func = OPS[op_name(ctx)];
  const n = src.length();
  const BLOCK = 64;

  const num_full_blocks = Math.floor(n / BLOCK);
  for (let b = 0; b < num_full_blocks; b++) {
    const start = b * BLOCK;
    let total = src.read(start);
    dst.write(start, total);
    for (let i = start + 1; i < start + BLOCK; i++) {
      total = func(total, src.read(i));
      dst.write(i, total);
    }
    sums.write(b, total);
  }

  const remainder = n % BLOCK;
  if (remainder > 0) {
    const start = num_full_blocks * BLOCK;
    let total = src.read(start);
    dst.write(start, total);
    for (let i = start + 1; i < n; i++) {
      total = func(total, src.read(i));
      dst.write(i, total);
    }
    sums.write(num_full_blocks, total);
  }
}

function cpu_totals(ctx) {
  const src = ctx.inputs[0].view;
  const dst = ctx.outputs[0].view;
  const op = op_name(ctx);
  const func = OPS[op];
  const n = src.length();
  if (n === 0) return;
  let acc = NEUTRAL[op];
  dst.write(0, acc);
  for (let i = 1; i < n; i++) {
    acc = func(acc, src.read(i - 1));
    dst.write(i, acc);
  }
}

function cpu_final(ctx) {
  const local = ctx.inputs[1].view;
  const prefix = ctx.inputs[2].view;
  const dst = ctx.outputs[0].view;
  const func = OPS[op_name(ctx)];
  const n = local.length();
  const BLOCK = 64;

  for (let i = 0; i < n; i++) {
    const block_id = Math.floor(i / BLOCK);
    if (block_id > 0) {
      dst.write(i, func(local.read(i), prefix.read(block_id)));
    } else {
      dst.write(i, local.read(i));
    }
  }
}

module.exports = { cpu_local, cpu_totals, cpu_final };
