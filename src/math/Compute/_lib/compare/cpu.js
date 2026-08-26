/**
 * Compare CPU — JS reference: out[i] = a[i] op b[i], result 0/1 (uint32).
 * op: string ("gt"/"ge"/"lt"/"le"/"eq"/"ne") or numeric code (0..5).
 */

"use strict";

const CODES = { 0: "gt", 1: "ge", 2: "lt", 3: "le", 4: "eq", 5: "ne" };

function resolveOp(op) {
  if (typeof op === "string") {
    return ["gt", "ge", "lt", "le", "eq", "ne"].includes(op) ? op : "gt";
  }
  return CODES[op] || "gt";
}

function _compare(a, b, op) {
  switch (op) {
    case "gt": return a > b ? 1 : 0;
    case "ge": return a >= b ? 1 : 0;
    case "lt": return a < b ? 1 : 0;
    case "le": return a <= b ? 1 : 0;
    case "eq": return a === b ? 1 : 0;
    case "ne": return a !== b ? 1 : 0;
    default: return 0;
  }
}

function cpu(ctx) {
  const op = resolveOp(ctx.uniforms.op || "gt");
  const dst = ctx.outputs[0].view;
  const use_sa = Number(ctx.uniforms.use_scalar_a || 0);
  const use_sb = Number(ctx.uniforms.use_scalar_b || 0);
  const scalar_a = Number(ctx.uniforms.scalar_a || 0.0);
  const scalar_b = Number(ctx.uniforms.scalar_b || 0.0);

  let input_idx = 0;
  const a_view = use_sa ? null : ctx.inputs[input_idx].view;
  if (!use_sa) input_idx++;
  const b_view = use_sb ? null : ctx.inputs[input_idx].view;

  const n = dst.length();
  for (let i = 0; i < n; i++) {
    const a_val = use_sa ? scalar_a : a_view.read(i);
    const b_val = use_sb ? scalar_b : b_view.read(i);
    dst.write(i, _compare(a_val, b_val, op));
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { cpu, _compare, resolveOp };
}
