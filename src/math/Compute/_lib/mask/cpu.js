/**
 * Mask CPU — JS reference: out[i] = cond[i] ? a[i] : b[i].
 * cond is uint32, a/b can be arrays or scalars.
 */

"use strict";

function cpu(ctx) {
  const cond = ctx.inputs[0].view;
  const dst = ctx.outputs[0].view;
  const use_sa = Number(ctx.uniforms.use_scalar_a || 0);
  const use_sb = Number(ctx.uniforms.use_scalar_b || 0);
  const scalar_a = Number(ctx.uniforms.scalar_a || 0.0);
  const scalar_b = Number(ctx.uniforms.scalar_b || 0.0);

  let idx = 1;
  const a_view = use_sa ? null : ctx.inputs[idx].view;
  if (!use_sa) idx++;
  const b_view = use_sb ? null : ctx.inputs[idx].view;

  const n = dst.length();
  for (let i = 0; i < n; i++) {
    const a_val = use_sa ? scalar_a : a_view.read(i);
    const b_val = use_sb ? scalar_b : b_view.read(i);
    dst.write(i, cond.read(i) !== 0 ? a_val : b_val);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { cpu };
}
