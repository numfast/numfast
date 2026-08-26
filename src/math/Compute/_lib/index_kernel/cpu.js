/**
 * IndexKernel CPU oracle — JS port of Compute/_lib/index_kernel/cpu.py.
 * np.full / np.arange / np.linspace / np.tile / np.repeat equivalents.
 *
 * dtype semantics (owner decision): uniform flag int_mode=1 -> integer path:
 * values in i32 domain (EXACT up to 2^31, f32 2^24 limit not applied),
 * arithmetic i32 WRAP (two's complement), linspace — C-trunc division.
 *
 * f32 arange note: np.arange(p0, p0+p1*n, p1, f32) generates start + i*step in
 * double precision then casts to f32 (the Python length-mismatch fallback uses
 * the same formula), so a single formula covers both branches.
 * f32 linspace standard case mirrors np.linspace(endpoint=True): last element
 * forced to p1 before the f32 cast.
 */

"use strict";

function _wrap_i32(x) {
  // Wrap to int32 domain (two's complement, WGSL parity). Exact for |x| < 2^53.
  return x | 0;
}

function _cpu_int32(ctx) {
  const dst = ctx.outputs[0].view;
  const n = Math.trunc(Number(ctx.uniforms.n));
  const mode = Math.trunc(Number(ctx.uniforms.mode));
  const p0 = Math.trunc(Number(ctx.uniforms.p0));
  const p1 = Math.trunc(Number(ctx.uniforms.p1));
  const p2 = Math.trunc(Number(ctx.uniforms.p2));

  for (let i = 0; i < n; i++) {
    let v;
    if (mode === 0) {
      v = p0;                                            // const(p0)
    } else if (mode === 1) {
      v = (p0 + Math.imul(p1, i)) | 0;                   // arange: WRAP i32
    } else if (mode === 3 || mode === 4) {
      const k = Math.trunc(Number(ctx.uniforms.k));
      const rep = Math.trunc(Number(ctx.uniforms.rep));
      const pat = new Array(k);
      for (let j = 0; j < k; j++) {
        pat[j] = Math.trunc(ctx.inputs[0].view.read(j)); // host-cast parity
      }
      if (mode === 3) {
        v = pat[i % k];                                  // tile
      } else {
        const q = Math.floor(i / rep);
        v = pat[Math.min(q, k - 1)];                     // repeat (u32 div)
      }
    } else {
      // linspace integer-trunc semantics: C-style trunc division, all steps
      // in i32 domain (WRAP on overflow, documented).
      const num = (p1 - p0) * i;
      v = (p0 + Math.trunc(num / p2)) | 0;
    }
    dst.write(i, _wrap_i32(v));
  }
}

function cpu(ctx) {
  if (Math.trunc(Number(ctx.uniforms.int_mode || 0))) {
    return _cpu_int32(ctx);
  }

  const dst = ctx.outputs[0].view;
  const n = Math.trunc(Number(ctx.uniforms.n));
  const mode = Math.trunc(Number(ctx.uniforms.mode));
  const p0 = Number(ctx.uniforms.p0);
  const p1 = Number(ctx.uniforms.p1);
  const p2 = Number(ctx.uniforms.p2);

  for (let i = 0; i < n; i++) {
    let v;
    if (mode === 0) {
      v = p0;                                            // const(p0)
    } else if (mode === 1) {
      v = Math.fround(p0 + p1 * i);                      // arange(start, step)
    } else if (mode === 3) {
      // tile(pattern, n): out[i] = pat[i % k] — pattern is an input buffer
      const k = Math.trunc(Number(ctx.uniforms.k));
      v = ctx.inputs[0].view.read(i % k);
    } else if (mode === 4) {
      // repeat(pattern, rep): out[i] = pat[min(i / rep, k - 1)] (u32 div)
      const k = Math.trunc(Number(ctx.uniforms.k));
      const rep = Math.trunc(Number(ctx.uniforms.rep));
      const q = Math.min(Math.floor(i / rep), k - 1);
      v = ctx.inputs[0].view.read(q);
    } else if (p2 === Number(Math.max(n - 1, 1))) {
      // linspace(a=p0, b=p1, denom=p2); standard case: endpoint=True parity
      const step = (p1 - p0) / p2;
      v = Math.fround(i === n - 1 ? p1 : p0 + i * step);
    } else {
      v = Math.fround(p0 + ((p1 - p0) * i) / p2);
    }
    dst.write(i, v);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { cpu, _wrap_i32 };
}
