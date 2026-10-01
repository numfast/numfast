// RColumnSink: column-at-a-time exact-R draws over an RBlockSource.
//
// Oracle: tools/r_rng.mjs (FROZEN — read only). Invariant: the consumed u32
// sequence (INCLUDING rejection redraws) is identical to the scalar oracle,
// so a sink column === scalar column element-wise for the same stream position.
// Exact-R only: rejection sampling (R >= 4.x default). No dead kind switch,
// no per-draw closures, bits/dn/mod hoisted per column, fast path bits<=16
// (1-2 u32 chunks, integer arithmetic). Numeric columns are TypedArrays only
// (no strings): sampleReplaceBlock -> Int32Array (1-based, like oracle),
// runifBlock -> Float64Array, rRoundBlock -> Float64Array.
// Pure ESM, no imports, no node: APIs. src is duck-typed { nextU32Block(n) }.
const I2_32M1 = 2.3283064365386963e-10; // 2^-32 (same constants as oracle)
const FIX_LO = 0.5 * I2_32M1;
const FIX_HI = 1.0 - 0.5 * I2_32M1;
const REFILL = 65536; // u32s per source pull
const WIN1 = 4096; // candidate window (1 chunk per candidate)
const WIN2 = 8192; // candidate window (2 chunks per candidate)

export class RColumnSink {
  constructor(src) {
    if (!src || typeof src.nextU32Block !== "function")
      throw new Error("RColumnSink: src must provide nextU32Block(n)");
    this._src = src;
    this._buf = new Uint32Array(0); // leftover u32s persist across columns: never dropped
    this._pos = 0;
    this._end = 0;
  }

  _need(m) {
    const avail = this._end - this._pos;
    if (avail >= m) return;
    const nb = this._src.nextU32Block(Math.max(m - avail, REFILL));
    if (avail === 0) {
      this._buf = nb;
      this._pos = 0;
      this._end = nb.length;
    } else {
      const buf = new Uint32Array(avail + nb.length);
      buf.set(this._buf.subarray(this._pos, this._end), 0);
      buf.set(nb, avail);
      this._buf = buf;
      this._pos = 0;
      this._end = buf.length;
    }
  }

  // R do_sample replace path: R_unif_index(dn)+1 per draw, rejection redraws
  // consume stream exactly like scalar rbits (chunk = u>>>16 === floor(unif()*65536)).
  sampleReplaceBlock(dn, k) {
    if (!Number.isInteger(dn) || dn <= 0) throw new Error("RColumnSink: dn must be positive int");
    if (!Number.isInteger(k) || k < 0) throw new Error("RColumnSink: k must be non-negative int");
    const out = new Int32Array(k);
    if (k === 0) return out;
    const bits = Math.ceil(Math.log2(dn)); // hoisted (scalar: per draw)
    const mod = 2 ** bits; // hoisted
    const nch = (bits >> 4) + 1; // u32s per candidate === scalar rbits loop count
    let i = 0;
    if (nch <= 2) {
      const two = nch === 2;
      while (i < k) {
        this._need(two ? WIN2 : WIN1);
        const buf = this._buf;
        let p = this._pos;
        const end = this._end;
        while (i < k) {
          if (p + nch > end) break;
          let dv;
          if (two) {
            dv = (65536 * (buf[p] >>> 16) + (buf[p + 1] >>> 16)) % mod;
            p += 2;
          } else {
            dv = (buf[p] >>> 16) % mod;
            p += 1;
          }
          while (dn <= dv) {
            if (p + nch > end) break;
            if (two) {
              dv = (65536 * (buf[p] >>> 16) + (buf[p + 1] >>> 16)) % mod;
              p += 2;
            } else {
              dv = (buf[p] >>> 16) % mod;
              p += 1;
            }
          }
          if (dn <= dv) break; // buffer ran out mid-rejection: refill, retry slot
          out[i++] = dv + 1;
        }
        this._pos = p;
      }
    } else {
      while (i < k) {
        this._need(WIN1);
        const buf = this._buf;
        let p = this._pos;
        const end = this._end;
        while (i < k) {
          if (p + nch > end) break;
          let v = 0;
          for (let c = 0; c < nch; c++) v = 65536 * v + (buf[p++] >>> 16);
          let dv = v % mod;
          while (dn <= dv) {
            if (p + nch > end) break;
            v = 0;
            for (let c = 0; c < nch; c++) v = 65536 * v + (buf[p++] >>> 16);
            dv = v % mod;
          }
          if (dn <= dv) break;
          out[i++] = dv + 1;
        }
        this._pos = p;
      }
    }
    return out;
  }

  // runif(n, min, max): min + (max-min) * unif_rand(), one u32 per output.
  runifBlock(n, min = 0, max = 1) {
    if (!Number.isInteger(n) || n < 0) throw new Error("RColumnSink: n must be non-negative int");
    const out = new Float64Array(n);
    if (n === 0) return out;
    const span = max - min;
    let i = 0;
    while (i < n) {
      this._need(WIN1);
      const buf = this._buf;
      let p = this._pos;
      const end = this._end;
      while (i < n && p < end) {
        const x = buf[p++] * I2_32M1;
        const f = x <= 0 ? FIX_LO : 1 - x <= 0 ? FIX_HI : x;
        out[i++] = min + span * f;
      }
      this._pos = p;
    }
    return out;
  }

  // R round(): half-to-even, pure element map. Same expression as oracle.
  rRoundBlock(values, digits = 0) {
    const p = 10 ** digits; // hoisted (scalar: per element)
    const out = new Float64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const x = values[i];
      if (!Number.isFinite(x)) {
        out[i] = x;
        continue;
      }
      const sgn = x < 0 || Object.is(x, -0) ? -1 : 1;
      const y = Math.abs(x) * p;
      const f = Math.floor(y);
      const d = y - f;
      let r;
      if (d < 0.5) r = f;
      else if (d > 0.5) r = f + 1;
      else r = f % 2 === 0 ? f : f + 1;
      out[i] = (sgn * r) / p;
    }
    return out;
  }
}
