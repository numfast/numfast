// RBlockSource: batched Mersenne-Twister stream, bit-identical to r_rng.mjs.
//
// Oracle: tools/r_rng.mjs (FROZEN — read only, never modified).
// Contract: init(seed) once; nextU32Block/nextUnifBlock return TypedArrays whose
// concatenated sequence === scalar makeR(seed) draws in the same call order
// (twist/temper identical, only batched). Pure ESM, no imports, no node: APIs.
const I2_32M1 = 2.3283064365386963e-10; // 2^-32
const MT_N = 624;
const MT_M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;
const FIX_LO = 0.5 * I2_32M1;
const FIX_HI = 1.0 - 0.5 * I2_32M1;

function lcgStep(s) {
  return (Math.imul(69069, s) + 1) | 0;
}

export class RBlockSource {
  constructor() {
    this._mt = new Uint32Array(MT_N);
    this._mti = MT_N;
    this._ready = false;
  }

  // Seed once (Ripley init, same as oracle). Second call throws.
  init(seed) {
    if (this._ready) throw new Error("RBlockSource: init(seed) must be called once");
    let s = seed | 0; // same coercion as oracle (seed | 0)
    for (let j = 0; j < 50; j++) s = lcgStep(s);
    s = lcgStep(s);
    for (let j = 0; j < MT_N; j++) {
      s = lcgStep(s);
      this._mt[j] = s >>> 0;
    }
    this._mti = MT_N; // force twist on first draw, like oracle
    this._ready = true;
    return this;
  }

  _ensure() {
    if (!this._ready) throw new Error("RBlockSource: init(seed) first");
  }

  _twist() {
    const mt = this._mt;
    for (let kk = 0; kk < MT_N - MT_M; kk++) {
      const y = ((mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)) >>> 0;
      mt[kk] = (mt[kk + MT_M] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
    }
    for (let kk = MT_N - MT_M; kk < MT_N - 1; kk++) {
      const y = ((mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)) >>> 0;
      mt[kk] = (mt[kk + (MT_M - MT_N)] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
    }
    const y = ((mt[MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK)) >>> 0;
    mt[MT_N - 1] = (mt[MT_M - 1] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
    this._mti = 0;
  }

  // Raw tempered uint32s in MT order. drain === n sequential genrandInt32().
  nextU32Block(n) {
    this._ensure();
    if (!Number.isInteger(n) || n < 0) throw new Error("RBlockSource: n must be non-negative int");
    const out = new Uint32Array(n);
    const mt = this._mt;
    let mti = this._mti;
    let i = 0;
    while (i < n) {
      if (mti >= MT_N) {
        this._mti = mti;
        this._twist();
        mti = 0;
      }
      const take = Math.min(n - i, MT_N - mti);
      for (let j = 0; j < take; j++) {
        let y = mt[mti++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
        out[i++] = y >>> 0;
      }
    }
    this._mti = mti;
    return out;
  }

  // unif_rand() in bulk: fixup(u * 2^-32), same expression as oracle.
  nextUnifBlock(n) {
    this._ensure();
    if (!Number.isInteger(n) || n < 0) throw new Error("RBlockSource: n must be non-negative int");
    const out = new Float64Array(n);
    const mt = this._mt;
    let mti = this._mti;
    let i = 0;
    while (i < n) {
      if (mti >= MT_N) {
        this._mti = mti;
        this._twist();
        mti = 0;
      }
      const take = Math.min(n - i, MT_N - mti);
      for (let j = 0; j < take; j++) {
        let y = mt[mti++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
        const x = (y >>> 0) * I2_32M1;
        out[i++] = x <= 0 ? FIX_LO : 1 - x <= 0 ? FIX_HI : x;
      }
    }
    this._mti = mti;
    return out;
  }
}
