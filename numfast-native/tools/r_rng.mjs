// R-compatible RNG (demo layer only, NOT core).
// Faithful to R 4.3.x src/main/RNG.c defaults:
//   RNGkind = Mersenne-Twister, Normal = Inversion, Sample = Rejection.
// Browser-safe: pure ESM, no imports, no node: APIs, no I/O.
//
// Chain per R source:
//   set.seed(s) -> Ripley init (LCG 69069 scrambling, NOT 1812433253)
//   unif_rand() -> fixup(MT_genrand()), MT_genrand = tempered uint32 * 2^-32
//   R_unif_index(dn) [rejection default] -> rbits() in 16-bit chunks,
//     bits = ceil(log2(dn)), redraw while dn <= dv; 0-based result
//   do_sample replace path -> R_unif_index(n) + 1 per draw
//   runif(n, min, max) -> min + (max - min) * unif_rand()
//   round(x, d) -> half-to-even (banker's, like R)
//
// Verified against R 4.3.2: set.seed(108); sample(100,10,TRUE) ->
//   [16,39,47,43,54,29,47,91,90,70] (see tools/r_ref_fixture.json).

const I2_32M1 = 2.3283064365386963e-10; // 2^-32
const MT_N = 624;
const MT_M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

function lcgStep(s) {
  return (Math.imul(69069, s) + 1) | 0;
}

export function makeR(seed, opts = {}) {
  const sampleKind = opts.sampleKind ?? 'rejection'; // 'rejection' (R >= 4.x default) | 'rounding'
  const mt = new Uint32Array(MT_N);
  // RNG_Init (Ripley): 50 LCG scrambles, then 625 fills; slot 0 is mti.
  let s = seed | 0;
  for (let j = 0; j < 50; j++) s = lcgStep(s);
  s = lcgStep(s); // i_seed[0] -> overwritten by FixupSeeds(initial=1) with mti=624
  for (let j = 0; j < MT_N; j++) {
    s = lcgStep(s);
    mt[j] = s >>> 0;
  }
  let mti = MT_N; // force twist on first draw

  function twist() {
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
    mti = 0;
  }

  function genrandInt32() {
    if (mti >= MT_N) twist();
    let y = mt[mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  function fixup(x) {
    if (x <= 0.0) return 0.5 * I2_32M1;
    if (1.0 - x <= 0.0) return 1.0 - 0.5 * I2_32M1;
    return x;
  }

  // R unif_rand() for Mersenne-Twister.
  function unif() {
    return fixup(genrandInt32() * I2_32M1);
  }

  // R rbits(): non-negative int < 2^bits built from 16-bit chunks.
  function rbits(bits) {
    let v = 0;
    for (let n = 0; n <= bits; n += 16) {
      v = 65536 * v + Math.floor(unif() * 65536);
    }
    const mod = 2 ** bits;
    return v % mod;
  }

  function unifIndexRound(dn) {
    return Math.floor(dn * unif());
  }

  // R R_unif_index(), 0-based double. Rejection is the R >= 4.x default.
  function unifIndex(dn) {
    if (sampleKind === 'rounding') return unifIndexRound(dn);
    if (!(dn > 0)) return 0;
    const bits = Math.ceil(Math.log2(dn));
    let dv;
    do {
      dv = rbits(bits);
    } while (dn <= dv);
    return dv;
  }

  // do_sample replace path for population 1..n (scalar first arg).
  function sampleReplace(n, k) {
    const out = new Array(k);
    for (let i = 0; i < k; i++) out[i] = unifIndex(n) + 1;
    return out;
  }

  function runif(n, min = 0, max = 1) {
    const out = new Array(n);
    const span = max - min;
    for (let i = 0; i < n; i++) out[i] = min + span * unif();
    return out;
  }

  return { unif, unifIndex, sampleReplace, runif, genrandInt32 };
}

// R round(): half-to-even on the scaled value.
export function rRound(x, digits = 0) {
  if (!Number.isFinite(x)) return x;
  const p = 10 ** digits;
  const sgn = x < 0 || Object.is(x, -0) ? -1 : 1;
  const y = Math.abs(x) * p;
  const f = Math.floor(y);
  const d = y - f;
  let r;
  if (d < 0.5) r = f;
  else if (d > 0.5) r = f + 1;
  else r = f % 2 === 0 ? f : f + 1; // exact tie -> even
  return (sgn * r) / p;
}
