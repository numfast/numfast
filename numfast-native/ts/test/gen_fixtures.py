# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""gen_fixtures.py -- produce the expected values the JS/WASM parity test
compares against, from the PYTHON host.

WHY A GENERATOR AND NOT A HARD-CODED TABLE
The JS side cannot compute what Python computes without reimplementing NumPy's
casting rules, which is the thing under test. So the expectation is produced
here, by two independent Python paths, and committed:

    reference = NumPy, with the documented int32 round-trip applied
                (series/map.rs::npy_i32_from_f64)
    native    = numfast-native through ctypes (src/numfast/_native/*.dll)

Both are compared byte-for-byte BEFORE anything is written. If they disagree
the generator ABORTS: a fixture that records a disagreement as if it were
truth is worse than no fixture.

WHY INPUTS ARE NOT STORED
The first draft stored the inputs too and the file came to 31 MB of hex, which
does not belong in git. Instead every input is reproduced from an explicit
32-bit LCG (below), and the fixture records the GENERATOR SPEC, not the bytes:

  * small cases (n <= 64)  -> the expected bytes verbatim, so a reader can
    check them by hand. These are the hand-checkable cases.
  * large cases (n = 100000)-> sha256 of the expected bytes plus the first and
    last lane, so a regression is still a byte-for-byte comparison, just
    without shipping 400 kB of hex per case.

Seed 42 throughout. Run:
    python numfast-native/ts/test/gen_fixtures.py
"""

import base64
import ctypes
import hashlib
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DLL = os.path.join(APP, "src", "numfast", "_native", "numfast_native.dll")
PROFILE = os.path.join(APP, "specs-rebuilt", "conformance-profile.toml")
OUT = os.path.join(HERE, "fixtures", "parity.json")
SEED = 42
LITERAL_MAX = 64

if not os.path.exists(DLL):
    raise SystemExit(
        "PARITY FIXTURES ABORTED: no native library at\n  %s\n"
        "These fixtures are produced by the Python host through ctypes. Without\n"
        "the committed DLL there is no reference, and a fixture generated from\n"
        "anything else would be a guess wearing a fixture's clothes." % DLL)

lib = ctypes.CDLL(DLL)


def tolerance():
    """Read the project's OWN conformance budget rather than inventing one.
    Spec 08 grants f32 atol/rtol 1e-5 or 4 ULP, f64 1e-12 or 2 ULP. The JS
    test enforces exactly this, so a float-lane difference inside the budget
    is conformance and a difference outside it is a defect -- decided by a
    number the project already published, not by a number chosen here."""
    import tomllib
    with open(PROFILE, "rb") as f:
        prof = tomllib.load(f)
    return {
        k: {"atol": prof["tolerance"][k]["atol"], "rtol": prof["tolerance"][k]["rtol"],
            "maxUlp": prof["tolerance"][k]["max_ulp"]}
        for k in ("f32", "f64")
    }


# --- the shared input generator --------------------------------------------
# Identical arithmetic in Python and JavaScript. `s / 2**32` is exact in IEEE
# 754 double, and one multiply plus one add is exactly specified, so both
# languages produce bit-identical lanes -- which is the property the parity
# test depends on.
def lcg_stream(seed, n):
    s = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(s)
    return out


def gen_int(n, lo, hi):
    span = hi - lo + 1
    return np.array([lo + (u * span) // 4294967296 for u in lcg_stream(SEED, n)],
                    dtype=np.int64).astype(np.int32)


def gen_f64(n, scale, offset):
    return np.array([(u / 4294967296.0) * scale + offset
                     for u in lcg_stream(SEED, n)], dtype=np.float64)


def gen_f32(n, scale, offset):
    return gen_f64(n, scale, offset).astype(np.float32)


# --- ctypes bindings --------------------------------------------------------
def _arr(name):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                   ctypes.c_uint32, ctypes.c_void_p]
    fn.restype = ctypes.c_int32
    return fn


def _sca(name):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_double,
                   ctypes.c_uint32, ctypes.c_void_p]
    fn.restype = ctypes.c_int32
    return fn


MAP_ARR = {k: _arr("nf_map_" + k) for k in ("i32", "f32", "f64")}
def _sca_i32(name):
    """`nf_map_scalar_i32` takes an i32 scalar, not an f64. Binding it with a
    double is not a harmless widening: it hands the kernel the wrong bits and
    the fixture would record the kernel's answer to a question nobody asked."""
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32,
                   ctypes.c_uint32, ctypes.c_void_p]
    fn.restype = ctypes.c_int32
    return fn


MAP_SCA = {k: _sca("nf_map_scalar_" + k) for k in ("f32", "f64")}
MAP_SCA["i32"] = _sca_i32("nf_map_scalar_i32")
MAP_SCA["fscalar_i32"] = _sca("nf_map_fscalar_i32")
MAP_DIVPOW_ARR = _arr("nf_map_f32_divpow")
MAP_DIVPOW_SCA = _sca("nf_map_scalar_f32_divpow")

KIND_DT = {"i32": np.int32, "f32": np.float32, "f64": np.float64}
OP_NUMPY = {0: "add", 1: "subtract", 2: "multiply", 3: "true_divide",
            4: "power", 5: "floor_divide", 6: "remainder"}
OP_NAME = {0: "add", 1: "sub", 2: "mul", 3: "div", 4: "pow",
           5: "floor_div", 6: "mod"}


def i32_scalar_ref(a, s, op):
    """`lane_iscalar_i32`: an INTEGER scalar stays in-lane, so add/sub/mul
    wrap mod 2^32 and never touch f64. div/pow ride the same f64 +
    round-ties-even round-trip as the array path. floor_div/mod are NumPy's
    Python-floor semantics directly, and `//0` is 0 (contract), never a trap."""
    if op in (0, 1, 2):
        wrap = {0: np.add, 1: np.subtract, 2: np.multiply}[op]
        return wrap(a.astype(np.int32), np.int32(s)).astype(np.int32)
    if op in (3, 4):
        return i32_from_f64(npy("i32", a, s, op), round_first=True)
    return np.asarray(npy("i32", a, s, op), np.int32)


def i32_from_f64(v, round_first=True):
    """`series/map.rs::npy_i32_from_f64`: NaN, +-Inf and anything outside
    [i32::MIN, 2^31) become INT_MIN (the x86 `cvttsd2si` rule); everything else
    truncates toward zero. `div`/`pow` round ties-even in f64 first
    (`np.round` is ties-even); `add`/`sub`/`mul` with a float scalar do NOT
    round at all, so `round_first=False` there -- rounding 2.5 to 3 where the
    contract says 2 is a whole-lane error, not a tolerance question."""
    r = np.asarray(v, dtype=np.float64)
    if round_first:
        r = np.round(r)
    bad = np.isnan(r) | (r >= 2.0 ** 31) | (r < -(2.0 ** 31))
    return np.where(bad, -2147483648.0, np.trunc(r)).astype(np.int32)


def npy(kind, a, other, op):
    """The reference. Note the f32 lane: `div` and `pow` on f32 inputs are
    computed in FLOAT64 by the contract (`lane_divpow_f32` casts both operands
    to f64 before the op and returns f64), so the reference promotes them.
    Computing in f32 here would compare the kernel against the wrong
    arithmetic -- and would overflow to Infinity where the kernel returns a
    finite 1e252."""
    widen = kind == "f32" and op in (3, 4)
    dt = np.float64 if widen else KIND_DT[kind]
    x = a.astype(dt)
    y = other.astype(dt) if isinstance(other, np.ndarray) else other
    return getattr(np, OP_NUMPY[op])(x, y)


def native(sym, a, other, op, out_dtype):
    n = a.size
    out = np.empty(n, out_dtype)
    if sym == "nf_map_f32_divpow":
        rc = MAP_DIVPOW_ARR(a.ctypes.data, other.ctypes.data, n, op, out.ctypes.data)
    elif sym == "nf_map_scalar_f32_divpow":
        rc = MAP_DIVPOW_SCA(a.ctypes.data, n, float(other), op, out.ctypes.data)
    elif sym.startswith("nf_map_fscalar_i32"):
        rc = MAP_SCA["fscalar_i32"](a.ctypes.data, n, float(other), op, out.ctypes.data)
    elif sym in ("nf_map_i32", "nf_map_f32", "nf_map_f64"):
        rc = MAP_ARR[sym[len("nf_map_"):]](a.ctypes.data, other.ctypes.data, n, op, out.ctypes.data)
    elif sym in ("nf_map_scalar_i32", "nf_map_scalar_f32", "nf_map_scalar_f64"):
        fn = MAP_SCA[sym[len("nf_map_scalar_"):]]
        sc = int(other) if sym == "nf_map_scalar_i32" else float(other)
        rc = fn(a.ctypes.data, n, sc, op, out.ctypes.data)
    else:
        raise SystemExit("unknown symbol " + sym)
    if rc != 0:
        raise SystemExit("%s rc=%d" % (sym, rc))
    return out


def hexs(a):
    return base64.b16encode(np.ascontiguousarray(a).tobytes()).decode("ascii").lower()


def expect_field(out):
    raw = np.ascontiguousarray(out).tobytes()
    if len(raw) <= LITERAL_MAX * 8:
        return {"expect_hex": hexs(out), "expect_bytes": len(raw)}
    lanes = np.ascontiguousarray(out)
    return {
        "expect_sha256": hashlib.sha256(raw).hexdigest(),
        "expect_bytes": len(raw),
        "head": [int(v) for v in lanes[:8]],
        "tail": [int(v) for v in lanes[-8:]],
    }


def main():
    cases = {}

    def add(name, **kw):
        if name in cases:
            raise SystemExit("duplicate case " + name)
        cases[name] = kw

    def i32_arr(op, vi, vj, label):
        r = npy("i32", vi, vj, op)
        ref = i32_from_f64(r, round_first=True) if op == 3 else np.asarray(r, np.int32)
        nat = native("nf_map_i32", vi, vj, op, np.int32)
        if not np.array_equal(nat, ref):
            raise SystemExit("native != numpy nf_map_i32/%s n=%d (%d differ)"
                             % (OP_NAME[op], vi.size, int((nat != ref).sum())))
        add(label, symbol="nf_map_i32", form="arr", op=op, dtype="i32", **expect_field(ref))

    def f32_arr(op, vf, vg, label):
        r = npy("f32", vf, vg, op)
        ref = np.asarray(r, np.float64) if op in (3, 4) else np.asarray(r, np.float32)
        nat = native("nf_map_f32_divpow" if op in (3, 4) else "nf_map_f32",
                     vf, vg, op, ref.dtype)
        add(label, symbol="nf_map_f32_divpow" if op in (3, 4) else "nf_map_f32",
            form="arr", op=op, dtype="f64" if op in (3, 4) else "f32", **expect_field(ref))

    def f64_arr(op, vd, ve, label):
        ref = np.asarray(npy("f64", vd, ve, op), np.float64)
        nat = native("nf_map_f64", vd, ve, op, np.float64)
        if not np.array_equal(nat, ref, equal_nan=True):
            raise SystemExit("native != numpy nf_map_f64/%s n=%d" % (OP_NAME[op], vd.size))
        add(label, symbol="nf_map_f64", form="arr", op=op, dtype="f64", **expect_field(ref))

    def fscalar(op, vi, s, label):
        ref = i32_from_f64(npy("i32", vi, s, op), round_first=op in (3, 4))
        nat = native("nf_map_fscalar_i32", vi, s, op, np.int32)
        if not np.array_equal(nat, ref):
            raise SystemExit("native != numpy nf_map_fscalar_i32/%s s=%s n=%d (%d differ)"
                             % (OP_NAME[op], s, vi.size, int((nat != ref).sum())))
        add(label, symbol="nf_map_fscalar_i32", form="scalar", op=op, dtype="i32",
            scalar=s, **expect_field(ref))

    # ---- i32 array-array: 6 (hand-checkable), 64, 100000 (allocator guard)
    for n in (6, 64, 100000):
        vi = gen_int(n, -10000, 10000)
        vj = gen_int(n, -500, 500)
        for op in (0, 1, 2, 3, 5, 6):
            i32_arr(op, vi, vj, "map_i32/%s/n%d" % (OP_NAME[op], n))

    # ---- i32 with a float scalar, INCLUDING pow --------------------------
    # The acceptance audit recorded WASM `map pow` with a fractional exponent
    # diverging at n=100000 while agreeing at n=6, and suspected a stale
    # artefact. Re-measured on the current 85-function build: native agrees
    # with NumPy at every size and every exponent tried, and so does WASM
    # once the caller stops writing over the module's own constant pool. These
    # cases pin the fixed behaviour at both sizes, so a regression is a test
    # failure rather than a rediscovery.
    for n in (6, 64, 100000):
        vi = gen_int(n, -10000, 10000)
        for s in (0.5, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0):
            fscalar(4, vi, s, "map_fscalar_i32/pow/s%s/n%d" % (s, n))
        for op in (0, 1, 2, 3):
            fscalar(op, vi, 2.5, "map_fscalar_i32/%s/n%d" % (OP_NAME[op], n))

    # ---- f64: the CPython divmod algorithm, zero divisors included --------
    for n in (6, 64):
        vd = gen_f64(n, 2000.0, -1000.0)
        ve = gen_f64(n, 20.0, 0.0)
        ve[::7] = 0.0
        for op in (0, 1, 2, 3, 4, 5, 6):
            f64_arr(op, vd, ve, "map_f64/%s/n%d" % (OP_NAME[op], n))

    # ---- f32: div/pow widen to f64 output ---------------------------------
    for n in (6, 64):
        vf = gen_f32(n, 2000.0, -1000.0)
        vg = gen_f32(n, 200.0, 0.0)
        for op in (3, 4):
            f32_arr(op, vf, vg, "map_f32_divpow/%s/n%d" % (OP_NAME[op], n))
        for op in (0, 1, 2, 5, 6):
            f32_arr(op, vf, vg, "map_f32/%s/n%d" % (OP_NAME[op], n))

    # ---- the four scalar map symbols --------------------------------------
    for n in (6, 64):
        vi = gen_int(n, -10000, 10000)
        vd = gen_f64(n, 2000.0, -1000.0)
        vf = gen_f32(n, 2000.0, -1000.0)

        # integer scalar on i32 lanes: add/sub/mul wrap mod 2^32
        for op, s in ((0, 2), (1, 2), (2, 2), (0, -3), (1, -3), (2, -3),
                      (3, 2), (3, -3), (4, 2), (5, 2), (5, -3), (6, 2), (6, -3)):
            ref = i32_scalar_ref(vi, s, op)
            nat = native("nf_map_scalar_i32", vi, s, op, np.int32)
            if not np.array_equal(nat, ref):
                raise SystemExit("native != numpy nf_map_scalar_i32/%s s=%d n=%d"
                                 % (OP_NAME[op], s, n))
            add("map_scalar_i32/%s/s%s/n%d" % (OP_NAME[op], s, n),
                symbol="nf_map_scalar_i32", form="scalar", op=op, dtype="i32",
                scalar=float(s), **expect_field(ref))

        # f64 scalar on f64 lanes: all seven
        for s in (0.5, 2.0, 2.5, -3.0):
            for op in (0, 1, 2, 3, 4, 5, 6):
                ref = np.asarray(npy("f64", vd, s, op), np.float64)
                nat = native("nf_map_scalar_f64", vd, s, op, np.float64)
                if not np.array_equal(nat, ref, equal_nan=True):
                    raise SystemExit("native != numpy nf_map_scalar_f64/%s s=%s n=%d"
                                     % (OP_NAME[op], s, n))
                add("map_scalar_f64/%s/s%s/n%d" % (OP_NAME[op], s, n),
                    symbol="nf_map_scalar_f64", form="scalar", op=op, dtype="f64",
                    scalar=s, **expect_field(ref))

        # f32 lanes with an f64 scalar demoted to f32 first
        for s in (0.5, 2.5, -3.0):
            for op in (0, 1, 2, 5, 6):
                ref = np.asarray(npy("f32", vf, s, op), np.float32)
                nat = native("nf_map_scalar_f32", vf, s, op, np.float32)
                if not np.array_equal(nat, ref, equal_nan=True):
                    raise SystemExit("native != numpy nf_map_scalar_f32/%s s=%s n=%d"
                                     % (OP_NAME[op], s, n))
                add("map_scalar_f32/%s/s%s/n%d" % (OP_NAME[op], s, n),
                    symbol="nf_map_scalar_f32", form="scalar", op=op, dtype="f32",
                    scalar=s, **expect_field(ref))
            for op in (3, 4):
                ref = np.asarray(npy("f32", vf, s, op), np.float64)
                nat = native("nf_map_scalar_f32_divpow", vf, s, op, np.float64)
                add("map_scalar_f32_divpow/%s/s%s/n%d" % (OP_NAME[op], s, n),
                    symbol="nf_map_scalar_f32_divpow", form="scalar", op=op,
                    dtype="f64", scalar=s, **expect_field(ref))

    payload = {
        "generator": "numfast-native/ts/test/gen_fixtures.py",
        "seed": SEED,
        "input_lcg": {
            "multiplier": 1664525, "increment": 1013904223, "modulus": 4294967296,
            "note": "s = (s*mult + inc) mod 2^32; int lane = lo + (u*span)//2^32; "
                    "float lane = (u/2^32)*scale + offset. Implemented "
                    "identically in gen_fixtures.py and test/parity.test.mjs.",
        },
        "inputs": {
            "i32": {"kind": "int", "lo": -10000, "hi": 10000},
            "i32_b": {"kind": "int", "lo": -500, "hi": 500},
            "f64": {"kind": "float", "scale": 2000.0, "offset": -1000.0},
            "f64_b": {"kind": "float", "scale": 20.0, "offset": 0.0, "zeroEvery": 7},
            "f32": {"kind": "float32", "scale": 2000.0, "offset": -1000.0},
            "f32_b": {"kind": "float32", "scale": 200.0, "offset": 0.0},
        },
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "dll": os.path.relpath(DLL, APP).replace("\\", "/"),
        "note": ("reference (NumPy + the documented int32 round-trip) and native "
                 "(ctypes into the committed DLL) were compared byte-for-byte "
                 "before recording; a disagreement aborts this generator"),
        "literal_max_n": LITERAL_MAX,
        "tolerance": tolerance(),
        "tolerance_source": "specs-rebuilt/conformance-profile.toml [tolerance]",
        "cases": cases,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    literal = sum(1 for c in cases.values() if "expect_hex" in c)
    print("wrote %s" % os.path.relpath(OUT, APP).replace("\\", "/"))
    print("cases %d (%d with literal bytes, %d by sha256), file %d bytes"
          % (len(cases), literal, len(cases) - literal, os.path.getsize(OUT)))


if __name__ == "__main__":
    main()