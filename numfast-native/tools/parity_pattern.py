# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden parity: pattern_encode — Python-loop ref vs Rust-native vs WASM.

Corpus (seed 42): 10k H2O-like "id%03d"/"id%010d" rows mixed with edge cases
(wrong prefix, empty body, lone '-', non-digits, negative, unicode) +
a separate overflow case (rc=-2 + err_row). Pass: codes/valid/width exact.
Writes results/parity_pattern.json.
"""
import ctypes
import json
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
NODE = r"C:\Program Files\nodejs\node.exe"
WASM = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm")
DLL = os.path.join(ROOT, "target", "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
MJS = os.path.join(ROOT, "tools", "wasm_pattern.mjs")

PREFIX = "id"


def ref_encode(strs, prefix):
    codes = np.zeros(len(strs), dtype=np.int32)
    valid = np.zeros(len(strs), dtype=np.uint8)
    width = -1
    for i, s in enumerate(strs):
        if not s.startswith(prefix):
            continue
        body = s[len(prefix):]
        neg = body.startswith("-")
        core = body[1:] if neg else body
        if not core or any(not ("0" <= ch <= "9") for ch in core):
            continue
        mag = int(core)
        if mag > (2**31 if neg else 2**31 - 1):
            raise OverflowError(i)
        codes[i] = -mag if neg else mag
        valid[i] = 1
        width = max(width, len(core))
    return codes, valid, width


def to_bufs(strs):
    enc = [s.encode("utf-8") for s in strs]
    offs = np.zeros(len(enc) + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(b) for b in enc])
    return b"".join(enc), offs


rng = np.random.default_rng(42)
strs = []
for _ in range(10_000):
    r = rng.random()
    if r < 0.70:
        strs.append("id%03d" % int(rng.integers(0, 1000)))
    elif r < 0.80:
        strs.append("id%010d" % int(rng.integers(0, 1000)))
    elif r < 0.84:
        strs.append("id-%d" % int(rng.integers(0, 100)))
    elif r < 0.88:
        strs.append(rng.choice(["xx1", "IDX5", "", "id", "id-", "id12a", "id 12", "id_12", "i", "id+3"]))
    elif r < 0.92:
        strs.append("id%d" % int(rng.integers(0, 10**9)))
    elif r < 0.96:
        strs.append("id" + "9" * int(rng.integers(11, 19)))  # long but < 2^31? may overflow
    else:
        strs.append(rng.choice(["id１２３", "id²", "ïd5"]))
# drop overflow rows from ok-case (separate case below)
ok_strs = []
for s in strs:
    try:
        ref_encode([s], PREFIX)
        ok_strs.append(s)
    except OverflowError:
        pass
ref_codes, ref_valid, ref_width = ref_encode(ok_strs, PREFIX)
n = len(ok_strs)
data, offs = to_bufs(ok_strs)
pfx = PREFIX.encode()

dll = ctypes.CDLL(DLL)
fn = dll.nf_pattern_encode
fn.argtypes = ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
               + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p] * 1
               + [ctypes.c_void_p] * 3)
# (data,total,offs,n,prefix,prefix_len,codes,valid,width_out,err_row_out)
fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
               ctypes.c_void_p, ctypes.c_void_p]
fn.restype = ctypes.c_int32

nat_c = np.zeros(n, dtype=np.int32)
nat_v = np.zeros(n, dtype=np.uint8)
nat_w = np.zeros(1, dtype=np.int32)
nat_e = np.zeros(1, dtype=np.int32)
rc = fn(np.frombuffer(data, dtype=np.uint8).ctypes.data, len(data),
        offs.ctypes.data, n, np.frombuffer(pfx, dtype=np.uint8).ctypes.data, len(pfx),
        nat_c.ctypes.data, nat_v.ctypes.data, nat_w.ctypes.data, nat_e.ctypes.data)
assert rc == 0, rc
assert fn(0, len(data), offs.ctypes.data, n,
          np.frombuffer(pfx, dtype=np.uint8).ctypes.data, len(pfx),
          nat_c.ctypes.data, nat_v.ctypes.data, nat_w.ctypes.data, nat_e.ctypes.data) == -1

# overflow case
ov = ["id1", "id" + "9" * 19, "id2", "id-2147483649"]
odata, ooffs = to_bufs(ov)
oc = np.zeros(4, dtype=np.int32)
ovd = np.zeros(4, dtype=np.uint8)
ow = np.zeros(1, dtype=np.int32)
oe = np.zeros(1, dtype=np.int32)
orc = fn(np.frombuffer(odata, dtype=np.uint8).ctypes.data, len(odata),
         ooffs.ctypes.data, 4, np.frombuffer(pfx, dtype=np.uint8).ctypes.data, len(pfx),
         oc.ctypes.data, ovd.ctypes.data, ow.ctypes.data, oe.ctypes.data)
assert orc == -2 and int(oe[0]) == 1, (orc, oe)

# malformed offsets
bad_offs = offs.copy()
bad_offs[5] = bad_offs[4] - 1
bc = np.zeros(n, dtype=np.int32)
bv = np.zeros(n, dtype=np.uint8)
bw = np.zeros(1, dtype=np.int32)
be = np.zeros(1, dtype=np.int32)
assert fn(np.frombuffer(data, dtype=np.uint8).ctypes.data, len(data),
          bad_offs.ctypes.data, n, np.frombuffer(pfx, dtype=np.uint8).ctypes.data, len(pfx),
          bc.ctypes.data, bv.ctypes.data, bw.ctypes.data, be.ctypes.data) == -3

with tempfile.TemporaryDirectory() as td:
    dp = os.path.join(td, "data.bin")
    op = os.path.join(td, "offs.i32")
    open(dp, "wb").write(data)
    offs.tofile(op)
    node_out = subprocess.run(
        [NODE, MJS, WASM, dp, op, PREFIX, str(n), "parity", RES],
        capture_output=True, text=True, check=True,
    )
nj = json.loads(node_out.stdout.strip())
w_c = np.fromfile(os.path.join(RES, "pat_codes.i32"), dtype=np.int32)
w_v = np.fromfile(os.path.join(RES, "pat_valid.u8"), dtype=np.uint8)

ok = True
for name, c, v, w in [("rust-native", nat_c, nat_v, int(nat_w[0])),
                       ("wasm-node", w_c, w_v, nj["width"])]:
    dc = int(np.sum(c != ref_codes))
    dv = int(np.sum(v != ref_valid))
    dw = (w != ref_width)
    bit = dc == 0 and dv == 0 and not dw
    ok &= bit
    print("%-12s code_bad=%d valid_bad=%d width=%d(ref %d) %s"
          % (name, dc, dv, w, ref_width, "EXACT" if bit else "DIFF"))
print("wasm-node call ms (1 run, cold): %.3f" % nj["ms"])
print("err contract: null=-1 ok, overflow=-2(err_row=1) ok, malformed=-3 ok")
print("corpus: n=%d valid_frac=%.3f width=%d" % (n, float(ref_valid.mean()), ref_width))
json.dump({"n": n, "pass": bool(ok), "width": ref_width,
           "wasm_cold_ms": nj["ms"]}, open(os.path.join(RES, "parity_pattern.json"), "w"), indent=1)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
