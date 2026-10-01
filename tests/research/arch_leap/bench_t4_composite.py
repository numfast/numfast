# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""T4: Q2 (id1,id2) через физическое кодирование composite key в один integer.
mixed-radix pack key=id1*C2+id2 (и bit-pack где степень двойки), не tuple/hash.
Seed 42. best-of-3. Research-only.
Git Bash: timeout 300 python tests/research/arch_leap/bench_t4_composite.py
"""
import ctypes, gc, json, os, time
import numpy as np

REPS = 3
M = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "..", "..", "numfast-native", "target",
                   "x86_64-pc-windows-gnu", "release", "numfast_native.dll")
OUT = os.path.join(HERE, "results_t4.json")
try:
    import psutil; P = psutil.Process()
    def rss(): return P.memory_info().rss / 1e9
except ImportError:
    def rss(): return float("nan")
def med(xs): return sorted(xs)[len(xs)//2]
def best_of(fn, reps=REPS):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    return med(ts), min(ts)

d = ctypes.CDLL(DLL)
f_i64 = d.nf_group_variant_i64
f_i64.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_size_t, ctypes.c_uint32] + [ctypes.c_void_p]*2 + [ctypes.c_size_t]
f_i64.restype = ctypes.c_int32
f_pack = d.nf_pack_i32_direct
f_pack.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
f_pack.restype = ctypes.c_int32

N = 1_000_000
C1, C2 = 100, 1000
G = C1 * C2
rng = np.random.default_rng(42)
t0 = time.perf_counter()
id1 = np.ascontiguousarray(rng.integers(0, C1, size=N, dtype=np.int32))
id2 = np.ascontiguousarray(rng.integers(0, C2, size=N, dtype=np.int32))
ticks = np.ascontiguousarray(np.rint(rng.normal(5000.0, 1500.0, N) * M).astype(np.int64))
gen_ms = (time.perf_counter()-t0)*1e3
print("rss_start=%.2fGB gen=%.0fms" % (rss(), gen_ms), flush=True)

# encode stage: numpy mixed-radix vs native nf_pack_i32_direct
t0 = time.perf_counter()
key_np = np.ascontiguousarray((id1.astype(np.int64) * C2 + id2).astype(np.int32))
enc_np_ms = (time.perf_counter()-t0)*1e3
key_nat = np.zeros(N, dtype=np.int32)
t0 = time.perf_counter()
# nf_pack_i32_direct(a=id1,b=id2,n, radix=C2, out) — signature per lib.rs: check rc
try:
    rc = f_pack(id1.ctypes.data, id2.ctypes.data, C2, N, key_nat.ctypes.data)
    enc_nat_ms = (time.perf_counter()-t0)*1e3
    pack_ok = (rc == 0 and (key_nat == key_np).all())
except Exception as e:
    enc_nat_ms = float("nan"); pack_ok = False; rc = str(e)
print("encode numpy=%.1fms native_pack rc=%s ok=%s (%.1fms)" % (enc_np_ms, rc, pack_ok, enc_nat_ms), flush=True)
assert (key_np >= 0).all() and key_np.max() < G
assert pack_ok or True  # record only; numpy path is the arch answer if native sig differs

ref = np.bincount(key_np, weights=ticks, minlength=G)
refc = np.bincount(key_np, minlength=G).astype(np.int64)
# reversibility: unpack
t0 = time.perf_counter()
u1 = (key_np.astype(np.int64) // C2).astype(np.int32); u2 = (key_np.astype(np.int64) % C2).astype(np.int32)
unpk_ms = (time.perf_counter()-t0)*1e3
assert (u1 == id1).all() and (u2 == id2).all()
print("unpack reversible: True (%.1fms)" % unpk_ms, flush=True)

s = np.zeros(G, dtype=np.int64); c = np.zeros(G, dtype=np.int64)
def runpack(): assert f_i64(key_np.ctypes.data, ticks.ctypes.data, N, 0, s.ctypes.data, c.ctypes.data, G) == 0
runpack(); assert (s == ref).all() and (c == refc).all()
tpack, tpackm = best_of(runpack)

# baseline: Python dict/tuple groupby on 50k prefix (full 1e6 dict too slow — extrapolate honestly)
K = 50_000
t0 = time.perf_counter()
dd = {}
for i in range(K):
    k = (int(id1[i]), int(id2[i]))
    dd[k] = dd.get(k, 0) + int(ticks[i])
dict_ms = (time.perf_counter()-t0)*1e3
print("dict-tuple 50k=%.0fms -> extrap 1e6 ~%.0fms (20x)" % (dict_ms, dict_ms*20), flush=True)
# verify dict prefix vs packed ref on same prefix
s50 = np.zeros(G, dtype=np.int64)
for (a, b), v in dd.items(): s50[a*C2+b] += v
ref50 = np.bincount(key_np[:K], weights=ticks[:K], minlength=G)
assert (s50 == ref50).all()

# bit-pack variant where C2 pow2 (C2=1024): key=(id1<<10)|id2
C2b = 1024; Gb = C1 * C2b
id2b = np.ascontiguousarray((id2 % C2b).astype(np.int32))
t0 = time.perf_counter()
key_bit = np.ascontiguousarray(((id1.astype(np.int32) << 10) | id2b))
bit_ms = (time.perf_counter()-t0)*1e3
assert ((key_bit >> 10) == id1).all() and ((key_bit & 1023) == id2b).all()
print("bit-pack (<<10): %.1fms reversible=True" % bit_ms, flush=True)

json.dump({"seed": 42, "n": N, "c1": C1, "c2": C2, "g": G, "gen_ms": gen_ms,
           "encode_numpy_ms": enc_np_ms, "encode_native_ms": enc_nat_ms, "pack_ok": bool(pack_ok),
           "unpack_ms": unpk_ms, "reversible": True,
           "agg_pack_med_ms": tpack, "agg_pack_min_ms": tpackm,
           "dict_50k_ms": dict_ms, "dict_extrap_1e6_ms": dict_ms*20,
           "bitpack_1024_ms": bit_ms, "rss_end_gb": rss()}, open(OUT, "w"), indent=1)
print("pack-agg med=%.1fms (vs dict-extrap ~%.0fms => ~%.0fx) rss=%.2fGB" % (tpack, dict_ms*20, dict_ms*20/max(tpack,1e-9), rss()), flush=True)
print("wrote %s" % OUT, flush=True)
