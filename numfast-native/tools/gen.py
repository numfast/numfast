# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate shared synthetic vectors (seed 42) for parity + bench.

Writes raw little-endian buffers (no headers) + meta.json:
  keys_*.i32 (int32 dense codes), values_*.f64 (float64)
- small: N=10_000, G=128   (golden parity)
- big:   N=1_000_000, G=256 (benchmark)
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectors")
os.makedirs(OUT, exist_ok=True)

CASES = {"small": (10_000, 128), "big": (1_000_000, 256)}

meta = {}
for name, (n, g) in CASES.items():
    rng = np.random.default_rng(42)
    keys = rng.integers(0, g, size=n).astype(np.int32)
    vals = rng.normal(loc=0.0, scale=100.0, size=n).astype(np.float64)
    assert keys.flags["C_CONTIGUOUS"] and vals.flags["C_CONTIGUOUS"]
    kp = os.path.join(OUT, "keys_%s.i32" % name)
    vp = os.path.join(OUT, "values_%s.f64" % name)
    keys.tofile(kp)
    vals.tofile(vp)
    meta[name] = {"n": n, "g": g, "keys": kp, "values": vp}
    print("%s: n=%d g=%d keys=%dB values=%dB" % (name, n, g, keys.nbytes, vals.nbytes))

with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("wrote", OUT)
