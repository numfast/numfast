# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Microstand block generators. Fixed seed 42. Keys int32, values int32.

LEVEL 0: raw sparse blocks 32K/64K/128K (M >> n or M ~ n -> high uniq).
LEVEL 1-like: blocks with ~90-100% key uniqueness (post-partial state shape).
Variants: uniform / skewed / clustered (locally repeating keys at globally
high cardinality) / sorted / random.
"""

import numpy as np

SEED = 42

# (name, n_rows, M_cardinality, pattern, level)
BLOCK_SUITE = [
    ("L0_32K_sparse_uniform",   32_768, 1_000_000, "uniform",   0),
    ("L0_64K_sparse_uniform",   65_536, 1_000_000, "uniform",   0),
    ("L0_128K_sparse_uniform", 131_072, 10_000_000, "uniform",  0),
    ("L0_64K_skewed",           65_536, 100_000,   "skewed",    0),
    ("L0_64K_clustered",        65_536, 1_000_000, "clustered", 0),
    ("L0_64K_sorted",           65_536, 100_000,   "sorted",    0),
    ("L0_64K_random",           65_536, 1_000_000, "random",    0),
    ("L0_64K_dense_M100",       65_536, 100,       "uniform",   0),
    ("L1_64K_uniq95_uniform",   65_536, 70_000,    "unique95",  1),
    ("L1_64K_uniq100_uniform",  65_536, 5_000_000, "unique100", 1),
    ("L1_64K_uniq95_clustered", 65_536, 70_000,    "clustered", 1),
    ("L1_128K_uniq90_sorted",  131_072, 150_000,   "sorted",    1),
]

# M sweep x pattern (fixed 128K probe blocks for BEST-per-regime).
M_SWEEP = [100, 10_000, 100_000, 1_000_000, 10_000_000]
PATTERNS = ["uniform", "skewed", "clustered", "sorted", "random"]


def gen_block(n, M, pattern, rng):
    if pattern == "uniform":
        keys = rng.integers(0, M, size=n, dtype=np.int64)
    elif pattern == "skewed":
        # power-law-ish: rank distribution via squared uniform -> few hot keys
        u = rng.random(n)
        keys = (np.floor(M * u * u)).astype(np.int64) % M
    elif pattern == "clustered":
        # locally repeating keys, globally high cardinality: each 1K chunk
        # draws from a small local pool inside the global space
        keys = np.empty(n, dtype=np.int64)
        chunk = 1024
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            base = int(rng.integers(0, M))
            pool = (base + rng.integers(0, 64, size=64)) % M
            keys[s:e] = pool[rng.integers(0, 64, size=e - s)]
    elif pattern == "sorted":
        keys = np.sort(rng.integers(0, M, size=n, dtype=np.int64))
    elif pattern == "random":
        keys = (rng.integers(-2**31, 2**31, size=n, dtype=np.int64) % M + M) % M
    elif pattern == "unique95":
        keys = rng.choice(M, size=n, replace=False) if M >= n else \
            rng.integers(0, M, size=n)
        # force ~95% uniqueness: duplicate 5% onto first keys
        keys = np.asarray(keys, dtype=np.int64)
        nd = n // 20
        keys[-nd:] = keys[:nd]
    elif pattern == "unique100":
        keys = (np.arange(n, dtype=np.int64) * 7919 + 13) % M
    else:
        raise ValueError(f"unknown pattern {pattern}")
    keys = keys.astype(np.int32, copy=False)
    vals = rng.integers(-1000, 1000, size=n).astype(np.int32)
    return keys, vals


def make_suite(seed=SEED):
    out = []
    for i, (name, n, M, pat, lvl) in enumerate(BLOCK_SUITE):
        rng = np.random.default_rng(seed + i)
        k, v = gen_block(n, M, pat, rng)
        out.append({"name": name, "n": n, "M": M, "pattern": pat,
                    "level": lvl, "keys": k, "vals": v})
    return out
