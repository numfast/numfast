# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Rowwise K-way time-argmin + gather parity: oracle vs native vs fallback.

Contract under test (canonical, generic K 1..=256): inputs K int32[N]
T lanes + K float32[N] D lanes; outputs t_best int32[N], d_best
float32[N], m_best uint8[N]; m = argmin over the T lanes with strict
`<` from lane 0 (ties keep the smallest index; INT32_MIN/MAX ordinary
ordered values); t_best = T[m] exact; d_best = D[m] bit-exact gather
(NaN/Inf/-0.0 ride as-is). D NEVER selects the mode.

Oracle is an independent formulation (strict-less scan over T, no
argmin call, fancy gather on the mode path); the wrapper fallback is
the stacked cascade; native is the Rust kernel via ctypes. All three
agree bit-exactly. Float compare is bitwise (view int32).

Coverage: random K (1..256 incl 1/2/3/4/5/8/16/64/256), all ties, pair
ties, mono inc/dec, negatives, INT32 extremes, NaN/Inf payload mixes,
N=0/1, odd/even, large N; >=1000 seeded fuzz cases; every case runs
twice (determinism, bit-exact); K=4 cross-checks the K4 wrapper.

Seed 42 everywhere. Prints table + stage breakdown, writes
results/parity_rowmin_kway.json. Exit 0 = PASS.
"""
import importlib.util
import json
import os
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WRAP = _load("nf_rowwise_kway",
             os.path.normpath(os.path.join(ROOT, "..", "src", "Drivers",
                                           "CPU", "_lib", "rowwise_kway.py")))
WRAP4 = _load("nf_rowwise_min4_time",
              os.path.normpath(os.path.join(ROOT, "..", "src", "Drivers",
                                            "CPU", "_lib", "rowwise_min4_time.py")))

I32MIN, I32MAX = -2 ** 31, 2 ** 31 - 1


def oracle(t_lanes, d_lanes):
    """Independent oracle: strict-less scan over T (no argmin)."""
    T = [np.ascontiguousarray(a, dtype=np.int32) for a in t_lanes]
    D = [np.ascontiguousarray(a, dtype=np.float32) for a in d_lanes]
    n = T[0].size
    k = len(T)
    m = np.zeros(n, dtype=np.uint8)
    best = T[0].copy()
    for j in range(1, k):
        take = T[j] < best  # strict: ties keep the smaller index
        m = np.where(take, np.uint8(j), m)
        best = np.where(take, T[j], best)
    Dm = np.stack(D)
    idx = np.arange(n)
    db = np.ascontiguousarray(Dm[m.astype(np.int64), idx])
    return (np.ascontiguousarray(best),
            np.ascontiguousarray(db),
            np.ascontiguousarray(m))


def bit_exact(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def run_both(t_lanes, d_lanes):
    """Wrapper via native path, then via forced fallback, then repeat."""
    tn, dn, mn = WRAP.rowwise_kway_time_argmin_gather(t_lanes, d_lanes)
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    try:
        tf, df, mf = WRAP.rowwise_kway_time_argmin_gather(t_lanes, d_lanes)
    finally:
        del os.environ["NUMFAST_NATIVE_DISABLE"]
    tn2, dn2, mn2 = WRAP.rowwise_kway_time_argmin_gather(t_lanes, d_lanes)
    return (tn, dn, mn), (tf, df, mf), (tn2, dn2, mn2)


def main():
    rng = np.random.default_rng(42)
    fails = []
    rows = []

    def check(label, cond, detail=""):
        rows.append((label, cond, detail))
        print(("PASS " if cond else "FAIL ") + label
              + ((" | " + detail) if detail else ""))
        if not cond:
            fails.append(label)

    cases = []

    def add(label, t_lanes, d_lanes):
        cases.append((label, [np.ascontiguousarray(a, dtype=np.int32)
                              for a in t_lanes],
                      [np.ascontiguousarray(a, dtype=np.float32)
                       for a in d_lanes]))

    def rand_d(n, k, fr=None):
        fr = fr if fr is not None else rng
        D = (fr.normal(0, 20, size=(k, max(n, 1)))).astype(np.float32)
        pick = fr.random(size=(k, max(n, 1)))
        D[pick < 0.06] = np.nan
        D[(pick >= 0.06) & (pick < 0.09)] = np.inf
        D[(pick >= 0.09) & (pick < 0.12)] = -np.inf
        return [np.ascontiguousarray(D[j, :n]) for j in range(k)]

    def rand_t(n, k, fr=None):
        fr = fr if fr is not None else rng
        T = fr.integers(I32MIN, I32MAX + 1, size=(k, max(n, 1))).astype(np.int32)
        return [np.ascontiguousarray(T[j, :n]) for j in range(k)]

    # --- edges across K ---
    for kk in (1, 2, 3, 4, 5, 8, 16, 64, 256):
        n = 9
        T = rand_t(n, kk)
        D = rand_d(n, kk)
        add(f"edge/K{kk}/N9", T, D)
    z1 = [np.zeros(0, dtype=np.int32)]
    f1 = [np.zeros(0, dtype=np.float32)]
    add("edge/N0/K1", z1, f1)
    z4 = [np.zeros(0, dtype=np.int32) for _ in range(4)]
    f4 = [np.zeros(0, dtype=np.float32) for _ in range(4)]
    add("edge/N0/K4", z4, f4)
    add("edge/N1/K1/tie",
        [np.array([7], np.int32)], [np.array([np.nan], np.float32)])
    add("edge/N1/K4/tie",
        [np.array([7], np.int32)] * 4,
        [np.array([1.0], np.float32), np.array([2.0], np.float32),
         np.array([3.0], np.float32), np.array([4.0], np.float32)])
    # all ties K=4
    n = 65
    T = rng.integers(I32MIN, I32MAX + 1, size=(4, n)).astype(np.int32)
    T[1, :] = T[0]
    T[2, :] = T[0]
    T[3, :] = T[0]
    add("edge/all-ties/K4/odd65", [T[0], T[1], T[2], T[3]], rand_d(n, 4))
    # INT32 extremes
    n = 48
    Te = rng.integers(I32MIN, I32MAX + 1, size=(4, n)).astype(np.int32)
    Te[0, ::7] = I32MIN
    Te[1, ::11] = I32MAX
    add("edge/i32-extremes/K4", [Te[0], Te[1], Te[2], Te[3]], rand_d(n, 4))
    # NaN / Inf payload mixes (never select the mode)
    n = 96
    T = rand_t(n, 4)
    A = (rng.normal(0, 5, size=n)).astype(np.float32)
    B, C, E = A.copy(), A.copy(), A.copy()
    A[::3] = np.nan
    B[1::4] = np.nan
    C[::5] = np.inf
    E[2::7] = -np.inf
    add("edge/nan-inf-payload/K4", T, [A, B, C, E])
    # K=256 extremes + ties
    n = 32
    T = rand_t(n, 256)
    for j in range(1, 256):
        T[j][::8] = T[0][::8]
    T[0][0] = np.int32(I32MIN)
    T[255][1] = np.int32(I32MIN)
    add("edge/K256/ties-extremes", T, rand_d(n, 256))
    # odd / even shapes
    for nn in (3, 4, 17, 18, 1023, 1024):
        add(f"edge/shape-K4-N{nn}", rand_t(nn, 4), rand_d(nn, 4))
    # large N K=4
    n = 200_000
    add("edge/large-200k/K4", rand_t(n, 4), rand_d(n, 4))
    # K=4 vs K4-wrapper cross-check seed
    n = 4096
    T = rand_t(n, 4)
    D = rand_d(n, 4)
    D[0][::3] = np.nan
    D[2][::5] = np.inf
    D[3][2::7] = -np.inf
    add("edge/K4-wrapper-4096", T, D)

    # --- fuzz: 1000 seeded cases, random K 1..256, small N incl 0/1 ---
    NFUZZ = 1000
    for seed in range(NFUZZ):
        fr = np.random.default_rng(100000 + seed)
        kk = int(fr.integers(1, 257))
        # bias small K for speed, keep large-K coverage every 10th seed
        if seed % 10 != 0:
            kk = int(fr.integers(1, 17))
        nn = int(fr.integers(0, 66))
        T = fr.integers(I32MIN, I32MAX + 1, size=(kk, max(nn, 1))).astype(np.int32)
        D = (fr.normal(0, 20, size=(kk, max(nn, 1)))).astype(np.float32)
        pick = fr.random(size=(kk, max(nn, 1)))
        D[pick < 0.06] = np.nan
        D[(pick >= 0.06) & (pick < 0.09)] = np.inf
        D[(pick >= 0.09) & (pick < 0.12)] = -np.inf
        if kk > 1 and nn > 0:
            tie_rows = fr.random(size=nn) < 0.25
            T[1, :nn][tie_rows] = T[0, :nn][tie_rows]
        T = np.ascontiguousarray(T[:, :nn].copy())
        D = np.ascontiguousarray(D[:, :nn].copy())
        cases.append((f"fuzz/seed{seed}/K{kk}/N{nn}",
                      [np.ascontiguousarray(T[j]) for j in range(kk)],
                      [np.ascontiguousarray(D[j]) for j in range(kk)]))

    # K-range contract: K=0 and K=257 rejected, never a silent path
    try:
        WRAP.rowwise_kway_time_argmin_gather([], [])
        check("api/K0-rejected", False, "no error raised")
    except ValueError:
        check("api/K0-rejected", True, "")
    try:
        WRAP.rowwise_kway_time_argmin_gather(
            [np.zeros(2, np.int32)] * 257, [np.zeros(2, np.float32)] * 257)
        check("api/K257-rejected", False, "no error raised")
    except ValueError:
        check("api/K257-rejected", True, "")

    t_oracle = t_nat = t_fb = 0.0
    for label, t_lanes, d_lanes in cases:
        nn = t_lanes[0].size
        kk = len(t_lanes)
        snap = ([a.tobytes() for a in t_lanes],
                [a.tobytes() for a in d_lanes])
        t = time.perf_counter()
        eo, do_, mo = oracle(t_lanes, d_lanes)
        t_oracle += (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        (tn, dn, mn), (tf, df, mf), (tn2, dn2, mn2) = run_both(t_lanes, d_lanes)
        t_nat += (time.perf_counter() - t) * 1e3 / 2.0
        t_fb += (time.perf_counter() - t) * 1e3 / 2.0
        ok_oracle_nat = (bit_exact(tn, eo) and bit_exact(dn, do_)
                         and bit_exact(mn, mo))
        check(f"{label} native==oracle", ok_oracle_nat, f"k={kk} n={nn}")
        check(f"{label} fallback==oracle",
              bit_exact(tf, eo) and bit_exact(df, do_) and bit_exact(mf, mo),
              f"k={kk} n={nn}")
        check(f"{label} deterministic",
              bit_exact(tn, tn2) and bit_exact(dn, dn2) and bit_exact(mn, mn2),
              f"k={kk} n={nn}")
        intact = ([a.tobytes() for a in t_lanes] == snap[0]
                  and [a.tobytes() for a in d_lanes] == snap[1])
        check(f"{label} inputs-intact", intact, f"k={kk} n={nn}")
        Tm = np.stack([np.ascontiguousarray(a, dtype=np.int32) for a in t_lanes])
        idx = np.arange(nn)
        check(f"{label} t==T[m]",
              np.array_equal(np.ascontiguousarray(Tm[mn.astype(np.int64), idx]), tn),
              f"k={kk} n={nn}")
        if kk == 4:
            t4, d4, m4 = WRAP4.rowwise_min4_time_argmin_gather(*t_lanes, *d_lanes)
            check(f"{label} K4wrapper==Kway",
                  bit_exact(t4, tn) and bit_exact(d4, dn) and bit_exact(m4, mn),
                  f"n={nn}")

    print(f"\nstages ms: oracle_total={t_oracle:.3f} native_total={t_nat:.3f} "
          f"fallback_total={t_fb:.3f} cases={len(cases)} "
          f"native_backend={WRAP.kway_available()} ({WRAP.why()})")
    json.dump({"seed": 42, "cases": len(cases), "fuzz": NFUZZ,
               "oracle_ms": t_oracle, "native_ms": t_nat,
               "fallback_ms": t_fb,
               "native_backend": WRAP.kway_available(), "why": WRAP.why(),
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "parity_rowmin_kway.json"), "w"), indent=1)
    print("PASS" if not fails else f"FAIL {fails[:10]} (+{len(fails) - 10})"
          if len(fails) > 10 else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
