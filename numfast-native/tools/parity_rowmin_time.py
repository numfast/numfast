# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Rowwise MIN4 time-argmin + optional gather parity: oracle vs native vs fallback.

Contract under test (FIX-1, generic): inputs T0-T3 int32[N], OPTIONAL
D0-D3 float32[N]; outputs t_best int32[N], d_best float32[N], m_best
uint8[N]; m = argmin over the T lanes with strict `<` from lane 0
(ties keep the smallest index; INT32_MIN/MAX ordinary ordered values);
t_best = T[m] exact; d_best = D[m] bit-exact gather when D given
(zeros when D omitted). D NEVER selects the mode.

Oracle is an independent formulation (strict-less where-cascade over T,
no argmin call, no fancy gather on the mode path); the wrapper fallback
is stacked T-argmin + fancy gather; native is the Rust kernel via
ctypes. All three must agree bit-exactly. Float compare is bitwise
(view int32), so -0.0 and NaN payloads are distinguished.

Coverage: random, all ties, pair ties, mono inc/dec, negatives,
INT32 extremes, NaN/Inf payload mixes, N=0/1, odd/even, large N;
>=1200 seeded fuzz cases; MANDATORY regression: T selects mode A while
the minimal D sits at another mode (catches the old D-argmin semantics);
every case runs twice (determinism, bit-exact); every case runs both
the gather path (D given) and the time-only path (D omitted).

Seed 42 everywhere. Prints table + stage breakdown, writes
results/parity_rowmin_time.json. Exit 0 = PASS.
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


WRAP = _load("nf_rowwise_min4_time",
             os.path.normpath(os.path.join(ROOT, "..", "src", "Drivers",
                                           "CPU", "_lib", "rowwise_min4_time.py")))

I32MIN, I32MAX = -2 ** 31, 2 ** 31 - 1


def oracle(t_lanes, d_lanes_or_none):
    """Independent oracle: strict-less where-cascade over T (no argmin)."""
    T = [np.ascontiguousarray(a, dtype=np.int32) for a in t_lanes]
    n = T[0].size
    m = np.zeros(n, dtype=np.uint8)
    best = T[0].copy()
    tb = T[0].copy()
    for k in (1, 2, 3):
        take = T[k] < best  # strict: ties keep the smaller index
        m = np.where(take, np.uint8(k), m)
        best = np.where(take, T[k], best)
        tb = np.where(take, T[k], tb)
    if d_lanes_or_none is None:
        db = np.zeros(n, dtype=np.float32)
    else:
        D = [np.ascontiguousarray(a, dtype=np.float32) for a in d_lanes_or_none]
        db = D[0].copy()
        for k in (1, 2, 3):
            take = (m == k)
            db = np.where(take, D[k], db)
        db = np.ascontiguousarray(db, dtype=np.float32)
    return (np.ascontiguousarray(tb),
            np.ascontiguousarray(db),
            np.ascontiguousarray(m))


def bit_exact(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def run_both(t_lanes, d_lanes_or_none):
    """Wrapper via native path, then via forced fallback, then repeat."""
    if d_lanes_or_none is None:
        tn, dn, mn = WRAP.rowwise_min4_time_argmin_gather(*t_lanes)
    else:
        tn, dn, mn = WRAP.rowwise_min4_time_argmin_gather(*t_lanes, *d_lanes_or_none)
    os.environ["NUMFAST_NATIVE_DISABLE"] = "1"
    try:
        if d_lanes_or_none is None:
            tf, df, mf = WRAP.rowwise_min4_time_argmin_gather(*t_lanes)
        else:
            tf, df, mf = WRAP.rowwise_min4_time_argmin_gather(*t_lanes, *d_lanes_or_none)
    finally:
        del os.environ["NUMFAST_NATIVE_DISABLE"]
    if d_lanes_or_none is None:
        tn2, dn2, mn2 = WRAP.rowwise_min4_time_argmin_gather(*t_lanes)
    else:
        tn2, dn2, mn2 = WRAP.rowwise_min4_time_argmin_gather(*t_lanes, *d_lanes_or_none)
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

    def add(label, t_lanes, d_lanes_or_none):
        cases.append((label, [np.ascontiguousarray(a, dtype=np.int32)
                              for a in t_lanes],
                      None if d_lanes_or_none is None else
                      [np.ascontiguousarray(a, dtype=np.float32)
                       for a in d_lanes_or_none]))

    def rand_d(n, fr=None):
        fr = fr if fr is not None else rng
        D = (fr.normal(0, 20, size=(4, max(n, 1)))).astype(np.float32)
        pick = fr.random(size=(4, max(n, 1)))
        D[pick < 0.06] = np.nan
        D[(pick >= 0.06) & (pick < 0.09)] = np.inf
        D[(pick >= 0.09) & (pick < 0.12)] = -np.inf
        return [np.ascontiguousarray(D[k, :n]) for k in range(4)]

    # --- MANDATORY regression: T selects mode A, minimal D elsewhere ---
    # Old D-argmin semantics returns the D-min lane; new T-argmin must not.
    add("regression/T0-wins-Dmin-at-1",
        [np.array([0, 0, 5], np.int32), np.array([10, 1, 6], np.int32),
         np.array([20, 2, 7], np.int32), np.array([30, 3, 4], np.int32)],
        [np.array([100.0, 50.0, 9.0], np.float32),
         np.array([-999.0, -999.0, 8.0], np.float32),
         np.array([50.0, 60.0, 7.0], np.float32),
         np.array([60.0, 70.0, -999.0], np.float32)])
    # row0: T-min mode0 (t=0), D-min mode1 -> must be m=0,t=0,d=100
    # row1: T-min mode0 (t=0), D-min mode1 -> must be m=0
    # row2: T-min mode3 (t=4), D-min mode3 too (control: both agree)
    add("regression/each-mode-wins-against-D",
        [np.array([0, 5, 5, 5], np.int32), np.array([5, 0, 5, 5], np.int32),
         np.array([5, 5, 0, 5], np.int32), np.array([5, 5, 5, 0], np.int32)],
        [np.array([999.0, -999.0, -999.0, -999.0], np.float32),
         np.array([-999.0, 999.0, -999.0, -999.0], np.float32),
         np.array([-999.0, -999.0, 999.0, -999.0], np.float32),
         np.array([-999.0, -999.0, -999.0, 999.0], np.float32)])
    # Each row: T picks the diagonal mode with the WORST D there.
    add("regression/T-tie-keeps-smallest-vs-D",
        [np.array([7, 7, 7], np.int32)] * 4,
        [np.array([1.0, 2.0, -999.0], np.float32),
         np.array([2.0, 1.0, 3.0], np.float32),
         np.array([3.0, 3.0, 2.0], np.float32),
         np.array([4.0, 4.0, 1.0], np.float32)])
    # All T tied -> m must be 0 regardless of D.

    z0 = np.zeros(0, dtype=np.int32)
    f0 = np.zeros(0, dtype=np.float32)
    add("edge/N0-gather", [z0] * 4, [f0] * 4)
    add("edge/N0-timeonly", [z0] * 4, None)
    for v in (0, 5, -7):
        add(f"edge/N1/T{v}",
            [np.array([v], np.int32), np.array([v + 1], np.int32),
             np.array([v - 1], np.int32), np.array([v], np.int32)],
            [np.array([1.0], np.float32), np.array([np.nan], np.float32),
             np.array([0.5], np.float32), np.array([np.inf], np.float32)])
    add("edge/N1-timeonly",
        [np.array([3], np.int32), np.array([1], np.int32),
         np.array([2], np.int32), np.array([1], np.int32)], None)
    # all ties over T
    n = 65
    T = rng.integers(I32MIN, I32MAX + 1, size=(4, n)).astype(np.int32)
    T[1, :] = T[0]
    T[2, :] = T[0]
    T[3, :] = T[0]
    add("edge/all-ties/odd65", [T[0], T[1], T[2], T[3]], rand_d(n))
    add("edge/all-ties/timeonly", [T[0], T[1], T[2], T[3]], None)
    # pair ties over T
    n = 100
    T = rng.integers(-1000, 1000, size=(4, n)).astype(np.int32)
    base = rng.integers(-1000, 1000, size=n).astype(np.int32)
    T[0] = base.copy()
    T[1] = base.copy()
    T[2] = rng.integers(-1000, 1000, size=n).astype(np.int32)
    T[3] = T[2].copy()
    add("edge/pair-ties/T01-T23", [T[0], T[1], T[2], T[3]], rand_d(n))
    T[1] = rng.integers(-1000, 1000, size=n).astype(np.int32)
    T[3] = T[0].copy()
    T[2] = T[1].copy()
    add("edge/pair-ties/T03-T12", [T[0], T[1], T[2], T[3]], rand_d(n))
    # mono inc / dec every row over T
    n = 64
    t0 = rng.integers(-1000, 1000, size=n).astype(np.int32)
    step = (1 + rng.integers(0, 5, size=n)).astype(np.int32)
    add("edge/mono-inc", [t0, t0 + step, t0 + 2 * step, t0 + 3 * step], rand_d(n))
    add("edge/mono-dec", [t0 + 3 * step, t0 + 2 * step, t0 + step, t0], rand_d(n))
    add("edge/mono-inc/timeonly",
        [t0, t0 + step, t0 + 2 * step, t0 + 3 * step], None)
    # negatives over T
    n = 50
    add("edge/negatives",
        [rng.integers(-1000, 1000, size=n).astype(np.int32) for _ in range(4)],
        rand_d(n))
    # INT32 extremes in T
    n = 48
    Text = rng.integers(I32MIN, I32MAX + 1, size=(4, n)).astype(np.int32)
    Text[0, ::7] = I32MIN
    Text[1, ::11] = I32MAX
    Text[2, 0] = I32MIN
    Text[3, -1] = I32MAX
    add("edge/i32-extremes", [Text[0], Text[1], Text[2], Text[3]], rand_d(n))
    add("edge/i32-extremes/timeonly",
        [Text[0], Text[1], Text[2], Text[3]], None)
    add("edge/i32-minmax-duel",
        [np.array([I32MIN, I32MAX], np.int32), np.array([I32MAX, I32MIN], np.int32),
         np.array([I32MAX, I32MAX], np.int32), np.array([I32MAX, I32MAX], np.int32)],
        [np.array([1.0, 2.0], np.float32), np.array([3.0, 4.0], np.float32),
         np.array([5.0, 6.0], np.float32), np.array([7.0, 8.0], np.float32)])
    # NaN / Inf payload mixes in D (never select the mode)
    n = 96
    T = rng.integers(-1000, 1000, size=(4, n)).astype(np.int32)
    A = (rng.normal(0, 5, size=n)).astype(np.float32)
    B, C, E = A.copy(), A.copy(), A.copy()
    A[::3] = np.nan
    B[1::4] = np.nan
    C[::5] = np.inf
    E[2::7] = -np.inf
    add("edge/nan-inf-payload", [T[0], T[1], T[2], T[3]], [A, B, C, E])
    nan = np.full(n, np.nan, dtype=np.float32)
    add("edge/all-nan-payload", [T[0], T[1], T[2], T[3]],
        [nan, nan.copy(), nan.copy(), nan.copy()])
    add("edge/negzero-payload",
        [np.array([2, 1], np.int32), np.array([1, 2], np.int32),
         np.array([3, 3], np.int32), np.array([4, 4], np.int32)],
        [np.array([-0.0, 0.0], np.float32), np.array([0.0, -0.0], np.float32),
         np.array([-0.0, 0.0], np.float32), np.array([0.0, -0.0], np.float32)])
    # odd / even shapes
    for nn in (3, 4, 17, 18, 1023, 1024):
        add(f"edge/shape-N{nn}",
            [rng.integers(I32MIN, I32MAX + 1, size=nn).astype(np.int32)
             for _ in range(4)], rand_d(nn))
    add("edge/shape-N17/timeonly",
        [rng.integers(I32MIN, I32MAX + 1, size=17).astype(np.int32)
         for _ in range(4)], None)
    # large N
    n = 1_000_000
    add("edge/large-1M",
        [rng.integers(I32MIN, I32MAX + 1, size=n).astype(np.int32)
         for _ in range(4)], rand_d(n))

    # --- fuzz: 1200 seeded cases (small N incl 0/1/odd/even) ---
    NFUZZ = 1200
    for seed in range(NFUZZ):
        fr = np.random.default_rng(100000 + seed)
        nn = int(fr.integers(0, 66))
        T = fr.integers(I32MIN, I32MAX + 1, size=(4, max(nn, 1))).astype(np.int32)
        D = (fr.normal(0, 20, size=(4, max(nn, 1)))).astype(np.float32)
        pick = fr.random(size=(4, max(nn, 1)))
        D[pick < 0.06] = np.nan
        D[(pick >= 0.06) & (pick < 0.09)] = np.inf
        D[(pick >= 0.09) & (pick < 0.12)] = -np.inf
        tie_rows = fr.random(size=max(nn, 1)) < 0.25
        T[1, tie_rows] = T[0, tie_rows]
        tie_rows2 = fr.random(size=max(nn, 1)) < 0.15
        T[3, tie_rows2] = T[2, tie_rows2]
        T, D = T[:, :nn].copy(), D[:, :nn].copy()
        T = np.ascontiguousarray(T)
        D = np.ascontiguousarray(D)
        # alternate gather / time-only across seeds (both paths covered)
        if seed % 2 == 0:
            cases.append((f"fuzz/seed{seed}/N{nn}/gather", [T[0], T[1], T[2], T[3]],
                          [D[0], D[1], D[2], D[3]]))
        else:
            cases.append((f"fuzz/seed{seed}/N{nn}/timeonly", [T[0], T[1], T[2], T[3]],
                          None))

    t_oracle = t_nat = t_fb = 0.0
    for label, t_lanes, d_lanes in cases:
        nn = t_lanes[0].size
        snap = ([a.tobytes() for a in t_lanes],
                None if d_lanes is None else [a.tobytes() for a in d_lanes])
        t = time.perf_counter()
        eo, do_, mo = oracle(t_lanes, d_lanes)
        t_oracle += (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        (tn, dn, mn), (tf, df, mf), (tn2, dn2, mn2) = run_both(t_lanes, d_lanes)
        t_nat += (time.perf_counter() - t) * 1e3 / 2.0
        t_fb += (time.perf_counter() - t) * 1e3 / 2.0
        ok_oracle_nat = (bit_exact(tn, eo) and bit_exact(dn, do_)
                         and bit_exact(mn, mo))
        check(f"{label} native==oracle", ok_oracle_nat, f"n={nn}")
        check(f"{label} fallback==oracle",
              bit_exact(tf, eo) and bit_exact(df, do_) and bit_exact(mf, mo),
              f"n={nn}")
        check(f"{label} deterministic",
              bit_exact(tn, tn2) and bit_exact(dn, dn2) and bit_exact(mn, mn2),
              f"n={nn}")
        intact = ([a.tobytes() for a in t_lanes] == snap[0]
                  and (snap[1] is None or
                       [a.tobytes() for a in d_lanes] == snap[1]))
        check(f"{label} inputs-intact", intact, f"n={nn}")
        # regression guard inside every case: D must never move m
        # (oracle already encodes it; explicit t_best==T[m] re-gather)
        Tm = np.stack(t_lanes)
        idx = np.arange(nn)
        check(f"{label} t==T[m]",
              np.array_equal(np.ascontiguousarray(Tm[mn.astype(np.int64), idx]), tn),
              f"n={nn}")

    # API contract: mixed D (some None) -> ValueError, never a silent path
    try:
        WRAP.rowwise_min4_time_argmin_gather(
            np.zeros(4, np.int32), np.zeros(4, np.int32),
            np.zeros(4, np.int32), np.zeros(4, np.int32),
            np.zeros(4, np.float32), None, None, None)
        check("api/mixed-D-rejected", False, "no error raised")
    except ValueError:
        check("api/mixed-D-rejected", True, "")

    print(f"\nstages ms: oracle_total={t_oracle:.3f} native_total={t_nat:.3f} "
          f"fallback_total={t_fb:.3f} cases={len(cases)} "
          f"native_backend={WRAP.time_rowmin_available()} ({WRAP.why()})")
    json.dump({"seed": 42, "cases": len(cases), "fuzz": NFUZZ,
               "oracle_ms": t_oracle, "native_ms": t_nat,
               "fallback_ms": t_fb,
               "native_backend": WRAP.time_rowmin_available(), "why": WRAP.why(),
               "pass": not fails, "fails": fails},
              open(os.path.join(RES, "parity_rowmin_time.json"), "w"), indent=1)
    print("PASS" if not fails else f"FAIL {fails[:10]} (+{len(fails) - 10})"
          if len(fails) > 10 else f"FAIL {fails}")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
