# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: fused elementwise indicators (SPEC-DELTA-11) — one dispatch, K outs.

Claims: fused[K] bit-exact vs separate[K x 1] (same emitters, seed 42);
tolerance vs numpy oracle (f32); edge cases; dispatch counts; boundary
rejections (groupby/sort/EMA never fuse silently); and the param-key contract
-- no key reaches the emitter unconsumed, and a cross-output dependency is
refused by name -- whose gate is STRUCTURAL (a count of cross-output reads in
the emitted WGSL), because the defect it guards (a dropped key re-rooting the
body to x) is invisible to any float comparison.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from harness import load_profile

APP_DIR = str(Path(__file__).resolve().parents[2])
PROFILE = load_profile()

FIVE = [
    {"name": "sma20", "op": "sma", "params": {"w": 20}},
    {"name": "rsi14", "op": "rsi", "params": {"w": 14}},
    {"name": "bup20", "op": "boll_up", "params": {"w": 20, "k": 2.0}},
    {"name": "blo20", "op": "boll_lo", "params": {"w": 20, "k": 2.0}},
    {"name": "stk14", "op": "stoch_k", "params": {"w": 14}},
]

ALL_OPS = [
    {"name": "sma5", "op": "sma", "params": {"w": 5}},
    {"name": "rsum5", "op": "rsum", "params": {"w": 5}},
    {"name": "rmin5", "op": "rmin", "params": {"w": 5}},
    {"name": "rmax5", "op": "rmax", "params": {"w": 5}},
    {"name": "mom3", "op": "mom", "params": {"lag": 3}},
    {"name": "ret", "op": "returns"},
]  # rstd/rsi/boll/zscore/stoch/axpb/map_div/signals: covered below in pairs


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](APP_DIR)


def _quotes(n, seed=42, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=n).astype(np.float64)
    x = (100.0 * np.exp(np.cumsum(rets))).astype(np.float32)
    if nan_frac:
        m = rng.random(n) < nan_frac
        x[m] = np.nan
    return x


def _assert_close(a, b, label, atol=None, max_ulp=None):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, f"{label}: shape/dtype"
    if a.dtype == np.int32:
        bad = np.where(a != b)[0]
        assert bad.size == 0, f"{label}: {bad.size} int mismatches"
        return {"maxdiff": 0}
    sec = PROFILE["tolerance"]["f32"]
    atol = sec["atol"] if atol is None else atol
    max_ulp = sec["max_ulp"] if max_ulp is None else max_ulp
    af, bf = a.astype(np.float64), b.astype(np.float64)
    both_nan = np.isnan(af) & np.isnan(bf)
    assert not (np.isnan(af) ^ np.isnan(bf)).any(), f"{label}: NaN placement"
    with np.errstate(invalid="ignore"):
        diff = np.abs(af - bf)
    diff[both_nan] = 0.0
    tol = np.maximum(atol, sec["rtol"] * np.abs(bf))
    ok = (diff <= tol) | both_nan
    if ok.all():
        return {"maxdiff": float(diff.max(initial=0.0))}
    bad = np.where(~ok)[0]
    import struct as _st

    def _ord(f):
        (i,) = _st.unpack("<i", _st.pack("<f", float(f)))
        i &= 0xFFFFFFFF
        return i ^ (0xFFFFFFFF if i & 0x80000000 else 0x80000000)

    ulps = [abs(_ord(af[i]) - _ord(bf[i])) for i in bad]
    assert max(ulps) <= max_ulp, (
        f"{label}: {bad.size} beyond tol, max ULP {max(ulps)}")
    return {"maxdiff": float(diff.max(initial=0.0)), "max_ulp": max(ulps)}


# Cancellation-prone ops: (ratio-1)*100 on O(100) quotes amplifies f32 loop
# noise to ~2e-5 abs; zscore (x-mean)/sd with mean err ~W*|x|*eps needs 1e-3.
# Bounds derived from f32 eps, not tuned to data.
_RELAX = {"roc3": {"atol": 5e-5}, "rc": {"atol": 5e-5},
          "zs20": {"atol": 1e-3}, "z": {"atol": 1e-3}}


def _assert_bitexact(a, b, label):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, f"{label}: shape/dtype"
    ab, bb = a.view(np.uint32), b.view(np.uint32)
    bad = np.where(ab != bb)[0]
    assert bad.size == 0, (f"{label}: {bad.size}/{a.size} bits differ, "
                           f"first @{bad[:5]}")


@pytest.mark.fast
def test_fused_five_bitexact_vs_separate(kernel):
    a = kernel.alias
    x = _quotes(8192)
    fused, fi = a["fused_run"](x, FIVE)
    assert fi["dispatches"] == 1
    n_sep = 0
    for spec, f in zip(FIVE, fused):
        (s,), si = a["fused_run"](x, [spec])
        n_sep += si["dispatches"]
        _assert_bitexact(f, s, spec["name"])
    assert n_sep == len(FIVE) == 5


@pytest.mark.fast
def test_fused_five_vs_oracle(kernel):
    a = kernel.alias
    x = _quotes(8192)
    fused, _ = a["fused_run"](x, FIVE)
    ref = a["fused_oracle"](x, FIVE)
    for f, r, spec in zip(fused, ref, FIVE):
        m = _assert_close(f, r, spec["name"], **_RELAX.get(spec["name"], {}))
        print(f"\n{spec['name']}: maxdiff={m['maxdiff']:.3g}")


@pytest.mark.fast
def test_fused_nan_inputs_match(kernel):
    a = kernel.alias
    x = _quotes(4096, nan_frac=0.02)
    spec = [FIVE[0], FIVE[1], FIVE[4],
            {"name": "zs20", "op": "zscore", "params": {"w": 20}},
            {"name": "sd20", "op": "rstd", "params": {"w": 20, "ddof": 1}}]
    fused, _ = a["fused_run"](x, spec)
    ref = a["fused_oracle"](x, spec)
    for f, r, s in zip(fused, ref, spec):
        _assert_close(f, r, f"nan:{s['name']}", **_RELAX.get(s["name"], {}))
        (one,), _ = a["fused_run"](x, [s])
        _assert_bitexact(f, one, f"nan-bit:{s['name']}")


@pytest.mark.fast
def test_fused_all_ops_vs_oracle_and_separate(kernel):
    a = kernel.alias
    x = _quotes(2048)
    spec = (ALL_OPS
            + [{"name": "roc3", "op": "roc", "params": {"lag": 3}}])
    fused, fi = a["fused_run"](x, spec)
    assert fi["dispatches"] == 1
    ref = a["fused_oracle"](x, spec)
    for f, r, s in zip(fused, ref, spec):
        _assert_close(f, r, s["name"], **_RELAX.get(s["name"], {}))
        (one,), _ = a["fused_run"](x, [s])
        _assert_bitexact(f, one, f"bit:{s['name']}")


@pytest.mark.fast
def test_fused_maps_and_signals(kernel):
    a = kernel.alias
    x = _quotes(2048)
    spec = [
        {"name": "dbl", "op": "axpb", "params": {"a": 2.0, "b": 1.0}},
        {"name": "half", "op": "map_div", "params": {"v": 2.0}},
        {"name": "up", "op": "above_sma", "params": {"w": 20}},
        {"name": "lo", "op": "rsi_lt", "params": {"w": 14, "level": 70.0}},
        {"name": "hi", "op": "rsi_gt", "params": {"w": 14, "level": 30.0}},
        {"name": "sig", "op": "cond_and", "params": {"a": "up", "b": "lo"}},
    ]
    fused, fi = a["fused_run"](x, spec)
    assert fi["dispatches"] == 1
    assert fused[2].dtype == np.int32 and fused[5].dtype == np.int32
    ref = a["fused_oracle"](x, spec)
    for f, r, s in zip(fused, ref, spec):
        _assert_close(f, r, s["name"])


@pytest.mark.fast
def test_fused_edges(kernel):
    a = kernel.alias
    # n = 0: no dispatch, empty outs
    res, info = a["fused_run"](np.zeros(0, dtype=np.float32), FIVE[:2])
    assert info["dispatches"] == 0
    assert all(r.size == 0 for r in res)
    # n = 1, n < w: all-NaN windows
    x1 = np.array([3.5], dtype=np.float32)
    res, _ = a["fused_run"](x1, FIVE[:2])
    assert np.isnan(res[0][0]) and np.isnan(res[1][0])
    # constant series: rsi -> 50, stoch -> 50 (exact branches)
    xc = np.full(64, 7.0, dtype=np.float32)
    spec = [{"name": "r", "op": "rsi", "params": {"w": 14}},
            {"name": "s", "op": "stoch_k", "params": {"w": 14}}]
    res, _ = a["fused_run"](xc, spec)
    assert float(res[0][-1]) == 50.0
    assert float(res[1][-1]) == 50.0
    # zero series: mean/ss/sd exactly 0 -> zscore 0.0 via sd==0 branch.
    # (nonzero constants keep f32 loop-sum noise: e/|e| = +/-1, by design)
    xz0 = np.zeros(64, dtype=np.float32)
    (z,), _ = a["fused_run"](xz0, [{"name": "z", "op": "zscore",
                                    "params": {"w": 14}}])
    assert float(z[-1]) == 0.0
    # w = 1: sma == x
    xr = _quotes(128)
    (s,), _ = a["fused_run"](xr, [{"name": "s1", "op": "sma",
                                   "params": {"w": 1}}])
    _assert_bitexact(s, xr, "w=1 identity")
    # all-NaN input
    xn = np.full(64, np.nan, dtype=np.float32)
    res, _ = a["fused_run"](xn, FIVE[:2])
    assert np.isnan(res[0][14:]).all()
    # div-by-zero roc: 0/0 -> NaN, x/0 -> inf (matches numpy oracle)
    xz = np.array([0.0, 0.0, 2.0, 4.0], dtype=np.float32)
    (r,), _ = a["fused_run"](xz, [{"name": "rc", "op": "roc",
                                   "params": {"lag": 1}}])
    (o,), _ = a["fused_run"](xz, [{"name": "o", "op": "roc",
                                   "params": {"lag": 1}}])
    ref = a["fused_oracle"](xz, [{"name": "rc", "op": "roc",
                                  "params": {"lag": 1}}])
    _assert_close(r, ref[0], "roc-div0")
    assert np.isnan(o[1]) and np.isinf(o[2]) and o[3] == pytest.approx(100.0)
    # lag >= n: all NaN
    (m,), _ = a["fused_run"](xr[:8], [{"name": "m", "op": "mom",
                                       "params": {"lag": 8}}])
    assert np.isnan(m).all()


@pytest.mark.fast
def test_fused_boundary_rejections(kernel):
    a = kernel.alias
    x = _quotes(64)
    for bad in ("ema", "groupby", "groupby_multi", "sort", "join", "reduce",
                "median"):
        with pytest.raises(ValueError, match="NOT fused|unknown fused op"):
            a["fused_run"](x, [{"name": "q", "op": bad,
                                "params": {"w": 5}}])
    with pytest.raises(ValueError, match="unique name"):
        a["fused_build"]([{"name": "d", "op": "sma", "params": {"w": 5}},
                          {"name": "d", "op": "sma", "params": {"w": 5}}])
    with pytest.raises(ValueError, match="EARLIER"):
        a["fused_build"]([{"name": "sig", "op": "cond_and",
                           "params": {"a": "up", "b": "up"}},
                          {"name": "up", "op": "above_sma",
                           "params": {"w": 5}}])
    with pytest.raises(ValueError, match="window w must be"):
        a["fused_build"]([{"name": "s", "op": "sma", "params": {"w": 0}}])
    many = [{"name": f"s{i}", "op": "sma", "params": {"w": i + 2}}
            for i in range(8)]
    with pytest.raises(ValueError, match="budget exceeded"):
        a["fused_build"](many)
    with pytest.raises(ValueError, match="float32"):
        a["fused_run"](np.arange(16, dtype=np.int32),
                       [{"name": "s", "op": "sma", "params": {"w": 5}}])
    with pytest.raises(ValueError, match="rank-1"):
        a["fused_run"](np.zeros((4, 4), dtype=np.float32),
                       [{"name": "s", "op": "sma", "params": {"w": 5}}])
    c = a["fused_build"](FIVE)
    assert c["source"].count("@compute") == 1
    assert c["n_int"] == 5 and set(c["dtypes"]) == {"f32"}


# ---- param-key contract: nothing is dropped, cross-output deps refuse -------
#
# The defect this guards is STRUCTURAL, not numeric: compile_spec read only
# iparams/fparams and silently ignored every other key, so a declared
# dependency vanished and the consumer's body stayed rooted at x -- the
# composite came out a bit-identical copy of the producer and every float
# comparison still "passed". Hence the discriminator here is the emitted WGSL
# (a re-rooted body reads the producer ZERO times) plus a refusal that names
# the output, the key and the op, never a silent re-root.
#
# Two exact oracles, both required, and the structural one is the gate:
#   1. emitted source: cross-output reads counted in the WGSL text;
#   2. values: the composition of the same f32 primitive, bit-for-bit.


def _out_reads(src, j):
    """(writes, reads) of storage o{j} in an emitted source, by text form.

    `o0[i] =` is the only write shape a fused body emits; every other `o0[...]`
    is a cross-output read. Counting text keeps the check independent of dtype
    and of any float tolerance.
    """
    return (len(re.findall(r"o%d\[i\] =" % j, src)),
            len(re.findall(r"o%d\[(?!i\] =)" % j, src)))


#: The spec that used to compile to two copies of sma(3) and return rc=0.
DEP_SMA = [{"name": "inner", "op": "sma", "params": {"w": 3}},
           {"name": "outer", "op": "sma", "params": {"w": 3, "src": "inner"}}]


@pytest.mark.fast
def test_fused_refuses_unconsumed_param_key(kernel):
    """No param key may reach compile_spec unconsumed. The refusal must name
    the output, the offending key, the op and the op's accepted keys -- a bare
    ValueError leaves the caller guessing which key was dropped, which is how
    the silent re-rooting survived."""
    a = kernel.alias
    for spec, keys in (
        ([{"name": "half", "op": "map_div", "params": {"v": 2.0, "a": 1.0}}],
         ("half", "a", "map_div", "v")),
        ([{"name": "inner", "op": "sma", "params": {"win": 3}}],
         ("inner", "win", "sma", "w")),
        ([{"name": "s", "op": "rstd", "params": {"w": 5, "k": 2.0}}],
         ("s", "k", "rstd")),
        ([{"name": "r", "op": "returns", "params": {"lag": 1}}],
         ("r", "lag", "returns")),
    ):
        with pytest.raises(ValueError) as ei:
            a["fused_build"](spec)
        m = str(ei.value)
        assert "unconsumed" in m, m
        missing = [k for k in keys if k not in m]
        assert not missing, f"refusal must name {missing}: {m}"


@pytest.mark.fast
def test_fused_refuses_cross_output_dependency(kernel):
    """A dep key naming another output is refused BY NAME, with the structural
    reason -- never ignored. It used to be ignored, which is the whole defect."""
    a = kernel.alias
    for spec, keys in (
        (DEP_SMA, ("outer", "src", "sma", "no barrier")),
        ([{"name": "half", "op": "map_div",
           "params": {"v": 2.0, "src": "inner"}},
          {"name": "inner", "op": "sma", "params": {"w": 5}}],
         ("half", "src", "map_div")),
        ([{"name": "m", "op": "mom", "params": {"lag": 2, "src": "inner"}},
          {"name": "inner", "op": "sma", "params": {"w": 5}}],
         ("m", "src", "mom")),
    ):
        with pytest.raises(ValueError) as ei:
            a["fused_build"](spec)
        m = str(ei.value)
        assert "refused" in m, m
        missing = [k for k in keys if k not in m]
        assert not missing, f"refusal must name {missing}: {m}"
    # the one dependency that holds stays accepted: cond_and, same row
    ok = a["fused_build"]([{"name": "u", "op": "above_sma", "params": {"w": 5}},
                           {"name": "c", "op": "cond_and",
                            "params": {"a": "u", "b": "u"}}])
    assert ok["source"].count("@compute") == 1


@pytest.mark.fast
def test_fused_consumer_body_never_rerooted(kernel):
    """ORACLE 1 (structural, exact): the ONLY cross-output reads an emitted
    source may contain are the ones cond_and declares. Every f32 body must
    read the input and write its own storage, nothing else. The pre-fix
    composite read the producer zero times and still returned rc=0, so this
    count is what separates an honoured dep from a dropped one."""
    a = kernel.alias

    def declared_reads(spec):
        names = [s["name"] for s in spec]
        exp = {}
        for s in spec:
            if s["op"] == "cond_and":
                for k in ("a", "b"):
                    j = names.index(s["params"][k])
                    exp[j] = exp.get(j, 0) + 1
        return exp

    for spec in ([{"name": "a", "op": "sma", "params": {"w": 3}}],
                 [{"name": "a", "op": "sma", "params": {"w": 3}},
                  {"name": "b", "op": "sma", "params": {"w": 7}},
                  {"name": "c", "op": "rsi", "params": {"w": 14}}],
                 [{"name": "a", "op": "axpb", "params": {"a": 2.0, "b": 1.0}},
                  {"name": "b", "op": "map_div", "params": {"v": 2.0}},
                  {"name": "c", "op": "map_div", "params": {"v": 4.0}},
                  {"name": "d", "op": "above_sma", "params": {"w": 5}},
                  {"name": "e", "op": "cond_and",
                   "params": {"a": "d", "b": "d"}}]):
        src = a["fused_build"](spec)["source"]
        exp = declared_reads(spec)
        for j in range(len(spec)):
            w, r = _out_reads(src, j)
            assert w == 1, f"o{j}: {w} writes"
            assert r == exp.get(j, 0), (
                f"o{j} is read {r}x, declared {exp.get(j, 0)}x: an f32 body "
                f"must never read another output\n{src}")
        assert src.count("var<storage") == len(spec) + 1
    # the refused spec has no source at all -- that is the point
    with pytest.raises(ValueError, match="refused"):
        a["fused_build"](DEP_SMA)


@pytest.mark.fast
def test_fused_composite_two_dispatches_is_bitexact(kernel):
    """ORACLE 1 (exact, values): the supported route to a composite is two
    fused_run calls, and the second is bit-for-bit the composition of the same
    f32 primitive over the producer's own result. This is what a params.src
    caller must get instead of a plausible wrong value."""
    a = kernel.alias
    x = _quotes(4096)
    producer_spec = {"name": "inner", "op": "sma", "params": {"w": 3}}
    consumer_spec = {"name": "outer", "op": "sma", "params": {"w": 3}}
    (producer,), pi = a["fused_run"](x, [producer_spec])
    (comp,), ci = a["fused_run"](producer, [consumer_spec])
    assert pi["dispatches"] == 1 and ci["dispatches"] == 1
    # the producer stage is untouched by being fed back in
    _assert_bitexact(producer, a["fused_run"](x, [producer_spec])[0][0],
                     "producer stage is stable")
    # the composite is exact AND is not a copy of the producer: a silently
    # re-rooted single-dispatch form would return the producer, and this is
    # the number that says so
    live = ~np.isnan(comp)
    assert live.sum() > 4000, "windows degenerate: the check below is vacuous"
    assert np.array_equal(np.isnan(comp)[:4], [True, True, True, True])
    gap = np.abs(comp[live].astype(np.float64)
                 - producer[live].astype(np.float64)).max()
    assert gap > 1e-3, ("sma(sma3) indistinguishable from sma(3): a silently "
                        f"re-rooted implementation would satisfy this ({gap})")
    # and the refused single-dispatch form would NOT have produced it
    with pytest.raises(ValueError, match="refused"):
        a["fused_build"]([producer_spec,
                          {"name": "outer", "op": "sma",
                           "params": {"w": 3, "src": "inner"}}])


@pytest.mark.fast
def test_fused_src_dep_pandas_sanity(kernel):
    """ORACLE 2 (sanity, NOT the gate). f32 vs pandas f64 over a 3-wide window
    of O(100) quotes: each f32 layer (eps 1.19e-7) contributes <= eps/2
    relative, and the two stages give < 1e-6 relative in exact arithmetic.
    rtol=1e-4 sits ~100x above that and ~2 orders BELOW the 6.08-absolute gap
    between sma(sma3) and sma(3), so it cannot admit a re-rooted body -- but it
    is a float bound, so it is not what the suite relies on."""
    pd = pytest.importorskip("pandas")
    a = kernel.alias
    x = _quotes(4096)
    s = pd.Series(x.astype(np.float64))
    inner_ref = s.rolling(3, min_periods=3).mean()
    comp_ref = inner_ref.rolling(3, min_periods=3).mean()
    (producer,), _ = a["fused_run"](x, [{"name": "i", "op": "sma",
                                        "params": {"w": 3}}])
    (comp,), _ = a["fused_run"](producer, [{"name": "s", "op": "sma",
                                           "params": {"w": 3}}])
    rtol = 1e-4
    for got, ref, nm in ((producer, inner_ref.to_numpy(np.float32), "inner"),
                         (comp, comp_ref.to_numpy(np.float32), "outer")):
        g, r = got.astype(np.float64), np.asarray(ref, dtype=np.float64)
        assert np.array_equal(np.isnan(g), np.isnan(r)), f"{nm}: NaN placement"
        both = ~np.isnan(g)
        rel = np.abs(g[both] - r[both]) / np.abs(r[both])
        assert rel.max() <= rtol, f"{nm}: max rel {rel.max():.3g} > {rtol:.0e}"
        print(f"\n{nm}: max rel vs pandas = {rel.max():.3g}")


@pytest.mark.fast
def test_fused_i32_chain_unchanged(kernel):
    """Non-regression: the i32 mask chain (2-deep, self-referencing) still
    compiles to one dispatch with one storage binding per output and is
    bit-exact, and the three pre-existing refusals still refuse."""
    a = kernel.alias
    x = _quotes(4096)
    spec = [{"name": "a", "op": "above_sma", "params": {"w": 3}},
            {"name": "b", "op": "above_sma", "params": {"w": 5}},
            {"name": "c", "op": "cond_and", "params": {"a": "a", "b": "b"}},
            {"name": "c2", "op": "cond_and", "params": {"a": "a", "b": "c"}}]
    src = a["fused_build"](spec)["source"]
    assert src.count("@compute") == 1
    assert src.count("var<storage,read_write>") == 4
    outs, fi = a["fused_run"](x, spec)
    assert fi["dispatches"] == 1 and len(outs) == 4
    assert all(o.dtype == np.int32 for o in outs)
    ref = a["fused_oracle"](x, spec)
    for got, r, s in zip(outs, ref, spec):
        _assert_bitexact(got, r, f"i32:{s['name']}")
    # composites are exact int arithmetic on the returned masks
    _assert_bitexact(outs[2], (outs[0] * outs[1]).astype(np.int32), "c")
    _assert_bitexact(outs[3], (outs[0] * outs[2]).astype(np.int32), "c2")

    pd = pytest.importorskip("pandas")
    s = pd.Series(x.astype(np.float64))
    m3 = s.rolling(3, min_periods=3).mean().to_numpy()
    m5 = s.rolling(5, min_periods=5).mean().to_numpy()
    pa = (x.astype(np.float64) > m3).astype(np.int32)
    pb = (x.astype(np.float64) > m5).astype(np.int32)
    for got, r in zip(outs, (pa, pb, pa * pb, pa * (pa * pb))):
        _assert_bitexact(got, r.astype(np.int32), "i32:pandas")

    # the three refusals that predate the param-key contract
    with pytest.raises(ValueError, match="EARLIER"):
        a["fused_build"]([{"name": "s", "op": "cond_and",
                           "params": {"a": "u", "b": "u"}},
                          {"name": "u", "op": "above_sma", "params": {"w": 5}}])
    with pytest.raises(ValueError, match="not an i32 mask"):
        a["fused_build"]([{"name": "f", "op": "sma", "params": {"w": 3}},
                          {"name": "c", "op": "cond_and",
                           "params": {"a": "f", "b": "f"}}])
    with pytest.raises(ValueError, match="EARLIER"):
        a["fused_build"]([{"name": "c", "op": "cond_and",
                           "params": {"a": "ghost", "b": "ghost"}},
                          {"name": "u", "op": "above_sma", "params": {"w": 5}}])


