# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""RNG + unique public helpers over the packaged kernel (thin, IR-only).

Every op lowers to IR jobs and runs compile -> optimize -> evaluate/cpu_execute.
No driver imports (flat assembly: kernel.alias only). No H2O sugar.
"""

import numpy as np

from .series import Series


def check_seed(seed):
    """rng_seed contract: uint64 int, bool rejected, explicit error."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"rng_seed must be an int in [0, 2**64), got {seed!r}. "
            "Fix: pass seed=42 (or any uint64 int). "
            "See specs/02-semantic-ir.md"
        )
    if not 0 <= seed < 2 ** 64:
        raise ValueError(
            f"rng_seed must be an int in [0, 2**64), got {seed!r}. "
            "Fix: pass seed=42 (or any uint64 int). "
            "See specs/02-semantic-ir.md"
        )
    return int(seed)


def _check_domain(stream, offset):
    for name, v in (("stream", stream), ("offset", offset)):
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2 ** 64:
            raise ValueError(
                f"rng {name} must be an int in [0, 2**64), got {v!r}. "
                f"Fix: pass {name}=0 (or any uint64 int)."
            )
    return int(stream), int(offset)


def _series_from(kernel, name, arr, logical, validity=None):
    return Series(kernel, name, np.ascontiguousarray(arr), logical,
                  validity=validity)


def fill_i32(kernel, n, seed, stream=0, offset=0, lo=0, hi=100,
             name="rng", backend="cpu"):
    a = kernel.alias
    seed = check_seed(seed)
    stream, offset = _check_domain(stream, offset)
    jobs = [a["ir_rng_fill_i32"]("r", n, seed, stream, offset, lo, hi)]
    graph = a["optimize"](a["compile"](jobs))
    res = a["evaluate"](graph, backend, max(0, int(n)))
    buf = res["result"]
    if res["execution_info"]["actual"] != backend and backend != "auto":
        raise RuntimeError(
            f"rng_fill_i32 routed to '{res['execution_info']['actual']}', "
            f"'{backend}' required."
        )
    return _series_from(kernel, name, buf, "int32")


def fill_f64(kernel, n, seed, stream=0, offset=0, lo=0.0, hi=1.0,
             name="rng", backend="cpu"):
    a = kernel.alias
    seed = check_seed(seed)
    stream, offset = _check_domain(stream, offset)
    jobs = [a["ir_rng_fill_f64"]("r", n, seed, stream, offset, lo, hi)]
    graph = a["optimize"](a["compile"](jobs))
    res = a["evaluate"](graph, backend, max(0, int(n)))
    return _series_from(kernel, name, res["result"], "float64")


def sample(kernel, n, k, seed, stream=0, offset=0, name="sample",
           backend="cpu"):
    a = kernel.alias
    seed = check_seed(seed)
    stream, offset = _check_domain(stream, offset)
    jobs = [a["ir_rng_sample_no_replace"]("r", n, k, seed, stream, offset)]
    graph = a["optimize"](a["compile"](jobs))
    res = a["evaluate"](graph, backend, max(0, int(n)))
    return _series_from(kernel, name, res["result"], "int32")


def permutation(kernel, n, seed, stream=0, offset=0, name="perm",
                backend="cpu"):
    a = kernel.alias
    seed = check_seed(seed)
    stream, offset = _check_domain(stream, offset)
    jobs = [a["ir_rng_permutation"]("r", n, seed, stream, offset)]
    graph = a["optimize"](a["compile"](jobs))
    res = a["evaluate"](graph, backend, max(0, int(n)))
    return _series_from(kernel, name, res["result"], "int32")


def compat(kernel, n, seed, kind="runif", lo=0.0, hi=1.0, m=None,
           name="compat"):
    """R-compat draw (CPU-only by contract; backend fixed to cpu)."""
    a = kernel.alias
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"rng_compat seed must be an int, got {seed!r}. Fix: pass seed=42."
        )
    jobs = [a["ir_rng_compat"]("r", n, seed, kind, lo, hi, m)]
    graph = a["optimize"](a["compile"](jobs))
    res = a["evaluate"](graph, "cpu", max(0, int(n)))
    logical = "float64" if kind == "runif" else "int32"
    return _series_from(kernel, name, res["result"], logical)


def map_round(kernel, series, ndigits=0, name=None):
    """Half-even round of a Series -> float64 Series (+validity sidecar)."""
    a = kernel.alias
    if not isinstance(series, Series):
        raise ValueError(
            f"map_round needs a NumFast Series, got {type(series).__name__}. "
            "Fix: pass nf.from_numpy(arr)."
        )
    jobs = [series._source("s"), a["ir_map_round"]("m", "s", ndigits)]
    graph = a["optimize"](a["compile"](jobs))
    bufs = a["cpu_execute"](graph["nodes"])
    valid = bufs.get("m#validity")
    return Series(kernel, name or series.name, bufs["m"], "float64",
                  validity=None if valid is None else np.ascontiguousarray(
                      np.asarray(valid, dtype=bool)))


def unique(kernel, series, name=None):
    """Sorted unique + inverse: {uniq, inv, ng}.

    uniq[inv] == keys (valid rows); invalid rows -> inv -1, excluded
    from uniq; ng = distinct count (metadata). Chains through the
    existing Series._source/IR path (no new machinery).
    """
    a = kernel.alias
    if not isinstance(series, Series):
        raise ValueError(
            f"unique needs a NumFast Series, got {type(series).__name__}. "
            "Fix: pass nf.from_numpy(arr)."
        )
    jobs = [series._source("s"), a["ir_unique_inverse"]("u", "s")]
    graph = a["optimize"](a["compile"](jobs))
    bufs = a["cpu_execute"](graph["nodes"])
    tag = name or series.name
    uniq = bufs["u"]
    logical = series.dtype if series.dtype in ("int32", "int64") else (
        "int32" if uniq.dtype == np.dtype(np.int32) else "int64")
    return {"uniq": Series(kernel, tag, uniq, logical),
            "inv": Series(kernel, (name + "#inv" if name else series.name + "#inv"),
                          bufs["u#inv"], "int32"),
            "ng": int(bufs["u#ng"])}


def shift(kernel, series, periods, name=None):
    """Positional shift N->N right by periods (thin over ir_shift).

    Chains through the existing Series._source/IR path (no new machinery).
    """
    if not isinstance(series, Series):
        raise ValueError(
            f"shift needs a NumFast Series, got {type(series).__name__}. "
            "Fix: pass nf.from_numpy(arr)."
        )
    return series.shift(periods, name=name)


def cumsum(kernel, series, name=None):
    """Inclusive prefix sum N->N (thin over ir_cumsum).

    Chains through the existing Series._source/IR path (no new machinery).
    """
    if not isinstance(series, Series):
        raise ValueError(
            f"cumsum needs a NumFast Series, got {type(series).__name__}. "
            "Fix: pass nf.from_numpy(arr)."
        )
    return series.cumsum(name=name)


def rolling_mean(kernel, series, window, min_periods=None, name=None):
    """Rolling mean N->N (thin over ir_rolling_sum + ir_map div).

    Chains through the existing Series._source/IR path (no new machinery).
    """
    if not isinstance(series, Series):
        raise ValueError(
            f"rolling_mean needs a NumFast Series, got {type(series).__name__}. "
            "Fix: pass nf.from_numpy(arr)."
        )
    return series.rolling_mean(window, min_periods=min_periods, name=name)


def returns(kernel, series, name=None):
    """Simple returns N->N (thin over ir_shift + ir_map div/sub).

    Chains through the existing Series._source/IR path (no new machinery).
    """
    if not isinstance(series, Series):
        raise ValueError(
            f"returns needs a NumFast Series, got {type(series).__name__}. "
            "Fix: pass nf.from_numpy(arr)."
        )
    return series.returns(name=name)


def lookup(kernel, build, probe, name=None):
    """Build-probe lookup -> {positions, hit, k} (thin over ir_lookup).

    Inputs must both be int32 Series (pass nf.from_numpy(arr); int64
    narrows on ingest, float/bool/text have no key semantics). Build
    keys MUST be unique (valid rows): dupes raise the same ValueError
    contract as the Join build ("not unique ... dedupe ..."); probe
    keys may repeat freely. Output positions index the SORTED-UNIQUE
    build order (miss -> -1, int32 Series: int64 driver output narrowed
    losslessly, max pos = k-1) + hit bool Series (True = match) + k
    (distinct valid build count, metadata like unique ng). Payload take
    is composition: payload_np[to_numpy(positions)[to_numpy(hit)]] or
    an ir_gather job over hit-filtered positions -- lookup never
    carries payloads. Invalid rows never match (miss, not error):
    invalid build rows are excluded from the table, invalid probe rows
    are forced to (-1, False). Chains through the existing
    Series._source/IR path (no new machinery, no IR change).
    """
    a = kernel.alias
    for label, s in (("build", build), ("probe", probe)):
        if not isinstance(s, Series):
            raise ValueError(
                f"lookup {label} needs a NumFast Series, got "
                f"{type(s).__name__}. "
                "Fix: pass nf.from_numpy(arr)."
            )
        if s.dtype != "int32":
            raise ValueError(
                f"lookup {label} needs int32 keys, got {s.dtype!r}. "
                "Fix: encode categoricals to int32 codes first "
                "(float/bool/text have no key semantics). "
                "See specs/02-semantic-ir.md"
            )
    bvals = np.ascontiguousarray(build.to_numpy(), dtype=np.int32)
    bvalid = build.validity
    if bvalid is not None and not bool(bvalid.all()):
        bvals = np.ascontiguousarray(bvals[bvalid])
    bsrc = Series(kernel, "__lookup_build", bvals, "int32")
    jobs = [bsrc._source("b"), probe._source("p"),
            a["ir_lookup"]("L", "b", "p")]
    graph = a["optimize"](a["compile"](jobs))
    bufs = a["cpu_execute"](graph["nodes"])
    pos = np.ascontiguousarray(np.asarray(bufs["L"], dtype=np.int64))
    hit = np.ascontiguousarray(np.asarray(bufs["L#hit"], dtype=bool))
    pvalid = probe.validity
    if pvalid is not None and not bool(pvalid.all()):
        pos = np.where(pvalid, pos, -1).astype(np.int64)
        hit = np.ascontiguousarray(hit & pvalid)
    k = int(bufs["L#k"])
    if int(pos.max(initial=-1)) >= k or int(pos.min(initial=0)) < -1:
        raise ValueError(
            f"lookup positions out of range for k={k}. "
            "Fix: report as a bug (driver contract violation)."
        )
    tag = name or probe.name
    return {"positions": Series(kernel, tag,
                                np.ascontiguousarray(pos.astype(np.int32)),
                                "int32"),
            "hit": Series(kernel, tag + "#hit", hit, "bool"),
            "k": k}
