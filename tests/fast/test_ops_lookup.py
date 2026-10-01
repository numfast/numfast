# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Golden: IR lookup (build-side unique -> probe positions + hit mask).

Semantics (fixed): build keys MUST be unique (dupes -> explicit error,
never collapse -- collapse is the GPU semi-lookup set semantics in
Drivers/GPU lookup_mask/LookupTable, mask-only, NOT this node); probe
dupes independent; positions index the SORTED-UNIQUE build order (-1 on
miss); sidecar <out>#hit bool mask; payload take is composition via
gather. chunkable=false (global build, like sort/groupby).
Gates: ctor contract; CPU exact incl. int32 extremes; parity vs Join
mechanics (JoinBuild/join_inner, seed 42) + vs numpy isin / GPU
lookup_ref oracle; capability (CPU yes / GPU semi-only declaration).
"""

import importlib.util as _ilu
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_FORK / "src" / "Drivers" / "CPU"))

from _lib import cpu as _C  # noqa: E402


def _mod(name, rel):
    spec = _ilu.spec_from_file_location(
        name, str(_FORK / rel))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_N = _mod("nflookup_nodes", "src/Semantic/IR/_lib/nodes.py")
_J = _mod("nflookup_join", "src/Relational/Join/_lib/join.py")
_G = _mod("nflookup_gpu", "src/Drivers/GPU/_lib/gpu.py")


def _dt():
    table = {"int32": "int32", "float32": "float32", "float64": "float64"}

    def canonical_dtype(name):
        if name == "bool":
            return {"logical": "bool", "width": 1}
        if name not in table:
            raise ValueError(f"unknown dtype '{name}'")
        return {"logical": table[name]}

    return np.dtype("int64"), canonical_dtype


def _run(jobs):
    acc, canon = _dt()
    nodes = [{"kernel_id": j["op"], "inputs": list(j["inputs"]),
              "params": dict(j["params"]), "out": j["out"]} for j in jobs]
    s = time.perf_counter()
    bufs = _C.cpu_execute_impl(nodes, acc, canon)
    dt = (time.perf_counter() - s) * 1000
    print(f"\nlookup stages ms: execute={dt:.3f}")
    return bufs


def test_lookup_ctor_contract():
    j = _N.ir_lookup("L", "b", "p")
    assert j == {"op": "lookup", "inputs": ["b", "p"],
                 "params": {}, "out": "L"}
    with pytest.raises(ValueError):
        _N.ir_lookup("L", "b", "b")
    with pytest.raises(ValueError):
        _N.ir_lookup("L", "", "p")
    with pytest.raises(ValueError):
        _N.ir_lookup("L", "b", 42)


def test_lookup_basic_positions_hit():
    bufs = _run([_N.ir_series("b", [30, 10, 20]),
                 _N.ir_series("p", [20, 99, 10, 20]),
                 _N.ir_lookup("L", "b", "p")])
    # sorted-U order: U=[10,20,30]
    assert list(bufs["L"]) == [1, -1, 0, 1]
    assert list(bufs["L#hit"]) == [True, False, True, True]
    assert bufs["L#k"] == 3
    assert bufs["L"].dtype == np.int64


def test_lookup_edges():
    z = np.zeros(0, dtype=np.int32)
    b = _run([_N.ir_series("b", [7]), _N.ir_series("p", [7]),
              _N.ir_lookup("L", "b", "p")])
    assert list(b["L"]) == [0] and list(b["L#hit"]) == [True]
    m = _run([_N.ir_series("b", [1, 2]), _N.ir_series("p", z),
              _N.ir_lookup("L", "b", "p")])
    assert m["L"].size == 0 and m["L#hit"].size == 0
    lo = np.array([-2147483648, 2147483647], dtype=np.int32)
    e = _run([_N.ir_series("b", lo), _N.ir_series("p", lo),
              _N.ir_lookup("L", "b", "p")])
    assert list(e["L#hit"]) == [True, True]
    n = np.arange(500, dtype=np.int32)
    a = _run([_N.ir_series("b", n), _N.ir_series("p", n + 1000),
              _N.ir_lookup("L", "b", "p")])
    assert not a["L#hit"].any() and (a["L"] == -1).all()
    # probe dupes independent
    d = _run([_N.ir_series("b", [5, 7]), _N.ir_series("p", [5, 5]),
              _N.ir_lookup("L", "b", "p")])
    assert list(d["L"]) == [0, 0] and list(d["L#hit"]) == [True, True]


def test_lookup_dupe_build_explicit_error():
    with pytest.raises(ValueError, match="not unique"):
        _run([_N.ir_series("b", [5, 5, 7]), _N.ir_series("p", [5]),
              _N.ir_lookup("L", "b", "p")])


def test_lookup_parity_join_mechanics_seed42():
    """Lookup+gather reconstructs Join inner exactly (probe order kept)."""
    rng = np.random.default_rng(42)
    bad = 0
    for t in range(200):
        n = int(rng.integers(0, 300))
        s = int(rng.integers(1, 300))
        xk, xv, rk, rv = _J.make_pair(seed=42 + t, n=n, s=s)
        build = _J.JoinBuild(rk, rv)  # unique-checked oracle side
        bufs = _run([_N.ir_series("b", rk), _N.ir_series("p", xk),
                     _N.ir_lookup("L", "b", "p")])
        pos, hit = np.asarray(bufs["L"]), np.asarray(bufs["L#hit"])
        # positions index sorted-U == build.u; payload take == gather
        got_v2 = np.ascontiguousarray(build.pu[pos[hit]])
        (ok, o1, o2), _ = _J.join_inner(xk, xv, build, threads=2)
        if not (np.array_equal(xk[hit], ok) and np.array_equal(xv[hit], o1)
                and np.array_equal(got_v2, o2)):
            bad += 1
    assert bad == 0, f"{bad}/200 mismatch vs Join mechanics"


def test_lookup_parity_mask_oracle_seed42():
    """Hit mask == numpy isin == GPU lookup_ref (mask-only oracle, no device)."""
    rng = np.random.default_rng(42)
    bad = 0
    for t in range(200):
        m = int(rng.integers(0, 300))
        n = int(rng.integers(0, 300))
        B = rng.integers(-50, 50, max(m, 1)).astype(np.int32)
        P = rng.integers(-50, 50, max(n, 1)).astype(np.int32)
        B = np.unique(B)  # unique-build contract domain
        bufs = _run([_N.ir_series("b", B), _N.ir_series("p", P),
                     _N.ir_lookup("L", "b", "p")])
        hit = np.asarray(bufs["L#hit"])
        ref = np.isin(P, B)
        gref = _G.lookup_ref(B, P).astype(bool)
        if not (np.array_equal(hit, ref) and np.array_equal(hit, gref)):
            bad += 1
    assert bad == 0, f"{bad}/200 mask mismatch"


def test_lookup_capability_cpu_yes_gpu_semi_only():
    cap = _C.cpu_capability_impl()
    assert "lookup" in cap["ops"]
    assert cap["chunkable_hints"]["lookup"] is False
    gspec = _ilu.spec_from_file_location(
        "nflookup_gpu2",
        str(_FORK / "src" / "Drivers" / "GPU" / "_lib" / "gpu.py"))
    # GPU module exposes semi-lookup standalone (mask-only, dupe-collapse);
    # full IR lookup (positions + unique contract) is NOT a GPU op.
    assert hasattr(_G, "lookup_mask") and hasattr(_G, "LookupTable")
    assert "lookup" not in _G.gpu_capability_impl()["ops"]
    assert _G.gpu_capability_impl()["chunkable_hints"].get("lookup") is None
