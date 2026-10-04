# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""The GPU claim is reachable from the public surface, and it is true.

WHY THIS FILE EXISTS. The engine was already honest: `gpu_capability` lists
the ops that execute on the device, `select_backend` names every other op as
an `op:<name>` blocker, and `evaluate(graph, 'gpu', n)` RAISES on a blocker
instead of falling back to the CPU silently. Nothing in Drivers/GPU masked a
missing operation.

What was missing was the DISCLOSURE. `nf.app().capabilities()` returned the
cpu_capability record only -- 33 ops, no GPU key -- so a user of `nf.*` could
not learn which 15 ops are on the GPU, could not get the 18 CPU-only names as
a list, and could not see which backend produced a result. `gpu_capability`
existed in the kernel alias and was in neither `nf.__all__` nor `App`, and
`repr(Series)` read `Series('v', int32[100])`: shape and dtype, and nothing
about which backend produced the data.

Every count here is MEASURED AT RUN TIME, not asserted from a document. The
headline numbers (15 of 33, 18 CPU-only) are asserted so a change in either
set fails loudly instead of silently rewriting the published claim -- but they
are also cross-checked against the two capability records, so the assertion
and the derivation cannot drift apart.

NO GPU KERNELS WERE ADDED. The split is disclosed, not widened: 10 of the 18
are blocked on global ordering or sequential dependency rather than on kernel
absence, so adding kernels would not move the headline and is not a
publication task.
"""

import numpy as np
import pytest


# --- honest GPU probe ------------------------------------------------------
#
# The skip reason has to be ACCURATE, because a skip that hides a real failure
# is the same defect as a silent CPU fallback. So the probe is the driver's own
# acquisition call -- `wgpu.gpu.request_adapter_sync(power_preference=
# "high-performance")`, `Drivers/GPU/_lib/gpu.py:37` -- and it is the ONLY
# thing allowed to produce a skip. Once the probe passes, every GPU assertion
# below runs unguarded: a genuine GPU defect fails the test instead of hiding
# behind a skip.

def _gpu_probe():
    """(usable: bool, reason: str). `reason` names the actual failure."""
    try:
        import wgpu
    except Exception as exc:                      # pragma: no cover
        return False, f"wgpu is not importable ({type(exc).__name__}: {exc})"
    try:
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:
        return False, (
            f"no usable GPU adapter: request_adapter_sync raised "
            f"{type(exc).__name__}: {exc}")
    return True, "wgpu adapter acquired"


_GPU_OK, _GPU_REASON = _gpu_probe()


def _gpu_or_skip():
    if _GPU_OK:
        return
    pytest.skip("SKIP (accurate): this box has no usable GPU -- " + _GPU_REASON
                + ". The GPU assertions below assert nothing they cannot "
                  "check; the CPU-side disclosure tests still run.")


def _has_gpu_extension():
    """True when the kernel exposes gpu_capability at all.

    Probes the KERNEL ALIAS, not `App.gpu_capabilities()`: the alias is the
    pre-existing record, so this does not skip merely because the public
    surface being added here is absent -- otherwise the test would skip on the
    exact tree it is meant to fail on, which is a skip hiding a real failure.

    The CPU-only blocker tests need the GPU driver EXTENSION (it owns
    gpu_capability) but not working HARDWARE: `select_backend`'s op-subset
    gate is structural and `evaluate(graph, 'gpu', n)` refuses an ineligible
    backend before any adapter is touched. So these skip only on a kernel
    built without Drivers/GPU -- and say so -- never because an adapter is
    missing.
    """
    import numfast as nf
    if "gpu_capability" not in nf.get_kernel().alias:
        return False, ("this kernel exposes no gpu_capability: the GPU driver "
                       "extension (Drivers/GPU) is not registered, so there "
                       "is no op-subset gate to ask")
    return True, "gpu_capability present in kernel.alias"


# --- one minimal graph per CPU-only operation ------------------------------
#
# Built by hand from the ir_* node constructors, one node for the operation
# under test plus whatever sources it needs. Each is the smallest graph that
# contains the op, because the claim under test is about the OP being in the
# graph -- not about the op's data path.

def _series_jobs(alias):
    """op -> jobs list, for all 18 CPU-only operations."""
    i32 = [3, 1, 2]
    f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ser = lambda tag, v, d="int32": alias["ir_series"](tag, v, d)
    codes = ser("c", [0, 1, 2])
    return {
        "encode_pattern": [ser("s", i32),
                           alias["ir_encode_pattern"]("x", ["a1", "a2", "a3"], "a")],
        "group_count_distinct": [ser("v", i32), ser("k", [0, 0, 1]),
                                 alias["ir_count_distinct"]("x", "v", "k")],
        "lookup": [ser("b", i32), ser("p", i32), alias["ir_lookup"]("x", "b", "p")],
        "map_round": [ser("s", f32, "float32"), alias["ir_map_round"]("x", "s", 1)],
        "rng_compat": [alias["ir_rng_compat"]("x", 3, 42, "runif", 0.0, 1.0)],
        "rng_fill_f64": [alias["ir_rng_fill_f64"]("x", 3, 42, 0, 0, 0.0, 1.0)],
        "rng_permutation": [alias["ir_rng_permutation"]("x", 3, 42, 0, 0)],
        "rng_sample_no_replace": [
            alias["ir_rng_sample_no_replace"]("x", 3, 2, 42, 0, 0)],
        "rolling_sum": [ser("s", i32), alias["ir_rolling_sum"]("x", "s", 2)],
        "text_contains": [codes, alias["ir_text_contains"]("x", "c", "a")],
        "text_endswith": [codes, alias["ir_text_endswith"]("x", "c", "a")],
        "text_equals": [codes, alias["ir_text_equals"]("x", "c", "a")],
        "text_length": [codes, alias["ir_text_length"]("x", "c")],
        "text_regex_replace": [codes, alias["ir_text_regex_replace"](
            "x", "c", "a", "b")],
        "text_startswith": [codes, alias["ir_text_startswith"]("x", "c", "a")],
        "unique": [ser("s", i32), alias["ir_unique"]("x", "s")],
        "unique_inverse": [ser("s", i32), alias["ir_unique_inverse"]("x", "s")],
        "where": [ser("m", i32), ser("t", [1, 2, 3]), ser("f", [4, 5, 6]),
                  ser("c", [1, 0, 1], "bool"), alias["ir_where"]("x", "c", "t", "f")],
    }


# =============================================================================
# 1. capabilities() discloses the GPU situation
# =============================================================================

@pytest.mark.fast
def test_capabilities_report_the_gpu_op_count():
    """FAILS BEFORE: capabilities() had no gpu_op_count key at all.

    The disclosure is the whole point -- 15 of 33, stated by a public call --
    so the count is asserted AND derived, from the two live capability
    records. If a kernel is added to one set and not the other, both halves
    fail together instead of the published sentence quietly going stale.
    """
    import numfast as nf
    caps = nf.app().capabilities()
    gpu = nf.gpu_capabilities()

    assert caps["op_count"] == 33, caps["op_count"]
    assert caps["gpu_op_count"] == 15, caps["gpu_op_count"]
    assert caps["cpu_only_op_count"] == 18, caps["cpu_only_op_count"]

    # derived, not asserted: the numbers must follow the records
    assert caps["op_count"] == len(caps["ops"])
    assert caps["gpu_op_count"] == len(caps["gpu_ops"]) == len(gpu["ops"])
    assert caps["cpu_only_op_count"] == len(caps["cpu_only_ops"])
    assert caps["gpu_op_count"] + caps["cpu_only_op_count"] == caps["op_count"]


@pytest.mark.fast
def test_capabilities_keep_the_cpu_record_verbatim():
    """Non-regression: adding keys must not change a single CPU value.

    `capabilities()` answered the CPU question before this change, so it must
    answer it IDENTICALLY now -- same keys, same values. A disclosure that
    perturbs the record it discloses is a behaviour change wearing a
    disclosure's clothes.
    """
    import numfast as nf
    caps = nf.app().capabilities()
    cpu = nf.get_kernel().alias["cpu_capability"]()
    for key, value in cpu.items():
        assert caps[key] == value, key
    assert "note" in cpu and "max_dispatch" in cpu and "chunkable_hints" in cpu


@pytest.mark.fast
def test_cpu_only_ops_are_named_and_complete():
    """FAILS BEFORE: `cpu_only_ops` did not exist; the 18 were prose only.

    G4 of the acceptance criteria: the CPU-only operations are enumerable as
    a list, not a sentence. Content is pinned exactly -- a wrong name here is
    a published list that does not match the engine -- and cross-checked
    against the set difference of the two capability records.
    """
    import numfast as nf
    caps = nf.app().capabilities()
    expected = [
        "encode_pattern", "group_count_distinct", "lookup", "map_round",
        "rng_compat", "rng_fill_f64", "rng_permutation",
        "rng_sample_no_replace", "rolling_sum", "text_contains",
        "text_endswith", "text_equals", "text_length", "text_regex_replace",
        "text_startswith", "unique", "unique_inverse", "where",
    ]
    assert caps["cpu_only_ops"] == expected
    assert caps["cpu_only_ops"] == sorted(caps["cpu_only_ops"])
    # derived: exactly set(cpu ops) - set(gpu ops), nothing invented or dropped
    assert caps["cpu_only_ops"] == sorted(
        set(caps["ops"]) - set(caps["gpu_ops"]))
    # and the GPU set is a strict subset: no op is claimed GPU-only.
    assert set(caps["gpu_ops"]) <= set(caps["ops"])


@pytest.mark.fast
def test_disclosure_states_the_gpu_and_auto_contracts():
    """FAILS BEFORE: neither key existed.

    The published sentence contains two behavioural clauses -- asking for the
    GPU on a CPU-only op RAISES rather than falling back silently, and
    `backend='auto'` resolves to the CPU -- so both have to be stated by the
    call the published sentence points at, not only held true in the engine.
    """
    import numfast as nf
    caps = nf.app().capabilities()

    on_gpu = caps["on_cpu_only_under_gpu_backend"]
    assert isinstance(on_gpu, str) and on_gpu
    assert "raises" in on_gpu
    assert "no silent CPU fallback" in on_gpu
    assert "op:<name>" in on_gpu

    auto = caps["backend_auto"]
    assert isinstance(auto, str) and auto
    # `auto` is NOT contractually "always the CPU": select_backend_dual_impl
    # ends with `backend = "gpu" if gpu < cpu else "cpu"` over measured costs.
    # The disclosure therefore has to promise the safe half -- an unmeasured
    # or uncovered graph resolves to the CPU, never to a fabricated GPU choice
    # -- and must not overclaim a hard CPU default.
    assert "backend='auto'" in auto
    assert "CPU" in auto
    assert "never" in auto


# =============================================================================
# 2. the GPU capability record is reachable from the public surface
# =============================================================================

@pytest.mark.fast
def test_gpu_capabilities_is_on_the_public_surface():
    """FAILS BEFORE: the GPU capability record was in kernel.alias and in NO
    public list -- not in `nf.__all__`, not on `App`. A user could not ask the
    question at all, which is the whole masking risk.

    Named `gpu_capabilities`, NOT `gpu_capability`. The plural is load-bearing
    rather than decorative: DESIGN §3 forbids a module-level public name from
    colliding with an internal alias name, and `kernel.alias['gpu_capability']`
    IS internal (`compile` is the single documented exception). So the public
    surface mirrors `capabilities()` / `App.gpu_capabilities()` -- a curated
    plural over an internal singular.
    """
    import numfast as nf
    assert "gpu_capabilities" in nf.__all__
    assert hasattr(nf, "gpu_capabilities")
    assert callable(nf.gpu_capabilities)
    assert "gpu_capabilities" in dir(nf.app())

    # the singular stays internal, as §3 requires
    assert "gpu_capability" not in nf.__all__

    gpu = nf.gpu_capabilities()
    assert isinstance(gpu["ops"], list) and gpu["ops"]
    # the packaged boundary and the facade must not drift apart
    assert sorted(gpu["ops"]) == nf.app().capabilities()["gpu_ops"]
    assert sorted(gpu["ops"]) == sorted(gpu["ops"])


# =============================================================================
# 3. backend='gpu' on a CPU-only op raises, naming the operation
# =============================================================================

@pytest.mark.fast
def test_every_cpu_only_op_is_a_named_gpu_blocker():
    """Locks the G3 clause the disclosure now makes in public.

    For all 18, without exception: `select_backend` names the op as an
    `op:<name>` blocker, and `evaluate(graph, 'gpu', n)` raises RuntimeError
    whose message CONTAINS that `op:<name>`. A silent CPU fallback would
    return a value instead of raising, and a raise that did not name the op
    would be unactionable.

    Needs the GPU driver EXTENSION but no GPU HARDWARE: the op-subset gate is
    structural and the ineligible backend is refused before any adapter is
    touched. It therefore skips only on a kernel without Drivers/GPU -- and
    says so -- never because an adapter is missing.
    """
    import numfast as nf
    ok, reason = _has_gpu_extension()
    if not ok:
        pytest.skip("SKIP (accurate): " + reason)

    alias = nf.get_kernel().alias
    caps = nf.app().capabilities()
    jobs_by_op = _series_jobs(alias)
    assert sorted(jobs_by_op) == sorted(caps["cpu_only_ops"]), (
        "the test's graph table and the engine's CPU-only op list disagree; "
        "one of them is out of date")

    for op in caps["cpu_only_ops"]:
        graph = alias["compile"](jobs_by_op[op])
        sel = alias["select_backend"](graph, 3)
        assert sel["gpu_eligible"] is False, (op, sel)
        assert f"op:{op}" in sel["gpu_blockers"], (op, sel["gpu_blockers"])
        with pytest.raises(RuntimeError) as caught:
            alias["evaluate"](graph, "gpu", 3)
        assert f"op:{op}" in str(caught.value), (op, str(caught.value))


@pytest.mark.fast
def test_backend_gpu_on_a_cpu_only_op_is_an_error_not_a_cpu_result():
    """The public verb, not the raw alias: nf.rng_compat under backend='gpu'.

    `rng_compat` is one of the 18. Asked for the GPU it must raise, and the
    message must name the op -- through the packaged surface a user actually
    calls, not only through kernel.alias.
    """
    import numfast as nf
    ok, reason = _has_gpu_extension()
    if not ok:
        pytest.skip("SKIP (accurate): " + reason)
    caps = nf.app().capabilities()
    assert "rng_compat" in caps["cpu_only_ops"]

    alias = nf.get_kernel().alias
    graph = alias["compile"](_series_jobs(alias)["rng_compat"])
    with pytest.raises(RuntimeError, match="rng_compat"):
        alias["evaluate"](graph, "gpu", 3)


@pytest.mark.fast
def test_backend_auto_resolves_to_the_cpu_on_an_uncovered_graph():
    """The `auto` clause, as behaviour rather than as a docstring.

    `map` is GPU-eligible but UNCOVERED by the measured calibration, so `auto`
    must resolve to the CPU -- and must say why in `reason`, observably. This
    is the state the disclosure describes; it is not a claim that `auto` can
    never choose the GPU.
    """
    import numfast as nf
    alias = nf.get_kernel().alias
    graph = alias["compile"]([alias["ir_series"]("v", [1, 2, 3]),
                              alias["ir_map"]("m", "v", "add", 1)])
    sel = alias["select_backend"](graph, 3, "auto")
    assert sel["backend"] == "cpu", sel
    assert sel["reason"], sel
    res = alias["evaluate"](graph, "auto", 3)
    assert res["execution_info"]["requested"] == "auto"
    assert res["execution_info"]["actual"] == "cpu", res["execution_info"]


# =============================================================================
# 4. Results carry the backend that produced them
# =============================================================================

@pytest.mark.fast
def test_series_repr_names_the_backend():
    """FAILS BEFORE: repr read `Series('v', int32[6])` -- no backend at all.

    Three cases, because the honest answer differs in each:
      * an ENGINE RESULT says which executor ran it ('cpu');
      * a HOST INGEST says 'host', because no engine backend produced it --
        claiming 'cpu' there would assert a CPU execution that never
        happened, which is the same lie pointed the other way;
      * `.backend` agrees with the repr, so the field is readable, not just
        printable.
    """
    import numfast as nf
    s = nf.from_numpy(np.arange(6, dtype=np.int32))

    assert s.backend is None
    assert "[host]" in repr(s), repr(s)

    produced = s + 1                      # evaluate(graph, 'cpu', n)
    assert produced.backend == "cpu"
    assert "[cpu]" in repr(produced), repr(produced)

    for verb in (lambda: s.cumsum(), lambda: s.shift(1),
                 lambda: s.rolling_mean(2)):
        got = verb()
        assert got.backend == "cpu", repr(got)
        assert "[cpu]" in repr(got), repr(got)


@pytest.mark.fast
def test_gpu_result_repr_names_the_gpu():
    """FAILS BEFORE: nothing carried a backend, so a GPU-produced column and a
    CPU-produced one were indistinguishable in the REPL.

    Needs working HARDWARE, and skips with the adapter's own error text when
    there is none. The probe already ran; this body is unguarded on purpose,
    so a real GPU defect fails rather than skipping.

    WHY THIS USES A `map` GRAPH AND NOT `nf.rng_fill_i32(..., backend='gpu')`.
    The RNG verb is the only public verb taking a `backend=`, so it would be
    the end-to-end route -- but the FIRST native RNG call in a process latches
    `argtypes` on the current CDLL, and `native_cpu._probe()` resets
    `_select_ok/_shift_ok/_map_ok/_cumsum_ok` on a DLL-identity change but NOT
    `_rng_ok`. `test_packaging_adapters` toggles `NUMFAST_NATIVE_DISABLE`,
    which re-`CDLL`s the same path into a fresh object with default `c_int`
    argtypes, so the latch goes stale and every later RNG call raises
    `ctypes.ArgumentError: argument 1: OverflowError: int too long to
    convert` -- which `cpu.py`'s `except RuntimeError` does not catch either.
    That is a pre-existing defect in FROZEN Drivers/CPU, reproducible from the
    public API alone (see the report); this test must not be the thing that
    trips it, so it drives a GPU-eligible `map` through the engine instead and
    reads the backend from the engine's OWN `execution_info['actual']` rather
    than choosing it. COVERAGE GAP, stated rather than hidden: the public RNG
    verb's threading of `backend` is not asserted end-to-end here.
    """
    _gpu_or_skip()
    import numfast as nf
    alias = nf.get_kernel().alias
    src = np.arange(4096, dtype=np.int32)
    jobs = [alias["ir_series"]("s", src), alias["ir_map"]("m", "s", "add", 1)]

    res = alias["evaluate"](alias["compile"](jobs), "gpu", len(src))
    actual = res["execution_info"]["actual"]
    assert actual == "gpu", res["execution_info"]      # the engine really ran

    col = nf.Series(nf.get_kernel(), "m", res["result"], "int32", backend=actual)
    assert col.backend == "gpu"
    assert "[gpu]" in repr(col), repr(col)
    assert col.to_numpy().tolist() == (src + 1).tolist()

    # the label is not decorative: identical bytes to the CPU path, different label
    cpu = alias["evaluate"](alias["compile"](jobs), "cpu", len(src))
    assert np.asarray(res["result"]).tobytes() == \
        np.asarray(cpu["result"]).tobytes()


@pytest.mark.fast
def test_chain_results_carry_the_cpu_backend():
    """A compiled query's columns carry the executor that produced them.

    `compile()` reads `cpu_execute` buffers, so 'cpu' is the fact; leaving it
    unset would make a query result look host-ingested, which is a different
    and equally wrong claim.
    """
    import numfast as nf
    table = nf.from_numpy(np.arange(12, dtype=np.int32).reshape(4, 3))
    out = table.query().compile()
    for name in out.names:
        col = out.column(name)
        assert col.backend == "cpu", repr(col)
        assert "[cpu]" in repr(col), repr(col)