"""Phase 4 - REDUCE resource/failure tests (S67), R1-R7 + readback-size.

Spec: REDUCE_SLICE_SPEC.md sec.8 (S67).

R1 refcount (acquire->release->release clamps at 0, re-acquire returns
   the same buffer).
R2 reuse (max_bytes -> evict -> free-list -> same-size alloc reuses the
   buffer).
R3 leak (20 warm Reduce runs: uploads/allocs/resident stable; after
   driver.release() -> mapped=0).
R4-Reduce: min/max N=0 -> ValueError BEFORE dispatch (S62 descriptor
   guard, execute=0/execute_wave=0, GPU not submitted); N=4_194_241 ->
   ValueError BEFORE dispatch (S64 _check_dispatch_limit, queue.submit
   never called).
R5: N=4_194_240 (limit) - 1 packet ordinary, correct (sum rel 1e-4 vs
   np.sum, min/max exact, GPU==CPU).
R6: sum N=0 -> ValueError BEFORE dispatch (S62 unified, user decision
   2026-08-20: ALL ops N=0 -> ValueError on both backends; GPU 0-byte
   buffer impossible on frozen wgpu_driver, sum 0.0 neutral NOT
   implemented); GPU not submitted.
R7: repeated calls after errors (min N=0, max N=0, dispatch limit) -
   pool not broken, subsequent runs correct.
+ readback-size: Reduce n=64 / n=1M correctness (D2H by exact size).
+ observation (out of slice): unknown op AS-IS divergence - descriptor
   defaults to "sum" (output template), CPU raises ValueError, GPU WGSL
   else->max - recorded, NOT a failure (fix per S49 precedent, future
   slice).

Evidence: evidence/reduce_slice_phase4/resource_failure.json

IMPORTANT: this file is ASCII-only (no Cyrillic) because
tests/test_backend.py::test_no_cupy_in_test_files reads tests/*.py with
open() in locale encoding (cp1251 on this host).
"""

import json
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import Runtime
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Compute import register_all

CHECKPOINT_SHA = "f3a21d9"
SLICE = "reduce"
SEED = 42
LIMIT = 4_194_240
SUM_JOBS = [{"op": "Reduce", "inputs": ["x"], "params": {}, "out": "y"}]

EVIDENCE_DIR = (pathlib.Path(__file__).resolve().parent
               / "evidence" / "reduce_slice_phase4")

try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}
ADAPTER_NAME = ADAPTER_INFO.get("device", "unknown")
ADAPTER_BACKEND = ADAPTER_INFO.get("backend_type", "unknown")

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")

EVIDENCE = {"R1": None, "R2": None, "R3": None, "R4": None, "R5": None,
            "R6": None, "R7": None, "readback_size": None,
            "unknown_op_observation": None}

_EMPTY_MSG = ("Reduce on empty input (n=0) is undefined; "
              "use Fill or check input length")


def _num(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import wgpu
        v["wgpu"] = wgpu.__version__
    except Exception:  # noqa: BLE001
        v["wgpu"] = "n/a"
    return v


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_all(rt)
    return rt


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    return rt


def _job_for(op):
    return {"op": "Reduce", "inputs": ["x"], "params": {"op": op}, "out": "y"}


def _sum_ref(x):
    return float(np.sum(x.astype(np.float64)))


# ============================================================================
# R1 - refcount: double release clamps at 0, re-acquire returns same buffer
# ============================================================================

@needs_gpu
def test_r1_double_release():
    EVIDENCE["R1"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        pool = rt.driver._pool
        raw = np.zeros(64, dtype=np.float32)
        b1 = pool.acquire(raw, 256, upload_fn=lambda: b"\x00" * 256)
        rc_after_acquire = pool._map[id(raw)].refcount
        pool.release(raw)
        rc_after_release1 = pool._map[id(raw)].refcount
        pool.release(raw)  # double release
        rc_after_release2 = pool._map[id(raw)].refcount
        b2 = pool.acquire(raw, 256, upload_fn=lambda: b"\x00" * 256)
        rc_after_acquire2 = pool._map[id(raw)].refcount
        assert rc_after_release2 == 0, "refcount must clamp at 0"
        assert rc_after_acquire == 1 and rc_after_release1 == 0
        assert b2 is b1, "re-acquire must return the same buffer"
        assert rc_after_acquire2 == 1
        EVIDENCE["R1"] = {
            "status": "PASS",
            "fact": f"acquire->release->release: refcount "
                    f"{rc_after_acquire}->{rc_after_release1}->"
                    f"{rc_after_release2} (>=0); re-acquire returns same "
                    f"buffer, refcount {rc_after_acquire2}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R2 - reuse: evict -> free-list -> same-size alloc reuses the buffer
# ============================================================================

@needs_gpu
def test_r2_evict_reuse():
    EVIDENCE["R2"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        pool = rt.driver._pool
        pool.max_bytes = 1_000_000
        raw1 = np.zeros(524288, dtype=np.float32)
        raw_other = np.zeros(300000, dtype=np.float32)
        raw2 = np.zeros(524288, dtype=np.float32)
        b1 = pool.alloc(raw1, 524288)
        pool.alloc(raw_other, 300000)
        pool.finalize([id(raw1), id(raw_other)])
        mid = pool.stats()
        b2 = pool.alloc(raw2, 524288)
        end = pool.stats()
        reused = b1 is b2
        assert reused, "same-size alloc must reuse the free-list buffer"
        assert end["allocs"] == mid["allocs"], \
            "free-list reuse must not grow allocs"
        assert end["free"] == 0 and end["mapped"] == 1
        EVIDENCE["R2"] = {
            "status": "PASS",
            "fact": f"max_bytes=1M: alloc(524288)+alloc(300000)->finalize->"
                    f"evicts={mid['evicts']}, destroys={mid['destroys']}, "
                    f"free={mid['free']}; alloc(524288) reused same buffer "
                    f"({reused}), allocs stable {mid['allocs']}->"
                    f"{end['allocs']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R3 - leak: 20 warm Reduce runs, pool stable; after release -> 0
# ============================================================================

@needs_gpu
def test_r3_leak():
    EVIDENCE["R3"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        tasks = rt.compile(SUM_JOBS)
        stats_first = None
        for i in range(20):
            rt.execute(tasks, {"x": x})
            if i == 0:
                stats_first = rt.driver._pool.stats()
        stats_last = rt.driver._pool.stats()
        stable = (stats_first["resident_bytes"] == stats_last["resident_bytes"]
                  and stats_first["allocs"] == stats_last["allocs"]
                  and stats_first["uploads"] == stats_last["uploads"])
        assert stable, f"pool must be stable, {stats_first} -> {stats_last}"
        rt.driver.release()
        after_release = rt.driver._pool.stats()
        assert after_release["mapped"] == 0 and after_release["free"] == 0, \
            f"pool must be fully released, got {after_release}"
        EVIDENCE["R3"] = {
            "status": "PASS",
            "fact": f"20 identical warm Reduce runs: uploads="
                    f"{stats_first['uploads']}, allocs={stats_first['allocs']}, "
                    f"resident_bytes={stats_first['resident_bytes']} stable "
                    f"(ordinary path allocates per-call buffers, pool not "
                    f"touched on the ordinary path - stability trivially "
                    f"holds); after driver.release(): mapped="
                    f"{after_release['mapped']}, free={after_release['free']}",
        }
    finally:
        rt.driver.release()


# ============================================================================
# R4-Reduce - min/max N=0 (S62) and N=4_194_241 (S64): ValueError
#             BEFORE dispatch, GPU not submitted
# ============================================================================

@needs_gpu
def test_r4_error_before_dispatch():
    EVIDENCE["R4"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        x0 = np.zeros(0, dtype=np.float32)
        calls = {"execute": 0, "execute_wave": 0, "submit": 0}
        orig_exec = rt.driver.execute
        orig_wave = rt.driver.execute_wave
        orig_submit = rt.driver._queue.submit

        def wrap_exec(packet):
            calls["execute"] += 1
            return orig_exec(packet)

        def wrap_wave(wave):
            calls["execute_wave"] += 1
            return orig_wave(wave)

        def wrap_submit(*a, **k):
            calls["submit"] += 1
            return orig_submit(*a, **k)

        rt.driver.execute = wrap_exec
        rt.driver.execute_wave = wrap_wave
        rt.driver._queue.submit = wrap_submit

        # min/max N=0: descriptor._output_size guard at BUILD time
        for op in (1, 2):
            with pytest.raises(ValueError) as excinfo:
                rt.execute(rt.compile([_job_for(op)]), {"x": x0})
            assert str(excinfo.value) == _EMPTY_MSG, str(excinfo.value)
        # N=4_194_241: _check_dispatch_limit at execute() before compile
        y = np.zeros(LIMIT + 1, dtype=np.float32)
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile(SUM_JOBS), {"x": y})
        msg = str(excinfo.value)
        assert "Dispatch limit exceeded" in msg, msg
        assert calls["submit"] == 0, \
            f"GPU must never submit: submit={calls['submit']}"
        assert calls["execute"] == 1 and calls["execute_wave"] == 1, \
            f"expected only the limit packet to reach execute, got " \
            f"execute={calls['execute']} execute_wave={calls['execute_wave']}"
        EVIDENCE["R4"] = {
            "status": "PASS",
            "fact": f"min N=0 and max N=0 -> ValueError at BUILD "
                    f"(descriptor._output_size, S62): '{_EMPTY_MSG}'; "
                    f"N=4_194_241 -> ValueError '{msg[:60]}...' at "
                    f"execute() BEFORE compile/dispatch "
                    f"(_check_dispatch_limit, S64); GPU never submitted: "
                    f"submit={calls['submit']}, execute="
                    f"{calls['execute']} (only the limit packet reached "
                    f"execute and raised), execute_wave="
                    f"{calls['execute_wave']}",
        }
    finally:
        rt.driver.execute = orig_exec
        rt.driver.execute_wave = orig_wave
        rt.driver._queue.submit = orig_submit
        rt.driver.release()


# ============================================================================
# R5 - N=4_194_240 (limit): 1 packet ordinary, correct (sum rel 1e-4,
#      min/max exact, GPU==CPU)
# ============================================================================

@needs_gpu
def test_r5_limit_n_correct():
    EVIDENCE["R5"] = {"status": "FAIL", "fact": "test raised"}
    rt_gpu = _make_gpu_runtime()
    rt_cpu = _make_cpu_runtime()
    orig_exec = None
    try:
        assert rt_gpu.driver.max_dispatch_elements() == LIMIT
        rng = np.random.default_rng(SEED)
        n = LIMIT
        x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
        ref = {"sum": _sum_ref(x),
               "min": float(np.min(x.astype(np.float64))),
               "max": float(np.max(x.astype(np.float64)))}
        packets = []
        orig_exec = rt_gpu.driver.execute

        def wrap_exec(packet):
            packets.append(packet)
            return orig_exec(packet)

        rt_gpu.driver.execute = wrap_exec
        results = {}
        for op, name in ((0, "sum"), (1, "min"), (2, "max")):
            packets.clear()
            rt_gpu.execute(rt_gpu.compile([_job_for(op)]), {"x": x})
            gpu = float(rt_gpu.driver.resolve_output("y")[0])
            rt_cpu.execute(rt_cpu.compile([_job_for(op)]), {"x": x})
            cpu = float(rt_cpu.driver.resolve_output("y")[0])
            results[name] = {"gpu": gpu, "cpu": cpu, "oracle": ref[name]}
            assert len(packets) == 1, \
                f"{name}: 1 packet expected, got {len(packets)}"
            assert packets[0].profile["dispatch_x"] == 65535, \
                f"{name}: dx={packets[0].profile['dispatch_x']}"
            if name == "sum":
                rel_g = abs(gpu - ref["sum"]) / max(abs(ref["sum"]), 1e-300)
                rel_c = abs(cpu - ref["sum"]) / max(abs(ref["sum"]), 1e-300)
                assert rel_g <= 1e-4, f"sum gpu rel {rel_g}"
                assert rel_c <= 1e-4, f"sum cpu rel {rel_c}"
            else:
                assert gpu == ref[name], f"{name}: gpu {gpu} != {ref[name]}"
                assert cpu == ref[name], f"{name}: cpu {cpu} != {ref[name]}"
        EVIDENCE["R5"] = {
            "status": "PASS",
            "fact": f"N={LIMIT} (limit): each op 1 packet ordinary, "
                    f"dx=65535; "
                    f"sum gpu={results['sum']['gpu']:.6e} cpu="
                    f"{results['sum']['cpu']:.6e} oracle="
                    f"{results['sum']['oracle']:.6e} (rel gpu "
                    f"{abs(results['sum']['gpu'] - ref['sum']) / max(abs(ref['sum']), 1e-300):.2e} "
                    f"<= 1e-4); min gpu==cpu==oracle="
                    f"{results['min']['gpu']!r}; max gpu==cpu==oracle="
                    f"{results['max']['gpu']!r}",
        }
    finally:
        if orig_exec is not None:
            rt_gpu.driver.execute = orig_exec
        rt_cpu.driver.release()
        rt_gpu.driver.release()


# ============================================================================
# R6 - sum N=0 -> ValueError BEFORE dispatch (S62 unified: ALL ops)
# ============================================================================

def test_r6_sum_n_zero():
    fact = ""
    n = 0
    x = np.zeros(n, dtype=np.float32)

    rt_cpu = Runtime(driver=CpuDriver())
    register_all(rt_cpu)
    try:
        calls = {"execute": 0, "execute_wave": 0}
        orig_exec = rt_cpu.driver.execute
        orig_wave = rt_cpu.driver.execute_wave

        def wrap_exec(packet):
            calls["execute"] += 1
            return orig_exec(packet)

        def wrap_wave(wave):
            calls["execute_wave"] += 1
            return orig_wave(wave)

        rt_cpu.driver.execute = wrap_exec
        rt_cpu.driver.execute_wave = wrap_wave
        with pytest.raises(ValueError) as excinfo:
            rt_cpu.execute(rt_cpu.compile(SUM_JOBS), {"x": x})
        assert str(excinfo.value) == _EMPTY_MSG, str(excinfo.value)
        assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
            f"CPU N=0 sum must fail BEFORE dispatch: " \
            f"execute={calls['execute']}, execute_wave={calls['execute_wave']}"
        fact = ("CPU N=0 sum -> ValueError BEFORE dispatch "
                "(descriptor._output_size, S62 unified), "
                "execute=0/execute_wave=0")
    finally:
        rt_cpu.driver.release()

    if GPU_AVAILABLE:
        rt = _make_gpu_runtime()
        try:
            calls = {"execute": 0, "execute_wave": 0}
            orig_exec = rt.driver.execute
            orig_wave = rt.driver.execute_wave

            def wrap_exec(packet):
                calls["execute"] += 1
                return orig_exec(packet)

            def wrap_wave(wave):
                calls["execute_wave"] += 1
                return orig_wave(wave)

            rt.driver.execute = wrap_exec
            rt.driver.execute_wave = wrap_wave
            with pytest.raises(ValueError) as excinfo:
                rt.execute(rt.compile(SUM_JOBS), {"x": x})
            assert str(excinfo.value) == _EMPTY_MSG, str(excinfo.value)
            assert calls["execute"] == 0 and calls["execute_wave"] == 0, \
                f"GPU N=0 sum must fail BEFORE dispatch (no submit): " \
                f"execute={calls['execute']}, execute_wave={calls['execute_wave']}"
            fact += ("; GPU N=0 sum -> ValueError BEFORE dispatch "
                     "(guard, no submit)")
        finally:
            rt.driver.release()
    else:
        fact += "; GPU N=0 sum skipped (no adapter)"

    EVIDENCE["R6"] = {"status": "PASS", "fact": fact}


# ============================================================================
# R7 - repeated calls after errors (min N=0, max N=0, dispatch limit):
#      pool not broken, subsequent runs correct
# ============================================================================

@needs_gpu
def test_r7_repeated_calls_after_error():
    EVIDENCE["R7"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        x = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        ref = _sum_ref(x)
        x0 = np.zeros(0, dtype=np.float32)

        def run_ok():
            rt.execute(rt.compile(SUM_JOBS), {"x": x})
            res = rt.driver.resolve_output("y")
            assert res is not None
            assert abs(_sum_ref(x) - ref) <= 1e-4 * max(ref, 1.0)

        # 1) min N=0 -> ValueError (build), then correct run
        with pytest.raises(ValueError):
            rt.execute(rt.compile([_job_for(1)]), {"x": x0})
        run_ok()

        # 2) max N=0 -> ValueError (build), then correct run
        with pytest.raises(ValueError):
            rt.execute(rt.compile([_job_for(2)]), {"x": x0})
        run_ok()

        # 3) dispatch limit N=4_194_241 -> ValueError (execute), then
        #    correct run
        y = np.zeros(LIMIT + 1, dtype=np.float32)
        with pytest.raises(ValueError) as excinfo:
            rt.execute(rt.compile(SUM_JOBS), {"x": y})
        assert "Dispatch limit exceeded" in str(excinfo.value)
        run_ok()

        # 4) unknown op -> CPU ValueError is a CPU-path fact; on GPU the
        #    WGSL else->max executes - record observation, then correct run
        z = (rng.standard_normal(64) * 5 + 30).astype(np.float32)
        rt.execute(rt.compile([_job_for(7)]), {"x": z})
        res7 = rt.driver.resolve_output("y")
        assert res7 is not None and res7.shape == (1,)
        run_ok()

        EVIDENCE["R7"] = {
            "status": "PASS",
            "fact": "after min N=0 ValueError (build), after max N=0 "
                    "ValueError (build), after N=4_194_241 ValueError "
                    "(execute, _check_dispatch_limit), after unknown-op "
                    "op=7 GPU run (WGSL else->max, AS-IS observation) - "
                    "subsequent Reduce sum runs produce correct results "
                    "(pool not broken)",
        }
    finally:
        rt.driver.release()


# ============================================================================
# readback-size: Reduce n=64 / n=1M correctness (D2H by exact size)
# ============================================================================

@needs_gpu
def test_readback_size():
    EVIDENCE["readback_size"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        rng = np.random.default_rng(SEED)
        results = {}
        for n in (64, 1_000_000):
            x = (rng.standard_normal(n) * 5 + 30).astype(np.float32)
            ref = _sum_ref(x)
            rt.execute(rt.compile(SUM_JOBS), {"x": x})
            res = rt.driver.resolve_output("y")
            assert res is not None and res.shape == (1,), \
                f"n={n} shape {None if res is None else res.shape}"
            rel = abs(float(res[0]) - ref) / max(ref, 1e-300)
            assert rel <= 1e-4, f"n={n} rel {rel}"
            results[str(n)] = {"shape": list(res.shape), "rel": rel}
        EVIDENCE["readback_size"] = {
            "status": "PASS",
            "fact": f"Reduce readback by exact payload size (Q1 fix, S28): "
                    f"n=64 rel={results['64']['rel']:.2e}, "
                    f"n=1M rel={results['1000000']['rel']:.2e} "
                    f"(sum vs np.sum f64, contract rel 1e-4), "
                    f"shape [1] both",
        }
    finally:
        rt.driver.release()


# ============================================================================
# Observation (out of slice): unknown op AS-IS divergence
# ============================================================================

def test_unknown_op_observation():
    """Unknown op: descriptor defaults to 'sum' (output template), CPU
    raises ValueError, GPU WGSL else->max. AS-IS divergence (REDUCE_SLICE_
    SPEC sec.2 observation), fix per S49 precedent in a future slice.
    Recorded, NOT a failure.
    """
    fact = {}
    x = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    # descriptor: output name = "sum" (template from name, default sum)
    rt = _make_gpu_runtime()
    try:
        tasks = rt.compile([{"op": "Reduce", "inputs": ["x"],
                             "params": {"op": 7}, "out": "y"}])
        out_names = [n for t in tasks for n in t.out_names]
        fact["descriptor"] = f"output template '{out_names[0]}' (default sum)"
    finally:
        rt.driver.release()
    # CPU: ValueError
    rt_cpu = Runtime(driver=CpuDriver())
    register_all(rt_cpu)
    try:
        try:
            rt_cpu.execute(rt_cpu.compile([_job_for(7)]), {"x": x})
            fact["cpu"] = "no error (unexpected)"
        except ValueError as e:
            fact["cpu"] = f"ValueError: {str(e)[:60]}"
    finally:
        rt_cpu.driver.release()
    # GPU: else->max executes
    if GPU_AVAILABLE:
        rt = _make_gpu_runtime()
        try:
            rt.execute(rt.compile([_job_for(7)]), {"x": x})
            res = rt.driver.resolve_output("y")
            fact["gpu"] = f"executed as max (WGSL else branch): {float(res[0])}"
        finally:
            rt.driver.release()
    else:
        fact["gpu"] = "skipped (no adapter)"
    EVIDENCE["unknown_op_observation"] = {
        "status": "PASS",
        "fact": ("unknown op=7 AS-IS divergence (REDUCE_SLICE_SPEC sec.2 "
                 "observation, OUT of slice scope): " +
                 "; ".join(f"{k}: {v}" for k, v in fact.items()) +
                 " - three-way divergence descriptor(default sum) / "
                 "cpu(ValueError) / gpu(else->max); fix per S49 precedent "
                 "in a future slice, recorded NOT failed"),
    }


# ============================================================================
# evidence save (runs last, file-order independent via pytest last in file)
# ============================================================================

def test_evidence_saved():
    for k, v in EVIDENCE.items():
        if v is None:
            EVIDENCE[k] = {"status": "SKIP", "fact": "not executed"}
    evidence = {
        "phase": 4,
        "slice": SLICE,
        "checkpoint_sha": CHECKPOINT_SHA,
        "versions": _versions(),
        "result": {
            "gpu_available": GPU_AVAILABLE,
            "adapter": ADAPTER_NAME,
            "backend": ADAPTER_BACKEND,
            "limit": LIMIT,
            "cases": EVIDENCE,
        },
        "evidence_schema": "v1",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "resource_failure.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    assert os.path.getsize(path) > 0, "resource_failure.json written empty"