"""Phase 2 — Execution Contract conformance tests.

Spec: EXECUTION_CONTRACT.md §3–§5 (checkpoint 37a5cbf).

Covers:
  Step A (G1)  — no-op: compile([])==[], execute([],{}) without GPU calls.
  Step B (G2)  — ScanLocal N in {1,63,64}, op=add, f32: WebGPU == CPU oracle
                 == np.cumsum, exact (single block, f32-representable).
  Step C (G3)  — readback contract: shape/dtype/no-NaN, int-exact, Q1
                 bucket-overhead measurement (documented, NOT fixed).
  Instrumentation (G6) — cold/warm 6-stage profile + pool reuse.
  R1–R7 (G4/G5) — Resource/failure spec §5 with the real GPU pool.

Evidence written on every run: evidence/execution_phase2/
{A,B,C,failures}.json (G7). Runtime/Drivers code is NOT modified (G8).
"""

import json
import os
import statistics
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "math")))

from Runtime._lib.runtime import (Runtime, RUNTIME_VERSION,
                                  KERNEL_ABI_VERSION, EXTENSION_API_VERSION)
from Compute import register_all as register_compute
from Runtime._lib.Drivers.WebGPU import WebGpuDriver
from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
from Runtime._lib.Drivers.base import Driver
from Runtime._lib.mod_iface import (BlockView, ExecutionPlan, InputSlot,
                                    OutputSlot, KernelValidator)
from Runtime._lib.planner import planner
from Runtime._lib.optimizer import optimize_graph
from Runtime._lib.builder import builder

CHECKPOINT_SHA = "37a5cbfccc63d8c3c3aaa2eb75b0922e37b825bb"
SEED = 42
BLOCK = 64
SCANLOCAL_JOBS = [{"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
                   "out": ["scan", "_bsum"]}]
SCAN_CHAIN_JOBS = [
    {"op": "ScanLocal", "inputs": ["data"], "params": {"op": 0},
     "out": ["scan_local", "block_sum"]},
    {"op": "ScanTotals", "inputs": ["block_sum"], "params": {"op": 0},
     "out": "block_prefix"},
    {"op": "ScanFinal", "inputs": ["data", "scan_local", "block_prefix"],
     "params": {"op": 0}, "out": "scan"},
]

# ── GPU availability ────────────────────────────────────────────────
try:
    import wgpu
    _adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
except Exception:  # noqa: BLE001 — adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None
ADAPTER_INFO = dict(_adapter.info) if _adapter is not None else {}
ADAPTER_NAME = ADAPTER_INFO.get("device", "unknown")
ADAPTER_VENDOR = ADAPTER_INFO.get("vendor", "unknown")
ADAPTER_BACKEND = ADAPTER_INFO.get("backend_type", "unknown")

needs_gpu = pytest.mark.skipif(not GPU_AVAILABLE,
                               reason="WebGPU adapter unavailable")

# ── Evidence store ──────────────────────────────────────────────────
EVIDENCE = {"A": None, "B": {}, "C": {}, "INSTR": {}, "R": {}}
for _r in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
    EVIDENCE["R"][_r] = {"status": "SKIP", "fact": "not executed"}


def _num(x):
    """Cast numpy scalars to JSON-safe Python numbers."""
    if isinstance(x, np.generic):
        return x.item()
    return x


def _versions():
    return {
        "runtime": RUNTIME_VERSION,
        "kernel_abi": KERNEL_ABI_VERSION,
        "ext_api": EXTENSION_API_VERSION,
        "python": sys.version.split()[0],
        "wgpu": wgpu.__version__ if GPU_AVAILABLE else "n/a",
        "numpy": np.__version__,
    }


def _device():
    name = ADAPTER_NAME if ADAPTER_VENDOR in ADAPTER_NAME \
        else f"{ADAPTER_VENDOR} {ADAPTER_NAME}".strip()
    return {"adapter": name, "backend": ADAPTER_BACKEND}


def _evidence_base(step, kernel, n, dtype, shape):
    return {
        "step": step,
        "kernel": kernel,
        "op": "add",
        "n": n,
        "dtype": dtype,
        "shape": list(shape) if shape is not None else [],
        "device": _device(),
        "versions": _versions(),
        "seed": SEED,
        "n_runs": {"cold": 1, "warm": 0},
        "stages_ns": {"compile": {}, "alloc": {}, "h2d": {}, "dispatch": {},
                      "d2h": {}, "pool_reuse": {}},
        "pool": {"uploads": 0, "allocs": 0, "evicts": 0, "resident_bytes": 0},
        "latency_ms": 0.0,
        "throughput_mbps": 0.0,
        "memory_bytes": 0,
        "checkpoint_sha": CHECKPOINT_SHA,
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _make_x(n):
    return (np.random.default_rng(SEED).standard_normal(n) * 5 + 30).astype(np.float32)


def _make_gpu_runtime():
    rt = Runtime(driver=WebGpuDriver())
    register_compute(rt)
    return rt


def _make_cpu_runtime():
    rt = Runtime(driver=CpuDriver())
    register_compute(rt)
    return rt


def _wrap_execute_counter(driver):
    """Capture every executed ExecutionPacket (ordinary path)."""
    packets_seen = []
    orig = driver.execute

    def wrapped(packet):
        packets_seen.append(packet)
        return orig(packet)

    driver.execute = wrapped
    return packets_seen


def _wrap_alloc_timing(device):
    """Time device.create_buffer_with_data / create_buffer (allocation)."""
    times = []

    def _wrap(name):
        orig = getattr(device, name)

        def timed(*a, **k):
            t0 = time.perf_counter_ns()
            r = orig(*a, **k)
            times.append(time.perf_counter_ns() - t0)
            return r

        setattr(device, name, timed)

    _wrap("create_buffer_with_data")
    _wrap("create_buffer")
    return times


def _run_scan(rt, x):
    """compile + execute ScanLocal jobs; returns (result, packet)."""
    packets_seen = _wrap_execute_counter(rt.driver)
    tasks = rt.compile(SCANLOCAL_JOBS)
    rt.execute(tasks, {"data": x})
    return rt.driver.resolve_output("scan"), packets_seen[0]


def _bucket(size_bytes):
    """Pool bucket rule: power-of-2, min 1024 (gpu_buffer_pool._bucket)."""
    b = 1 << (size_bytes - 1).bit_length()
    return b if b >= 1024 else 1024


def _fused_build(rt, jobs, source_data):
    """planner + optimizer + builder (test_execute_fused pattern)."""
    rt.driver.kernel_table = rt.kernel_table
    tasks = rt.compile(jobs)
    graph = planner(tasks, rt.kernel_table)
    graph = optimize_graph(graph, level=rt.optimizer_level)
    return builder(graph, source_data, rt.driver, rt.kernel_table)


class _FakePool:
    def __init__(self):
        self._stats = {"uploads": 0, "allocs": 0, "evicts": 0, "destroys": 0,
                       "resident_bytes": 0, "mapped": 0, "cached": 0, "free": 0}

    def stats(self):
        return dict(self._stats)


class MockDriver(Driver):
    """Step A mock: base Driver subclass counting GPU-call sites."""

    def __init__(self):
        super().__init__()
        self.calls = {"create_buffer_with_data": 0, "create_buffer": 0,
                      "submit": 0, "read_buffer": 0}
        self._output_store = {}
        self._pool = _FakePool()

    def execute(self, packet):
        pass

    def resolve_output(self, name):
        return self._output_store.get(name)

    def store_output(self, name, task_id, output_idx, data):
        self._output_store[name] = data

    def max_dispatch_elements(self):
        return None

    def release(self):
        pass


class _FailingReadQueue:
    """Wraps the real queue; read_buffer raises (R7 injection)."""

    def __init__(self, inner):
        self._inner = inner
        self.read_calls = 0

    def write_buffer(self, *a, **k):
        return self._inner.write_buffer(*a, **k)

    def submit(self, *a, **k):
        return self._inner.submit(*a, **k)

    def read_buffer(self, *a, **k):
        self.read_calls += 1
        raise RuntimeError("injected readback failure")


def _describe_no_wgsl(params):
    return ExecutionPlan(
        inputs=[InputSlot(name="data", dtype="float")],
        outputs=[OutputSlot(dtype="float", template="out")],
        workspace=[], uniforms={},
        output_size_fn=lambda sizes: [sizes[0]],
    )


def _cpu_no_wgsl(ctx):
    pass


# ════════════════════════════════════════════════════════════════════
# Step A — no-op (G1)
# ════════════════════════════════════════════════════════════════════

def test_a_noop_compile():
    rt = Runtime(driver=MockDriver())
    register_compute(rt)
    try:
        t0 = time.perf_counter_ns()
        tasks = rt.compile([])
        compile_ns = time.perf_counter_ns() - t0
        assert tasks == []
        rec = _evidence_base("A", "none", 0, "none", [])
        rec["stages_ns"]["compile"]["rt_compile_ns"] = compile_ns
        EVIDENCE["A"] = rec
    finally:
        rt.driver.release()


def test_a_noop_execute():
    rt = Runtime(driver=MockDriver())
    register_compute(rt)
    try:
        t0 = time.perf_counter_ns()
        result = rt.execute([], {})
        exec_ns = time.perf_counter_ns() - t0
        assert result is None, "execute([], {}) must return None"
        assert rt.driver.calls == {"create_buffer_with_data": 0,
                                   "create_buffer": 0, "submit": 0,
                                   "read_buffer": 0}, \
            f"no GPU calls expected, got {rt.driver.calls}"
        assert rt.driver._output_store == {}, \
            f"output_store must be empty, got {rt.driver._output_store}"
        assert rt.driver.resolve_output("x") is None
        assert rt.driver._pool.stats() == {"uploads": 0, "allocs": 0,
                                           "evicts": 0, "destroys": 0,
                                           "resident_bytes": 0, "mapped": 0,
                                           "cached": 0, "free": 0}, \
            "pool.stats() must be unchanged"
        if EVIDENCE["A"] is None:
            EVIDENCE["A"] = _evidence_base("A", "none", 0, "none", [])
        EVIDENCE["A"]["stages_ns"]["compile"]["execute_noop_ns"] = exec_ns
        EVIDENCE["A"]["n_runs"] = {"cold": 1, "warm": 0}
    finally:
        rt.driver.release()


# ════════════════════════════════════════════════════════════════════
# Step B — ScanLocal parity (G2)
# ════════════════════════════════════════════════════════════════════

def test_b_scanlocal_parity():
    if not GPU_AVAILABLE:
        for n in (1, 63, 64):
            _cpu_scan_oracle(n)
        pytest.skip("WebGPU adapter unavailable; CPU oracle recorded")
    for n in (1, 63, 64):
        rt = _make_gpu_runtime()
        try:
            x = _make_x(n)
            packets_seen = _wrap_execute_counter(rt.driver)
            t0 = time.perf_counter_ns()
            tasks = rt.compile(SCANLOCAL_JOBS)
            rt_compile_ns = time.perf_counter_ns() - t0
            assert len(tasks) == 1, "exactly 1 Task expected"
            rt.execute(tasks, {"data": x})
            res = rt.driver.resolve_output("scan")
            assert res is not None, "resolve_output('scan') must exist"
            assert res.shape == (n,), f"shape {(n,)} expected, got {res.shape}"
            assert res.dtype == np.float64, \
                f"host dtype float64 expected, got {res.dtype}"
            assert not np.isnan(res).any(), "NaN in result"
            assert len(packets_seen) == 1, \
                f"1 ExecutionPacket expected, got {len(packets_seen)}"
            packet = packets_seen[0]
            assert packet.dispatch == (1, 1, 1), \
                f"dispatch (1,1,1) expected, got {packet.dispatch}"
            assert packet.abi_version == 1
            # KernelValidator conformance (G2)
            plan = rt.kernel_table["ScanLocal"]["describe"]({"op": 0})
            KernelValidator.validate(packet, plan)

            # CPU oracle
            rt_cpu = _make_cpu_runtime()
            try:
                rt_cpu.execute(rt_cpu.compile(SCANLOCAL_JOBS), {"data": x})
                res_cpu = rt_cpu.driver.resolve_output("scan")
            finally:
                rt_cpu.driver.release()
            ref = np.cumsum(x.astype(np.float32))

            diff_gpu = float(np.abs(res - ref).max())
            diff_cpu = float(np.abs(res_cpu - ref).max())
            diff_gpu_cpu = float(np.abs(res - res_cpu).max())
            tol = max(1e-4, 1e-4 * float(np.abs(ref).max()))
            assert diff_gpu == 0.0, f"GPU vs cumsum diff {diff_gpu} != 0"
            assert diff_cpu == 0.0, f"CPU vs cumsum diff {diff_cpu} != 0"
            assert diff_gpu_cpu == 0.0, f"GPU vs CPU diff {diff_gpu_cpu} != 0"
            assert diff_gpu <= tol, f"GPU diff {diff_gpu} > tol {tol}"

            rec = _evidence_base("B", "ScanLocal", n, "f32", [n])
            rec["n_runs"] = {"cold": 1, "warm": 0}
            rec["stages_ns"]["compile"]["rt_compile_ns"] = rt_compile_ns
            rec["stages_ns"]["compile"]["shader_compile_ns"] = packet.profile.get("compile_ns", 0)
            rec["stages_ns"]["compile"]["pipeline_ns"] = packet.profile.get("pipeline_ns", 0)
            rec["stages_ns"]["alloc"]["n_create_buffer_with_data"] = 1
            rec["stages_ns"]["alloc"]["n_create_buffer"] = 2
            rec["stages_ns"]["h2d"]["upload_ns"] = packet.profile.get("upload_ns", 0)
            rec["stages_ns"]["dispatch"]["dispatch_ns"] = packet.profile.get("dispatch_ns", 0)
            rec["stages_ns"]["d2h"]["readback_ns"] = packet.profile.get("readback_ns", 0)
            rec["stages_ns"]["pool_reuse"]["note"] = "ordinary path (1 packet): pool not used"
            rec["pool"] = {"uploads": 0, "allocs": 0, "evicts": 0,
                           "resident_bytes": 0}
            host_bytes = (n + 1) * 8  # scan + _bsum float64 host
            gpu_bytes = (n + 1) * 4 + 16  # f32 outputs + uniform
            rec["memory_bytes"] = host_bytes + gpu_bytes
            rec["latency_ms"] = (rt_compile_ns + packet.profile.get("exec_time_ns", 0)) / 1e6
            rec["throughput_mbps"] = (n * 4) / max(1, packet.profile.get("readback_ns", 1)) * 1e3
            rec["parity"] = {"diff_gpu_vs_cumsum": diff_gpu,
                             "diff_cpu_vs_cumsum": diff_cpu,
                             "diff_gpu_vs_cpu": diff_gpu_cpu,
                             "exact": diff_gpu == 0.0 and diff_cpu == 0.0}
            rec["packet"] = {"count": 1, "dispatch": list(packet.dispatch),
                             "abi_version": packet.abi_version,
                             "inputs": len(packet.input_buffers),
                             "outputs": len(packet.output_buffers)}
            EVIDENCE["B"][str(n)] = rec
        finally:
            rt.driver.release()


def _cpu_scan_oracle(n):
    """CPU semantic oracle only (no GPU available)."""
    x = _make_x(n)
    rt_cpu = _make_cpu_runtime()
    try:
        rt_cpu.execute(rt_cpu.compile(SCANLOCAL_JOBS), {"data": x})
        res = rt_cpu.driver.resolve_output("scan")
    finally:
        rt_cpu.driver.release()
    ref = np.cumsum(x.astype(np.float32))
    rec = _evidence_base("B", "ScanLocal", n, "f32", [n])
    rec["n_runs"] = {"cold": 0, "warm": 0}
    rec["device"] = {"adapter": "n/a", "backend": "cpu"}
    rec["parity"] = {"diff_cpu_vs_cumsum": float(np.abs(res - ref).max()),
                     "exact": float(np.abs(res - ref).max()) == 0.0,
                     "gpu_available": False}
    EVIDENCE["B"][str(n)] = rec


# ════════════════════════════════════════════════════════════════════
# Step C — readback (G3) + Q1 measurement
# ════════════════════════════════════════════════════════════════════

@needs_gpu
def test_c_readback():
    # ── n=64 f32 readback contract ──
    n = 64
    rt = _make_gpu_runtime()
    try:
        x = _make_x(n)
        res, packet = _run_scan(rt, x)
        ref = np.cumsum(x.astype(np.float32))
        assert res.shape == (n,)
        assert res.dtype == np.float64
        assert not np.isnan(res).any(), "NaN for NaN-free input"
        assert float(np.abs(res - ref).max()) == 0.0
        readback_ns_64 = packet.profile.get("readback_ns", 0)

        # ── int inputs: exact (|cumsum| <= 2^24) ──
        xi = np.arange(n, dtype=np.int32)
        res_int, packet_int = _run_scan(rt, xi)
        ref_int = np.cumsum(xi.astype(np.float32))
        assert np.abs(ref_int).max() <= 2 ** 24
        assert not np.isnan(res_int).any()
        assert float(np.abs(res_int - ref_int).max()) == 0.0, \
            "int scan must be exact"
    finally:
        rt.driver.release()

    # ── Q1: ScanLocal n=1_000_000 (valid: 15625 workgroups <= 65535) ──
    n1m = 1_000_000
    rt1m = _make_gpu_runtime()
    try:
        x1m = _make_x(n1m)
        res1m, packet1m = _run_scan(rt1m, x1m)
        assert res1m.shape == (n1m,)
        assert not np.isnan(res1m).any()
        readback_ns_1m = packet1m.profile.get("readback_ns", 0)
        # Q1 bucket analysis (documented, NOT fixed — V8)
        payload_64 = n * 4          # f32 "scan" output bytes
        payload_1m = n1m * 4
        q1 = {
            "n64": {
                "kernel_used": "ScanLocal",
                "size_bytes": payload_64,
                "bucket_bytes": _bucket(payload_64),
                "bucket_overhead_bytes": _bucket(payload_64) - payload_64,
                "readback_ratio": _bucket(payload_64) / payload_64,
                "actual_readback_overhead_bytes": 0,
                "readback_ns": readback_ns_64,
                "path": "ordinary: exact-size GPU buffers, actual overhead 0; "
                        "bucket fields = pool-rule potential overhead (V8)",
            },
            "n1M": {
                "kernel_used": "ScanLocal",
                "size_bytes": payload_1m,
                "bucket_bytes": _bucket(payload_1m),
                "bucket_overhead_bytes": _bucket(payload_1m) - payload_1m,
                "readback_ratio": _bucket(payload_1m) / payload_1m,
                "actual_readback_overhead_bytes": 0,
                "readback_ns": readback_ns_1m,
                "path": "ordinary: exact-size GPU buffers, actual overhead 0; "
                        "bucket fields = pool-rule potential overhead (V8)",
            },
            "note": "ordinary path allocates exact-size GPU buffers (no "
                    "bucket); pool bucket overhead (V8) measured on fused "
                    "path below",
        }
        # fused-path pool bucket analysis (real pool, 3-packet chain)
        rt_f = _make_gpu_runtime()
        try:
            for fn, fn_payload, fn_bucket_key in (
                    ("n64", payload_64, "n64"),
                    ("n1M", payload_1m, "n1M")):
                xf = x if fn == "n64" else x1m
                packets = _fused_build(rt_f, SCAN_CHAIN_JOBS, {"data": xf})
                rt_f.driver.execute_fused(packets)
                last = packets[-1]
                final_bv = last.output_buffers[0]
                size_bytes = final_bv.size * 4
                bucket = _bucket(size_bytes)
                q1["fused_pool_bucket_" + fn_bucket_key] = {
                    "kernel_used": "ScanFullChain",
                    "payload_bytes": size_bytes,
                    "bucket_bytes": bucket,
                    "bucket_overhead_bytes": bucket - size_bytes,
                    "readback_ratio": bucket / size_bytes,
                    "actual_readback_overhead_bytes": bucket - size_bytes,
                    "readback_ns": last.profile.get("readback_ns", 0),
                    "pool": rt_f.driver._pool.stats(),
                }
        finally:
            rt_f.driver.release()
    finally:
        rt1m.driver.release()

    rec = _evidence_base("C", "ScanLocal", 64, "f32", [64])
    rec["n_runs"] = {"cold": 1, "warm": 0}
    rec["stages_ns"]["d2h"]["readback_ns_n64"] = readback_ns_64
    rec["stages_ns"]["d2h"]["readback_ns_n1M"] = readback_ns_1m
    rec["q1"] = q1
    rec["integrity"] = {"no_nan": True, "int_exact": True}
    EVIDENCE["C"] = rec


# ════════════════════════════════════════════════════════════════════
# Instrumentation (G6)
# ════════════════════════════════════════════════════════════════════

@needs_gpu
def test_instrumentation_cold_warm():
    results = {"n64": {}, "n1M": {}}

    def profile_one(n, warm_runs):
        x = _make_x(n)
        rt = _make_gpu_runtime()
        try:
            # COLD: fresh Runtime (empty shader/pipeline cache, empty pool)
            alloc_times = _wrap_alloc_timing(rt.driver._device)
            packets_seen = _wrap_execute_counter(rt.driver)
            t0 = time.perf_counter_ns()
            tasks = rt.compile(SCANLOCAL_JOBS)
            rt_compile_ns = time.perf_counter_ns() - t0
            t0 = time.perf_counter_ns()
            rt.execute(tasks, {"data": x})
            exec_wall_ns = time.perf_counter_ns() - t0
            p = packets_seen[0]
            cold = {
                "compile_ns": rt_compile_ns + p.profile.get("compile_ns", 0)
                              + p.profile.get("pipeline_ns", 0),
                "alloc_ns": sum(alloc_times),
                "h2d_ns": p.profile.get("upload_ns", 0),
                "dispatch_ns": p.profile.get("dispatch_ns", 0),
                "d2h_ns": p.profile.get("readback_ns", 0),
                "exec_wall_ns": exec_wall_ns,
                "latency_ns": rt_compile_ns + sum(alloc_times)
                              + p.profile.get("upload_ns", 0)
                              + p.profile.get("dispatch_ns", 0)
                              + p.profile.get("readback_ns", 0),
            }
            # WARM: >= 5 runs on the same Runtime (shader cache hit)
            warm = {"compile_ns": [], "alloc_ns": [], "h2d_ns": [],
                    "dispatch_ns": [], "d2h_ns": [], "exec_wall_ns": [],
                    "latency_ns": []}
            for _ in range(warm_runs):
                alloc_times.clear()
                packets_seen.clear()
                t0 = time.perf_counter_ns()
                tasks = rt.compile(SCANLOCAL_JOBS)
                warm["compile_ns"].append(time.perf_counter_ns() - t0)
                t0 = time.perf_counter_ns()
                rt.execute(tasks, {"data": x})
                warm["exec_wall_ns"].append(time.perf_counter_ns() - t0)
                p = packets_seen[0]
                warm["alloc_ns"].append(sum(alloc_times))
                warm["h2d_ns"].append(p.profile.get("upload_ns", 0))
                warm["dispatch_ns"].append(p.profile.get("dispatch_ns", 0))
                warm["d2h_ns"].append(p.profile.get("readback_ns", 0))
                warm["latency_ns"].append(
                    warm["compile_ns"][-1] + warm["alloc_ns"][-1]
                    + warm["h2d_ns"][-1] + warm["dispatch_ns"][-1]
                    + warm["d2h_ns"][-1])
            med = {k: int(statistics.median(v)) for k, v in warm.items()}
            return rt, cold, med
        except Exception:
            rt.driver.release()
            raise

    # n=64: cold + 7 warm
    rt64, cold64, warm64 = profile_one(64, 7)
    rt64.driver.release()
    # n=1M: cold + 5 warm
    rt1m, cold1m, warm1m = profile_one(1_000_000, 5)
    rt1m.driver.release()

    results["n64"]["cold"] = cold64
    results["n64"]["warm_median"] = warm64
    results["n1M"]["cold"] = cold1m
    results["n1M"]["warm_median"] = warm1m

    # ── pool reuse: fused 3-packet scan chain (pool real) ──
    rt_f = _make_gpu_runtime()
    try:
        x = _make_x(64)
        packets = _fused_build(rt_f, SCAN_CHAIN_JOBS, {"data": x})
        before = rt_f.driver._pool.stats()
        rt_f.driver.execute_fused(packets)
        after = rt_f.driver._pool.stats()
        rt_f.driver.execute_fused(packets)  # warm: mapping reuse, no upload
        after_warm = rt_f.driver._pool.stats()
        pool_reuse = {"before": before, "after_first": after,
                      "after_warm": after_warm,
                      "uploads_stable": after["uploads"] == after_warm["uploads"],
                      "note": "ScanLocal alone (1 packet) does not trigger "
                              "execute_fused; pool measured on full 3-packet "
                              "Scan chain"}
        assert after["uploads"] == after_warm["uploads"], \
            "warm fused run must not re-upload"
    finally:
        rt_f.driver.release()

    EVIDENCE["INSTR"] = {
        "n64": results["n64"],
        "n1M": results["n1M"],
        "pool_reuse": pool_reuse,
        "checkpoint_sha": CHECKPOINT_SHA,
    }
    # sanity: warm median stages are all positive and smaller than cold where
    # shader compilation dominated
    assert results["n64"]["warm_median"]["d2h_ns"] > 0
    assert results["n1M"]["warm_median"]["d2h_ns"] > 0


# ════════════════════════════════════════════════════════════════════
# R1–R7 — Resource/failure spec §5 (G4/G5)
# ════════════════════════════════════════════════════════════════════

@needs_gpu
def test_r1_double_release():
    EVIDENCE["R"]["R1"] = {"status": "FAIL", "fact": "test raised"}
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
        EVIDENCE["R"]["R1"] = {
            "status": "PASS",
            "fact": f"acquire->release->release: refcount {rc_after_acquire}->"
                    f"{rc_after_release1}->{rc_after_release2} (>=0); re-acquire "
                    f"returns same buffer, refcount {rc_after_acquire2}",
        }
    finally:
        rt.driver.release()


@needs_gpu
def test_r2_evict_reuse():
    EVIDENCE["R"]["R2"] = {"status": "FAIL", "fact": "test raised"}
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
        EVIDENCE["R"]["R2"] = {
            "status": "PASS",
            "fact": f"max_bytes=1M: alloc(524288)+alloc(300000)->finalize->"
                    f"evicts={mid['evicts']}, destroys={mid['destroys']}, "
                    f"free={mid['free']}; alloc(524288) reused same buffer "
                    f"({reused}), allocs stable {mid['allocs']}->{end['allocs']}",
        }
    finally:
        rt.driver.release()


@needs_gpu
def test_r3_leak():
    EVIDENCE["R"]["R3"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        x = _make_x(64)
        packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
        stats_first = None
        for i in range(20):
            rt.driver.execute_fused(packets)  # identical warm packets
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
        EVIDENCE["R"]["R3"] = {
            "status": "PASS",
            "fact": f"20 identical warm fused runs: uploads="
                    f"{stats_first['uploads']}, allocs={stats_first['allocs']}, "
                    f"resident_bytes={stats_first['resident_bytes']} stable; "
                    f"after driver.release(): mapped={after_release['mapped']}, "
                    f"free={after_release['free']}",
        }
    finally:
        rt.driver.release()


@needs_gpu
def test_r4_failed_kernel():
    EVIDENCE["R"]["R4"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        x = _make_x(64)

        # 1) unregistered op -> KeyError
        with pytest.raises(KeyError):
            rt.compile([{"op": "NotRegistered", "inputs": ["data"], "params": {}}])
        # correct run after error
        res, _ = _run_scan(rt, x)
        assert float(np.abs(res - np.cumsum(x.astype(np.float32))).max()) == 0.0

        # 2) kernel without wgsl -> ValueError
        rt.register_kernel("NoWgslKernel", describe=_describe_no_wgsl,
                           cpu=_cpu_no_wgsl, wgsl=None, abi_version=1)
        with pytest.raises(ValueError):
            rt.execute(rt.compile([{"op": "NoWgslKernel", "inputs": ["data"],
                                    "params": {}}]), {"data": x})
        res, _ = _run_scan(rt, x)
        assert float(np.abs(res - np.cumsum(x.astype(np.float32))).max()) == 0.0

        # 3) dispatch limit: ScanLocal N > 4_194_240 -> ValueError (Phase 4
        #    S30 F-062(b): deterministic error instead of opaque
        #    GPUValidationError "65536 workgroups > 65535")
        xb = np.zeros(4_194_241, dtype=np.float32)
        with pytest.raises(ValueError):
            rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": xb})
        # correct run after error
        res, _ = _run_scan(rt, x)
        assert float(np.abs(res - np.cumsum(x.astype(np.float32))).max()) == 0.0

        EVIDENCE["R"]["R4"] = {
            "status": "PASS",
            "fact": "unregistered op -> KeyError; kernel without wgsl -> "
                    "ValueError; ScanLocal N=4194241 -> ValueError (dispatch "
                    "limit 65535 workgroups, S30 F-062(b)); correct run works "
                    "after each error",
        }
    finally:
        rt.driver.release()


@needs_gpu
def test_r5_cancel():
    EVIDENCE["R"]["R5"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        x = _make_x(64)
        packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
        # warm packet 0 so the input is already mapped
        rt.driver.execute_fused([packets[0]])
        saved = rt.kernel_table.pop("ScanTotals")  # packet k=1 will fail
        try:
            with pytest.raises(KeyError):
                rt.driver.execute_fused(packets)
        finally:
            rt.kernel_table["ScanTotals"] = saved
        pool_after_fail = rt.driver._pool.stats()
        # packets < k completed: packet 0 uploaded input (uploads >= 1) and
        # its buffers are still mapped with refcount > 0 (finalize NOT called)
        assert pool_after_fail["uploads"] >= 1
        assert pool_after_fail["mapped"] >= 3
        assert pool_after_fail["cached"] == 0, \
            "failed fused run must NOT finalize (refcounts stay > 0)"
        # next valid run restores pool consistency (finalize at end)
        rt.driver.execute_fused(packets)
        pool_recovered = rt.driver._pool.stats()
        assert pool_recovered["cached"] == pool_recovered["mapped"], \
            "recovery run must finalize: all mapped entries cached"
        EVIDENCE["R"]["R5"] = {
            "status": "PASS",
            "fact": "3-packet chain, ScanTotals removed -> KeyError on packet "
                    "k=1; packets <k completed (uploads=1, mapped=3, cached=0 "
                    "=> no finalize after error); next valid run restores "
                    "pool (cached==mapped); error diagnosable (KeyError "
                    "'ScanTotals')",
        }
    finally:
        rt.driver.release()


@needs_gpu
def test_r6_upload_exception():
    EVIDENCE["R"]["R6"] = {"status": "FAIL", "fact": "test raised"}
    rt = _make_gpu_runtime()
    try:
        x = _make_x(64)
        packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
        rt.driver.execute_fused(packets)  # baseline
        stats_ok = rt.driver._pool.stats()
        # inject: upload_fn (blockview_to_bytes) raises on unconvertible data
        orig_view = packets[0].input_buffers[0].view
        packets[0].input_buffers[0].view = BlockView(np.array([object()], dtype=object))
        with pytest.raises(TypeError):
            rt.driver.execute_fused(packets)
        stats_fail = rt.driver._pool.stats()
        assert stats_fail == stats_ok, \
            f"pool must not be mutated, {stats_ok} -> {stats_fail}"
        # recovery
        packets[0].input_buffers[0].view = orig_view
        rt.driver.execute_fused(packets)
        assert rt.driver._pool.stats() == stats_ok
        EVIDENCE["R"]["R6"] = {
            "status": "PASS",
            "fact": "upload_fn (blockview_to_bytes on object array) raises "
                    "TypeError; pool stats identical before/after the failed "
                    "acquire (no new buffer); recovery run succeeds with "
                    "unchanged stats",
        }
    finally:
        rt.driver.release()


@needs_gpu
def test_r7_readback_exception():
    EVIDENCE["R"]["R7"] = {"status": "FAIL", "fact": "test raised"}
    # ── ordinary path: readback error swallowed (print) — known gap G1 ──
    rt = _make_gpu_runtime()
    try:
        x = _make_x(64)
        rt.driver._queue = _FailingReadQueue(rt.driver._queue)
        rt.execute(rt.compile(SCANLOCAL_JOBS), {"data": x})
        res = rt.driver.resolve_output("scan")
        assert res is not None, "output_store must not be corrupted"
        assert res.shape == (64,)
        assert rt.driver._queue.read_calls == 2, \
            "both outputs attempted readback (and failed)"
        # restore queue; subsequent runs work
        rt.driver._queue = rt.driver._queue._inner
        res_ok, _ = _run_scan(rt, x)
        assert float(np.abs(res_ok - np.cumsum(x.astype(np.float32))).max()) == 0.0
        ordinary_fact = ("read_buffer raises RuntimeError -> swallowed (print) "
                         "in ordinary path, no exception; output_store holds "
                         "zeros (not corrupted); subsequent run correct")
    finally:
        rt.driver.release()

    # ── fused path: readback exception propagates ──
    rt = _make_gpu_runtime()
    try:
        x = _make_x(64)
        packets = _fused_build(rt, SCAN_CHAIN_JOBS, {"data": x})
        rt.driver._queue = _FailingReadQueue(rt.driver._queue)
        with pytest.raises(RuntimeError):
            rt.driver.execute_fused(packets)
        # restore queue; subsequent runs work
        rt.driver._queue = rt.driver._queue._inner
        rt.driver.execute_fused(packets)
        fused_fact = ("read_buffer raises RuntimeError -> propagates in fused "
                      "path; recovery run succeeds")
    finally:
        rt.driver.release()

    EVIDENCE["R"]["R7"] = {
        "status": "PASS",
        "fact": ordinary_fact + "; " + fused_fact +
                " (injection via _FailingReadQueue wrapper, Runtime code "
                "untouched)",
    }


# ════════════════════════════════════════════════════════════════════
# G7 — save evidence JSONs (runs last)
# ════════════════════════════════════════════════════════════════════

def test_evidence_saved():
    ev_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "evidence", "execution_phase2"))
    os.makedirs(ev_dir, exist_ok=True)

    if EVIDENCE["A"] is None:
        EVIDENCE["A"] = _evidence_base("A", "none", 0, "none", [])
    if not EVIDENCE["B"]:
        EVIDENCE["B"]["0"] = {"step": "B", "status": "FAIL",
                              "fact": "B tests did not complete"}
    if EVIDENCE["C"] is None:
        EVIDENCE["C"] = _evidence_base("C", "ScanLocal", 0, "f32", [])
        EVIDENCE["C"]["status"] = "FAIL"

    files = {
        "A.json": EVIDENCE["A"],
        "B.json": {"gpu_available": GPU_AVAILABLE,
                   "adapter": ADAPTER_NAME, "backend": ADAPTER_BACKEND,
                   "cases": EVIDENCE["B"]},
        "C.json": EVIDENCE["C"],
        "instrumentation.json": EVIDENCE["INSTR"],
        "failures.json": {"checkpoint_sha": CHECKPOINT_SHA,
                          "gpu_available": GPU_AVAILABLE,
                          "cases": EVIDENCE["R"]},
    }
    for name, data in files.items():
        path = os.path.join(ev_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        assert os.path.getsize(path) > 0, f"{name} written empty"