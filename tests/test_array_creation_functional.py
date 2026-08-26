"""Creation functional tests (S200): numpy oracle, edges, dispatch, bridge."""

from contextlib import contextmanager

import numpy as np
import pytest

from core.Creation._lib.api import (arange, full, index, linspace, ones,
                                    repeat, random_integers, random_normal,
                                    random_uniform, tile, zeros)
from core.Creation._lib.bridge import to_series

try:
    import wgpu as _wgpu_mod
    _adapter = _wgpu_mod.gpu.request_adapter_sync(
        power_preference="high-performance")
except Exception:  # noqa: BLE001 - adapter probe must never crash collection
    _adapter = None
GPU_AVAILABLE = _adapter is not None


@contextmanager
def capture_jobs():
    """Collect job dicts handed to Runtime.compile (dispatch instrumentation)."""
    from Runtime.Runtime import _get_runtime
    rt = _get_runtime()
    seen = []
    orig = rt.compile

    def spy(jobs, *a, **k):
        seen.extend(jobs)
        return orig(jobs, *a, **k)

    rt.compile = spy
    try:
        yield seen
    finally:
        rt.compile = orig


def test_zeros_ones_full_exact():
    n = 257
    assert np.array_equal(zeros(n).to_numpy(), np.zeros(n, np.float32))
    assert np.array_equal(ones((n,)).to_numpy(), np.ones(n, np.float32))
    assert np.array_equal(full(n, -3.5).to_numpy(), np.full(n, -3.5, np.float32))


def test_arange_int_and_float():
    got = arange(0, 10, 2).to_numpy()
    assert np.array_equal(got, np.array([0, 2, 4, 6, 8], np.float32))
    assert np.array_equal(arange(5).to_numpy(), np.arange(5, dtype=np.float32))
    got = arange(1.0, 4.0, 0.5).to_numpy()
    assert np.array_equal(got, np.array([1, 1.5, 2, 2.5, 3, 3.5], np.float32))
    got = arange(-6, 6, 2, dtype="int32").to_numpy()
    assert got.dtype == np.int32
    assert np.array_equal(got, np.arange(-6, 6, 2, dtype=np.int32))


def test_linspace_endpoint():
    a, b, num = 2.0, 10.0, 7
    got = linspace(a, b, num).to_numpy()
    np.testing.assert_allclose(got, np.linspace(a, b, num, dtype=np.float32),
                               rtol=1e-6, atol=1e-6)
    assert float(got[0]) == pytest.approx(a, abs=1e-6)
    assert float(got[-1]) == pytest.approx(b, abs=1e-6)
    got = linspace(a, b, num, endpoint=False).to_numpy()
    np.testing.assert_allclose(
        got, np.linspace(a, b, num, endpoint=False, dtype=np.float32),
        rtol=1e-6, atol=1e-6)


def test_n0_empty_no_dispatch():
    with capture_jobs() as seen:
        for arr in (zeros(0), ones((0,)), full(0, 9.0), linspace(0, 1, 0),
                    arange(3, 3),
                    random_uniform(shape=(0,), seed=42),
                    random_normal(shape=(0,), seed=42),
                    random_integers(0, 10, shape=(0,), seed=42)):
            assert len(arr) == 0 and arr.raw.shape == (0,)
            assert arr.gpu_buffer is None
    assert seen == []  # N=0 -> empty host array, zero GPU dispatch


def test_n1():
    assert float(zeros(1).to_numpy()[0]) == 0.0
    assert float(ones(1).to_numpy()[0]) == 1.0
    assert float(full(1, 7.25).to_numpy()[0]) == np.float32(7.25)
    assert float(linspace(4, 8, 1).to_numpy()[0]) == 4.0
    u = random_uniform(2.0, 3.0, shape=(1,), seed=42).to_numpy()
    assert 2.0 <= float(u[0]) < 3.0
    z = random_normal(0.0, 1.0, shape=(1,), seed=42).to_numpy()
    assert np.isfinite(z[0])
    i = random_integers(-5, 5, shape=(1,), seed=42).to_numpy()
    assert -5 <= int(i[0]) < 5


def test_dtype_int32():
    for arr in (zeros(8, dtype="int32"), ones(8, dtype="int32"),
                full(8, 5, dtype="int32"), arange(8, dtype="int32"),
                random_integers(0, 100, shape=(8,), seed=42)):
        assert arr.dtype == "int32" and arr.raw.dtype == np.int32
    assert np.array_equal(arange(8, dtype="int32").to_numpy(),
                          np.arange(8, dtype=np.int32))


def test_determinism_bits():
    kw = dict(shape=999, seed=42)
    assert (random_uniform(**kw).raw.tobytes()
            == random_uniform(**kw).raw.tobytes())
    assert (random_normal(**kw).raw.tobytes()
            == random_normal(**kw).raw.tobytes())
    ikw = dict(shape=999, seed=42)
    assert (random_integers(0, 10 ** 6, **ikw).raw.tobytes()
            == random_integers(0, 10 ** 6, **ikw).raw.tobytes())


def test_tile_pattern():
    pat = [1.0, 2.0, 3.0]
    got = tile(pat, 9).to_numpy()
    exp = np.tile(np.asarray(pat, np.float32), 3)
    assert got.shape == (9,)
    assert np.array_equal(got, exp)
    # n не кратен k: усечение по out[i] = pat[i % k]
    got = tile(pat, 7).to_numpy()
    assert np.array_equal(got, np.asarray([1, 2, 3, 1, 2, 3, 1], np.float32))


def test_repeat_pattern():
    pat = [1.0, 2.0, 3.0]
    got = repeat(pat, repeats=2).to_numpy()
    assert got.shape == (6,)
    assert np.array_equal(got, np.array([1, 1, 2, 2, 3, 3], np.float32))
    assert np.array_equal(got, np.repeat(np.asarray(pat, np.float32), 2))


def test_tile_long_pattern():
    # pattern = GPU-resident вход, любой k (лимит 64 удалён)
    pat = (np.arange(3000, dtype=np.float32) % np.float32(17.0)) - np.float32(8.0)
    got = tile(pat, 9000).to_numpy()
    exp = np.tile(pat, 3)
    assert got.shape == (9000,)
    assert np.array_equal(got, exp)


def test_repeat_r1_identity():
    pat = [3.5, -2.25, 1000.5, 0.125, 7.0]
    got = repeat(pat, repeats=1).to_numpy()
    assert got.shape == (5,)
    assert np.array_equal(got, np.asarray(pat, np.float32))


def test_user_case():
    # пользовательский случай: 1 + (i % 3) == tile([1,2,3], n)
    n = 100
    via_tile = tile([1.0, 2.0, 3.0], n).to_numpy()
    base = arange(0, n).to_numpy()
    via_formula = (base % 3).astype(np.float32) + np.float32(1.0)
    assert np.array_equal(via_tile, via_formula)


def test_series_bridge():
    base = arange(0, 16)
    s = to_series(base)
    assert len(s) == 16
    expr = s * 2 + 1  # LazyExpr
    out = np.asarray(expr.data(), dtype=np.float64)
    exp = base.raw.astype(np.float64) * 2 + 1
    np.testing.assert_allclose(out, exp, rtol=1e-6, atol=1e-6)


# ============================================================================
# nf.index(n) — индекс как виртуальная Series (IndexKernel mode=1 arange)
# ============================================================================

def test_index_arange():
    # default dtype is NOW int32 (owner decision: real integer semantics)
    got = index(10)
    assert got.dtype == "int32" and got.raw.dtype == np.int32
    assert np.array_equal(got.to_numpy(), np.arange(10, dtype=np.int32))
    # legacy float path available explicitly
    got_f = index(10, dtype="float32").to_numpy()
    assert got_f.dtype == np.float32
    assert np.array_equal(got_f, np.arange(10))
    # n=0 -> пустой массив без dispatch (наследует arange-контракт)
    assert len(index(0)) == 0


def test_index_mod_tile():
    idx = index(9)
    s = to_series(idx)
    r = np.asarray((s % 3).compute(), dtype=np.float64)
    assert np.array_equal(r, np.tile([0, 1, 2], 3).astype(np.float64))


def test_index_squares():
    s = to_series(index(10))
    r = np.asarray((s * s).compute(), dtype=np.float64)
    assert np.array_equal(r[:5], [0, 1, 4, 9, 16])
    assert np.array_equal(r, (np.arange(10) ** 2).astype(np.float64))


@pytest.mark.skip(reason="mask/select ops not available in Series/LazyExpr v1")
def test_index_alternating():
    # ones-like чередование [0,1,0,1,...] требует mask/select — нет в v1
    pass


def test_mapbinary_mod_op():
    """MapBinary op=6 (mod) напрямую: CPU и GPU vs numpy fmod parity.

    Integer-valued f32 данные => trunc(a/b), a - trunc*b точны в f32
    (все значения <= 2^24) => exact equality с f64 oracle.
    Дробные данные => контрактный допуск max(1e-6*peak_a, 1e-6).
    """
    from Runtime._lib.runtime import Runtime
    from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
    from Compute import register_all

    rng = np.random.default_rng(42)
    # integer-valued: точный домен (Sterbenz: a < 2^24, trunc*b <= a)
    a_int = rng.permutation(np.arange(257, dtype=np.float32)).astype(np.float32)
    b_int = rng.integers(1, 17, 257).astype(np.float32)
    ref_int = np.fmod(a_int.astype(np.float64), b_int.astype(np.float64))
    # fractional: допуск-домен
    a_f = (rng.uniform(0, 100, 100)).astype(np.float32)
    b_f = rng.integers(1, 17, 100).astype(np.float32)
    ref_f = np.fmod(a_f.astype(np.float64), b_f.astype(np.float64))

    jobs_int = [{"op": "MapBinary", "inputs": ["a", "b"],
                 "params": {"op": 6}, "out": "y"}]
    jobs_f = [{"op": "MapBinary", "inputs": ["a", "b"],
               "params": {"op": 6}, "out": "y"}]

    def run(cls, data):
        rt = Runtime(driver=cls())
        register_all(rt)
        try:
            rt.execute(rt.compile(jobs_int if data is a_int else jobs_f),
                       {"a": data, "b": b_int if data is a_int else b_f})
            return np.asarray(rt.driver.resolve_output("y"), np.float64)
        finally:
            rt.driver.release()

    res_cpu_int = run(CpuDriver, a_int)
    assert res_cpu_int.shape == a_int.shape
    assert np.array_equal(res_cpu_int, ref_int), \
        float(np.abs(res_cpu_int - ref_int).max())
    res_cpu_f = run(CpuDriver, a_f)
    tol_f = max(1e-6 * float(np.abs(a_f).max()), 1e-6)
    assert float(np.abs(res_cpu_f - ref_f).max()) <= tol_f

    if not GPU_AVAILABLE:
        pytest.skip("WebGPU adapter unavailable")
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver

    res_gpu_int = run(WebGpuDriver, a_int)
    assert np.array_equal(res_gpu_int, ref_int), \
        float(np.abs(res_gpu_int - ref_int).max())
    res_gpu_f = run(WebGpuDriver, a_f)
    assert float(np.abs(res_gpu_f - ref_f).max()) <= tol_f


# ============================================================================
# int32 index semantics (owner decision): exact beyond 2^24, i32 domain ops
# ============================================================================

def test_index_int32_exact_20m():
    """nf.index(20M): values EXACT beyond 2^24 (f32 limit does not apply)."""
    n = 20_000_000
    got = index(n)
    assert got.dtype == "int32" and got.raw.dtype == np.int32
    tail = got.to_numpy()
    assert len(tail) == n
    assert int(tail[-1]) == n - 1
    assert int(tail[2 ** 24]) == 2 ** 24
    assert int(tail[2 ** 24 + 1]) == 2 ** 24 + 1
    assert int(tail[16_777_217]) == 16_777_217
    # proof the legacy f32 path cannot represent these values:
    assert int(np.float32(n - 1)) != n - 1


def test_index_mod_int_domain():
    """(idx % 3) stays in the int32 domain, correct for large i (> 2^24).

    Direct executor path: chained IndexKernelI32 -> MapBinaryI32 jobs.
    """
    from Runtime._lib.runtime import Runtime
    from Runtime._lib.Drivers.CPU._lib.cpu_driver import CpuDriver
    from Compute import register_all

    n = 16_800_000  # > 2**24, so f32 indices would already be lossy
    jobs = [
        {"op": "IndexKernelI32", "inputs": [],
         "params": {"n": n, "mode": 1, "dtype": "int32"}, "out": "idx"},
        {"op": "MapBinaryI32", "inputs": ["idx"],
         "params": {"op": 6, "dtype": "int32", "scalar_b": 3,
                    "use_scalar_b": 1}, "out": "m"},
    ]
    rt = Runtime(driver=CpuDriver())
    register_all(rt)
    try:
        rt.execute(rt.compile(jobs), {})
        raw = rt.driver.resolve_output("m")
    finally:
        rt.driver.release()
    m = np.asarray(raw, dtype=np.int32)
    assert m.dtype == np.int32
    exp = np.arange(n, dtype=np.int64) % 3
    for i in (n - 1, 2 ** 24, 2 ** 24 + 1, 2 ** 24 + 7, 12345):
        assert int(m[i]) == int(exp[i]), f"i={i}"

    if not GPU_AVAILABLE:
        return
    # GPU parity (small n, bit-exact i32 path end-to-end)
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver

    ng = 100_000
    jobs_g = [
        {"op": "IndexKernelI32", "inputs": [],
         "params": {"n": ng, "mode": 1, "dtype": "int32"}, "out": "idx"},
        {"op": "MapBinaryI32", "inputs": ["idx"],
         "params": {"op": 6, "dtype": "int32", "scalar_b": 3,
                    "use_scalar_b": 1}, "out": "m"},
    ]
    rtg = Runtime(driver=WebGpuDriver())
    register_all(rtg)
    try:
        rtg.execute(rtg.compile(jobs_g), {})
        mg_raw = rtg.driver.resolve_output("m")
    finally:
        rtg.driver.release()
    mg = np.asarray(mg_raw, dtype=np.int64)
    assert np.array_equal(mg, (np.arange(ng, dtype=np.int64) % 3))


def test_promotion_sin():
    """sin(idx*0.01) via Series auto-promote int32 -> f32 (documented loss)."""
    from Series._lib.math_ops import sin

    s = to_series(index(1000))
    got = np.asarray(sin(s * 0.01).compute(), dtype=np.float64)
    exp = np.sin(np.arange(1000, dtype=np.float64) * 0.01)
    assert got.shape == (1000,)
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-6)


def test_index_upper_bound():
    """n > 2**31-1 is outside the int32 domain -> ValueError."""
    with pytest.raises(ValueError, match="int32"):
        index(2 ** 31)
    with pytest.raises(ValueError, match="int32"):
        index(2 ** 31 + 123)
