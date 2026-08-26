"""H2D criterion: creation performs ZERO host->device uploads.

Primary metric: GpuBufferPool.stats()["uploads"] delta around each op (N=1M).
Fallback (documented): if pool API is unreachable from the Runtime driver,
the check degrades to captured jobs having inputs=[] -- no source buffers
exist at all, hence no host->device staging path.
Over-limit: N > 4_194_240 -> ValueError (chunking S209 planned).
"""

from contextlib import contextmanager

import pytest

from core.Creation._lib.api import (arange, full, linspace, ones,
                                    random_integers, random_normal,
                                    random_uniform, zeros)

N_BIG = 1_000_000
LIMIT = 4_194_240
OPS8 = ("zeros", "ones", "full", "arange", "linspace",
        "uniform", "normal", "integers")


def _runtime():
    from Runtime.Runtime import _get_runtime
    return _get_runtime()


def _pool_stats(rt):
    pool = getattr(rt.driver, "_pool", None)
    return pool.stats() if pool is not None else None


@contextmanager
def capture_jobs(rt):
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


def _run_op(name, n):
    if name == "zeros":
        return zeros(n)
    if name == "ones":
        return ones(n)
    if name == "full":
        return full(n, 3.5)
    if name == "arange":
        return arange(0, float(n), 1.0)
    if name == "linspace":
        return linspace(0.0, 1.0, n)
    if name == "uniform":
        return random_uniform(-1.0, 1.0, shape=(n,), seed=42)
    if name == "normal":
        return random_normal(0.0, 1.0, shape=(n,), seed=42)
    return random_integers(0, 1000, shape=(n,), seed=42)


@pytest.mark.parametrize("name", OPS8)
def test_h2d_zero_uploads_per_op(name):
    rt = _runtime()
    before = _pool_stats(rt)
    with capture_jobs(rt) as seen:
        out = _run_op(name, N_BIG)
    after = _pool_stats(rt)
    assert len(out) == N_BIG
    assert out.gpu_buffer is None  # no staging buffer attached to result
    jobs = [j for j in seen if j.get("op") in ("IndexKernel", "RandomKernel")]
    assert jobs, f"no dispatch observed for {name}"
    assert all(j.get("inputs") == [] for j in jobs), "creation must have 0 inputs"
    if before is not None:
        assert after["uploads"] - before["uploads"] == 0


def test_h2d_over_limit_valueerror():
    for call in (lambda: zeros(LIMIT + 1),
                 lambda: linspace(0, 1, LIMIT + 1),
                 lambda: random_normal(shape=(LIMIT + 1,), seed=42)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("N > 4_194_240 must raise ValueError")


def test_h2d_boundary_limit_allowed():
    out = zeros(LIMIT)
    assert len(out) == LIMIT
    out.free()
