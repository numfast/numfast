"""Creation resource/failure tests R1-R7 (ownership, limits, determinism)."""

import numpy as np
import pytest

from core.Creation._lib.api import (arange, full, linspace, ones,
                                    random_integers, random_normal,
                                    random_uniform, zeros)

OVER = 4_194_241  # первый N за лимитом single-dispatch 4_194_240


def test_r1_free_refcount():
    a = zeros(1024)
    assert a.gpu_buffer is None and not a._released
    a.free()
    a.free()  # повторный free идемпотентен
    assert a._released
    assert a.raw is None and a.gpu_buffer is None
    assert len(a) == 0
    assert a.to_numpy().size == 0


def test_r2_reuse_second_call():
    a, b = ones(4096), ones(4096)
    assert a is not b
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    u1 = random_uniform(-1.0, 1.0, shape=(512,), seed=42)
    u2 = random_uniform(-1.0, 1.0, shape=(512,), seed=42)
    assert u1 is not u2
    assert u1.raw.tobytes() == u2.raw.tobytes()
    for x in (a, b, u1, u2):
        x.free()


def test_r3_acquire_release_cycles_x20():
    for i in range(20):
        c = random_uniform(-1.0, 1.0, shape=(256,), seed=i)
        assert np.isfinite(c.to_numpy()).all()
        c.free()
    d = zeros(64)  # пул/драйвер живы после цикла acquire/release
    assert np.array_equal(d.to_numpy(), np.zeros(64, np.float32))
    d.free()


_OVER_CALLS = (
    lambda: zeros(OVER),
    lambda: ones(OVER),
    lambda: full(OVER, 1.0),
    lambda: arange(0, OVER),
    lambda: linspace(0, 1, OVER),
    lambda: random_uniform(shape=(OVER,), seed=42),
    lambda: random_normal(shape=(OVER,), seed=42),
    lambda: random_integers(0, 10, shape=(OVER,), seed=42),
)


@pytest.mark.parametrize("idx", range(len(_OVER_CALLS)))
def test_r4_dispatch_limit_valueerror(idx):
    with pytest.raises(ValueError):
        _OVER_CALLS[idx]()


def test_r5_big_n_ok():
    n = 4_000_000  # < 4_194_240
    z = zeros(n)
    assert len(z) == n and not z.to_numpy().any()
    z.free()
    u = random_uniform(0.0, 1.0, shape=(n,), seed=42)
    v = u.to_numpy()
    assert v.shape == (n,) and (v >= 0).all() and (v < 1).all()
    u.free()


def test_r6_n0_n1_all_ops():
    for arr in (zeros(0), ones((0,)), full(0, 1.0), arange(5, 5),
                linspace(0, 1, 0), random_uniform(shape=(0,), seed=42),
                random_normal(shape=(0,), seed=42),
                random_integers(0, 10, shape=(0,), seed=42)):
        assert len(arr) == 0
        arr.free()
    vals = (zeros(1).to_numpy(), ones(1).to_numpy(),
            full(1, 2.5).to_numpy(), arange(0, 1).to_numpy(),
            linspace(3, 9, 1).to_numpy())
    assert float(vals[4][0]) == 3.0
    i1 = random_integers(-2, 2, shape=(1,), seed=42).to_numpy()
    assert -2 <= int(i1[0]) < 2


def test_r7_determinism_after_errors():
    ref = random_uniform(-5.0, 5.0, shape=(128,), seed=42).raw.tobytes()
    with pytest.raises(ValueError):
        zeros(OVER)
    with pytest.raises(ValueError):
        arange(0, 1, 0)  # step == 0
    with pytest.raises(ValueError):
        random_uniform(shape=(4,), seed=None)  # seed обязателен
    with pytest.raises(NotImplementedError):
        zeros((2, 2))  # N-D planned
    again = random_uniform(-5.0, 5.0, shape=(128,), seed=42)
    assert again.raw.tobytes() == ref
    again.free()
