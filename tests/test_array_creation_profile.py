"""Creation profile: cold/warm wall ms + throughput vs numpy equivalents.

Ops: zeros / arange / random_uniform / random_normal at N=100k/1M/4M.
cold = first call of the combo in this session; warm = best of REPS;
numpy baseline = best of REPS. Fixed seed 42 everywhere.
"""

import time

import numpy as np
import pytest

from core.Creation._lib.api import arange, random_normal, random_uniform, zeros

NS = [100_000, 1_000_000, 4_000_000]
REPS = 5
SEED = 42
OPS = ("zeros", "arange", "uniform", "normal")
RESULTS = {}


def _ms(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def _best(fn):
    return min(_ms(fn) for _ in range(REPS))


def _pair(name, n):
    if name == "zeros":
        gpu = lambda: zeros(n).free()  # noqa: E731
        cpu = lambda: np.zeros(n, np.float32).tobytes()  # noqa: E731
    elif name == "arange":
        gpu = lambda: arange(0.0, float(n), 1.0).free()  # noqa: E731
        cpu = lambda: np.arange(n, dtype=np.float32).tobytes()  # noqa: E731
    elif name == "uniform":
        gpu = lambda: random_uniform(0.0, 1.0, shape=(n,), seed=SEED).free()  # noqa: E731
        cpu = lambda: np.random.default_rng(SEED).uniform(size=n).astype(np.float32).tobytes()  # noqa: E731
    else:
        gpu = lambda: random_normal(0.0, 1.0, shape=(n,), seed=SEED).free()  # noqa: E731
        cpu = lambda: np.random.default_rng(SEED).normal(size=n).astype(np.float32).tobytes()  # noqa: E731
    return gpu, cpu


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("n", NS)
def test_profile_creation_vs_numpy(name, n):
    gpu, cpu = _pair(name, n)
    cold = _ms(gpu)
    warm = _best(gpu)
    np_ms = _best(cpu)
    thr = n / (warm / 1e3)
    RESULTS[(name, n)] = (cold, warm, np_ms, thr)
    print(f"\n{name:8s} N={n:>9d} | cold={cold:10.3f} ms "
          f"| warm={warm:10.3f} ms | numpy={np_ms:10.3f} ms "
          f"| throughput={thr:.3e} elem/s")
    assert cold > 0.0 and warm > 0.0 and np_ms > 0.0
    assert thr > 1e5, f"throughput collapsed: {thr:.3e} elem/s"


def test_profile_summary_printed():
    assert RESULTS, "parametrized profile tests must run first"
    for (name, n), (cold, warm, np_ms, thr) in sorted(RESULTS.items()):
        print(f"SUMMARY {name:8s} N={n:>9d} cold={cold:9.3f}ms "
              f"warm={warm:9.3f}ms numpy={np_ms:9.3f}ms thr={thr:.3e}")
