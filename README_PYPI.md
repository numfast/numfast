# NumFast

**High-performance numerical computing for Python — GPU acceleration without writing CUDA.**

```bash
pip install numfast
```

```python
import numfast as nf

s = nf.series([1.0, 2.0, 3.0, 4.0])

y = (s * 2 + 1).compute()
print(y.data())          # [3.0, 5.0, 7.0, 9.0]

peaks = (s > 2).filter(s)   # comparison -> mask -> gather
print(peaks.data())         # [3.0, 4.0]

print(nf.mean(y))           # 6.0
```

## Creation — GPU-resident arrays, no host staging

```python
import numfast as nf

x = nf.zeros(1_000_000)                              # float32 zeros on device
o = nf.ones(500_000, dtype="int32")
g = nf.arange(0, 10, 2)                              # [0, 2, 4, 6, 8]
i = nf.index(1_000_000)                              # int32 0..n-1, exact up to 2^31
t = nf.tile([1, 2, 3], 9)                            # [1,2,3,1,2,3,...]
u = nf.random.uniform(shape=1_000_000, seed=42)      # deterministic, bit-exact
n = nf.random.normal(loc=0.0, scale=1.0, shape=100_000, seed=42)
r = nf.random.integers(low=1, high=7, shape=1_000, seed=42, dtype="int32")
```

Same `seed` + same index = same value — bit-identical across runs.

## Series + filter + stats — real outputs

Derived from `docs/EXAMPLES.md` (E1-E3, E5, E7), verified against the public API:

```python
import numfast as nf

# E1. Million-element arithmetic
s = nf.series(list(range(1_000_000)))
y = (s * 2 + 1).compute()
print(len(y), y.data()[:3])               # 1000000 [1.0, 3.0, 5.0]

# E3. Deterministic random
a = nf.random.uniform(shape=5, seed=42).to_numpy()
b = nf.random.uniform(shape=5, seed=42).to_numpy()
print((a == b).all())                     # True — same seed, same stream

# E5. Filter (comparison -> mask -> gather)
s = nf.series([1.0, 5.0, 2.0, 8.0, 3.0])
big = (s > 4).filter(s)
print(big.data())                         # [5.0, 8.0]

# E7. Statistics
s = nf.series([1.0, 2.0, 3.0])
print(nf.mean(s), nf.std(s), nf.var(s), nf.total(s), nf.count(s))
# 2.0 0.816... 0.666... 6.0 3

# Chained expressions are lazy DAGs, executed once on .compute()
s = nf.series([1.0, 2.0, 3.0, 4.0])
y = (s * 2 + 1).compute()
print(y.data())                           # [3.0, 5.0, 7.0, 9.0]
print(nf.mean(y), nf.minimum(y), nf.maximum(y))  # 6.0 3.0 9.0
```

Note: the first `.compute()` call includes pipeline warm-up (seconds on some
machines); subsequent calls reuse compiled plans and are fast.

## GroupBy / TopK + diagnostics

```python
import numfast as nf

# Top-K — largest k elements
s = nf.series([5.0, 1.0, 8.0])
print(nf.topk(s, 2))                      # [8. 5.]

# GroupBy — keys/vals are host arrays via .data()
keys = nf.series([0.0, 1.0, 0.0, 1.0])
vals = nf.series([1.0, 3.0, 2.0, 4.0])
res = nf.groupby(keys.data(), vals.data())
# res["keys"], res["count"], res["sum"], res["min"], res["max"], res["mean"]

# Device & profiling
print(nf.device_info())                   # {'backend': ..., 'platform': ..., 'version': ...}
nf.set_backend("cpu")                     # or "gpu" when available

with nf.profile():
    y = (s * 2 + 1).compute()             # stage breakdown printed (ms per stage)
```

Arrays above 4_194_240 elements raise `ValueError("chunking planned")` in v1.

## What's inside

| Group | API |
|---|---|
| Creation | `zeros ones full arange linspace index tile repeat` |
| Random | `random.uniform random.normal random.integers` (explicit `seed`) |
| Series | lazy expressions `+ - * / % **`, comparisons `< <= > >= == !=`, `.filter(mask)`, `.compute()` |
| Math | `sin cos tan exp log sqrt abs square neg` |
| Statistics | `mean std var total minimum maximum count` |
| Tabular | `topk(s, k)`, `groupby(keys, vals)` |
| Operations | `scan sort histogram matmul fft` |
| Diagnostics | `device_info() set_backend(...) profile()` |

## When NumFast is fast

Measured characterizations (RTX 2060, details in docs/PERFORMANCE.md):
- fusion-friendly pipelines and parameter batches: **8-31x** vs NumPy
- elementwise Filter compositions: up to **334x**
- recursive per-element chains (e.g. Wilder RSI): CPU may win — we say so honestly

> Honest limits (see docs/LIMITATIONS.md): rank-1 only in v1, no int64,
> log(0) raises, gather with int32 index not yet on GPU path, first call
> pays warm-up cost, serial IIR dependencies are latency-bound.

## Documentation

- [docs/QUICKSTART.md](https://github.com/numfast/numfast/blob/main/docs/QUICKSTART.md) — copy-paste quickstart
- [docs/EXAMPLES.md](https://github.com/numfast/numfast/blob/main/docs/EXAMPLES.md) — E1-E8 with real outputs
- [docs/ARCHITECTURE.md](https://github.com/numfast/numfast/blob/main/docs/ARCHITECTURE.md) — how it works + extensions
- [docs/LIMITATIONS.md](https://github.com/numfast/numfast/blob/main/docs/LIMITATIONS.md) — honest limits
- [docs/PERFORMANCE.md](https://github.com/numfast/numfast/blob/main/docs/PERFORMANCE.md) — measured numbers only

## Status

`1.0.0a1` · Python 3.10+ · License: see LICENSE
