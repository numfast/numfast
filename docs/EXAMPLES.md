# Examples (E1-E8)

All snippets verified against the public API. Real outputs shown.

## E1. Million-element arithmetic

```python
import numfast as nf
s = nf.series(list(range(1_000_000)))     # build once
y = (s * 2 + 1).compute()
print(len(y), y.data()[:3])               # 1000000 [1.0, 3.0, 5.0]
```

First expression call includes pipeline warm-up (seconds); subsequent calls are fast.

## E2. Index cycle (tile pattern)

```python
pat = nf.tile([1, 2, 3], 9)               # 1 2 3 1 2 3 ...
i = nf.index(9)
s = nf.series((i.to_numpy() % 3 + 1).tolist())
print(s.data())                           # [1.0, 2.0, 3.0, 1.0, ...]
```

Note: a concrete series is already materialized -- use `.data()`. `.compute()` exists on lazy expressions only.

## E3. Deterministic random

```python
a = nf.random.uniform(shape=5, seed=42).to_numpy()
b = nf.random.uniform(shape=5, seed=42).to_numpy()
print((a == b).all())                     # True — same seed, same stream
```

The same seed produces bit-identical streams in Python and JavaScript (PCG-u32 counter-based).

## E4. Harmonics

```python
import math
s = nf.series([0.0, 1.0, 2.0])
y = (nf.math.sin(s) + 0.5 * nf.math.cos(s * 2)).compute()
print(y.data())                           # [0.5, 0.63339752..., 0.58247554...]
```

## E5. Filter

```python
s = nf.series([1.0, 5.0, 2.0, 8.0, 3.0])
big = (s > 4).filter(s)
print(big.data())                         # [5.0, 8.0]
```

## E6. Top-K

```python
s = nf.series([5.0, 1.0, 8.0])
print(nf.topk(s, 2))                      # [8. 5.]
```

## E7. Statistics

```python
s = nf.series([1.0, 2.0, 3.0])
print(nf.mean(s), nf.std(s), nf.var(s), nf.total(s), nf.count(s))
# 2.0 0.816... 0.666... 6.0 3
```

## E8. Large GPU-resident pipeline

```python
x = nf.ones(1_000_000)                           # GPU-resident creation
s = nf.series(x.to_numpy().tolist())             # bridge into expressions
y = ((s + 1) * 2).compute()
mask = y > 2
kept = mask.filter(y)
print(len(y), len(kept))                         # 1000000 1000000
```

Note: with `nf.zeros` the pipeline yields `y = 2.0` everywhere, so the mask `y > 2` selects nothing (`len(kept) == 0`) -- composition is correct, data choice matters.

Note: arrays above 4_194_240 elements return "chunking planned" ValueError in v1.
