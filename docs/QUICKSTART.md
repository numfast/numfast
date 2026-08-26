# Quickstart

## Install (Python)

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (.venv/bin/activate on Linux/macOS)
pip install numfast           # or: pip install -e . from repo root
```

Requires: Python 3.10+, numpy, wgpu (installed automatically).

## Import

```python
import numfast as nf
print(nf.__version__)         # 1.0.0a1
```

## Create arrays (GPU-resident, no host staging)

```python
x = nf.zeros(1_000_000)                    # float32 zeros
o = nf.ones(500_000, dtype="int32")
f = nf.full(100, 3.14)
g = nf.arange(0, 10, 2)                    # [0, 2, 4, 6, 8]
l = nf.linspace(0, 1, 5)                   # [0, 0.25, 0.5, 0.75, 1]
i = nf.index(1_000_000)                    # int32 indices 0..n-1
t = nf.tile([1, 2, 3], 9)                  # [1,2,3,1,2,3,...]
r = nf.repeat([1, 2], 4)                   # [1,1,1,1,2,2,2,2]
```

`x.to_numpy()` gives a numpy view/copy when you need host data.

## Deterministic random (seed required)

```python
u = nf.random.uniform(low=0.0, high=1.0, shape=1_000_000, seed=42)
n = nf.random.normal(loc=0.0, scale=1.0, shape=100_000, seed=42)
d = nf.random.integers(low=1, high=7, shape=1_000, seed=42, dtype="int32")
```

Same seed + same index => same value, bit-exact across Python and JavaScript.

## Series and expressions

```python
s = nf.series([1.0, 2.0, 3.0, 4.0])

y = (s * 2 + 1).compute()      # lazy DAG, executed once
print(y.data())                # [3.0, 5.0, 7.0, 9.0]

z = nf.math.sin(s).compute()   # sin, cos, tan, exp, log, sqrt, abs, square, neg
w = (s ** 2).compute()
m = (s % 2).compute()          # remainder
```

Note: expressions are lazy; `.data()` is the explicit materialization boundary.

## Comparisons, filter, topk

```python
s = nf.series([1.0, 2.0, 3.0, 4.0])

mask = s > 2                       # [0, 0, 1, 1]
peaks = mask.filter(s)             # gather: [3.0, 4.0]
top = nf.topk(s, 2)                # largest k: [4.0, 3.0]
```

## GroupBy

```python
keys = nf.series([0.0, 1.0, 0.0, 1.0])
vals = nf.series([1.0, 3.0, 2.0, 4.0])
res = nf.groupby(keys.data(), vals.data())
# res["keys"], res["count"], res["sum"], res["min"], res["max"], res["mean"]
```

## Statistics and operations

```python
print(nf.mean(s), nf.std(s), nf.total(s), nf.minimum(s), nf.maximum(s), nf.count(s))
c = nf.scan(s)          # prefix sums
srt = nf.sort(s)
h = nf.histogram(s, bins=4)
```

## Diagnostics and profiling

```python
print(nf.device_info())    # {'backend': ..., 'platform': ..., 'version': ...}
nf.set_backend("cpu")      # or "gpu" when available

with nf.profile():
    y = (s * 2 + 1).compute()
# stage breakdown printed
```

## JavaScript

See `examples/E1_creation.html` … `E8_device.html` (run `npx serve .` from repo root, open in Chrome/Edge 113+ for WebGPU; other browsers fall back to CPU automatically).
