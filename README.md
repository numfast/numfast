# NumFast

**High-performance numerical computing for Python and JavaScript — GPU acceleration without writing CUDA.**

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

GPU-resident arrays — created on device, no host staging:

```python
x = nf.zeros(1_000_000)
i = nf.index(1_000_000)                              # int32 indices, exact up to 2^31
u = nf.random.uniform(shape=1_000_000, seed=42)      # deterministic, bit-exact across Python/JS
```

## JavaScript / WebGPU

```bash
npm install @numfast/numfast
```

```js
import * as nf from "numfast";
const x = nf.zeros(1_000_000);
const y = nf.arange(0, 10);
console.log(nf.deviceInfo());   // { backend: "...", version: "..." }
```

Same conceptual API as Python. v1 runs on CPU backend; a WebGPU driver is included for browsers (see `examples/`). Browser examples require Chrome/Edge 113+.

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

## Documentation

- [docs/QUICKSTART.md](https://github.com/numfast/numfast/blob/main/docs/QUICKSTART.md) — copy-paste quickstart
- [docs/EXAMPLES.md](https://github.com/numfast/numfast/blob/main/docs/EXAMPLES.md) — E1-E8 with real outputs
- [docs/ARCHITECTURE.md](https://github.com/numfast/numfast/blob/main/docs/ARCHITECTURE.md) — how it works + extensions
- [docs/LIMITATIONS.md](https://github.com/numfast/numfast/blob/main/docs/LIMITATIONS.md) — honest limits
- [docs/PERFORMANCE.md](https://github.com/numfast/numfast/blob/main/docs/PERFORMANCE.md) — measured numbers only

## Status

`1.0.0a1` · Python 3.10+ · Node.js 18+ · License: see LICENSE
