# NumFast

**GPU-first numerical computing framework** — Runtime, Driver, Compute, High-Level API and Agent SDK.

```
pip install numfast
```

## Architecture

```
User (human or AI)              ─┐
    │                            │ Agent SDK
    ▼                            │ AI-agnostic
Agent SDK                        ─┘
    │
    ▼
Session — owns Runtime
    │
    ├──► Runtime — kernel table, compiler, executor
    ├──► Driver — CPU (reference) · WebGPU · CUDA (stub)
    ├──► Operations — scan(), matmul(), fft(), sort(), histogram()
    └──► Compute — 6 GPU algorithms with CPU==GPU conformance
```

## Quick start

```python
import numfast as nf
import numpy as np

# Prefix sum
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = nf.scan(x)
print(y)  # [1. 3. 6. 10. 15.]

# Bitonic sort
x = np.random.random(8)
s = nf.sort(x)

# Matrix multiplication
A = np.ones((4, 4), dtype=np.float64)
B = np.ones((4, 4), dtype=np.float64)
C = nf.matmul(A.flatten(), B.flatten(), M=4, N=4, K=4)
```

## Agent SDK

```python
import numfast as nf

sdk = nf.AgentSDK()

with sdk.session() as session:
    # Run a script
    result = session.run("x = 2 + 2")

    # Compare two arrays
    result = session.diff(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    # Compare CPU vs GPU
    cpu_result = ...
    gpu_result = ...
    session.compare(cpu_result, gpu_result)

    # Run benchmarks
    session.benchmark("matmul_4x4", "matmul(A, B, M=4, N=4, K=4)")
```

## Tools

| Tool | Description |
|------|-------------|
| `scan()` | Inclusive prefix sum (CPU/GPU) |
| `matmul()` | Tiled matrix multiplication |
| `fft()` | Fast Fourier Transform |
| `sort()` | Bitonic sort (power of 2) |
| `histogram()` | Binned histogram with atomics |

## Backends

- **CPU** — reference implementation, float64, always available
- **WebGPU** — GPU via `wgpu-py` (Windows, Linux, macOS)
- CUDA, OpenCL, Metal — driver interface ready (stubs)

Every GPU kernel has CPU == GPU conformance tests.

## Requirements

- Python >= 3.11
- NumPy (CPU)
- `wgpu-py` (optional, for GPU)

## Development

```bash
# Run all tests
pytest

# Run conformance suite
python -m tests.conformance.test_compute
```

## License

AGPL-3.0-only

## Status

Pre-alpha. API is stable (frozen per ADR-005). Active development.
