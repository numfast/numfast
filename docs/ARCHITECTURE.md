# Architecture

## Pipeline

```
creation / data
      ↓
Series + expressions (lazy DAG)
      ↓
execution plan (compile once per graph shape)
      ↓
CPU backend (numpy)  or  GPU backend (WebGPU/WGSL kernels)
      ↓
result (explicit materialization boundary: .data())
```

NumFast compiles expression graphs into small GPU kernels (Map, MapBinary, Compare, Gather, Scan, Reduce, Sort, StateKernel, Random...) and executes them with zero host staging between steps. Arrays created by `nf.zeros/nf.index/...` never round-trip through host memory.

## Extensions

Every component is an extension with a manifest. Applications can register their own extensions at runtime (`load_extension`) — adding new operations, kernels, CPU references and implementations without modifying NumFast. See ARCHITECTURE_ROLES.md for the split between NumFast runtime and APP Builder (a separate developer tool; NOT a runtime dependency).

## Backends

- CPU (numpy) — always available, reference semantics
- WebGPU — Python: wgpu; JavaScript: browser navigator.gpu
- CUDA — planned after ABI stabilization

## Design rules

- f32 compute, int32 logical/index domain, explicit materialization boundary
- single canonical execution path (no parallel NumPy shortcuts in hot paths)
- deterministic PCG-u32 random, bit-exact across Python/JS
