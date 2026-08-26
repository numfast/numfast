# Limitations (honest)

- **GroupBy in JavaScript**: deferred to v1.1 (Python has it).
- **Gather with int32 index buffers**: not yet supported on the GPU path (driver limitation); CPU oracle is exact. Tracked as D-5.
- **Multi-output kernels**: partially supported in the JS executor.
- **Arrays > 4_194_240 elements**: creation raises "chunking planned" ValueError in v1 (both languages).
- **Rank-1 shapes only** in v1; N-D planned.
- **int64 dtype**: not supported (int32 logical domain; WRAP rules documented for index()).
- **Browser smoke**: WebGPU examples require Chrome/Edge 113+; other browsers fall back to CPU. Automated browser testing is not part of CI yet.
- **Recursive per-element chains** (Wilder RSI-style IIR): GPU may not speed these up; CPU wins on latency-bound serial dependencies (measured).
- **log(0)** raises ValueError (no masking).
- **First expression call** includes pipeline warm-up (seconds on some machines); subsequent calls reuse compiled plans.
