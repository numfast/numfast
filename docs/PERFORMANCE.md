# Performance (measured only)

Hardware: NVIDIA RTX 2060 (Vulkan/WebGPU), seed 42, float32. Full methodology in `evidence/`.

| Workload | Result | Conditions |
|---|---|---|
| Fused SMA20→signal→PnL (1 dispatch) | **12.7x / 28.7x / 31.3x** vs NumPy-f64 (N=100k/1M/4.19M); **4.75x** vs NumPy-f32 at 4.19M | single fold, memory ≈1:1 |
| SearchSpace batch: 1000 SMA params × 1M rows | **8-10.6x**, memory **×10 less** (404 MB vs 4 GB), parity 1000/1000 | 1 upload resident, batches of 100 |
| Multi-symbol backtest Stage A (3 coins × 3 months, real data) | **3.45x**, parity 180/180 | 9 datasets × 1M |
| Full WalkForward 11×29 (319 datasets) | **~3-4x honest** (cold/warm), parity 6380/6380 | mega-fused kernel, 1 upload/dispatch/readback |
| Filter compositions | avg **23x** (N≥100k), up to **334x** on specific consumers | exact L∞=0 |
| TopK (Sort→Gather) | ~**0.9-1.0x** at N≥1M vs numpy argsort | correctness PASS; CPU baseline strong |
| GroupBy (composition) | **0.38-0.56x** vs pandas | characterization; segmented GPU reduce is future work |
| Recursive IIR (Wilder RSI fused walk) | **0.91x** vs CPU-f32 | serial dependency is the bottleneck |

Takeaways:
- GPU wins on fusion-friendly graphs, large batches and parameter sweeps.
- Latency-bound recursive chains and strong C baselines (pandas groupby, np.argsort) may not benefit — characterized, not hidden.
- All numbers reproducible: fixed seeds, committed benchmarks under `benchmarks/`.
