# RESULTS — already measured, packaged verbatim

Seed 42 everywhere. All figures copied from the source artifacts (see `suite.toml`).
`run_all.py` does not remeasure; it only validates + prints.

## H2O GROUPBY @ 10M (`bench_h2o_hll.json`, + `bench_join_distinct_hll.json`)

| Query | Correctness | HLL ms | A (oracle) ms | Note |
|---|---|---|---|---|
| Q1 | EXACT (diff 0 vs A + GOLD) | 31.3 | 30.7 | 2% delta = noise; claim = parity |
| Q2 | EXACT (multiset-compared) | 2692.2 | 1531.9 | values exact, HLL slower (hash vs pack labels) |
| Q3–Q5 | BLOCKED_BY_MVP (0 jobs) | — | — | not measured, not claimed |
| Join | HLL stub (0 jobs) / A oracle ok | — | inner 3.56 / left 2.31 | not claimed |

Ingest: CSV path in artifact (`csv_load_ms` stored there).

## ClickBench Q1–Q43 @ ~1M (`bench_clickbench_hll.json`, rows=999978)

Ingest split: parquet_read **494.81ms** + to_numpy **288.19ms** + dict_url **3751.00ms** + date **66.76ms**.

| Query | Status | Match | Detail |
|---|---|---|---|
| Q1 | supported | EXACT | count=999978 (A 3.0ms / HLL 2.44ms / duck 8.64ms) |
| Q2 | supported | EXACT | count=14174 (A 8.93ms / HLL 6.65ms / duck 9.92ms) |
| Q3 | supported | EXACT+TOL | sum=1604053554 (A 5.3ms / HLL 4.77ms / duck 10.41ms) |
| Q30 | supported | EXACT | 181→2 affine (A 243.1ms / HLL 5.98ms / duck 35.01ms; precise split in Q30 suite) |
| +4 more | supported | EXACT | see artifact matrix |
| 8 | unsupported | N/A | 0 jobs, nothing claimed |
| 27 | BLOCKED_BY_MVP | N/A | 0 jobs, nothing claimed (incl. Q5/Q6 DISTINCT, Q21 LIKE, Q29 regex, Q36 tuple, Q43 DATE_TRUNC) |

Measured AB-probes (EXACT, not HLL-supported claims): Q36 A 220.58ms vs B 199.09ms
(groups 68330, vs duck 33.29ms) · Q5 DISTINCT A 85.54ms vs duck 23.35ms (value 79842)
· Q9 A 161.9ms vs B 145.92ms (top0 [229,27961]).

## HLL Stage-5 FULL (`HLL/bench_stage5_full.json`, seed 42, small_n=20, CPU-only)

| Wave | Match | Detail |
|---|---|---|
| w1 packet | EXACT | hash `c303b9330a12e6b2`, deterministic/immutable/serializable, zero_copy_ref |
| w2 float | EXACT | caller schema, NaN/inf pass-through, mix rejected |
| w3 join | EXACT | inner/left/outer, dupe=error, null=miss, zero-copy |
| w4 window | EXACT | per-partition rolling/shifted/cumsum, boundaries never cross |
| w5 text | EXACT | restage contract; LIKE `_`/interior-`%` BLOCKED |
| w6 multi/distinct | EXACT | grouped + global distinct |
| w7 unified | EXACT | all waves + Q30 intact in one run |

Proof-only (small-N semantics). No throughput claim.

## Stage-6 GPU proof (`bench_stage6_gpu_proof.json`, N=1M K=7, reps=5 warm=2)

Same packet both backends, hash `82502e6d937d349c`. Parity **EXACT**
(main 56468897 = 56468897, 12/12 edges EXACT).

| Stage | ms (med) |
|---|---|
| CPU end-to-end (warm med / cold) | **2.84** / 2.61 |
| GPU H2D | 2.53 |
| GPU exec | 1.12 |
| GPU D2H | 2.04 |
| GPU finalize | 0.03 |
| GPU total wall (warm med / cold) | **5.76** / 694.33 (cold = compile) |

Residency: H2D 4000000B, resident 4000000B, D2H 15628B (only W partials cross;
`m` never downloaded; 1 readback, 2 dispatches, 2 pipelines compiled once).
**CPU wins N=1M (2.03x).** Claim = parity + residency + method, not speedup.
Regression: stage-5 all EXACT + Q30 181→2 intact. Untouched: RoadGraph/Planner/prod/ABI/HLL.

## Q30 affine (`bench_q30_auto_affine.json`, rows=999978, warm=6) — UNCONFIRMED prototype-only

Current code CSE+DCE: 181 nodes → 181 nodes EXACT (compile ~24ms exec ~208ms).
Reference artifact (develop/OptimizerAffine, NOT reproduced by current code):

181 nodes → 2 nodes (affine+cse+dce). All diffs 0 (A/AUTO/B vs duck, A vs AUTO vs B).

| Path | cold total | warm-med total | rows/s |
|---|---|---|---|
| A (181 nodes) | 211.22ms | 215.89ms | 4,656,217 |
| AUTO (2 nodes) | 5.19ms | **5.02ms** | 304,792,355 |
| B hand-written (2 nodes) | 3.96ms | 3.39ms | 301,012,326 |
| DuckDB (ref / meas best-of-3) | 31.59 | 35.52 | — |

**UNCONFIRMED prototype-only 43.02x** (215.89 → 5.02, reference artifact). NOT claimed as current result. AUTO within 1.5x of hand-written B (reference only).
Rewrite ~1.7ms included in reference totals. DuckDB 7.1x on this shape (35.52 → 5.02) — reference only, UNCONFIRMED.
