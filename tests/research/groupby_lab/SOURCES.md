# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# SOURCES (group 10): fresh ideas → lab mapping. No foreign code copied.

| # | Source | Mechanism | Already have | What's new | Benefit | Cost | Genericity |
|---|---|---|---|---|---|---|---|
| S1 | DuckDB aggregate HT (2022 blog): salted hash bits in pointer array, 4/8B pointers | Fingerprint pre-filter before pointer chase | J quotient/fingerprint (measured fp_rate=0) | Pointer-width adaptivity (4B vs 8B by table size) | Fewer random accesses, less memory | Extra build pass over pointer array | Generic, int/str |
| S2 | DuckDB parallel agg (Leis et al.): per-thread radix-partitioned HTs, partition-wise merge, no giant final HT | Same-hash → same partition → independent merges | C (radix partition) + E (tree merge) | Per-thread MULTIPLE small HTs to bound pointer array (= our B-partial!) | Bounded memory/thread, skew resilience | Duplicate groups across HTs, merge later | Generic |
| S3 | DuckDB perfect-hash aggregate (stats: few distinct groups) | Dense array indexed by group id, no hashing | F/O dense branch (span-gated bincount) | Trigger from stored stats instead of per-block scan | O(1) probe, zero hash cost | Needs cardinality stats upfront | Low-card only |
| S4 | DuckDB external aggregation (0.9+) | Spill partitions to storage when groups > RAM | NOTHING (untested) | Disk-spill path for >RAM group counts | Handles 1B+ groups | I/O cost, complexity | Only >RAM regime |
| S5 | Polars streaming engine (morsel-driven, 2025-26) | Morsel chunks, per-morsel pre-agg, streaming collect | pipelined() runners (overlap ≤0.58 measured) | Engine-level morsel scheduling (our pipe is thread-pool, not morsel scheduler) | Constant memory, overlap | Scheduler complexity | Generic |
| S6 | Polars sorted-key fast path ("keys are sorted → sorted fast path") | Skip hash entirely on sorted input | D sorted/run path (fastest on sorted everywhere) | Auto-detect inside operator (we do per-block) | 3-6x on sorted (measured D vs A) | Sorted-check scan (~0.1ms/64K) | Sorted only |
| S7 | Xue et al. 2025 "Global Hash Tables Strike Back": two-phase ticketing (ticket morsel → aggregate morsel), global concurrent HT | Decouple dedup (ticket) from accumulation | N batched (hash+scatter split, hash=5%) | Global table, no merge phase; lock-free tickets | No merge, vectorized phases | Atomic contention on skew (hot tickets) | Needs atomics; skew-sensitive |
| S8 | HyPer/Umbra morsel-driven + compiled pipelines | Push-based morsels, LLVM-compiled tight loops | runners threads (pull-based ThreadPool) | Push model + codegen (out of scope for research CPU lab) | Less scheduling overhead | Compiler infra | Infra-level |
| S9 | Polars Categorical overhaul (2025) | Dictionary-encode strings once → int group-by | NOTHING for strings (lab is int32/int64 only) | Encode-once then dense/int path (like resident_prepare codes!) | String group-by → int speed | Encode pass, dict memory | String keys |
| S10 | DataFusion/Arrow: vectorized hash into pre-sized table, two-pass | Pre-size from estimated cardinality, then fill | M flat + estimate probes | Cardinality estimate → exact pre-size (we use 2x) | No resize, better load | Estimate cost | Generic |
| S11 | MonetDB/Vectorwise: radix-clustered group-by | Full radix cluster then per-cluster dense agg | C + O-dense composition (untested as pipeline) | Cluster-then-dense pipeline | Cache-sized clusters | Partition pass | Mid-card |
| S12 | ClickHouse: two-level (partial per-thread → merge) + specialized low-cardinality | Same shape as our H1→funnel | A→funnel, hier results | LowCardinality type = S9 equivalent | Same as S3/S9 | Same | Low-card |

## Lab verdicts already measured (this mission)

- S1-filter: works (fp=0) but doesn't pay at load 0.5 (J 2-3x slower than flat M).
- S2-multi-HT: B-partial loses on high-uniq (17.5-21 vs 13.3ms direct); pays only on
  cross-segment duplication. E pair-funnel tree loses to D linear merge everywhere.
- S3/S6: confirmed winners — O/F dense on low-card, D on sorted.
- S5: overlap 0.2-0.58, D pipe 9.92M rows/s; qmax-insensitive.
- S7-vectorized-hash: numpy slower than scalar njit (9.84 vs 1.45ms) — naive SIMD no.
- NEW TO TEST: S4 spill (>RAM), S9 strings/dict-encoding, S10 estimate-presize,
  S11 cluster-then-dense pipeline, S7 lock-free ticketing under skew.
