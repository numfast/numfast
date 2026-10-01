# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""CACHE-PART research lab (rebuilt branch, research-only).

Question: can range partitioning turn random writes over a huge dense
accumulator (G up to 1M) into cache-local writes without global sorting?

NOT a production Extension: no .toml, no setup(kernel), never imported by
production. Pure numpy/numba + stdlib threads. Research-only.

Variants (one contract each, golden exact vs flat reference):
  A flat       — plain dense bincount over full G (baseline).
  B range-part — P range partitions via mask scans; small accumulator per
                 partition; merge = concat (disjoint ranges).
  C blocklocal — row-chunks, full-G accumulator per block, dense-add merge
                 (control: expected to lose at high G, kept as checkpoint).
  D scatter-part (hash/partition hybrid) — one counting scatter by coarse
                 pid=(k*P)//G groups rows by partition, then small
                 accumulators; merge = concat like B.

Cost split mandatory: total = partition + aggregate + merge (timed apart).
Gen: uniform keys in [0,G), int32 vals, seed 42. N<=10M (1B forbidden).
"""
