# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""FUSED-PART research lab (rebuilt branch, research-only).

Hypothesis under test: fused scatter-directly-into-partition-accumulators.
Instead of input -> partition(buffers) -> aggregate (B/D pay a separate
partition phase: ~1190ms B / ~194ms D @N=10M G=1M), do ONE pass: compute
pid from the key inside the loop and update the owning partition
accumulator directly. No pid/key/val intermediate buffers.

Question: is partition locality then practically free vs flat dense
(274ms @G=1M T=1)? If pid computation itself is expensive, this branch
closes with an honest measurement.

NOT a production Extension: no .toml, no setup(kernel), never imported
by production. Pure numpy/numba + stdlib threads. Research-only.

Variants (one contract, golden exact vs flat bincount reference):
  A flat    - gate probe + full-G bincount + compact (baseline).
  Fshift    - fused, pid = key>>S (high bits), range blocks of 2**S.
  Fscaled   - fused, pid = (key*P)//G (multiply+div, ceil-split bounds).
  Fdiv      - fused, pid = key//ws (range division, uniform blocks).
  Fmask     - fused, pid = key&(P-1) (low-bit mask, strided partition,
              local idx = key>>log2(P); merge = concat + argsort).
  Pshift/Pscaled/Pdiv/Pmask - pid-computation cost alone (numpy pass,
              vectorized round-trip exactness: key_of(pid(k)) == k).

Cost split mandatory: total = part + agg + merge (timed apart). Fused
reports part~0 (pid inside agg loop); P* cells report pid time as part.
Gen: uniform keys in [0,G), int32 vals, seed 42. N<=10M (1B forbidden).
"""
