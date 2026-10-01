# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""DIRECT-DENSE research lab (rebuilt branch, research-only).

NOT a production Extension: no .toml, no setup(kernel), never imported by
production (builder loads only paths listed in full.toml). Pure numpy/numba.

Contract: keys int32 (resident codes 0..M-1 or generic range), vals int32
-> State(ukeys sorted int64, sums/counts int64 accumulators, optional
mins/maxs int32 for fused Q4). One golden verifier (golden.py). No
Python dict/list on hot paths. Legacy provenance: AdaptiveGroupBy dense
global bincount + njit scatter + fused/AoS idea, re-implemented clean
(no code copied).
"""
