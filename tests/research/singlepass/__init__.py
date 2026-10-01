# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""SINGLE-PASS research lab (research-only, never imported by prod).

E-track: single-pass aggregation over resident int32 codes with minimal
state (F2 numba kernels), fast dict materialization, small-start adaptive
hash, sorted-run kernels. Promoted: fused dense + sorted-sp + small-start
hash + in-place pack + tolist dicts. Negatives: int32-bincount (narrow-dtype
loops slower), inline-hash-accumulate (random RMW stalls).
"""
