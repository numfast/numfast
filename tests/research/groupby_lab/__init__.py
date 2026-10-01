# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU RESEARCH LAB: hierarchical adaptive groupby aggregation (rebuilt branch).

NOT a production Extension: no .toml, no setup(kernel), never imported by
production (builder loads only paths listed in full.toml). Pure numpy/numba
algorithm lab with one shared contract for all candidates.

Contract: input integer keys + aggregate values -> output (unique keys +
aggregate states). One golden verifier for all (see verifier.py).
"""
