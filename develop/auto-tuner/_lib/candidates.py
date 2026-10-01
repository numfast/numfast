# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""17 candidates over EXISTING Planner alternatives only (no new kernels).

Dimensions (all pre-existing):
- exec policy ST/MT(2,4,8): P2/MT chunked policy, same semantics.
- groupby strategy: dense_fused / dense_shift / hash / sorted / unique
  (names mirror Planner plan_groupby + native groupby variants).
- data path: materialize (filter output alloc) vs fused (no intermediate).
Exactly 17 rows. Pure constructors, no execution here.
"""
CANDIDATES = [
    {"id": "Plan A", "exec": "ST", "threads": 1, "groupby": "hash", "path": "materialize"},
    {"id": "Plan B", "exec": "ST", "threads": 1, "groupby": "dense_shift", "path": "materialize"},
    {"id": "Plan C", "exec": "ST", "threads": 1, "groupby": "dense_fused", "path": "materialize"},
    {"id": "Plan D", "exec": "ST", "threads": 1, "groupby": "sorted", "path": "materialize"},
    {"id": "Plan E", "exec": "ST", "threads": 1, "groupby": "unique", "path": "materialize"},
    {"id": "Plan F", "exec": "ST", "threads": 1, "groupby": "dense_fused", "path": "fused"},
    {"id": "Plan G", "exec": "ST", "threads": 1, "groupby": "dense_shift", "path": "fused"},
    {"id": "Plan H", "exec": "ST", "threads": 1, "groupby": "hash", "path": "fused"},
    {"id": "Plan I", "exec": "ST", "threads": 1, "groupby": "sorted", "path": "fused"},
    {"id": "Plan J", "exec": "ST", "threads": 1, "groupby": "unique", "path": "fused"},
    {"id": "Plan K", "exec": "MT", "threads": 4, "groupby": "hash", "path": "materialize"},
    {"id": "Plan L", "exec": "MT", "threads": 4, "groupby": "dense_shift", "path": "materialize"},
    {"id": "Plan M", "exec": "MT", "threads": 4, "groupby": "dense_fused", "path": "materialize"},
    {"id": "Plan N", "exec": "MT", "threads": 4, "groupby": "dense_fused", "path": "fused"},
    {"id": "Plan O", "exec": "MT", "threads": 4, "groupby": "hash", "path": "fused"},
    {"id": "Plan P", "exec": "MT", "threads": 2, "groupby": "dense_fused", "path": "fused"},
    {"id": "Plan Q", "exec": "MT", "threads": 8, "groupby": "dense_fused", "path": "fused"},
]


def list_candidates():
    return [dict(c) for c in CANDIDATES]
