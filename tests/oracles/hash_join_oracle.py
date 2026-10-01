# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Test oracle: reference hash-join (pure Python, small-N only).

Restored P4: callers (scratch/j1_small_broadcast.py, scratch/j2_medium_join.py,
tests/fast/test_ops_cpujoin.py) import join_inner / join_left_outer /
check_exact_sum / assert_unique_keys from this path. Never tracked in git
(P1 research artifact); reconstructed from caller contract + CpuJoin semantics.

Scope: tiny reference checks only (fuzz N<=200, parity N<=20K). Mass compute
stays on the production Join path. Do NOT optimize, do NOT wire into prod.
"""

from __future__ import annotations


def assert_unique_keys(keys, name="build"):
    seen = set()
    for k in keys:
        k = int(k)
        if k in seen:
            raise ValueError(f"oracle {name} keys not unique: dupe {k}")
        seen.add(k)


def join_inner(x_keys, x_v1, right_keys, right_v2):
    """INNER join, probe order. Returns [(key, v1, v2)] int tuples."""
    rmap = {int(k): int(v) for k, v in zip(right_keys, right_v2)}
    if len(rmap) != len(list(right_keys)):
        raise ValueError("oracle build keys not unique: dupes collapse")
    rows = []
    for k, v1 in zip(x_keys, x_v1):
        v2 = rmap.get(int(k))
        if v2 is not None:
            rows.append((int(k), int(v1), int(v2)))
    return rows


def join_left_outer(x_keys, x_v1, right_keys, right_v2):
    """LEFT join, probe order. Returns [(key, v1, v2|None)]."""
    rmap = {int(k): int(v) for k, v in zip(right_keys, right_v2)}
    if len(rmap) != len(list(right_keys)):
        raise ValueError("oracle build keys not unique: dupes collapse")
    return [(int(k), int(v1), rmap.get(int(k)))
            for k, v1 in zip(x_keys, x_v1)]


def check_exact_sum(rows):
    """Exact chk: (sum(v1), sum(v2, misses skipped)) as int."""
    s1 = sum(int(r[1]) for r in rows) if rows else 0
    s2 = sum(int(r[2]) for r in rows if r[2] is not None) if rows else 0
    return int(s1), int(s2)
