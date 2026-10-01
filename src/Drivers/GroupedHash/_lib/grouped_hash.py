# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Generic grouped COUNT DISTINCT entry: pair-hash lane retired (NATIVE-ALL).

Canonical lane is the caller-owned sorted_dedup path (valid-mask compact +
sort over (keys, values) + one vector scan; C-speed, bit-exact). This module
keeps the Extension API (grouped_distinct_hash raises GroupedHashMiss so the
caller takes the proven lane) and never raises past _HashMiss. WASM-gate:
browser uses wasm sort/unique.
"""


class _HashMiss(Exception):
    """Fast-lane miss: caller must use the proven sorted_dedup lane."""


GroupedHashMiss = _HashMiss


def grouped_distinct_hash(kc, vc):
    """(ukeys int64 sorted, nunique int64, hash_ms, scan_ms).

    NATIVE-ALL: pair-hash lane retired (both the MT JIT vehicle in
    GroupedHashMT and the ST python-loop lane here). Always raises
    _HashMiss: the caller runs the proven sorted_dedup lane verbatim
    (C-speed sort + vector scan, bit-exact).
    """
    raise _HashMiss("pair-hash retired NATIVE-ALL: sorted_dedup canonical")
