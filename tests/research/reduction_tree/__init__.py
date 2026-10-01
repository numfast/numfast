# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""REDUCTION-TREE research lab (research-only, no .toml, never in prod).

Static balanced tree (fixed pairs B1+B2->N1, barrier per round) vs dynamic
ready-queue (child done -> atomic completion counter -> parent runnable ->
one worker takes it; bounded queue, blocking sync, normal locks, NO busy
wait, NO registers). Nodes are SORTED (keys+states) + linear merge variant.
"""
