# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""BUFFER-REUSE research lab (research-only, no .toml, never imported by prod).

Hypothesis: immutable input + MUTABLE reduction workspace with capacity vs
logical length (buf[:K]), read/write cursors, in-place compaction allowed
ONLY when the write frontier never destroys unread state (checked by the
algorithm itself: forward compaction r>=w invariant).
"""
