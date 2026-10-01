# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
# ADR: Scan Chunking

Status: accepted.

- `scan` is `chunkable=false`.
- Backends must not split `scan` into chunks.
- Chunking applies only to operations with `chunkable=true`.
- The Planner routes `scan` as a single wave within backend limits.
