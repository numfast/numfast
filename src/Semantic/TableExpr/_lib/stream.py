# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""stream.py -- the nfs_stream_* wrapper (E6, DESIGN §1.1 row 36).

The engine already has a real budgeted lazy block reader
(`nfs_stream_open/plan/read_block`). What it does NOT have is an iterator
and a plan surface, so this file adds exactly those two things and nothing
else -- no new format, no new reader, no IR nodes.

Honest scope, unchanged from DESIGN §5.6: the format is `nfs-stream-v1` with
a hard-wired 8-column schema (`k1 k2 k3 id4 id6 v1 v2 v3`), no `jobs`, no
expression chains.
"""

from pathlib import Path

from _lib import plan


class Stream:
    """Budgeted lazy block iterator over an nfs-stream-v1 file."""

    def __init__(self, handle, path):
        self._handle = handle
        self._path = str(path)

    def __repr__(self):
        return (f"Stream({self._path!r}, mode={self._handle['mode']!r}, "
                f"blocks={len(self._handle['index'])})")

    # -- facts -----------------------------------------------------------
    @property
    def mode(self):
        return self._handle["mode"]

    @property
    def path(self):
        return self._path

    @property
    def nblocks(self):
        return len(self._handle["index"])

    @property
    def columns(self):
        return [c["name"] for c in self._handle["service"]["columns"]]

    def __len__(self):
        return self.nblocks

    # -- the three engine verbs, verbatim -------------------------------
    def plan(self, budget_frac=0.10):
        """Chunk plan from the RAM budget (blocks_per_chunk / nchunks)."""
        return plan.prepass("nfs_stream_plan")(self._handle, budget_frac)

    def read_block(self, idx):
        """One block's arrays as {column: ndarray}, no transforms."""
        idx = int(idx)
        if idx < 0 or idx >= self.nblocks:
            raise plan.fail(
                "open_stream",
                f"block {idx} out of range (file has {self.nblocks} blocks).",
                "read_block(0 .. nblocks-1)")
        return plan.prepass("nfs_stream_read_block")(self._handle, idx)

    def __iter__(self):
        for i in range(self.nblocks):
            yield self.read_block(i)


def _resolve_file(path):
    """`open_stream` takes the stream file; a directory must hold exactly one."""
    p = Path(str(path))
    if p.is_file():
        return p
    if p.is_dir():
        found = sorted(q for q in p.iterdir() if q.is_file())
        if len(found) != 1:
            raise plan.fail(
                "open_stream",
                f"directory {str(p)!r} holds {len(found)} files; exactly one "
                "nfs-stream-v1 file is required.",
                "pass the file path directly")
        return found[0]
    raise plan.fail(
        "open_stream",
        f"no such stream path: {str(p)!r}",
        "build it first with nfs_stream_build(csv, out), then pass that file")


def open_stream(path, budget_frac=0.25, force_lazy=False):
    """Open an nfs-stream-v1 file lazily under a memory budget."""
    target = _resolve_file(path)
    handle = plan.prepass("nfs_stream_open")(str(target), budget_frac,
                                             bool(force_lazy))
    return Stream(handle, target)