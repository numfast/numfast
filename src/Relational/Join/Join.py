# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
from _lib.join import JoinBuild as JoinBuild
from _lib.join import build_timed as _oj_build
from _lib.join import chk_inner as _chk_inner
from _lib.join import chk_left as _chk_left
from _lib.join import join_inner as _oj_inner
from _lib.join import join_left as _oj_left
from _lib.join import make_pair as join_pair
from _lib.native import prod_build as _prod_build
from _lib.native import prod_inner as _prod_inner
from _lib.native import prod_left as _prod_left

_box = {}


def join_build(right_keys, right_v2):
    """Production build: native when available, else proven NumPy.

    Same API/contracts/errors as _lib.join (ValueError on size/dupe).
    Backend-absence (OSError/RuntimeError) -> proven path, never loud.
    """
    try:
        return _prod_build(right_keys, right_v2)
    except (OSError, RuntimeError):
        return _oj_build(right_keys, right_v2)


def join_inner(x_keys, x_v1, build, threads=16):
    """Production INNER: fused -> probe+gather -> proven NumPy."""
    try:
        return _prod_inner(x_keys, x_v1, build, threads=threads)
    except (OSError, RuntimeError):
        return _oj_inner(x_keys, x_v1, build, threads=threads)


def join_left(x_keys, x_v1, build, threads=16):
    """Production LEFT: fused -> probe+gather -> proven NumPy."""
    try:
        return _prod_left(x_keys, x_v1, build, threads=threads)
    except (OSError, RuntimeError):
        return _oj_left(x_keys, x_v1, build, threads=threads)


def join_chk(o1, o2, valid=None):
    if valid is None:
        return _chk_inner(o1, o2)
    return _chk_left(o1, o2, valid)


def setup(kernel):
    _box["kernel"] = kernel
    kernel.metadata.setdefault("Join", {})["version"] = "0.1.0"


PUBLIC = {"join_build": join_build, "join_inner": join_inner,
          "join_left": join_left, "join_pair": join_pair,
          "join_chk": join_chk}
