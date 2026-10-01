# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""DISPOSABLE shim: old `src/Compute/CpuJoin` import -> `src/Relational/Join`.

SPEC 12 relocation: implementation moved 1:1 to src/Relational/Join/_lib/join.py.
This shim keeps stale `src/Compute/CpuJoin` import paths working until callers
migrate. No logic, re-export only. DELETE after migration.
"""

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

_FORK = _Path(__file__).resolve().parents[1]
_NEW = _FORK / "src" / "Relational" / "Join"

if str(_NEW) not in _sys.path:
    _sys.path.insert(0, str(_NEW))

import _lib.join as _J  # noqa: E402

JoinBuild = _J.JoinBuild
build_timed = _J.build_timed
join_inner = _J.join_inner
join_left = _J.join_left
make_pair = _J.make_pair
chk_inner = _J.chk_inner
chk_left = _J.chk_left
