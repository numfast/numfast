# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""M2c: load_profile memoizes the parse, but a genuine on-disk edit wins.

load_profile() is called once per ExecutionGraph (Runtime.evaluate ->
Planner.select_backend -> _load_profile). The parsed calibration tables are
memoized per (path, st_mtime_ns, st_size); only the parse is cached, and
every call still os.stat()s the file, so a real rewrite is picked up. The
volatile keys (_path, _age_s) are overlaid per call, so two calls return
equal values but distinct top-level dicts sharing the memoized tables.

This test never writes the real calibration.toml: it works on a tmp copy.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK / "src" / "Runtime" / "Planner"))

from _lib.calibrate import load_profile as _load_profile  # noqa: E402

COPY_KEY = "compare_cpu_b"


def _same_size_edit(text, key):
    """One significant digit flipped, byte length preserved."""
    old = next(ln for ln in text.splitlines() if ln.startswith(key + " ="))
    tok = old.split("=", 1)[1].strip()
    i = tok.index(".") + 2
    new_tok = tok[:i] + ("9" if tok[i] != "9" else "8") + tok[i + 1:]
    return raw_replace(text, old, key + " = " + new_tok), new_tok


def raw_replace(text, old, new):
    out = text.replace(old, new)
    assert len(out) == len(text), "edit must preserve file size"
    return out


@pytest.mark.fast
def test_profile_memo_picks_up_on_disk_edit(tmp_path):
    """A same-second, same-size rewrite must invalidate the memo."""
    src = FORK / "calibration.toml"
    cp = tmp_path / "calibration.toml"
    shutil.copy2(src, cp)

    before = _load_profile(str(cp))
    assert before is not None, "the real calibration.toml must load"
    again = _load_profile(str(cp))
    assert again is not None
    assert again["cost"] == before["cost"]
    # Memoized tables are shared by reference; only the top-level dict and
    # the volatile keys are per-call, so consumers cannot corrupt the memo
    # by writing their own top-level keys.
    assert again["cost"] is before["cost"]
    assert again is not before
    assert again["_path"] == before["_path"] == str(cp)

    # newline="" on BOTH the read and the write: this test's subject is a
    # same-SIZE rewrite, so it must compare bytes and not let the text layer
    # rewrite them. Without it the invariant holds only on Windows, and for an
    # accident: read_text() folds CRLF to LF and write_text() then expands LF
    # back to os.linesep, so a 5821-byte CRLF file round-trips to 5821 bytes on
    # Windows and to 5659 bytes on Linux. The assertion below therefore failed
    # on WSL2 (5821 != 5659) while the memo it was checking worked perfectly --
    # a test measuring the platform's newline convention, not the engine.
    #
    # open() rather than Path.read_text(newline=...): that keyword arrived in
    # 3.13, and pyproject declares >=3.11. Verified -- on WSL2's Python 3.12.3
    # read_text(newline="") raises TypeError, which is how this was caught.
    raw = cp.read_bytes().decode("utf-8")
    edited, new_tok = _same_size_edit(raw, COPY_KEY)
    # copy2 keeps the source mtime; pin it to now so the rewrite below lands
    # in the same wall-clock second (mtime granularity is coarse on both hosts).
    os.utime(cp, None)
    for _ in range(64):
        st0 = cp.stat()
        cp.write_bytes(edited.encode("utf-8"))
        st1 = cp.stat()
        if int(st0.st_mtime) == int(st1.st_mtime):
            break
        os.utime(cp, None)
        _load_profile(str(cp))          # warm, so the retry is a real hit
    else:
        pytest.fail("could not land the edit within one wall-clock second")
    assert st0.st_size == st1.st_size, (
        f"the edit was not size-preserving: {st0.st_size} -> {st1.st_size}")

    after = _load_profile(str(cp))
    assert after is not None
    assert after["cost"][COPY_KEY] == float(new_tok)
    assert after["cost"][COPY_KEY] != before["cost"][COPY_KEY]

    # Rejected content is never memoized: same answer on every call.
    cp.write_bytes(b'model_version = "not_the_schema"\n')
    assert _load_profile(str(cp)) is None
    assert _load_profile(str(cp)) is None

    # A missing file is a miss, not a cached hit.
    cp.write_bytes(raw.encode("utf-8"))
    assert _load_profile(str(cp))["cost"][COPY_KEY] == \
        before["cost"][COPY_KEY]
    cp.unlink()
    assert _load_profile(str(cp)) is None