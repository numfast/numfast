# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""No error in this tree may point at a `specs/` path, or print a bare ` See `.

The 204 references this guards were not introduced by a decision. They were
inherited, they shipped in the wheel before anyone looked, and they resolved to
nothing for every user of the package: there is no `specs/` directory in the
wheel, in the sdist, or in the repository. Measured before the removal --
171 `doc="specs/..."` attributes and 33 `See specs/...` message tails, naming 17
distinct files, of which **zero** exist in any artefact this project ships.

The `doc=` parameter of the error contract is real and is kept; what it must
not carry is a path that resolves to nothing. Two failure shapes are named:

  1. `doc="specs/..."`, or a message ending `See specs/...` -- a reference a
     reader cannot follow. Removed as one mechanical rule; this is its teeth.
  2. `f"... See {doc}"` printed unconditionally, so a helper whose default is
     `doc=""` emits a message ending in a bare " See ". That is the shape the
     removal leaves behind, which is why it is the shape that must not survive.

This scans the SOURCE TREE and any built artefact, so the claim is checkable on
the channel where a reader meets it rather than only in the checkout. Pass a
wheel or sdist path and it reads that; otherwise it reads `src/`.
"""

import re
import tarfile
import zipfile
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2]

DOC_PATH = re.compile(r'doc\s*=\s*"(specs/[^"]*)"')
SEE_PATH = re.compile(r'See\s+(specs/[a-z0-9/._-]*\.md)')
TAIL = re.compile(r'See \{doc\}"\)')
DEF_LINE = re.compile(r"def \w+\(")


def _is_comment_or_docstring(text, index):
    """True when line `index` is inside a comment or a string literal.

    This file has to describe the shapes it forbids in prose, and that prose
    must not trip its own rule -- otherwise the guard cannot be written down,
    and a guard nobody can write is a guard nobody maintains.
    """
    line = text.splitlines()[index]
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    # Inside a docstring: count quotes on every preceding line.
    before = "\n".join(text.splitlines()[:index])
    triple = before.count('"""') + before.count("'''")
    return triple % 2 == 1


def _scan_text(text, name, hits):
    lines0 = text.splitlines()
    for i, line in enumerate(lines0):
        if _is_comment_or_docstring(text, i):
            continue
        m = DOC_PATH.search(line)
        if m:
            hits.append(f"{name}:{i + 1}: doc={m.group(1)!r} names a specs/ path")
        m = SEE_PATH.search(line)
        if m:
            hits.append(f"{name}:{i + 1}: message points at {m.group(1)}")
    # Function bodies, located by their own `def` line, so what is checked is
    # the helper's text rather than the whole file's.
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines)
              if DEF_LINE.match(ln) and not ln.startswith(" ")]
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        body = "\n".join(lines[s:end])
        if TAIL.search(body) and "if doc else" not in body:
            hits.append(
                f"{name}:{s + 1}: prints ' See {{doc}}' unconditionally, so a "
                f"default of doc=\"\" emits a bare ' See '")


def _artefact_texts(path):
    """{member name: text} for every .py in a wheel or an sdist."""
    out = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".py"):
                    out[n] = z.read(n).decode("utf-8", "replace")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if m.name.endswith(".py"):
                    f = t.extractfile(m)
                    if f is not None:
                        out[m.name] = f.read().decode("utf-8", "replace")
    return out


def _artefacts():
    """Every numfast wheel and sdist this tree has built, if any.

    Only `numfast-*`: `dist/` may hold a vendored third-party wheel, and
    asserting about another project's sources is not this test's business.
    """
    return sorted([p for p in (*FORK.glob("dist*/*.whl"),
                               *FORK.glob("dist*/*.tar.gz"))
                   if p.name.startswith("numfast-")])


@pytest.mark.fast
def test_no_error_in_the_tree_points_at_a_specs_path():
    hits = []
    n = 0
    for p in sorted((FORK / "src").rglob("*.py")):
        n += 1
        _scan_text(p.read_bytes().decode("utf-8"), p.relative_to(FORK).as_posix(),
                   hits)
    assert not hits, (
        f"{len(hits)} dangling specs/ reference(s) in src/ ({n} files scanned):\n  "
        + "\n  ".join(hits)
        + "\n\nThe error contract is what + fix + an optional doc link, and no "
          "artefact this project ships contains a specs/ directory. Repoint at "
          "something a reader can open, or drop the link.")
    assert n > 100, f"only {n} .py files found under src/; the scan is not running"


@pytest.mark.fast
def test_no_built_artefact_carries_a_specs_reference():
    """The same rule on the channel where a user meets the message.

    Skipped when nothing has been built -- not because the check is
    unimportant, but because there is nothing to check, and a test that
    invents a pass out of an absent artefact is the thing this whole file is
    about.
    """
    artefacts = _artefacts()
    if not artefacts:
        pytest.skip("no wheel or sdist under dist*/; run `python -m build` first")
    hits = []
    scanned = 0
    for a in artefacts:
        for name, text in _artefact_texts(a).items():
            scanned += 1
            _scan_text(text, f"{a.name}:{name}", hits)
    assert not hits, (
        f"{len(hits)} dangling specs/ reference(s) in the BUILT artefacts "
        f"({scanned} .py members across {len(artefacts)} archive(s)):\n  "
        + "\n  ".join(hits))


@pytest.mark.fast
def test_the_error_contract_still_has_a_doc_slot():
    """The reference was removed, not the capability.

    format_error keeps `doc`, and still prints it when given one: a caller can
    point at something real. What must not exist is a default naming a file
    that ships nowhere.
    """
    import sys

    sys.path.insert(0, str(FORK / "src" / "Core"))
    from _lib.errors import format_error  # noqa: E402

    # The default emits no doc link at all, and no bare " See ".
    plain = str(format_error("what happened", fix="do this"))
    assert plain == "what happened Fix: do this.", plain
    assert "See" not in plain, plain
    # A caller can still supply one, and it appears verbatim.
    linked = str(format_error("what happened", fix="do this", doc="docs/API.md"))
    assert linked.endswith("See docs/API.md"), linked
    # And the doc-link tail is exactly the one the contract documents.
    import inspect

    sig = inspect.signature(format_error)
    assert "doc" in sig.parameters, sig
    assert sig.parameters["doc"].default == "", sig