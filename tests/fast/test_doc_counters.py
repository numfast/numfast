# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Counters the documentation states and the code derives: they must agree.

build.mjs already refuses to build when `Cargo.toml` disagrees with
`pyproject.toml`, so version drift cannot happen on the npm channel. This file
extends that idea to the counters that are GENERATED or CHEAPLY DERIVABLE, and
only to those.

WHAT WOULD HAVE CAUGHT THE FOUR-LANE DEFECT, AND WHAT WOULD NOT.

It would NOT have been caught. That README claimed `ssspCsr` returns
[0, 10, 30, <inf>] where the kernel returns [0, 10, 30] -- a wrong VALUE, not a
moved one. No counter mechanism sees a wrong value; only executing the
documented example against the kernel does, and that is what
`tests/fast/test_planner_profile_validity.py`'s sibling
`numfast-native/ts/test/readme.test.mjs` now does. This file is therefore NOT
the fix for that defect and does not claim to be; it is the part that is cheap,
and the two are recorded together because the distinction is the useful part.

WHY THESE AND NOT THE OTHERS. A counter qualifies here if the source of truth
already exists in the tree and reading it costs nothing at test time:

  * `len(V0)` is the registry the Extension builds at import. Seven separate
    assertions already pin it to 44 (test_consumer_surface,
    test_consumer_is_null, test_consumer_silent_regressions,
    test_release_blockers). What no assertion did was check the six places the
    DOCUMENTATION states 44 -- README.md, README_PYPI.md, docs/API.md. A name
    added without a doc edit passes every test in this repository.
  * `TOTAL_EXPORTS` / `WRAPPED.length` / `SCOPE.statement` are generated in
    index.ts at module load, and the README quotes the sentence word for word.
    Pinned on the npm side by test/readme.test.mjs.

WHAT IS DELIBERATELY NOT HERE. The ClickBench and H2O figures are measurements,
not derivations: pinning them would freeze a number on a machine that is not
the one they were measured on, which is a worse lie than a stale prose figure.
The parity-fixture case counts (203) come from a generated JSON whose own
generator is the authority, and test/parity.test.mjs already reports them on
every run. Anything requiring a refactor to make derivable is out of scope by
the instruction, and there is a lot of it.
"""

import importlib.util
import re
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2]


def _load_v0():
    """The Extension's own _lib/names.py, by path -- the import-guard's one
    permitted route, and the same one its own tests use."""
    path = FORK / "src" / "Semantic" / "TableExpr" / "_lib" / "names.py"
    spec = importlib.util.spec_from_file_location("_lib.names_rb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.V0


def _text(rel):
    return (FORK / rel).read_text(encoding="utf-8")


#: Every place the documentation states the CONSUMER-SURFACE count, found by
#: search and then registered one by one -- (file, regex with one capture group,
#: what that site is claiming). All 12 were located by grepping the three prose
#: files for `\d+[- ]names?`; the ones that are NOT surface widths are excluded
#: deliberately and listed in OTHER_COUNTS below, because a bare number search
#: cannot tell them apart and pretending otherwise produces a test that fails on
#: correct prose.
DOC_CLAIMS = (
    ("README.md", r"The consumer surface is (\d+) names", "the headline claim"),
    ("README.md", r"the (\d+)-name v0 registry", "the registry's width"),
    ("README.md", r"consumer facade; (\d+) public names", "the tree diagram"),
    ("README.md", r"the consumer facade \((\d+) public names\)", "the layout map"),
    ("README.md", r"the (\d+)-name surface, the kernel-level names",
     "the docs index table"),
    ("README_PYPI.md", r"\*\*The API is (\d+) names\.\*\*", "the PyPI headline"),
    ("README_PYPI.md", r"A (\d+)-name consumer facade", "the feature list"),
    ("README_PYPI.md", r"the (\d+)-name surface and every guard", "the links list"),
    ("docs/API.md", r"the (\d+)-name v0 registry", "the layer table"),
    ("docs/API.md", r"the consumer facade \((\d+) names\)", "the layer-1 heading"),
    ("docs/API.md", r"not one of the (\d+) registry entries", "pandas' status"),
    ("docs/API.md", r"is not one of the (\d+) names", "lookup's status"),
    ("docs/API.md", r"not part of the (\d+)-name registry", "__all__'s status"),
)

#: Counts in the same files that are NOT the surface width, registered so that
#: adding one is a deliberate act and not an accident this file would then flag.
#: Each was confirmed by reading its line during the search.
OTHER_COUNTS = (
    ("README.md", r"Absent \| (\d+) names", "the four names deliberately left out"),
    ("README_PYPI.md", r"a (\d+)-name expression vocabulary",
     "the expression vocabulary, a subset of the surface"),
)


@pytest.mark.fast
def test_every_documented_surface_count_is_the_real_one():
    """Each of the 13 sites that states the width, against `len(V0)`.

    Not "the docs agree with each other" -- against the registry the code builds,
    so a name added to the registry without a documentation edit fails here and
    names both numbers.
    """
    n = len(_load_v0())
    for rel, pattern, claim in DOC_CLAIMS:
        text = _text(rel)
        m = re.search(pattern, text)
        assert m, (
            f"{rel} no longer states the counter in the spelling this file checks "
            f"({claim}). Teach this file the new wording rather than dropping the "
            f"check: the point is that the number is not maintained by hand.")
        assert int(m.group(1)) == n, (
            f"{rel} states a {m.group(1)}-name consumer surface ({claim}); the "
            f"registry has {n}. Either the registry grew without a doc edit, or "
            f"the doc is wrong.")


@pytest.mark.fast
def test_the_non_surface_counts_are_still_what_the_docs_say():
    """The two neighbouring counts, checked so they are not silently edited.

    These are pinned to their CURRENT values, which is weaker than the surface
    check and deliberately so: neither is derivable from the registry. Their job
    is to say "this file knows these numbers exist", so that a future edit to one
    of them is a decision made here rather than an oversight nobody flagged.
    """
    for rel, pattern, claim in OTHER_COUNTS:
        m = re.search(pattern, _text(rel))
        assert m, f"{rel} no longer states {claim}"
        # 4 = names absent from v0 (README); 23 = the expression vocabulary.
        assert int(m.group(1)) in (4, 23), (
            f"{rel} changed {claim} to {m.group(1)}. If that is a correction, "
            f"update this file with the new value and say why in the commit.")