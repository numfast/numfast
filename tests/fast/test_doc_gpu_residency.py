# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""The GPU-residency claim, pinned against the surface that does not have it.

WHAT THIS FILE EXISTS FOR. Four prose files said the GPU "buys parity and
residency" / "the GPU claim is parity and residency". There is no residency
mechanism in the surface to buy: `to_gpu` is ABSENT from the kernel alias
table, `nf.app()` exposes no GPU-residency verb, and `Table`/`Series` expose
none either. `gpu_execute` is a batch execute with per-node read-back, so
every result returns to host memory. The word promised an API that does not
exist, and a reader who trusted it would build a usage model the library
cannot honour -- the specific failure this project exists to prevent.

So the claim is gone and the honest wording is pinned twice over: the surface
is asserted NOT to carry a residency verb, and the corrected sentence is
asserted to be present in each file that used to carry the false one. A wrong
word that can drift back is not fixed, so the check is here.

WHAT IS NOT ASSERTED. This file does not pin the performance wording, the
15-of-33 split or the CPU-only count: `test_gpu_disclosure.py` measures those
at run time from the capability records. Pinning prose about a number that the
code derives is `test_doc_counters.py`'s job and this file's non-job; this one
pins a sentence that no derivation can restore.
"""

import re
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2]


def _text(rel):
    return (FORK / rel).read_text(encoding="utf-8")


#: The corrected sentence, as a regex over the shipped prose. Registered one
#: site at a time, each found by reading the line it replaced.
CORRECTED = (
    ("README.md",
     r"current API does \*\*not\*\* provide a GPU-resident\s+Table/Series"),
    ("README_PYPI.md",
     r"current API does \*\*not\*\* provide a GPU-resident\s+Table/Series"),
    ("docs/ARCHITECTURE.md",
     r"current API does \*\*not\*\* provide a GPU-resident\s+Table/Series"),
)

#: The false claim in every wording it was published in. Absent from all
#: public-facing markdown, so it cannot be reintroduced by an edit that only
#: had one of the four spellings in mind.
FALSE_CLAIM = re.compile(
    r"parity and residency|buys parity|GPU claim is parity|GPU buys residency",
    re.IGNORECASE)

#: Public-facing markdown, and the subtrees that are generated or vendored --
#: `corresp_src` is a copy of the Rust crate and `node_modules` is not ours.
_SKIP = {"node_modules", "target", "dist", ".git", "corresp_src",
         "_corresp_src", "egg-info", "__pycache__"}


def _public_markdown():
    return [p for p in sorted(FORK.rglob("*.md"))
            if not any(part in _SKIP for part in p.relative_to(FORK).parts)]


@pytest.mark.fast
def test_the_surface_has_no_gpu_residency_verb():
    """`to_gpu` and friends are absent from every name the user can reach."""
    import numfast as nf

    kernel = nf.get_kernel()
    aliases = set(kernel.alias)
    assert "to_gpu" not in aliases
    assert "gpu_resident" not in aliases
    assert "stay_on_gpu" not in aliases

    for verb in ("to_gpu", "gpu_resident", "stay_on_gpu", "on_gpu", "to_device"):
        assert not hasattr(nf.app(), verb), verb
        assert not hasattr(nf.Table, verb), verb
        assert not hasattr(nf.Series, verb), verb


@pytest.mark.fast
def test_the_corrected_wording_is_in_every_file_that_carried_the_false_claim():
    """The replacement sentence is present where the false one used to be."""
    for rel, pattern in CORRECTED:
        text = _text(rel)
        assert re.search(pattern, text), (
            f"{rel} no longer states that the API provides no GPU-resident "
            f"Table/Series. If the wording changed deliberately, teach this "
            f"file the new spelling rather than dropping the check: the claim "
            f"is the one a reader would otherwise take as an API guarantee.")


@pytest.mark.fast
def test_no_public_document_claims_gpu_residency():
    """The false claim is absent from every public-facing document, not just
    the three corrected above."""
    for path in _public_markdown():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = FALSE_CLAIM.search(line)
            assert m is None, (
                f"{path.relative_to(FORK).as_posix()}:{n} claims GPU residency "
                f"({m.group(0) if m else ''!r}); there is no residency mechanism "
                f"in the surface -- test_the_surface_has_no_gpu_residency_verb.")