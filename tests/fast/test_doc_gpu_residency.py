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

THE NOTEBOOK HOLE, and why it was one. `_public_markdown()` globs `*.md`, so
before this revision the guard covered the README and the docs but not a single
notebook. `showcase/` ships five `.ipynb` files whose whole job is to be the
evidence a reader runs -- and a notebook can assert residency in three
different places: a markdown cell, a source cell, and, least visibly, an
OUTPUT cell. A committed output is prose too: an executed `print()` of a claim
is a sentence in the document. Guarding the prose while leaving the main
showcase unguarded is the same defect one directory over, so notebooks are
scanned here on all three channels.

The scanner is written to FAIL LOUDLY on a notebook it cannot parse. A guard
that skips an unreadable document is a guard with a hole shaped like the file
it could not open, so an unparseable notebook is an error, not a pass.

WHAT IS NOT ASSERTED. This file does not pin the performance wording, the
15-of-33 split or the CPU-only count: `test_gpu_disclosure.py` measures those
at run time from the capability records. Pinning prose about a number that the
code derives is `test_doc_counters.py`'s job and this file's non-job; this one
pins a sentence that no derivation can restore.
"""

import json
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


#: The three places a notebook can carry a claim, named as the failure messages
#: name them. A committed OUTPUT is prose: an executed `print()` of a claim is a
#: sentence in the document, and it is the channel least likely to be edited by
#: hand and therefore the most likely to survive a fix to the markdown around it.
NOTEBOOK_CHANNELS = ("markdown", "source", "output")


def _public_notebooks():
    return [p for p in sorted(FORK.rglob("*.ipynb"))
            if not any(part in _SKIP for part in p.relative_to(FORK).parts)]


def _label(path):
    """Repo-relative when the file is in the fork, absolute when it is not.

    The negative control writes its synthetic notebooks to pytest's tmp_path,
    which is outside the fork, so `relative_to` would raise on exactly the
    inputs this scanner most needs to name.
    """
    try:
        return path.relative_to(FORK).as_posix()
    except ValueError:
        return path.as_posix()


def notebook_claims(notebook_path, claim=FALSE_CLAIM):
    """-> [(channel, cell_index, text)] where `claim` appears in a notebook.

    Reads the notebook's JSON directly rather than importing nbformat: nbformat
    is not a test dependency of this repository, and a guard that only runs when
    an optional package is present is a guard with a hole shaped like that
    package. A notebook that cannot be parsed raises -- see the module docstring.

    `claim` is a parameter so the negative control below can drive this exact
    scanner over a synthetic notebook, which is what makes the check trustworthy:
    a scanner nobody has seen catch anything is not known to catch anything.
    """
    text = notebook_path.read_text(encoding="utf-8")
    try:
        nb = json.loads(text)
    except json.JSONDecodeError as err:
        raise AssertionError(
            f"{_label(notebook_path)} is not parseable JSON "
            f"({err}); this guard refuses to skip a document it cannot read."
        ) from err
    if not isinstance(nb.get("cells"), list):
        raise AssertionError(
            f"{_label(notebook_path)} has no 'cells' list; "
            f"this guard refuses to skip a document it cannot read."
        )

    found = []
    for index, cell in enumerate(nb["cells"]):
        kind = cell.get("cell_type")
        if kind in ("markdown", "code"):
            source = "".join(cell.get("source", []))
            if claim.search(source):
                # Reported as `source`, not as the nbformat `code`: the channel
                # names are the three places a CLAIM can live, and a reader
                # grepping this test for the word "code" should not conclude
                # the source channel is unchecked.
                found.append(("markdown" if kind == "markdown" else "source",
                              index, source))
        if kind != "code":
            continue
        for output in cell.get("outputs", []):
            if claim.search(json.dumps(output, ensure_ascii=False)):
                found.append(("output", index, json.dumps(output,
                                                          ensure_ascii=False)))
    return found


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


@pytest.mark.fast
def test_no_public_notebook_claims_gpu_residency():
    """The false claim is absent from notebooks too, on all three channels.

    Separate from the markdown check above rather than folded into it: the
    markdown test's assertion and its failure message are unchanged, and a
    guard that grows by editing the thing it guards is harder to review.
    """
    notebooks = _public_notebooks()
    # The showcase is what this exists for. A glob that silently matches
    # nothing would pass forever, so the check that it finds something is
    # asserted here, not assumed.
    assert notebooks, (
        "no .ipynb found under the fork -- the notebook guard would be "
        "asserting nothing")
    for path in notebooks:
        for channel, index, text in notebook_claims(path):
            m = FALSE_CLAIM.search(text)
            assert m is None, (
                f"{path.relative_to(FORK).as_posix()} cell {index} "
                f"({channel}) claims GPU residency "
                f"({m.group(0) if m else ''!r}); there is no residency mechanism "
                f"in the surface -- test_the_surface_has_no_gpu_residency_verb.")


@pytest.mark.fast
def test_the_notebook_guard_catches_a_claim_on_every_channel(tmp_path):
    """THE NEGATIVE CONTROL: the scanner above is shown to fail.

    A guard with no test that it fires proves nothing -- `test_no_public_...`
    would still be green if the scanner matched nothing at all, or if the glob
    returned an empty list, or if the three channels were read from the wrong
    key. So a synthetic notebook is written carrying the false claim in each
    channel in turn, and `notebook_claims` is required to find all three.

    Also pins the second half of the hole: a notebook whose outputs are checked
    only by re-reading the source is unguarded, so the `output` channel is
    asserted here on a notebook whose SOURCE is entirely innocent.
    """
    innocent_source = ["print('gpu_capabilities() reports 15 of 33 ops')\n"]

    for channel in NOTEBOOK_CHANNELS:
        cells = []
        if channel == "markdown":
            cells.append({"cell_type": "markdown",
                          "source": ["The GPU buys parity and residency.\n"]})
        elif channel == "source":
            cells.append({"cell_type": "code",
                          "source": ["CLAIM = 'GPU claim is parity and residency'\n"]})
        else:
            # The source cell is clean; only the executed OUTPUT carries it.
            cells.append({"cell_type": "code", "source": list(innocent_source),
                          "outputs": [{"output_type": "stream", "name": "stdout",
                                       "text": ["the GPU buys parity and residency\n"]}]})
        path = tmp_path / f"guard_control_{channel}.ipynb"
        path.write_text(json.dumps({"cells": cells, "metadata": {},
                                    "nbformat": 4, "nbformat_minor": 5}),
                        encoding="utf-8")

        found = notebook_claims(path)
        assert found, (
            f"the notebook guard did NOT fire on a {channel} cell carrying the "
            f"false claim -- the guard is not scanning that channel, so "
            f"test_no_public_notebook_claims_gpu_residency is asserting nothing "
            f"for it")
        assert {f[0] for f in found} == {channel}, found


@pytest.mark.fast
def test_the_notebook_guard_passes_a_clean_notebook(tmp_path):
    """The other half: a notebook that says the honest thing must pass.

    Without this, a scanner that flagged EVERYTHING would satisfy the control
    above. Together the two say the scanner discriminates.
    """
    cells = [
        {"cell_type": "markdown",
         "source": ["The current API does **not** provide a GPU-resident "
                    "Table/Series: results return to host memory.\n"]},
        {"cell_type": "code",
         "source": ["print(nf.gpu_capabilities()['ops'][:3])\n"],
         "outputs": [{"output_type": "stream", "name": "stdout",
                      "text": ["['series', 'pack_keys', 'compare']\n"]}]},
    ]
    path = tmp_path / "guard_control_clean.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {},
                                "nbformat": 4, "nbformat_minor": 5}),
                    encoding="utf-8")
    assert notebook_claims(path) == []


@pytest.mark.fast
def test_an_unreadable_notebook_is_an_error_not_a_skip(tmp_path):
    """A notebook that cannot be parsed must fail the guard, not pass it."""
    path = tmp_path / "broken.ipynb"
    path.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(AssertionError, match="not parseable JSON"):
        notebook_claims(path)

    path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    with pytest.raises(AssertionError, match="no 'cells' list"):
        notebook_claims(path)