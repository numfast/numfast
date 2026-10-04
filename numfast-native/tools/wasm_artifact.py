# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""wasm_artifact.py -- resolve AND VALIDATE the build a parity script runs.

A parity script that cannot say WHICH artefact it validated is not a test; it
is a script that prints PASS. That is not hypothetical here:

`numfast-native/tools/numfast_native.wasm` is a TRACKED 56-function build
(57 exports) sitting in git beside `target/wasm32-unknown-unknown/release/
numfast_native.wasm`, the current 85-function build (86 exports). 29 symbols
apart. Four parity scripts -- `parity_map`, `parity_cumsum`, `parity_shift`,
`parity_unique` -- pointed at the tracked copy and reported green. A reader
would reasonably conclude WASM parity was established against the current
kernels. It was not; it was established against a build that predates 29 of
them.

So `resolve()` refuses, LOUDLY, in four cases:

  * node is absent -- the export count cannot be read without it, so the
    script cannot say what it validated;
  * the current build is absent -- `target/` is gitignored, so a clean clone
    has no artefact at all, and a cryptic node error is not a report;
  * the path still resolves to the stale tracked copy (57 exports);
  * the export count is not the expected one.

Nothing is skipped and nothing is auto-repaired. `NUMFAST_WASM` overrides the
path; `NUMFAST_WASM_EXPECT_EXPORTS` overrides the expected count, for the day
a kernel is legitimately added and this constant must move with it. A wrong
artefact must STOP the run: a PASS printed against the wrong bytes is worse
than no output at all.
"""

import os
import shutil
import subprocess

TOOLS = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.normpath(os.path.join(TOOLS, ".."))

# The build under test. Gitignored by `numfast-native/.gitignore` (`target/`),
# which is exactly why a stale copy was committed next to it in the first
# place -- and why "the artefact is missing" must be a stated failure rather
# than an incidental one.
CURRENT = os.path.join(
    NATIVE, "target", "wasm32-unknown-unknown", "release", "numfast_native.wasm")

# The stale tracked copy. Named here only so the guard can RECOGNISE it and
# say what it is; it is never a fallback.
STALE = os.path.join(TOOLS, "numfast_native.wasm")

# 86 = 85 Func + 1 Memory. Measured on the current build; the acceptance
# audit independently counted 86 exports / 85 functions / 0 imports.
EXPECTED_EXPORTS = 86

# 57 = 56 Func + 1 Memory: the tracked build, 29 symbols behind.
STALE_EXPORTS = 57

BUILD_CMD = ("cargo build --manifest-path numfast-native/Cargo.toml "
             "--target wasm32-unknown-unknown --release")


def _node():
    node = shutil.which("node")
    if node is None:
        raise SystemExit(
            "PARITY ABORTED: node is not on PATH. A parity script that cannot "
            "read the artefact's export count cannot say which build it "
            "validated, so it does not get to report PASS. Install Node "
            ">= 22.18 and re-run.")
    return node


def exports(path, node=None):
    """Export count of a .wasm, or None when node cannot read it."""
    node = node or _node()
    out = subprocess.run(
        [node, "-e",
         "const fs=require('fs');"
         "const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));"
         "process.stdout.write(String(WebAssembly.Module.exports(m).length));",
         path],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def resolve(expect=None):
    """Validated path to the WASM build under test, or SystemExit.

    Returns the path on success. Calls `SystemExit` with a stated reason on
    every failure -- there is no path through this function that returns a
    path it has not verified.
    """
    path = os.environ.get("NUMFAST_WASM") or CURRENT
    want = int(os.environ.get("NUMFAST_WASM_EXPECT_EXPORTS") or expect
               or EXPECTED_EXPORTS)
    node = _node()

    if not os.path.exists(path):
        near = ""
        if os.path.exists(STALE):
            near = ("\n  A stale %d-export build IS present at\n    %s\n"
                    "  -- it is %d symbols behind and is NOT what these "
                    "scripts validate.\n    Point NUMFAST_WASM at a current "
                    "build, or build one:\n    %s"
                    % (STALE_EXPORTS, STALE, EXPECTED_EXPORTS - STALE_EXPORTS,
                       BUILD_CMD))
        raise SystemExit(
            "PARITY ABORTED: no WASM artefact at\n  %s\n"
            "The release build is a build product and is gitignored, so a "
            "clean clone has none. These scripts will not fall back to a "
            "different .wasm and will not skip: a parity script that cannot "
            "name the artefact it validated cannot report PASS.%s"
            % (path, near))

    got = exports(path, node)
    if got is None:
        raise SystemExit(
            "PARITY ABORTED: could not read the export section of\n  %s\n"
            "node could not instantiate it. Refusing to report PASS against "
            "an unreadable artefact." % path)
    if got == STALE_EXPORTS and os.path.normcase(os.path.abspath(path)) == \
            os.path.normcase(os.path.abspath(STALE)):
        raise SystemExit(
            "PARITY ABORTED: %s is the STALE TRACKED build -- %d exports "
            "(56 functions), %d symbols behind the current build. Four parity "
            "scripts used to run this and report green, which is how a suite "
            "came to validate the wrong artefact.\n  Build the current one:\n"
            "    %s\n  or set NUMFAST_WASM=<path>."
            % (path, got, EXPECTED_EXPORTS - STALE_EXPORTS, BUILD_CMD))
    if got != want:
        raise SystemExit(
            "PARITY ABORTED: %s exports %d, expected %d.\n"
            "The build moved. If that is legitimate (a kernel was added or "
            "removed), set NUMFAST_WASM_EXPECT_EXPORTS=%d -- do not delete the "
            "check, because an unchecked export count is how the 56-function "
            "artefact survived in git unnoticed."
            % (path, got, want, got))
    return path


def banner(path):
    """One line naming the validated artefact, for the script's output."""
    return "wasm artefact: %s (%d exports, validated)" % (path, exports(path))