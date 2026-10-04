# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""wasm_artifact.py -- resolve AND VALIDATE the build a parity script runs.

A parity script that cannot say WHICH artefact it validated is not a test; it
is a script that prints PASS. That is not hypothetical here:

`numfast-native/tools/numfast_native.wasm` is a TRACKED 56-function build
(57 exports) sitting in git beside `target/wasm32-unknown-unknown/release/
numfast_native.wasm`, the current 86-function build (87 exports). 30 symbols
apart. Four parity scripts -- `parity_map`, `parity_cumsum`, `parity_shift`,
`parity_unique` -- pointed at the tracked copy and reported green. A reader
would reasonably conclude WASM parity was established against the current
kernels. It was not; it was established against a build that predates 30 of
them.

So `resolve()` refuses, LOUDLY, in four cases:

  * node is absent -- the export surface cannot be read without it, so the
    script cannot say what it validated;
  * the current build is absent -- `target/` is gitignored, so a clean clone
    has no artefact at all, and a cryptic node error is not a report;
  * the path still resolves to the stale tracked copy (57 exports);
  * the function exports do not match `abi/census.json` BY NAME.

That last one is derived, not typed in. The constant this replaced
(`EXPECTED_EXPORTS = 86`, commented "86 = 85 Func + 1 Memory") compared a
TOTAL export count against a FUNCTIONS-derived number, so it was wrong by the
one Memory export the module also carries, and a build of 87 symbols could not
pass it. Nothing was wrong with the build; the comparison was. The expectation
now comes from the same source-derived census the JS guard reads
(`ts/abi-surface.mjs`), and the kinds are compared separately, so the Memory
export can no longer be mistaken for a missing kernel.

Nothing is skipped and nothing is auto-repaired. `NUMFAST_WASM` overrides the
path. `NUMFAST_WASM_EXPECT_EXPORTS` additionally asserts a total count, for the
day the census is knowingly ahead of a build -- the preferred response to a
legitimate kernel addition is to regenerate `abi/census.json`. A wrong
artefact must STOP the run: a PASS printed against the wrong bytes is worse
than no output at all.
"""

import json
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

#: The source-derived authority, generated FROM the `#[no_mangle]` bodies and the
#: six `macro_rules!` templates in `numfast-native/src/{lib.rs,router.rs}`. This
#: is the SAME file and the SAME derivation `ts/abi-surface.mjs` reads (commit
#: a15d56d), so the JS guard and this one cannot disagree about the surface.
CENSUS = os.path.join(NATIVE, "abi", "census.json")

#: The stale tracked copy exports 57 symbols: 56 functions + 1 memory. Kept as a
#: number because it names a SPECIFIC committed artefact, not an expectation
#: about the build under test.
STALE_EXPORTS = 57

BUILD_CMD = ("cargo build --manifest-path numfast-native/Cargo.toml "
             "--target wasm32-unknown-unknown --release")


def census_funcs():
    """The function names the Rust source declares, from `abi/census.json`.

    Derived, not typed in. A hand-kept integer here is a number only the author
    can update, and the author is not the person running the suite -- which is
    exactly how `EXPECTED_EXPORTS = 86` came to sit one symbol behind the build
    it was supposed to police, and took all eight parity scripts down with it.
    """
    try:
        with open(CENSUS, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(
            "PARITY ABORTED: cannot read the source-derived ABI census at\n"
            "  %s\n  %s\nIt is the authority for which exports a build must "
            "have. Without it there is nothing to check the artefact against, "
            "and a hand-typed count is what this replaced." % (CENSUS, e))
    symbols = raw.get("symbols") or {}
    if not symbols:
        raise SystemExit(
            "PARITY ABORTED: %s carries no symbols; it is not a usable "
            "authority for the export surface." % (CENSUS,))
    return sorted(symbols)


def _node():
    node = shutil.which("node")
    if node is None:
        raise SystemExit(
            "PARITY ABORTED: node is not on PATH. A parity script that cannot "
            "read the artefact's export count cannot say which build it "
            "validated, so it does not get to report PASS. Install Node "
            ">= 22.18 and re-run.")
    return node


def export_info(path, node=None):
    """Exports of a .wasm split BY KIND, or None when node cannot read it.

    Returns {"func": [names...], "other": {kind: [names...]}, "total": int}.

    The split is the point. `WebAssembly.Module.exports()` reports a Memory
    export alongside the functions, so a single total compared against a
    functions-derived expectation is off by one by construction -- which is how
    `EXPECTED_EXPORTS = 86` ("86 = 85 Func + 1 Memory") came to demand 86 of a
    build that exports 87, and aborted all eight parity scripts. Comparing
    functions to functions cannot have that error.
    """
    node = node or _node()
    out = subprocess.run(
        [node, "-e",
         "const fs=require('fs');"
         "const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));"
         "process.stdout.write(JSON.stringify("
         "WebAssembly.Module.exports(m).map(e=>[e.kind,e.name])));",
         path],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        pairs = json.loads(out.stdout.strip())
    except ValueError:
        return None
    info = {"func": [], "other": {}, "total": len(pairs)}
    for kind, name in pairs:
        if kind == "function":
            info["func"].append(name)
        else:
            info["other"].setdefault(kind, []).append(name)
    info["func"].sort()
    return info


def exports(path, node=None):
    """Total export count of a .wasm, or None when node cannot read it."""
    info = export_info(path, node)
    return None if info is None else info["total"]


def resolve(expect=None):
    """Validated path to the WASM build under test, or SystemExit.

    Returns the path on success. Calls `SystemExit` with a stated reason on
    every failure -- there is no path through this function that returns a
    path it has not verified.

    The surface is checked BY NAME against `abi/census.json`, the authority
    generated from the `#[no_mangle]` bodies, exactly as `ts/abi-surface.mjs`
    does for the JS side. A set difference names the kernel that went missing;
    a count only says that something did.
    """
    path = os.environ.get("NUMFAST_WASM") or CURRENT
    want = os.environ.get("NUMFAST_WASM_EXPECT_EXPORTS") or expect
    node = _node()
    declared = census_funcs()

    if not os.path.exists(path):
        near = ""
        if os.path.exists(STALE):
            near = ("\n  A stale %d-export build IS present at\n    %s\n"
                    "  -- it is %d symbols behind and is NOT what these "
                    "scripts validate.\n    Point NUMFAST_WASM at a current "
                    "build, or build one:\n    %s"
                    % (STALE_EXPORTS, STALE, len(declared) + 1 - STALE_EXPORTS,
                       BUILD_CMD))
        raise SystemExit(
            "PARITY ABORTED: no WASM artefact at\n  %s\n"
            "The release build is a build product and is gitignored, so a "
            "clean clone has none. These scripts will not fall back to a "
            "different .wasm and will not skip: a parity script that cannot "
            "name the artefact it validated cannot report PASS.%s"
            % (path, near))

    info = export_info(path, node)
    if info is None:
        raise SystemExit(
            "PARITY ABORTED: could not read the export section of\n  %s\n"
            "node could not instantiate it. Refusing to report PASS against "
            "an unreadable artefact." % path)
    if info["total"] == STALE_EXPORTS and \
            os.path.normcase(os.path.abspath(path)) == \
            os.path.normcase(os.path.abspath(STALE)):
        raise SystemExit(
            "PARITY ABORTED: %s is the STALE TRACKED build -- %d exports "
            "(56 functions), %d symbols behind the current build. Four parity "
            "scripts used to run this and report green, which is how a suite "
            "came to validate the wrong artefact.\n  Build the current one:\n"
            "    %s\n  or set NUMFAST_WASM=<path>."
            % (path, info["total"], len(declared) + 1 - STALE_EXPORTS,
               BUILD_CMD))

    have = set(info["func"])
    missing = [s for s in declared if s not in have]
    extra = sorted(have.difference(declared))
    if missing or extra:
        parts = []
        if missing:
            parts.append("the Rust source declares %d function(s) this artefact "
                         "does not export: %s" % (len(missing), ", ".join(missing)))
        if extra:
            parts.append("the artefact exports %d function(s) the source does "
                         "not declare: %s" % (len(extra), ", ".join(extra)))
        raise SystemExit(
            "PARITY ABORTED: %s does not match the source.\n  %s.\n"
            "  The .wasm is a gitignored build product: rebuild it with\n"
            "    %s\n"
            "  If the source really is ahead of abi/census.json, regenerate the "
            "census -- do not delete this check, because an unchecked export "
            "surface is how the 56-function artefact survived in git unnoticed."
            % (path, "; ".join(parts), BUILD_CMD))

    # Opt-in override, for the day the census is knowingly ahead of a build.
    if want not in (None, ""):
        want = int(want)
        if info["total"] != want:
            raise SystemExit(
                "PARITY ABORTED: %s exports %d symbols, NUMFAST_WASM_EXPECT_EXPORTS"
                "=%d.\nThe build moved. If that is legitimate, regenerate "
                "abi/census.json and drop the override -- do not delete the "
                "check." % (path, info["total"], want))
    return path


def banner(path):
    """One line naming the validated artefact, for the script's output."""
    info = export_info(path)
    other = sum(len(v) for v in info["other"].values())
    return ("wasm artefact: %s (%d exports = %d functions + %d non-function, "
            "surface matches abi/census.json, validated)"
            % (path, info["total"], len(info["func"]), other))