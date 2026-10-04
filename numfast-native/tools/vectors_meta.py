# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""vectors_meta.py -- read `vectors/meta.json` with its paths resolved HERE.

`meta.json` is generated (`tools/gen.py`, seed 42) and gitignored, so it is a
build product of whatever machine ran gen.py. It used to store ABSOLUTE paths,
which meant a meta.json generated on one checkout made every reader die with a
FileNotFoundError naming a tree that does not exist on the next one -- and in a
log that reads exactly like a parity failure.

`gen.py` now writes basenames; this resolves them against THIS checkout's
`vectors/`, and refuses rather than guessing when a file named in meta.json is
absent. A missing vector is a stated failure, never a fallback to some other
buffer.

Usage, replacing the old pattern:

    import vectors_meta
    meta = vectors_meta.load("small")
    keys = np.fromfile(meta["keys"], dtype=np.int32)
"""

import json
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.normpath(os.path.join(TOOLS, ".."))
VEC = os.path.join(NATIVE, "vectors")
META = os.path.join(VEC, "meta.json")

GEN_CMD = "python numfast-native/tools/gen.py"


def load(case="small", *fields):
    """The meta.json entry for `case`, with the named fields made absolute.

    @param fields which fields to resolve and check. Defaults to both.
    """
    if not os.path.exists(META):
        raise SystemExit(
            "VECTORS ABORTED: no %s\n"
            "The shared synthetic vectors are a build product and are gitignored, "
            "so a clean clone has none. Produce them with:\n  %s\n"
            "These scripts will not substitute another buffer and will not skip."
            % (META, GEN_CMD))
    with open(META, encoding="utf-8") as f:
        cases = json.load(f)
    entry = cases.get(case)
    if entry is None:
        raise SystemExit(
            "VECTORS ABORTED: %s has no case %r (it has %s)"
            % (META, case, ", ".join(sorted(cases)) or "none"))
    for field in (fields or ("keys", "values")):
        name = entry.get(field)
        if not name:
            raise SystemExit(
                "VECTORS ABORTED: %s[%r] has no %r" % (META, case, field))
        # A basename resolves against this checkout's vectors/; an absolute path
        # is honoured only if it exists, so a meta.json left over from another
        # machine fails loudly instead of reading something unexpected.
        resolved = name if os.path.isabs(name) else os.path.join(VEC, name)
        if not os.path.exists(resolved):
            raise SystemExit(
                "VECTORS ABORTED: %s[%r][%r] names\n  %s\nwhich does not exist.\n"
                "If that is an absolute path from another checkout, regenerate the "
                "vectors here:\n  %s" % (META, case, field, resolved, GEN_CMD))
        entry[field] = resolved
    return entry