# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Native DLL env boot: packaged first, cargo build tree second.

Boot resolution and the boot-time REPORTING of the same file are one decision
here, so they cannot disagree: `ensure_native_env` pins NUMFAST_NATIVE_DLL and
`default_native_path` is the same probe for callers that only want to report.

The binary NAME is derived from the running platform, never a list of candidate
names. A list makes every platform take the first name it recognises, and in a
checkout `src/numfast/_native/` holds the tracked Windows `numfast_native.dll`,
so Linux was offered -- and loaded -- a Windows artefact (measured on Ubuntu
24.04, `sys.platform == "linux"`, NUMFAST_NATIVE_DLL unset:
`native_info()["dll"] -> src/numfast/_native/numfast_native.dll`,
`dll_exists: True`). The same list would have offered that file on macOS too.

WHERE the file may live is a list, because that is genuinely a list: an
installed wheel ships the binary in `numfast/_native/`, and a checkout keeps it
in the cargo build tree until someone `cp`s it into place. Both shapes are
probed, in that order, and the list is returned rather than discarded so a
caller that found nothing can say WHERE it looked. That is the
`NUMFAST_CALIBRATION_DIR` / fork-root shape the Planner already uses for the
cost profile: a named override, a fork root found by walking, and a failure
that names the paths instead of reporting "absent".

The suffix mapping mirrors `setup.py:_suffix_for_tag`, which hard-fails the
build when a `.dll` is packed for a Linux host; that guard is build-time only and
never covered this run-time path. setup.py is unchanged.
"""

#: The fork root marker, the same one `_package_root` and
#: `Runtime/Planner/_lib/calibrate.py:_fork_root` walk to. Nearest ancestor
#: holding it: the repository in a checkout, the installed package directory
#: in a wheel (numfast/full.toml beside numfast/_ext/).
FORK_MARKER = "full.toml"


def native_binary_name(stem="numfast_native"):
    """The native binary FILENAME this platform may load, or None.

    None means this platform ships no binary with the project, and the caller
    then offers nothing at all rather than naming a file it cannot load.
    """
    import sys
    if sys.platform.startswith("win"):
        return stem + ".dll"
    if sys.platform == "darwin":
        return stem + ".dylib"
    if sys.platform.startswith("linux"):
        return stem + ".so"
    return None


def fork_root(pkg_dir):
    """Nearest STRICT ancestor of `pkg_dir` holding FORK_MARKER, else pkg_dir.

    Strictly a parent: in a checkout `src/numfast/` sits under `<repo>/src`,
    so the repository is `parents[1]` and holds `full.toml`. In a wheel the
    marker is at `numfast/full.toml` -- the package directory ITSELF, never an
    ancestor -- so this walk finds nothing and returns the package directory.
    That is the right answer there for the wrong-looking reason, and it is why
    the marker is not also tested against `pkg_dir`: a checkout whose package
    directory ever carried its own `full.toml` would resolve the fork root to
    `src/numfast` and the build tree one level too high.
    """
    for p in pkg_dir.resolve().parents:
        if (p / FORK_MARKER).exists():
            return p
    return pkg_dir.resolve()


def probed_paths(pkg_dir):
    """Every place this platform's binary may live, in precedence order.

    Package-relative first (the installed wheel), then the checkout's cargo
    build tree. Both `target/release/` -- what a plain `cargo build --release`
    writes, the recipe docs/INSTALL.md gives -- and each `target/<triple>/release/`
    are considered, because `cargo build --release --target <triple>` is the
    other one-line build and which one ran is not recorded anywhere. The triple
    is read off the directory rather than hardcoded: it is the host's business,
    and the FILE name already pins the platform, so a build tree carrying
    `x86_64-pc-windows-gnu/` contributes nothing on Linux.

    Both the bare and `lib`-prefixed names are probed in EVERY location,
    `_native/` included. Cargo names a cdylib `libnumfast_native.so` on a Unix
    host and `numfast_native.dll` on Windows, while the staged copy is whatever
    an operator `cp`'d -- and setup.py checks only the SUFFIX
    (`resolve_flavour` -> `_suffix_for_tag`), never the prefix, so a wheel may
    legitimately carry either spelling.
    """
    name = native_binary_name()
    if name is None:
        return []
    names = (name, "lib" + name)
    out = [pkg_dir / "_native" / n for n in names]
    target = fork_root(pkg_dir) / "numfast-native" / "target"
    out += [target / "release" / n for n in names]
    try:
        triples = sorted(p for p in target.iterdir() if (p / "release").is_dir())
    except OSError:
        triples = []
    for t in triples:
        out += [t / "release" / n for n in names]
    return out


def default_native_path(pkg_dir):
    """-> the platform's own binary inside the package, or None.

    What discovery OFFERS when NUMFAST_NATIVE_DLL names nothing. Package-relative
    only: the checkout build tree is `ensure_native_env`'s business, and a caller
    that only reports facts does not want a build tree.
    """
    name = native_binary_name()
    if name is None:
        return None
    cand = pkg_dir / "_native" / name
    return str(cand) if cand.exists() else None


def ensure_native_env(pkg_dir):
    """Pin NUMFAST_NATIVE_DLL to this platform's binary, or name where it looked.

    -> the list of paths this decision CONSIDERED, in order. Never discarded and
    never a bare `None`, because "not found" on its own is the answer that
    costs the most to reconstruct by hand: the reader has to guess which of the
    package-relative and build-tree shapes was meant and then go looking.

    Three outcomes, and the list says which:

    * `NUMFAST_NATIVE_DISABLE=1` -> `[]`. The operator asked for the numpy
      fallback, so nothing was looked for and there is nothing to report.
    * `NUMFAST_NATIVE_DLL` already set -> `[that file]`. An operator naming a
      file is an override, not a hint: discovery does not run and does not get
      to disagree with it.
    * otherwise -> `probed_paths(pkg_dir)`, and the first existing entry is
      pinned. When NOTHING exists the same list is returned AND a warning names
      every entry -- this is the `Join/native.py:_why_missing` contract, at the
      severity the situation deserves: the native backend is optional there, so
      this degrades to numpy rather than raising.
    """
    import os
    if os.environ.get("NUMFAST_NATIVE_DISABLE") == "1":
        return []
    if "NUMFAST_NATIVE_DLL" in os.environ:
        return [os.environ["NUMFAST_NATIVE_DLL"]]
    probed = probed_paths(pkg_dir)
    for cand in probed:
        if cand.exists():
            os.environ["NUMFAST_NATIVE_DLL"] = str(cand)
            return probed
    # Nothing found. Silence here is the failure this branch exists to stop:
    # the engine falls back to numpy, every lane stays CORRECT, and nothing in
    # the output says the native path was never taken.
    import warnings
    warnings.warn(
        "numfast native binary %r not found; the numpy fallback will be used. "
        "Probed in order: %s. Fix: cargo build --release in numfast-native/ "
        "(or copy the binary into numfast/_native/), or set "
        "NUMFAST_NATIVE_DLL."
        % (native_binary_name(), "; ".join(str(p) for p in probed)),
        RuntimeWarning, stacklevel=2)
    return probed
