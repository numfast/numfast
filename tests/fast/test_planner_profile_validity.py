# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""A shipped cost profile may drive the planner only where it was measured.

calibration.toml ships inside both wheels carrying the [hardware] of the one
machine that measured it. Before this rule the Planner honoured it on every
machine: a laptop, a phone and a server all routed on timings taken on one
Windows host with one GPU. The library was making a routing decision from data
that does not describe its user, silently. That is the defect class this engine
spends its life eliminating, except it sat in the router rather than in a
returned value.

The order this file pins:

  1. NUMFAST_CALIBRATION_DIR -- measured here by whoever pointed at it.
     Authoritative whatever its [hardware] says: naming the file IS the
     assertion.
  2. The shipped profile -- only when its [hardware] matches this machine.
  3. Neither -- stub costs, and the refusal says which fields differed, so the
     fallback is discoverable rather than silent.

The machine-identity fields are compared by plain equality, so two "unknown"s
agree and unknown-vs-known does not: a live probe that cannot name the device
must not be taken as proof it is the device the numbers came from.

The tests never write the real calibration.toml: rule 1 works on a copy in
tmp_path, rules 2 and 3 read the shipped file and change what the MACHINE
reports instead.
"""

import shutil
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK / "src" / "Runtime" / "Planner"))

from _lib.calibrate import (  # noqa: E402
    HARDWARE_IDENTITY, calibrate_impl as C_calibrate, hardware_match,
    hardware_now, profile_path, resolve_routing_profile, routing_reject_warning)


@pytest.fixture(scope="module")
def kernel():
    from builder import MAIN

    return MAIN["build"](str(FORK))


@pytest.fixture(scope="module")
def real_capability(kernel):
    return kernel.alias["gpu_capability"]()


def _graph(alias, op):
    import numpy as np
    rng = np.random.default_rng(42)
    v = rng.integers(0, 1000, 4096).astype(np.int32)
    g = alias["compile"]([{"op": "series", "inputs": [], "params": {
        "values": v, "dtype": "int32"}, "out": "s"}])
    g["nodes"].append({"kernel_id": op, "inputs": ["s"],
                       "params": {"a": 7, "b": 3}, "out": "r"})
    g["outputs"].append("r")
    return g


#: A recorded value that cannot be this machine's, per identity field. Chosen
#: to be different from ANY plausible live probe rather than from one host: the
#: previous literal set hard-coded `python = "3.12.3"`, which is exactly what
#: WSL2 reports, so the "every field disagrees" assertion passed with 4 of 5
#: fields differing there and the test did not notice it was weaker.
_NOT_THIS_MACHINE = {
    "cpu": "not-this-machine (synthetic)",
    "platform": "not-this-machine (synthetic platform)",
    "python": "0.0.0-not-this-machine",
    "gpu_device": "NOT THIS MACHINE GPU",
    "gpu_backend": "NOT-THIS-MACHINE-BACKEND",
}


def _foreign_profile(tmp_path, capability):
    """A byte-valid copy of the shipped profile whose [hardware] is elsewhere.

    Every identity field is rewritten to a value derived from THIS machine's
    live probe, so the copy is foreign by construction on whatever host runs
    the test rather than foreign relative to one hard-coded one.
    """
    d = tmp_path / "caldir"
    d.mkdir()
    text = (FORK / "calibration.toml").read_text(encoding="utf-8")
    now = hardware_now(capability)
    for f in HARDWARE_IDENTITY:
        foreign = _NOT_THIS_MACHINE[f]
        assert foreign != str(now.get(f, "unknown")), (
            f"the synthetic foreign value for {f} equals what this machine "
            f"reports ({foreign!r}), so the copy would not be foreign here")
        old = None
        for line in text.splitlines():
            if line.startswith(f + " ="):
                old = line
                break
        assert old is not None, f"shipped profile no longer records {f}"
        text = text.replace(old, '%s = "%s"' % (f, foreign), 1)
    (d / "calibration.toml").write_text(text, encoding="utf-8")
    return d


# 1. The machine this profile was measured on still gets it. If this fails,
#    the recorded ClickBench / H2O figures stop being reproducible here.
#
#    WHICH MACHINE this is, is the point. The shipped profile names one host,
#    so on any other machine the correct answer is `prof is None` and a
#    differ-list that names every field which disagrees. Pre-fix the test
#    asserted the measuring machine's answer unconditionally, so it failed on
#    every host that is not it -- measured on WSL2: 3 of 5 identity fields
#    differ and the assertion was `dec["differ"] == []`. It now asserts the
#    rule on whichever machine runs it, which is stronger: BOTH branches are
#    checked, and the measuring machine is still covered because that is the
#    branch it takes.
@pytest.mark.fast
def test_shipped_profile_applies_only_on_the_machine_that_measured_it(
        real_capability):
    prof, dec = resolve_routing_profile(gpu_capability=real_capability)
    recorded, live = dec["profile_hardware"], dec["hardware"]
    disagree = sorted(f for f in HARDWARE_IDENTITY
                      if str(recorded.get(f, "unknown")) !=
                      str(live.get(f, "unknown")))
    if disagree:
        # A foreign machine: refused, and the refusal names what differs.
        assert prof is None, dec
        assert dec["routing"] is False and dec["matched"] is False
        assert dec["origin"] == "shipped", dec
        assert [f for f, _, _ in dec["differ"]] == disagree, dec["differ"]
    else:
        # The measuring machine: honoured, and every identity field agrees.
        assert prof is not None, dec
        assert dec["routing"] is True and dec["origin"] == "shipped"
        assert dec["differ"] == []
    # Either way the compared fields are exactly the identity set.
    assert sorted({f for f, _, _ in dec["differ"]}) == disagree


# 2. vram is not an identity field. A profile measured where the capability
#    note already named the device records vram="unknown" (nvidia-smi is only
#    a fallback), so comparing it would reject a perfectly good profile.
@pytest.mark.fast
def test_vram_is_not_part_of_machine_identity(real_capability):
    assert "vram_mb" not in HARDWARE_IDENTITY
    assert "vram_source" not in HARDWARE_IDENTITY
    assert "backend" not in HARDWARE_IDENTITY  # the constant "webgpu" both sides
    now = hardware_now(real_capability)
    # Identical on every identity field, wildly different on the excluded ones:
    # still a match.
    profile_hw = dict(now, vram_mb="24576", vram_source="nvidia-smi",
                      backend="cuda")
    ok, differ = hardware_match(profile_hw, now)
    assert ok and differ == [], differ
    # And a difference inside the identity set is not a match.
    ok, differ = hardware_match(dict(now, gpu_device="some other card"), now)
    assert not ok and [f for f, _, _ in differ] == ["gpu_device"], differ


# 3. Unknown is not a wildcard: it agrees with unknown, never with a value.
@pytest.mark.fast
def test_unknown_hardware_never_counts_as_a_match(real_capability):
    now = hardware_now(real_capability)
    ok, differ = hardware_match({f: "unknown" for f in HARDWARE_IDENTITY}, now)
    assert not ok, "a wholly unknown profile must not be honoured"
    assert {f for f, _, _ in differ} == set(HARDWARE_IDENTITY)
    ok, differ = hardware_match(dict(dec_ := now), now)
    assert ok and differ == []


# 4. A shipped profile that describes another machine must NOT route, and the
#    refusal must name the fields that disagreed. The machine is made foreign by
#    patching the capability the driver reports, which is exactly what
#    hardware_now() reads -- the same technique test_calibration_guards.py uses.
@pytest.mark.fast
def test_shipped_profile_on_a_foreign_machine_routes_on_stub(kernel,
                                                             real_capability):
    a = kernel.alias
    real = a["gpu_capability"]
    foreign = dict(real_capability)
    foreign["note"] = "wgpu-py AMD Radeon Pro 5500M DX12: elementwise"
    a["gpu_capability"] = lambda: foreign
    try:
        for op in ("compare", "reduce", "sort"):
            for n in (1000, 100000, 1000000):
                sel = a["select_backend"](_graph(a, op), n)
                assert sel["backend"] == "cpu", (op, n, sel["backend"])
                assert sel["profile"]["version"] == "stub", sel["profile"]
                assert sel["cost_estimate"] == {"cpu": None, "gpu": None}
                w = sel["profile"]["warning"]
                assert "different machine" in w, w
                assert "gpu_device" in w and "gpu_backend" in w, w
                # The refusal is discoverable, not silent: the profile that
                # exists at the path is named as the one refused.
                assert sel["profile"]["source"] == "shipped", sel["profile"]
                assert "NUMFAST_CALIBRATION_DIR" in w and \
                    "calibrate(force=True)" in w, w
        info = a["calibrate_info"]()
        assert info["routing"]["routing"] is False
        assert info["routing"]["origin"] == "shipped"
        # The two GPU fields MUST be among the disagreements -- that is what
        # this test changes. The set need not be exactly those two: when the
        # host running the test is already not the machine that measured the
        # shipped profile, cpu/platform/python differ as well, and pre-fix
        # asserting `== {"gpu_device", "gpu_backend"}` made the test fail on
        # every non-measuring host while pinning nothing extra. Membership, not
        # equality: the GPU fields must be refused, and nothing may be
        # silently accepted.
        differ = {f for f, _, _ in info["routing"]["differ"]}
        assert {"gpu_device", "gpu_backend"} <= differ, info["routing"]["differ"]
        assert set(info["routing"]["profile_hardware"]) >= set(differ)
        # The FILE is still described -- complete, measured, just not ours.
        assert info["profile"]["version"] == "calibrated_v1"
        # And the same verdict, word for word, as the routing decision.
        assert routing_reject_warning(info["routing"]) == \
            sel["profile"]["warning"]
    finally:
        a["gpu_capability"] = real


# 5. Rule 1: a profile measured HERE is authoritative even if its [hardware]
#    block names somewhere else. Pointing the variable at a file is the assert.
@pytest.mark.fast
def test_local_profile_is_authoritative_whatever_its_hardware_says(
        kernel, real_capability, tmp_path, monkeypatch):
    a = kernel.alias
    d = _foreign_profile(tmp_path, real_capability)
    monkeypatch.setenv("NUMFAST_CALIBRATION_DIR", str(d))
    prof, dec = resolve_routing_profile(gpu_capability=real_capability)
    assert prof is not None, dec
    assert dec["routing"] is True and dec["origin"] == "local"
    # Every identity field disagrees, and it is honoured anyway.
    ok, differ = hardware_match(dec["profile_hardware"],
                               dec["hardware"])
    assert not ok and len(differ) == len(HARDWARE_IDENTITY), differ
    for op in ("compare", "reduce", "sort"):
        for n in (1000, 100000, 1000000):
            sel = a["select_backend"](_graph(a, op), n)
            assert sel["profile"]["version"] == "calibrated_v1", sel["profile"]
            assert sel["cost_estimate"]["cpu"] is not None, sel
            assert sel["profile"]["warning"] == "", sel["profile"]
    assert a["calibrate_info"]()["routing"]["origin"] == "local"


# 6. No profile anywhere is a different refusal from a profile that is not
#    ours, and must read differently -- one is "measure here", the other is
#    "the one you have was measured elsewhere".
@pytest.mark.fast
def test_absent_profile_and_foreign_profile_read_differently(kernel, tmp_path,
                                                             monkeypatch):
    a = kernel.alias
    real = a["gpu_capability"]
    foreign = dict(real())
    foreign["note"] = "wgpu-py AMD Radeon Pro 5500M DX12: elementwise"

    monkeypatch.setenv("NUMFAST_CALIBRATION_DIR", str(tmp_path))
    _p, absent = resolve_routing_profile(a["gpu_capability"]())
    assert absent["reason"] == "no calibration profile at this path"
    assert "calibrate(force=True)" in routing_reject_warning(absent)
    assert "different machine" not in routing_reject_warning(absent)
    assert profile_path() == str(tmp_path / "calibration.toml")

    monkeypatch.delenv("NUMFAST_CALIBRATION_DIR")
    a["gpu_capability"] = lambda: foreign
    try:
        _p, rejected = resolve_routing_profile(a["gpu_capability"]())
        assert "different machine" in rejected["reason"]
        assert "different machine" in routing_reject_warning(rejected)
    finally:
        a["gpu_capability"] = real


# 7. An explicitly passed dict still wins: it is the caller's assertion, and
#    it is how the suite exercises cost shapes no profile here records.
@pytest.mark.fast
def test_explicit_profile_dict_still_wins_over_resolution(kernel):
    a = kernel.alias
    sel = a["select_backend"](_graph(a, "compare"), 100000,
                              {"model_version": "calibrated_v1", "cost": {}})
    assert sel["profile"]["version"] != "stub", sel["profile"]


# 8. The refusal survives the EXPLAIN surface: format_explain prints the
#    profile warning, which is how a Chain user reaches the same fact.
@pytest.mark.fast
def test_rejection_reaches_the_explain_text(kernel, real_capability):
    from _lib.calibrate import format_explain
    a = kernel.alias
    real = a["gpu_capability"]
    foreign = dict(real_capability)
    foreign["note"] = "wgpu-py AMD Radeon Pro 5500M DX12: elementwise"
    a["gpu_capability"] = lambda: foreign
    try:
        sel = a["select_backend"](_graph(a, "compare"), 1000)
        text = format_explain(_graph(a, "compare"), {
            "actual": sel["backend"], "requested": "auto",
            "profile": sel["profile"], "coverage": sel["coverage"],
            "estimate": sel["estimate"]})
    finally:
        a["gpu_capability"] = real
    assert "different machine" in text, text
    assert "warning=" in text and "none" not in text.split("warning=")[1][:20]


# 9. calibrate records the SAME identity block it will later compare against:
#    one parser, so a profile can never be rejected for a spelling that
#    hardware_now() does not also produce.
@pytest.mark.fast
def test_recorded_and_compared_hardware_come_from_one_parser(kernel,
                                                             real_capability):
    import _lib.calibrate as C
    recorded = C._hardware(kernel.alias)
    compared = hardware_now(real_capability)
    for f in HARDWARE_IDENTITY:
        assert str(recorded.get(f, "unknown")) == \
            str(compared.get(f, "unknown")), f
    # The one documented difference is the deliberate one: _hardware() may fill
    # vram from nvidia-smi, hardware_now() never does, and vram is not identity.
    assert recorded.get("vram_mb") == compared.get("vram_mb"), "unexpected drift"


# --- 10. calibrate() may not claim someone else's profile ------------------
#
# The router was fixed above; calibrate() is the adjacent surface and had the
# same defect one level out: its reuse check was `Path(p).exists() and
# load_profile(p) is not None`, with no machine test at all, so on every
# foreign machine it answered {"status": "reused"} naming a profile taken on
# another host. Same class, different function.
#
# And the refusal it needed had a second half. Returning something other than
# "reused" means the call falls through to measurement and a write, and
# profile_path() resolves to the INSTALLED PACKAGE DIRECTORY in a wheel
# (numfast/full.toml beside numfast/_ext/). So fixing the reuse check without
# closing the write would have turned a false claim into a write into
# site-packages. Both halves are pinned here.


def _this_machine_profile(tmp_path, capability):
    """A copy of the shipped profile carrying THIS machine's [hardware].

    Built from hardware_now(), not hand-written, so the test holds on the
    measuring host and on any foreign one -- which is the point: "reused"
    must depend on the identity comparison, never on whose host runs the test.
    """
    text = (FORK / "calibration.toml").read_text(encoding="utf-8")
    now = hardware_now(capability)
    for f in HARDWARE_IDENTITY:
        old = None
        for line in text.splitlines():
            if line.startswith(f + " ="):
                old = line
                break
        assert old is not None, f"shipped profile no longer records {f}"
        text = text.replace(old, '%s = "%s"' % (f, now[f]), 1)
    p = tmp_path / "calibration.toml"
    p.write_text(text, encoding="utf-8")
    return p


def _no_measurement(monkeypatch):
    """Make the measurement matrix a hard error: it must not be reached."""
    import _lib.calibrate as C

    def _boom(*a, **kw):
        raise AssertionError("calibrate() reached the measurement matrix")

    monkeypatch.setattr(C, "measure_matrix", _boom)


@pytest.mark.fast
def test_calibrate_reuses_a_profile_that_describes_this_machine(
        kernel, real_capability, tmp_path):
    p = _this_machine_profile(tmp_path, real_capability)
    out = C_calibrate(dict(kernel.alias), path=str(p))
    assert out["status"] == "reused", out
    assert out["routing"]["matched"] is True, out["routing"]
    assert out["routing"]["reason"].startswith("shipped profile [hardware] "
                                               "matches"), out["routing"]


@pytest.mark.fast
def test_calibrate_on_a_foreign_machine_is_not_reused_and_writes_nothing(
        kernel, real_capability, tmp_path, monkeypatch):
    """The defect: pre-fix this returned {"status": "reused"} here.

    Asserted three ways, because each catches a different half-done fix:
    the status is not "reused", the byte content of the file is untouched, and
    the measurement matrix is never entered (no measurement, no write).
    """
    _no_measurement(monkeypatch)
    a = kernel.alias
    real = a["gpu_capability"]
    foreign = dict(real_capability)
    foreign["note"] = "wgpu-py AMD Radeon Pro 5500M DX12: elementwise"
    a["gpu_capability"] = lambda: foreign
    try:
        p = _this_machine_profile(tmp_path, real_capability)
        before = p.read_bytes()
        out = C_calibrate(dict(a), path=str(p))
    finally:
        a["gpu_capability"] = real
    assert out["status"] != "reused", out
    assert out["status"] == "foreign", out
    assert p.read_bytes() == before, "calibrate() rewrote a foreign profile"
    # It says WHICH machine, rather than merely declining.
    differ = {f for f, _, _ in out["routing"]["differ"]}
    assert differ == {"gpu_device", "gpu_backend"}, out["routing"]
    assert "NUMFAST_CALIBRATION_DIR" in out["note"], out["note"]
    assert "force=True" in out["note"], out["note"]


@pytest.mark.fast
def test_a_locally_measured_profile_is_authoritative_whatever_its_hardware_says(
        kernel, real_capability, tmp_path, monkeypatch):
    """Rule 1 applies to calibrate() as it does to the router: naming the
    directory IS the assertion, so a foreign [hardware] block is still reused
    rather than reported as someone else's."""
    d = _foreign_profile(tmp_path, real_capability)
    monkeypatch.setenv("NUMFAST_CALIBRATION_DIR", str(d))
    out = C_calibrate(dict(kernel.alias))
    assert out["status"] == "reused", out
    assert out["path"] == str(d / "calibration.toml"), out
    assert out["routing"]["origin"] == "local", out["routing"]
    assert out["routing"]["differ"] == [], out["routing"]


@pytest.mark.fast
def test_calibrate_refuses_to_write_into_an_installed_package_directory(
        kernel, tmp_path, monkeypatch):
    """A wheel's fork root IS the package directory, so the default write
    target for every pip-installed user is site-packages.

    Refused BEFORE the matrix runs, and the refusal names the way out.
    """
    _no_measurement(monkeypatch)
    pkg = tmp_path / "site-packages" / "numfast"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "full.toml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        C_calibrate(dict(kernel.alias), path=str(pkg / "calibration.toml"))
    msg = str(ei.value)
    assert "installed numfast package" in msg, msg
    assert "NUMFAST_CALIBRATION_DIR" in msg, msg
    assert not (pkg / "calibration.toml").exists()


@pytest.mark.fast
def test_calibrate_refuses_a_destination_that_cannot_be_written(
        kernel, tmp_path, monkeypatch):
    """Two shapes of the same refusal: the directory is not there, and it is
    there but nothing can be written into it.

    The second is checked against a real filesystem refusal (a file where a
    directory must be), not against a stubbed predicate -- os.access() answers
    for the caller's privileges, and root writes a 0o555 directory happily, so
    a permission-bit assertion would pass for the wrong reason on the machine
    that runs it.
    """
    _no_measurement(monkeypatch)
    missing = tmp_path / "no-such-dir" / "calibration.toml"
    with pytest.raises(ValueError) as ei:
        C_calibrate(dict(kernel.alias), path=str(missing))
    assert "does not exist" in str(ei.value), str(ei.value)

    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        C_calibrate(dict(kernel.alias), path=str(blocked / "calibration.toml"))
    assert "does not exist" in str(ei.value), str(ei.value)
    assert blocked.read_text(encoding="utf-8").startswith("this is a file"), \
        "calibrate() wrote into a path it was told not to"


@pytest.mark.fast
def test_the_public_calibrate_verb_never_rewrites_the_repo_profile(
        kernel, real_capability):
    """Through the kernel alias, on whatever host runs it, and the shipped
    calibration.toml comes back byte-identical either way.

    On the machine that measured it that is the gate for this change --
    calibrate() does exactly what it did before. On any other host it is the
    fix: the call reports `foreign` and stops, where before it reported
    `reused`. The assertion is on the file, not on which of the two happened,
    because the second outcome is the bug this test exists to catch.
    """
    a = kernel.alias
    real_profile = FORK / "calibration.toml"
    _p, decision = resolve_routing_profile(real_capability)
    before = real_profile.read_bytes()
    out = a["calibrate"](quick=True)
    assert out["path"] == str(real_profile), out
    assert out["status"] == ("reused" if decision["matched"] else "foreign"), \
        out
    assert real_profile.read_bytes() == before, "calibrate() wrote the profile"
    assert not list(FORK.glob(".numfast-write-probe-*")), \
        "the writability probe left a file behind"
