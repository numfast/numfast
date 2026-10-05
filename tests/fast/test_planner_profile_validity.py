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
    HARDWARE_IDENTITY, hardware_match, hardware_now, profile_path,
    resolve_routing_profile, routing_reject_warning)


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


def _foreign_profile(tmp_path):
    """A byte-valid copy of the shipped profile whose [hardware] is elsewhere."""
    d = tmp_path / "caldir"
    d.mkdir()
    text = (FORK / "calibration.toml").read_text(encoding="utf-8")
    for old, new in (
            ('cpu = "Intel64 Family 6 Model 79 Stepping 1, GenuineIntel"',
             'cpu = "aarch64 (foreign host)"'),
            ('platform = "Windows-11-10.0.26100-SP0"',
             'platform = "Linux-6.8.0-x86_64"'),
            ('python = "3.14.6"', 'python = "3.12.3"'),
            ('gpu_device = "NVIDIA GeForce RTX 2060"',
             'gpu_device = "AMD Radeon Pro 5500M"'),
            ('gpu_backend = "Vulkan"', 'gpu_backend = "DX12"')):
        assert old in text, f"shipped profile no longer says {old!r}"
        text = text.replace(old, new)
    (d / "calibration.toml").write_text(text, encoding="utf-8")
    return d


# 1. The machine this profile was measured on still gets it. If this fails,
#    the recorded ClickBench / H2O figures stop being reproducible here.
@pytest.mark.fast
def test_shipped_profile_applies_on_the_machine_that_measured_it(real_capability):
    prof, dec = resolve_routing_profile(gpu_capability=real_capability)
    assert prof is not None, (
        "the shipped profile must still drive routing on the machine that "
        f"measured it; refused: {dec['reason']}")
    assert dec["routing"] is True and dec["origin"] == "shipped"
    assert dec["differ"] == []
    # The identity fields are exactly the ones compared, and all of them match.
    recorded = dec["profile_hardware"]
    for f in HARDWARE_IDENTITY:
        assert str(recorded.get(f, "unknown")) == \
            str(dec["hardware"].get(f, "unknown")), f


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
        assert {f for f, _, _ in info["routing"]["differ"]} == \
            {"gpu_device", "gpu_backend"}, info["routing"]["differ"]
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
    d = _foreign_profile(tmp_path)
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
