# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Loader Extension — P0-L file->Storage. Builder entry point."""

from Loader._lib.dzst import load_dzst


def setup(kernel):
    kernel.metadata.setdefault("Loader", {})
    kernel.metadata["Loader"]["version"] = "0.1.0"
    kernel.metadata["Loader"]["description"] = "P0-L file->Storage"
    kernel.metadata["Loader"]["types"] = ["load_dzst"]
    kernel.metadata["Loader"]["alias"] = "load_dzst"
    kernel.load_dzst = load_dzst
