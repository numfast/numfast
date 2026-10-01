# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only
"""Test gates: маркеры fast/heavy. Default = только fast; heavy — только явно (-m heavy)."""

import sys
from pathlib import Path

import pytest

# Test tooling: собственный _builder пакета; dev-запуск из корня репо через src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    config.addinivalue_line("markers", "fast: seconds-scale tests, default selection")
    config.addinivalue_line("markers", "heavy: memory/load tests, run only with -m heavy")


def pytest_collection_modifyitems(config, items):
    if config.option.markexpr.strip():
        return
    skip_heavy = pytest.mark.skip(reason="heavy: run explicitly with pytest -m heavy")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
