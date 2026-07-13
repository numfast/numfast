# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

import pytest
from _core.backend import get_xp, set_active, device_info
from _core.backend._numpy import AVAILABLE as NUMPY_AVAILABLE


def test_get_xp_returns_module():
    xp = get_xp()
    assert xp.__name__ == "numpy"


def test_default_backend_is_numpy_on_windows():
    xp = get_xp()
    import numpy as np
    assert xp is np


def test_set_active_numpy():
    set_active("numpy")
    xp = get_xp()
    import numpy as np
    assert xp is np


def test_set_active_invalid_raises():
    with pytest.raises(ImportError, match="not available"):
        set_active("nonexistent")


def test_numpy_driver_available():
    assert NUMPY_AVAILABLE is True


def test_device_info():
    info = device_info()
    assert info["backend"] == "numpy"
    assert info["platform"] == "cpu"


def test_no_cupy_in_lib_files():
    import os
    lib_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "Series", "_lib"),
        os.path.join(os.path.dirname(__file__), "..", "src", "math", "Stats", "_lib"),
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "Tables", "_lib"),
    ]
    import re
    for d in lib_dirs:
        for f in os.listdir(d):
            if f.endswith(".py"):
                fp = os.path.join(d, f)
                with open(fp) as fh:
                    content = fh.read()
                    if re.search(r"import\s+cupy", content):
                        assert False, f"CuPy import found in {fp}"


def test_no_cupy_in_test_files():
    import os
    test_dir = os.path.join(os.path.dirname(__file__))
    import re
    for f in os.listdir(test_dir):
        if f.endswith(".py"):
            fp = os.path.join(test_dir, f)
            with open(fp) as fh:
                content = fh.read()
                if re.search(r"import\s+cupy", content):
                    assert False, f"CuPy import found in {fp}"
