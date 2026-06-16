# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

AVAILABLE = True
PRIORITY = 10


def get_xp():
    import numpy as np
    return np


def get_device():
    return None


def device_info() -> dict:
    import numpy as np
    return {
        "backend": "numpy",
        "version": np.__version__,
        "platform": "cpu",
    }
