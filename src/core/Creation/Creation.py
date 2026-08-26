"""Creation Extension -- публичный API создания массивов (S200).
Builder entry point.
"""

from _lib.api import (
    arange,
    full,
    index,
    linspace,
    ones,
    random_integers,
    random_normal,
    random_uniform,
    repeat,
    tile,
    zeros,
)
from _lib.array import CreatedArray
from _lib.bridge import to_series


def setup(kernel):
    """Register in kernel metadata."""
    kernel.metadata.setdefault("Creation", {})
    kernel.metadata["Creation"]["version"] = "0.1.0"
    kernel.metadata["Creation"]["types"] = [
        "zeros", "ones", "full", "arange", "index", "linspace",
        "tile", "repeat",
        "random_uniform", "random_normal", "random_integers",
        "CreatedArray", "to_series",
    ]
