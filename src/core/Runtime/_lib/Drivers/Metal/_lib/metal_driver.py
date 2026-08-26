"""Metal Driver stub."""
from Runtime._lib.Drivers.base import Driver


class MetalDriver(Driver):
    def __init__(self):
        super().__init__()
        raise NotImplementedError(
            "Metal Driver is not yet implemented. "
            "See docs/ARCHITECTURE.md for the interface contract."
        )

    def execute(self, packet):
        raise NotImplementedError("Metal Driver: execute")
