"""OpenCL Driver stub."""
from Runtime._lib.Drivers.base import Driver


class OpenClDriver(Driver):
    def __init__(self):
        super().__init__()
        raise NotImplementedError(
            "OpenCL Driver is not yet implemented. "
            "See docs/ARCHITECTURE.md for the interface contract."
        )

    def execute(self, packet):
        raise NotImplementedError("OpenCL Driver: execute")
