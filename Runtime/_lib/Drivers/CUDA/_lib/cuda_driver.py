"""CUDA Driver stub — NotImplementedError until CUDA backend is implemented."""
from ...base import Driver


class CudaDriver(Driver):
    """CUDA Driver — заглушка.
    
    Будет реализован после стабилизации Driver API.
    Использует PyCUDA или CUDA Python.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError(
            "CUDA Driver is not yet implemented. "
            "See _specs/numfast/docs_ru/Driver_API.md for the interface contract."
        )

    def execute(self, packet):
        raise NotImplementedError("CUDA Driver: execute")
