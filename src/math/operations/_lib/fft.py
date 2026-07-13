"""High-level FFT operation."""
import math
import numpy as np
from Runtime import Runtime
from Compute import register_all


def fft(data: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Fast Fourier Transform (complex interleaved).

    Creates Runtime ONCE and reuses it across BitReverse + log2N butterfly stages.
    Each stage compiles a single job and executes it; output chains to next.

    Args:
        data: 1D float64 array, interleaved [real0, imag0, real1, imag1, ...]
              or 1D float64 real-only array
        use_gpu: use WebGPU if True

    Returns:
        result: 1D float64 array, interleaved [real, imag, ...]
    """
    from Runtime._lib.Drivers.WebGPU import WebGpuDriver

    # Handle real-only input
    if data.ndim == 1 and data.dtype in (np.float32, np.float64):
        n = len(data)
        complex_data = np.empty(2 * n, dtype=np.float64)
        complex_data[0::2] = data
        complex_data[1::2] = 0.0
    else:
        complex_data = data

    N = len(complex_data) // 2
    log2N = int(math.log2(N))

    # --- Runtime created ONCE ---
    driver = WebGpuDriver() if use_gpu else None
    rt = Runtime(driver=driver)
    register_all(rt)

    # Bit-reverse
    rt.execute(
        rt.compile([{"op": "BitReverse", "inputs": ["complex"], "params": {"N": N}, "out": "rev"}]),
        {"complex": complex_data}
    )
    d = rt.driver.resolve_output("rev")

    # Butterfly stages — each stage uses same Runtime
    for stage in range(log2N):
        rt.execute(
            rt.compile([{"op": "FftStage", "inputs": ["complex"],
                         "params": {"stage": stage, "N": N}, "out": "complex"}]),
            {"complex": d}
        )
        d = rt.driver.resolve_output("complex")

    if driver:
        driver.release()

    return d
