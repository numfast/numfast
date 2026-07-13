"""Register test kernels (MACD, FFT, Blur, Stress) in a Runtime instance."""

from Trading.MACD import describe as _macd_desc, cpu as _macd_cpu
from .kernels.fft_test import describe as _fft_desc, cpu as _fft_cpu
from .kernels.blur_test import describe as _blur_desc, cpu as _blur_cpu
from .kernels.stress_test import describe as _stress_desc, cpu as _stress_cpu


def register_test_kernels(runtime):
    runtime.register_kernel("MACD", describe=_macd_desc, cpu=_macd_cpu, abi_version=1)
    runtime.register_kernel("FFT", describe=_fft_desc, cpu=_fft_cpu, abi_version=1)
    runtime.register_kernel("BLUR", describe=_blur_desc, cpu=_blur_cpu, abi_version=1)
    runtime.register_kernel("STRESS", describe=_stress_desc, cpu=_stress_cpu, abi_version=1)
