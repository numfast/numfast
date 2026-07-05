"""FFT — Fast Fourier Transform (iterative Cooley-Tukey, radix-2).

Complex numbers are stored as interleaved float pairs:
  [real0, imag0, real1, imag1, real2, imag2, ...]

Kernels:
  FftStage — one butterfly stage (called in loop: stage=0..log2(N)-1)
  BitReverse — bit-reversal permutation

Usage:
    # Bit-reverse first
    jobs_br = [{"op": "BitReverse", "inputs": ["complex"], "params": {"N": 1024}, "out": "rev"}]
    
    # Then loop over stages
    for stage in range(0, log2N):
        jobs_fft = [{"op": "FftStage", "inputs": ["data"], "params": {"stage": stage, "N": 1024}, "out": "data"}]
"""

from .descriptor import describe_stage, describe_bitreverse
from .cpu import cpu_stage, cpu_bitreverse
from .wgsl import wgsl_stage, wgsl_bitreverse

__all__ = [
    "describe_stage", "describe_bitreverse",
    "cpu_stage", "cpu_bitreverse",
    "wgsl_stage", "wgsl_bitreverse",
]
