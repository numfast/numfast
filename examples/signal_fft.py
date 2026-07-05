"""Signal frequency analysis using FFT."""
import sys, os
import numpy as np
_examples_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_examples_dir, '..'))
sys.path.insert(0, os.path.join(_examples_dir, '..', '..'))

from numfast import fft


def find_dominant_frequencies(signal: np.ndarray, sample_rate: float = 1.0, top_k: int = 3):
    """Find dominant frequencies in a signal.

    Args:
        signal: 1D real-valued signal
        sample_rate: samples per second
        top_k: number of dominant frequencies to return

    Returns:
        frequencies: list of (freq, magnitude) tuples
    """
    n = len(signal)
    spectrum = fft(signal)
    
    # Convert interleaved to complex
    complex_spec = spectrum[0::2] + 1j * spectrum[1::2]
    
    # Magnitude spectrum (only positive frequencies)
    magnitudes = np.abs(complex_spec[:n // 2])
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)[:n // 2]
    
    # Top K
    indices = np.argsort(magnitudes)[-top_k:][::-1]
    return [(freqs[i], magnitudes[i]) for i in indices]


if __name__ == "__main__":
    # Create test signal: 5Hz + 12Hz + noise (FFT requires power-of-2 length)
    sample_rate = 64.0
    t = np.linspace(0, 1.0, int(sample_rate), endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t) + 0.1 * np.random.random(len(t))
    
    print("Signal frequency analysis:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {len(t)}")
    
    freqs = find_dominant_frequencies(signal, sample_rate, top_k=4)
    print(f"  Dominant frequencies:")
    for f, m in freqs:
        print(f"    {f:.1f} Hz (magnitude: {m:.3f})")
