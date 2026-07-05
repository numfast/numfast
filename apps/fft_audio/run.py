"""Audio frequency analysis using FFT."""
import numpy as np
from numfast import fft

def find_tones(data, sample_rate=44100, top_k=5):
    """Find dominant frequencies in an audio signal."""
    spectrum = fft(data)
    n = len(spectrum) // 2
    magnitudes = np.abs(spectrum[0::2] + 1j * spectrum[1::2])[:n // 2]
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)[:n // 2]
    indices = np.argsort(magnitudes)[-top_k:][::-1]
    return [(freqs[i], magnitudes[i]) for i in indices]

if __name__ == "__main__":
    sample_rate = 44100
    # MUST be power of 2 for NumFast FFT
    n_samples = 8192
    duration = n_samples / sample_rate
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Mix of 440Hz (A4), 880Hz (A5), and noise
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.random.random(len(t))
    
    tones = find_tones(signal, sample_rate, top_k=4)
    print("Dominant frequencies:")
    for freq, mag in tones:
        print(f"  {freq:.1f} Hz  (magnitude: {mag:.3f})")
