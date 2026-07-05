import numpy as np

def expected_fft(data):
    return np.fft.fft(data)

if __name__ == "__main__":
    from run import find_tones
    n_samples = 8192
    t = np.linspace(0, n_samples / 44100, n_samples, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)
    tones = find_tones(signal, 44100, 3)
    print(f"Detected: {tones[0][0]:.1f} Hz (expected 440 Hz)")
