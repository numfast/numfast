"""Histogram equalization for grayscale images."""
import numpy as np
from numfast import histogram, scan

def equalize(image, bins=256):
    flat = image.flatten()
    hist = histogram(flat, bins=bins, min_val=0.0, max_val=1.0)
    cdf = scan(hist)
    cdf = cdf / cdf[-1]
    indices = np.clip((flat * (bins - 1)).astype(np.int64), 0, bins - 1)
    result = cdf[indices]
    return result.reshape(image.shape)

if __name__ == "__main__":
    np.random.seed(42)
    img = np.random.random((256, 256)).astype(np.float64) * 0.3 + 0.35
    eq = equalize(img)
    print(f"Original:  min={img.min():.3f}  max={img.max():.3f}  mean={img.mean():.3f}")
    print(f"Equalized: min={eq.min():.3f}  max={eq.max():.3f}  mean={eq.mean():.3f}")
    print(f"Shape: {img.shape} -> {eq.shape}")
