"""Image histogram equalization."""
import sys, os
import numpy as np
_examples_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_examples_dir, '..'))
sys.path.insert(0, os.path.join(_examples_dir, '..', '..'))

from numfast import histogram, scan


def equalize(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Apply histogram equalization to a grayscale image.

    Args:
        image: 2D float64 array, values in [0, 1]
        bins: number of histogram bins

    Returns:
        equalized: 2D float64 array, same shape
    """
    flat = image.flatten()
    
    # Histogram
    hist = histogram(flat, bins=bins, min_val=0.0, max_val=1.0)
    
    # Cumulative distribution function (CDF)
    cdf = scan(hist)
    cdf = cdf / cdf[-1]  # normalize to [0, 1]
    
    # Map each pixel
    indices = np.clip((flat * (bins - 1)).astype(np.int64), 0, bins - 1)
    result = cdf[indices]
    
    return result.reshape(image.shape)


if __name__ == "__main__":
    # Create synthetic low-contrast image
    np.random.seed(42)
    img = np.random.random((64, 64)).astype(np.float64) * 0.2 + 0.4  # narrow range
    equalized = equalize(img)
    
    print(f"Image histogram equalization:")
    print(f"  Original:  min={img.min():.3f}  max={img.max():.3f}  mean={img.mean():.3f}")
    print(f"  Equalized: min={equalized.min():.3f}  max={equalized.max():.3f}  mean={equalized.mean():.3f}")
    print(f"  Shape: {img.shape} -> {equalized.shape}")
