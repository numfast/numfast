import numpy as np
from scipy.ndimage import uniform_filter

def expected_blur(image, kernel_size=5):
    return uniform_filter(image, size=kernel_size)

if __name__ == "__main__":
    from run import box_blur
    img = np.random.random((16, 16)).astype(np.float64)
    r1 = box_blur(img, kernel_size=3)
    r2 = expected_blur(img, kernel_size=3)
    print(f"Match: {np.allclose(r1, r2, atol=1e-6)}")
