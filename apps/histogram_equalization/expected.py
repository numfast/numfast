import numpy as np

def expected_histogram(image, bins=256):
    flat = image.flatten()
    hist, _ = np.histogram(flat, bins=bins, range=(0, 1))
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    indices = np.clip((flat * (bins - 1)).astype(np.int64), 0, bins - 1)
    return cdf[indices].reshape(image.shape)

if __name__ == "__main__":
    from run import equalize
    img = np.random.random((64, 64)).astype(np.float64)
    r1 = equalize(img)
    r2 = expected_histogram(img)
    print(f"Match: {np.allclose(r1, r2, atol=1e-10)}")
