"""Image box blur using matmul (separable convolution)."""
import numpy as np
import numfast as nf

def box_blur(image, kernel_size=5):
    """Apply box blur via separable convolution using matmul.
    
    Box blur = 1/(k^2) * ones(k,k) is separable:
    row blur: rows_avg = image @ (1/k * ones(k,1))
    col blur: result = (1/k * ones(1,k)) @ rows_avg
    """
    h, w = image.shape
    k = kernel_size
    
    # Build averaging matrix (w × w)
    avg_mat = np.zeros((w, w), dtype=np.float64)
    for i in range(w):
        left = max(0, i - k // 2)
        right = min(w, i + k // 2 + 1)
        n = right - left
        avg_mat[i, left:right] = 1.0 / n
    
    # Row blur: (h×w) @ (w×w) = h×w
    row_blurred = nf.matmul(image.flatten(), avg_mat.flatten(), M=h, N=w, K=w)
    row_blurred = row_blurred.reshape(h, w)
    
    # Col blur: (w×h) → same matrix applied to transpose
    col_blurred = nf.matmul(row_blurred.T.flatten(), avg_mat.flatten(), M=w, N=w, K=w)
    col_blurred = col_blurred.reshape(w, h).T
    
    return col_blurred

if __name__ == "__main__":
    np.random.seed(42)
    img = np.random.random((64, 64)).astype(np.float64)
    blurred = box_blur(img, kernel_size=5)
    print(f"Original:  {img.min():.3f} .. {img.max():.3f}, shape={img.shape}")
    print(f"Blurred:   {blurred.min():.3f} .. {blurred.max():.3f}, shape={blurred.shape}")
    print(f"Diff:      {np.abs(img - blurred).max():.3f} (edges preserved, interior smoothed)")
