"""Sobel edge detection using matmul for convolution."""
import numpy as np
import numfast as nf

def sobel_edge(image):
    """Detect edges using Sobel operator via matmul.
    
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    
    Implemented as matrix multiplication for separable parts.
    """
    h, w = image.shape
    
    # Build derivative matrices
    Dx = np.zeros((w, w), dtype=np.float64)
    Dy = np.zeros((h, h), dtype=np.float64)
    
    for i in range(w):
        if i == 0:
            Dx[i, i+1] = 1.0
        elif i == w - 1:
            Dx[i, i-1] = -1.0
        else:
            Dx[i, i-1] = -1.0
            Dx[i, i+1] = 1.0
    
    for i in range(h):
        if i == 0:
            Dy[i, i+1] = 1.0
        elif i == h - 1:
            Dy[i, i-1] = -1.0
        else:
            Dy[i, i-1] = -1.0
            Dy[i, i+1] = 1.0
    
    # Gx = image @ Dx  (horizontal gradient via matmul)
    Gx = nf.matmul(image.flatten(), Dx.flatten(), M=h, N=w, K=w).reshape(h, w)
    
    # Gy = Dy @ image  (vertical gradient via matmul)
    Gy = nf.matmul(Dy.flatten(), image.flatten(), M=h, N=w, K=h).reshape(h, w)
    
    magnitude = np.sqrt(Gx**2 + Gy**2)
    return magnitude

if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic image with a vertical edge
    img = np.zeros((32, 32), dtype=np.float64)
    img[:, 16:] = 1.0
    
    edges = sobel_edge(img)
    print(f"Edge map:   min={edges.min():.3f}  max={edges.max():.3f}")
    print(f"Edge column 16 (expected high): {edges[16, 15:18].round(3)}")
    print(f"Edge column 0  (expected low):  {edges[16, 0:3].round(3)}")
