"""Mandelbrot set visualization — classic parallel compute demo."""
import numpy as np

def mandelbrot(width=800, height=600, max_iter=256, 
               x_min=-2.5, x_max=1.5, y_min=-1.5, y_max=1.5):
    """Compute Mandelbrot set.
    
    Each pixel is independent — perfect parallel workload.
    NumFast map() will accelerate the iteration loop on GPU.
    """
    x = np.linspace(x_min, x_max, width).astype(np.float64)
    y = np.linspace(y_min, y_max, height).astype(np.float64)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    
    Z = np.zeros_like(C, dtype=np.complex128)
    escape = np.full(C.shape, max_iter, dtype=np.int64)
    
    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask] ** 2 + C[mask]
        escape[mask & (np.abs(Z) > 2.0)] = i
    
    return escape

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    result = mandelbrot(400, 300, max_iter=64)
    t = time.perf_counter() - t0
    print(f"Mandelbrot 400x300x64: {t*1000:.1f} ms")
    print(f"  shape={result.shape}, range=[{result.min()}, {result.max()}]")
