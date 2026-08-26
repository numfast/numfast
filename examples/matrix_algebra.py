"""Matrix algebra using matmul."""
import sys, os
import numpy as np
_examples_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_examples_dir, '..'))
sys.path.insert(0, os.path.join(_examples_dir, '..', '..'))

import numfast as nf


def power_method(A: np.ndarray, iterations: int = 20) -> tuple:
    """Estimate dominant eigenvalue and eigenvector via power iteration.

    Uses matmul for the matrix-vector product.

    Args:
        A: 2D float64 matrix (n×n)
        iterations: number of iterations

    Returns:
        eigenvalue: float
        eigenvector: 1D array
    """
    n = A.shape[0]
    v = np.random.random(n).astype(np.float64)
    v = v / np.linalg.norm(v)
    
    A_flat = A.flatten()
    
    for _ in range(iterations):
        # Av via matmul (n×1 vector as n×1 matrix)
        Av = nf.matmul(A_flat, v, M=n, N=1, K=n)
        v_new = Av.flatten()
        norm = np.linalg.norm(v_new)
        v = v_new / norm
    
    # Rayleigh quotient
    Av = nf.matmul(A_flat, v, M=n, N=1, K=n).flatten()
    eigenvalue = float(np.dot(v, Av))
    
    return eigenvalue, v


if __name__ == "__main__":
    np.random.seed(42)
    n = 16
    # Symmetric matrix
    A = np.random.random((n, n)).astype(np.float64)
    A = A.T @ A  # make symmetric positive definite
    
    eigval, eigvec = power_method(A, iterations=30)
    np_eigvals = np.linalg.eigvalsh(A)
    
    print(f"Power iteration (n={n}):")
    print(f"  Estimated eigenvalue: {eigval:.6f}")
    print(f"  Largest numpy eigenvalue: {np_eigvals[-1]:.6f}")
    print(f"  Error: {abs(eigval - np_eigvals[-1]):.2e}")
