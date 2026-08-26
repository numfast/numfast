"""Linear regression using normal equations (matmul)."""
import numpy as np
import numfast as nf

def linear_regression(X, y):
    """Fit y = Xβ using normal equations: β = (X^T X)^{-1} X^T y.

    Uses matmul for matrix products.
    """
    XT = X.T
    n = X.shape[1]
    # X^T X (n×n)
    XTX = nf.matmul(XT.flatten(), X.flatten(), M=n, N=n, K=X.shape[0])
    XTX = XTX.reshape(n, n)
    # X^T y (n×1)
    XTy = nf.matmul(XT.flatten(), y.flatten(), M=n, N=1, K=X.shape[0])
    # Solve via numpy (inverse is a placeholder — future: operations.inv)
    beta = np.linalg.solve(XTX, XTy.flatten())
    return beta

if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_features = 100, 3
    X = np.random.random((n_samples, n_features)).astype(np.float64)
    true_beta = np.array([2.5, -1.3, 0.7])
    y = X @ true_beta + np.random.random(n_samples) * 0.1

    beta = linear_regression(X, y)
    print(f"True beta:    {true_beta}")
    print(f"Estimated:    {beta.round(4)}")
    print(f"Error:        {np.abs(beta - true_beta).max():.2e}")
