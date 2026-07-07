import numpy as np

def expected_lr(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

if __name__ == "__main__":
    from run import linear_regression
    X = np.random.random((50, 3)).astype(np.float64)
    y = X @ np.array([1.0, 2.0, 3.0]) + 0.01 * np.random.random(50)
    b1 = linear_regression(X, y)
    b2 = expected_lr(X, y)
    print(f"Match: {np.allclose(b1, b2, atol=1e-6)}")
