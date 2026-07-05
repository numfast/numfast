import numpy as np
from run import kmeans

if __name__ == "__main__":
    np.random.seed(42)
    data = np.vstack([
        np.random.randn(50, 2) * 0.5 + np.array(c)
        for c in [[0, 0], [5, 5], [10, 0]]
    ]).astype(np.float64)
    labels, centroids, iters = kmeans(data, k=3)
    print(f"K-Means: {iters} iterations, {len(np.unique(labels))} clusters found")
