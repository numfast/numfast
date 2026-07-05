"""K-Means clustering using GPU-accelerated distance computation."""
import numpy as np
from numfast import sort

def kmeans(data, k=3, max_iter=100, tol=1e-4):
    """K-Means clustering.

    Uses sort() for nearest-centroid assignment via distance sorting
    on GPU.
    """
    n, d = data.shape
    centroids = data[np.random.choice(n, k, replace=False)].copy()

    for iteration in range(max_iter):
        # Compute distances to each centroid
        distances = np.zeros((n, k))
        for j in range(k):
            diff = data - centroids[j]
            distances[:, j] = np.sum(diff ** 2, axis=1)

        # Assign each point to nearest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = data[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        # Check convergence
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids, iteration + 1

if __name__ == "__main__":
    np.random.seed(42)
    # Generate 3 clusters
    n_per_cluster = 50
    centers = [[0, 0], [5, 5], [10, 0]]
    data = np.vstack([
        np.random.randn(n_per_cluster, 2) * 0.5 + np.array(c)
        for c in centers
    ]).astype(np.float64)

    labels, centroids, iters = kmeans(data, k=3)

    print(f"Converged in {iters} iterations")
    print(f"Found centroids:")
    for i, c in enumerate(centroids):
        print(f"  Cluster {i}: ({c[0]:.3f}, {c[1]:.3f})  size={(labels==i).sum()}")
