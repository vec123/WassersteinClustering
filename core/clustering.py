import numpy as np
from typing import List, Optional

class WassersteinKMeans:
    def __init__(self, n_clusters: int = 2, max_iter: int = 100, tolerance: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centroids: Optional[List[np.ndarray]] = None

    def _wasserstein_distance(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        # P=1 Wasserstein Distance for empirical distributions
        return np.mean(np.abs(np.sort(dist1) - np.sort(dist2)))

    def _get_barycenter(self, distributions: List[np.ndarray]) -> np.ndarray:
        return np.median(np.sort(distributions, axis=0), axis=0)

    def fit(self, distributions: List[np.ndarray]):
        n = len(distributions)
        if n < self.n_clusters: return self
        
        # Initialize
        idx = np.random.choice(n, self.n_clusters, replace=False)
        self.centroids = [distributions[i].copy() for i in idx]

        for _ in range(self.max_iter):
            # Assign
            labels = np.array([np.argmin([self._wasserstein_distance(d, c) for c in self.centroids]) for d in distributions])
            
            # Update
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_pts = [distributions[i] for i in range(n) if labels[i] == k]
                new_centroids.append(self._get_barycenter(cluster_pts) if cluster_pts else distributions[np.random.randint(n)])

            if sum(self._wasserstein_distance(o, n) for o, n in zip(self.centroids, new_centroids)) < self.tolerance:
                break
            self.centroids = new_centroids
        
        # Sort centroids by variance (0: Bull/Low Vol, 1: Bear/High Vol)
        self.centroids.sort(key=lambda x: np.var(x))
        return self

    def predict(self, dist: np.ndarray) -> int:
        return np.argmin([self._wasserstein_distance(dist, c) for c in self.centroids])