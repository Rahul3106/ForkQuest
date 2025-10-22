# Implement K-Means from scratch

import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, n_iterations=100):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.n_iterations):
            # Assign clusters
            labels = self._assign_clusters(X)

            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence (if centroids do not change)
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)