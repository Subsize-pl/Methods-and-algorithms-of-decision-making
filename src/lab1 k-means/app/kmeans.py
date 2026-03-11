import numpy as np


# Object that stores the current state of the Lloyd algorithm
class KMeansState:
    def __init__(
        self,
        data: np.ndarray,
        k: int,
        rng: np.random.Generator | None = None,
    ):
        # ndim = number of array dimensions. Must be 2 (plane)
        # data.shape[1] = coordinates count. Must be 2 (x, y)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must be shape (n,2)")
        if k <= 0:
            raise ValueError("k must be > 0")

        self.data = data
        self.n, self.dim = data.shape
        self.k = k
        self.rng = rng or np.random.default_rng()

        # Select k random points
        self.centroids: np.ndarray = self._init_centroids()

        # Arr that stores the cluster number for each point.
        self.labels: np.ndarray = np.zeros(self.n, dtype=int)
        self.iteration: int = 0
        self.converged: bool = False

    def _init_centroids(self) -> np.ndarray:
        # Randomly select k unique indices from the range [0, n)
        idx = self.rng.choice(self.n, size=self.k, replace=False)
        return self.data[idx].astype(float).copy()

    def assign(self) -> None:
        # Compute Euclidean distance from every point to every centroid.
        # data[:, None, :]      -> shape (n_points, 1, dim)
        # centroids[None, :, :] -> shape (1, k, dim)
        # Broadcasting produces (n_points, k, dim) pairwise coordinate differences.
        dists = np.linalg.norm(
            self.data[:, None, :] - self.centroids[None, :, :],
            axis=2,
        )

        # For each point select the index of the nearest centroid.
        # Result: labels array of shape (n_points,)
        self.labels = np.argmin(dists, axis=1)

    def update(self) -> tuple[np.ndarray, list]:
        new_centroids = np.zeros_like(self.centroids)
        empty = []
        for i in range(self.k):
            members = self.data[self.labels == i]
            if members.shape[0] == 0:
                empty.append(i)
            else:
                new_centroids[i] = members.mean(axis=0)
        return new_centroids, empty

    def handle_empty(self, new_centroids: np.ndarray, empty: list) -> None:
        for c in empty:
            new_centroids[c] = self.data[self.rng.integers(0, self.n)]

    def step(self, tol: float = 1e-3, reinit_empty: bool = True) -> float:
        """
        Perform one K-means iteration: assign points to centroids and update centroids.

        :param tol: Threshold for maximum centroid shift to consider convergence (default 1e-3)
        :param reinit_empty: If True, reinitialize empty clusters with random points
        :return: Maximum centroid movement in this iteration. If <= tol, self.converged is set to True
        """

        if self.converged:
            return 0.0

        self.assign()

        new_centroids, empty = self.update()
        if empty and reinit_empty:
            self.handle_empty(new_centroids, empty)

        shifts = np.linalg.norm(self.centroids - new_centroids, axis=1)
        max_shift = float(shifts.max()) if shifts.size > 0 else 0.0

        self.centroids = new_centroids
        self.iteration += 1

        if max_shift <= tol:
            self.converged = True

        return max_shift

    def inertia(self) -> float:
        return float(np.sum((self.data - self.centroids[self.labels]) ** 2))
