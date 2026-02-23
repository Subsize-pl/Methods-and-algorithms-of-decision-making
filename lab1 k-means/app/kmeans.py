import numpy as np


class KMeansState:
    def __init__(
        self, data: np.ndarray, k: int, rng: np.random.Generator | None = None
    ):
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must be shape (n,2)")
        if k <= 0:
            raise ValueError("k must be > 0")

        self.data = data
        self.n, self.dim = data.shape
        self.k = k
        self.rng = rng or np.random.default_rng()
        self.centroids: np.ndarray = self._init_centroids()
        self.labels: np.ndarray = np.zeros(self.n, dtype=int)
        self.iteration: int = 0
        self.converged: bool = False

    def _init_centroids(self) -> np.ndarray:
        idx = self.rng.choice(self.n, size=self.k, replace=False)
        return self.data[idx].astype(float).copy()

    def assign(self) -> None:
        dists = np.linalg.norm(
            self.data[:, None, :] - self.centroids[None, :, :],
            axis=2,
        )
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
