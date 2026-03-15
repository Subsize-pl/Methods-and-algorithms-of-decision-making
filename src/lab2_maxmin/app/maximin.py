from __future__ import annotations
import numpy as np


class MaximinState:
    """
    Maximin clustering algorithm for 2D data.

    Algorithm steps:
    1. Randomly choose one data point as the first centroid C1.
    2. Select the point that is farthest from C1 and use it as the second centroid C2.
    3. Assign every data point to the nearest centroid.
    4. For each cluster:
       - find the point with the maximum distance from its centroid (delta_k).
    5. Let kp be the largest of these distances.
    6. Compute the average pairwise distance between all current centroids.
    7. If kp > 0.5 * average_centroid_distance:
       - add that farthest point as a new centroid
       - repeat from step 3.
    8. Otherwise stop — the set of centroids is considered stable.

    The resulting centroids can be used as good initial centers
    for algorithms like K-Means.
    """

    def __init__(self, data: np.ndarray, rng: np.random.Generator | None = None):
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("data must be shape (n,2)")
        self.data = data.astype(float)
        self.n, self.dim = data.shape
        self.rng = rng or np.random.default_rng()

        # centroids (m,2)
        self.centroids: np.ndarray = np.empty((0, 2), dtype=float)
        # labels: -1 unassigned, otherwise index of centroid
        self.labels: np.ndarray = np.full(self.n, -1, dtype=int)

        self.iteration: int = 0
        self.converged: bool = False

        # init with one random centroid
        idx0 = int(self.rng.integers(0, self.n))
        self.centroids = np.vstack(
            [self.centroids, self.data[idx0].copy().astype(float)]
        )

    def assign(self) -> None:
        """Assign each point to the nearest centroid (Euclidean)."""
        if self.centroids.shape[0] == 0:
            raise RuntimeError("No centroids")
        dists = np.linalg.norm(
            self.data[:, None, :] - self.centroids[None, :, :], axis=2
        )
        self.labels = np.argmin(dists, axis=1)

    def _pairwise_centroid_mean_distance(self) -> float:
        m = self.centroids.shape[0]
        if m < 2:
            return 0.0
        diffs = self.centroids[:, None, :] - self.centroids[None, :, :]
        dmat = np.linalg.norm(diffs, axis=2)
        iu = np.triu_indices(m, k=1)
        vals = dmat[iu]
        return float(vals.mean())

    def _find_cluster_max_deltas(self) -> tuple[np.ndarray, np.ndarray]:
        """For each cluster compute max distance to centroid and the index of that point."""
        m = self.centroids.shape[0]
        max_deltas = np.zeros(m, dtype=float)
        max_idx = np.full(m, -1, dtype=int)
        for k in range(m):
            members = np.where(self.labels == k)[0]
            if members.size == 0:
                continue
            dists = np.linalg.norm(self.data[members] - self.centroids[k], axis=1)
            j = int(np.argmax(dists))
            max_deltas[k] = float(dists[j])
            max_idx[k] = int(members[j])
        return max_deltas, max_idx

    def step(self) -> tuple[float, bool]:
        """
        Perform one iteration:
          - if only 1 centroid exists -> add the farthest point as 2nd centroid
          - else assign, find cluster max-deltas, take global kp and decide to add new centroid
        Returns (kp, added_flag)
        """
        if self.converged:
            return 0.0, False

        # special case: create second centroid
        if self.centroids.shape[0] == 1:
            dists = np.linalg.norm(self.data - self.centroids[0], axis=1)
            idx = int(np.argmax(dists))
            self.centroids = np.vstack(
                [self.centroids, self.data[idx].copy().astype(float)]
            )
            self.assign()
            self.iteration += 1
            return float(dists[idx]), True

        # assign points
        self.assign()

        # per-cluster maxima
        max_deltas, max_idx = self._find_cluster_max_deltas()
        kp = float(max_deltas.max()) if max_deltas.size > 0 else 0.0
        kp_cluster = int(np.argmax(max_deltas)) if max_deltas.size > 0 else -1
        kp_data_idx = (
            int(max_idx[kp_cluster])
            if (kp_cluster >= 0 and max_idx[kp_cluster] >= 0)
            else -1
        )

        avg_pairwise = self._pairwise_centroid_mean_distance()

        added = False
        if avg_pairwise > 0.0 and kp > 0.5 * avg_pairwise and kp_data_idx >= 0:
            # add new centroid
            self.centroids = np.vstack(
                [self.centroids, self.data[kp_data_idx].copy().astype(float)]
            )
            added = True
            self.iteration += 1
            self.assign()
        else:
            self.converged = True

        return kp, added

    def fit(self, max_iter: int = 1000) -> None:
        """Run until convergence or max_iter."""
        while not self.converged and self.iteration < max_iter:
            _, _ = self.step()

    def inertia(self) -> float:
        """Sum of squared distances to assigned centroids (compactness)."""
        if self.centroids.shape[0] == 0:
            return 0.0
        if np.any(self.labels < 0):
            self.assign()
        return float(np.sum((self.data - self.centroids[self.labels]) ** 2))
