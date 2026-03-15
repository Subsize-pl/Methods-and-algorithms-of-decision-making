from __future__ import annotations
from common.config import DgSettings
import numpy as np


# Class that generates synthetic 2D Gaussian-like clusters for K-means testing
class GaussianDataGenerator:

    def __init__(
        self,
        min_modes: int = DgSettings.MIN_MODES,  # min number of cluster centers (data generation centers)
        max_modes: int = DgSettings.MAX_MODES,  # max number of cluster centers
        min_x: int = DgSettings.MIN_X,  # min coordinate value for generated centers
        max_x: int = DgSettings.MAX_X,  # max coordinate value for generated centers
        min_scale: int = DgSettings.MIN_SCALE,  # min standard deviation of clusters (cluster spread)
        max_scale: int = DgSettings.MAX_SCALE,  # max standard deviation of clusters
    ) -> None:
        self.min_modes = min_modes
        self.max_modes = max_modes
        self.x_min = min_x
        self.x_max = max_x
        self.min_scale = min_scale
        self.max_scale = max_scale

    def generate_mixture(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # create many "data generation centers"
        n_modes = max(
            self.min_modes,
            min(self.max_modes, int(np.sqrt(n_samples) // 10)),
        )
        modes = rng.uniform(
            self.x_min,
            self.x_max,
            size=(n_modes, 2),
        )

        # how many points per center
        per = int(np.ceil(n_samples / n_modes))
        parts = []

        # dc = distribution center. E.g. [12.1, 43.1]
        for i, dc in enumerate(modes):
            count = per if i < n_modes - 1 else n_samples - len(parts) * per

            # scale - standard deviation
            scale = rng.uniform(
                self.min_scale,
                self.max_scale,
            )

            # generate points
            pts = rng.normal(loc=dc, scale=scale, size=(count, 2))
            parts.append(pts)

        # vertical stack
        data = np.vstack(parts)[:n_samples]
        return data
