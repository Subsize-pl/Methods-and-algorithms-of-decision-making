from __future__ import annotations
from typing import Optional
import numpy as np
from .settings import DataGeneratorSettings as DgSettings


# Class that generates synthetic 2D Gaussian-like clusters for K-means testing
class DataGenerator:

    @staticmethod
    def generate_mixture(
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # create many "data generation centers"
        n_modes = max(
            DgSettings.MODES_MIN,
            min(DgSettings.MODES_MAX, int(np.sqrt(n_samples) // 10)),
        )
        modes = rng.uniform(
            DgSettings.X_MIN,
            DgSettings.X_MAX,
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
                DgSettings.SCALE_MIN,
                DgSettings.SCALE_MAX,
            )

            # generate points
            pts = rng.normal(loc=dc, scale=scale, size=(count, 2))
            parts.append(pts)

        # vertical stack
        data = np.vstack(parts)[:n_samples]
        return data
