from __future__ import annotations
from typing import Optional
import numpy as np
from .settings import DataGeneratorSettings as DgSettings


class DataGenerator:

    @staticmethod
    def generate_mixture(n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_modes = max(
            DgSettings.MODES_MIN,
            min(DgSettings.MODES_MAX, int(np.sqrt(n_samples) // 10)),
        )
        modes = rng.uniform(
            DgSettings.X_MIN,
            DgSettings.X_MAX,
            size=(n_modes, 2),
        )
        per = int(np.ceil(n_samples / n_modes))
        parts = []

        for i, mu in enumerate(modes):
            count = per if i < n_modes - 1 else n_samples - len(parts) * per
            scale = rng.uniform(
                DgSettings.SCALE_MIN,
                DgSettings.SCALE_MAX,
            )
            pts = rng.normal(loc=mu, scale=scale, size=(count, 2))
            parts.append(pts)

        data = np.vstack(parts)[:n_samples]
        return data
