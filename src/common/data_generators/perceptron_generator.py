from __future__ import annotations

from typing import Tuple

import numpy as np


class PerceptronDataGenerator:
    # Generates simple 2D  separable clusters for demopurposes
    def __init__(self, radius: float = 8.0, spread: float = 1.0) -> None:
        self.radius = float(radius)
        self.spread = float(spread)

    def generate_2d(
        self,
        n_classes: int,
        samples_per_class: int,
        seed: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be > 0")

        rng = np.random.default_rng(seed)

        parts: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for cls in range(n_classes):
            angle = 2.0 * np.pi * cls / n_classes
            center = np.array(
                [self.radius * np.cos(angle), self.radius * np.sin(angle)],
                dtype=float,
            )

            # Cluster around the class center
            pts = rng.normal(
                loc=center,
                scale=self.spread,
                size=(samples_per_class, 2),
            )
            parts.append(pts)
            labels.append(np.full(samples_per_class, cls, dtype=int))

        data = np.vstack(parts)
        y = np.concatenate(labels)

        # Shuffle samples to mix classes
        idx = rng.permutation(data.shape[0])
        return data[idx], y[idx]
