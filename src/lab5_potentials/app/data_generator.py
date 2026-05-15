from dataclasses import dataclass
from typing import Optional

import numpy as np


# Small labeled set used by the potential method
@dataclass(slots=True)
class PotentialDataset:
    points: np.ndarray
    labels: np.ndarray


class PotentialDataGenerator:
    # Generates a tiny training set and a larger test set
    def __init__(
        self,
        class1_center: tuple[float, float] = (-1.8, 0.8),
        class2_center: tuple[float, float] = (1.9, 1.8),
        train_spread: float = 0.28,
        test_spread: float = 0.45,
    ) -> None:
        self.class1_center = np.array(class1_center, dtype=float)
        self.class2_center = np.array(class2_center, dtype=float)
        self.train_spread = float(train_spread)
        self.test_spread = float(test_spread)

    def generate_training_set(self, seed: Optional[int] = None) -> PotentialDataset:
        rng = np.random.default_rng(seed)

        # 6 training objects, 3 per class
        class1 = rng.normal(
            loc=self.class1_center,
            scale=self.train_spread,
            size=(3, 2),
        )
        class2 = rng.normal(
            loc=self.class2_center,
            scale=self.train_spread,
            size=(3, 2),
        )

        points = np.vstack([class1, class2])
        labels = np.array([1, 1, 1, -1, -1, -1], dtype=int)

        order = rng.permutation(points.shape[0])
        return PotentialDataset(points=points[order], labels=labels[order])

    def generate_test_set(
        self,
        n_samples: int = 250,
        seed: Optional[int] = None,
    ) -> PotentialDataset:
        if n_samples < 2:
            raise ValueError('n_samples must be >= 2')

        rng = np.random.default_rng(seed)

        n1 = n_samples // 2
        n2 = n_samples - n1

        class1 = rng.normal(
            loc=self.class1_center,
            scale=self.test_spread,
            size=(n1, 2),
        )
        class2 = rng.normal(
            loc=self.class2_center,
            scale=self.test_spread,
            size=(n2, 2),
        )

        points = np.vstack([class1, class2])
        labels = np.array([1] * n1 + [-1] * n2, dtype=int)

        order = rng.permutation(points.shape[0])
        return PotentialDataset(points=points[order], labels=labels[order])
