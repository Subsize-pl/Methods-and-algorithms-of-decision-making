from __future__ import annotations
from common.config import DgSettings
from typing import Optional
import numpy as np


# Class that generates random 2D points in a given coordinate range
class RandomDataGenerator:

    def __init__(
        self,
        min_x: int = DgSettings.MIN_X,  # min coordinate value for generated centers
        max_x: int = DgSettings.MAX_X,  # max coordinate value for generated centers
    ) -> None:
        self.min_x = min_x
        self.max_x = max_x

    def generate_mixture(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:

        rng = np.random.default_rng(seed)

        # generate completely random points in the square [min_x, max_x]
        data = rng.uniform(
            low=[self.min_x, self.min_x],
            high=[self.max_x, self.max_x],
            size=(n_samples, 2),
        )

        return data
