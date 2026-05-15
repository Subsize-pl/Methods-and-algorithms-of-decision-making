from dataclasses import dataclass
from typing import Optional

import numpy as np


# Small report for one training epoch
@dataclass(slots=True)
class TrainStepResult:
    updates: int
    accuracy: float
    support_vectors: int


class PotentialState:
    """
    Potential method for binary classification.

    Idea:
    1. Build a separating function as a sum of partial potentials.
    2. Check each training object.
    3. If it is misclassified, add its potential with the correct sign.
    4. Repeat until no errors remain.
    """

    def __init__(
        self,
        train_points: np.ndarray,
        train_labels: np.ndarray,
        test_points: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
    ) -> None:
        if train_points.ndim != 2 or train_points.shape[1] != 2:
            raise ValueError('train_points must have shape (n, 2)')
        if train_labels.ndim != 1:
            raise ValueError('train_labels must be a 1D array')
        if train_points.shape[0] != train_labels.shape[0]:
            raise ValueError('train_points and train_labels sizes must match')

        unique = set(np.unique(train_labels).tolist())
        if not unique.issubset({-1, 1}):
            raise ValueError('train_labels must contain only -1 and 1')

        self.train_points = train_points.astype(float)
        self.train_labels = train_labels.astype(int)
        self.test_points = None if test_points is None else test_points.astype(float)
        self.test_labels = None if test_labels is None else test_labels.astype(int)

        self.n_train = self.train_points.shape[0]
        self.weights = np.zeros(4, dtype=float)
        self.alphas = np.zeros(self.n_train, dtype=int)

        self.iteration = 0
        self.converged = False
        self.last_updates = 0
        self.train_history: list[int] = []

        self.test_predictions: Optional[np.ndarray] = None
        self.test_accuracy: Optional[float] = None

    @staticmethod
    def basis(point: np.ndarray) -> np.ndarray:
        # First Hermite-style basis terms for two variables
        x1 = float(point[0])
        x2 = float(point[1])
        return np.array([1.0, 2.0 * x1, 2.0 * x2, 4.0 * x1 * x2], dtype=float)

    def decision_function(self, point: np.ndarray) -> float:
        # Weighted sum of basis terms
        return float(self.weights @ self.basis(point))

    def decision_function_batch(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must have shape (n, 2)')

        phi = np.column_stack(
            [
                np.ones(points.shape[0], dtype=float),
                2.0 * points[:, 0],
                2.0 * points[:, 1],
                4.0 * points[:, 0] * points[:, 1],
            ]
        )
        return phi @ self.weights

    def predict(self, points: np.ndarray) -> np.ndarray | int:
        scores = self.decision_function_batch(points)
        labels = np.where(scores >= 0.0, 1, -1)
        if np.asarray(points).ndim == 1:
            return int(labels[0])
        return labels

    def support_vector_count(self) -> int:
        return int(np.count_nonzero(self.alphas))

    def decision_expression(self) -> str:
        w0, w1, w2, w3 = self.weights
        return f'{w0:.3f} + {w1:.3f}*2x1 + {w2:.3f}*2x2 + {w3:.3f}*4x1x2'

    def train_step(self) -> TrainStepResult:
        """
        Run one epoch.
        Returns number of corrections made in this epoch.
        """
        if self.converged:
            return TrainStepResult(
                updates=0,
                accuracy=self.train_accuracy(),
                support_vectors=self.support_vector_count(),
            )

        updates = 0

        # Online corrections, object by object
        for idx in range(self.n_train):
            point = self.train_points[idx]
            true_label = int(self.train_labels[idx])
            score = self.decision_function(point)

            if true_label * score <= 0.0:
                self.alphas[idx] += 1
                self.weights += true_label * self.basis(point)
                updates += 1

        self.iteration += 1
        self.last_updates = updates
        self.train_history.append(updates)

        if updates == 0:
            self.converged = True

        return TrainStepResult(
            updates=updates,
            accuracy=self.train_accuracy(),
            support_vectors=self.support_vector_count(),
        )

    def train_accuracy(self) -> float:
        preds = self.predict(self.train_points)
        return float(np.mean(preds == self.train_labels))

    def fit(self, max_epochs: int = 1000) -> None:
        # Stop when no corrections remain or max_epochs is reached
        while not self.converged and self.iteration < max_epochs:
            self.train_step()

    def classify_test_set(self) -> tuple[np.ndarray, Optional[float]]:
        if self.test_points is None:
            raise ValueError('test_points were not provided')

        preds = np.asarray(self.predict(self.test_points))
        self.test_predictions = preds

        if self.test_labels is None:
            self.test_accuracy = None
        else:
            self.test_accuracy = float(np.mean(preds == self.test_labels))

        return preds, self.test_accuracy
