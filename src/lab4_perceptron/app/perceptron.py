from __future__ import annotations

from typing import Optional

import numpy as np


class PerceptronState:
    """
    Multi-class perceptron state.

    Steps:
    1. Add bias to each vector.
    2. Compute N decision values.
    3. Choose the class with the maximum value.
    4. If the class is wrong, increase the true class weights
       and decrease the predicted class weights.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        n_classes: int,
        learning_rate: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if data.shape[0] != labels.shape[0]:
            raise ValueError("data and labels sizes must match")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        self.data = data.astype(float)
        self.labels = labels.astype(int)
        self.n_classes = int(n_classes)
        self.learning_rate = float(learning_rate)
        self.rng = rng or np.random.default_rng()

        self.n_samples, self.n_features = self.data.shape
        self.weights = np.zeros((self.n_classes, self.n_features + 1), dtype=float)

        self.iteration = 0
        self.converged = False
        self.error_history: list[int] = []

        if self.labels.min() < 0 or self.labels.max() >= self.n_classes:
            raise ValueError("labels must be in range [0, n_classes - 1]")

    def _add_bias(self, x: np.ndarray) -> np.ndarray:
        # Append 1 for the bias term
        return np.append(x, 1.0)

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        # Raw decision values d_j(x)
        x_aug = self._add_bias(np.asarray(x, dtype=float))
        return self.weights @ x_aug

    def predict(self, x: np.ndarray) -> np.ndarray | int:
        scores = self.decision_scores_batch(x)

        if np.asarray(x).ndim == 1:
            return int(np.argmax(scores[0]))

        return np.argmax(scores, axis=1)

    def decision_scores_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
        return x_aug @ self.weights.T

    def step_epoch(self) -> int:
        """
        Run one training epoch.
        Returns the number of corrections.
        """
        if self.converged:
            return 0

        updates = 0

        # Fixed order keeps the demo easy to follow
        for idx in range(self.n_samples):
            x_aug = self._add_bias(self.data[idx])
            true_class = int(self.labels[idx])

            scores = self.weights @ x_aug
            pred_class = int(np.argmax(scores))

            if pred_class != true_class:
                # Correct class goes up, wrong class goes down
                self.weights[true_class] += self.learning_rate * x_aug
                self.weights[pred_class] -= self.learning_rate * x_aug
                updates += 1

        self.iteration += 1
        self.error_history.append(updates)

        if updates == 0:
            self.converged = True

        return updates

    def train(self, max_epochs: int = 1000) -> None:
        # Stop when no corrections remain or max_epochs is reached
        while not self.converged and self.iteration < max_epochs:
            self.step_epoch()

    def accuracy(self) -> float:
        preds = self.predict(self.data)
        return float(np.mean(preds == self.labels))

    def decision_functions(self) -> np.ndarray:
        # Returns weights for all N decision functions
        return self.weights.copy()
