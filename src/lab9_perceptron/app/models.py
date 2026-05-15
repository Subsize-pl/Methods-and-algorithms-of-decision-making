from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(slots=True)
class TrainingSample:
    """One labeled training example."""

    features: List[float]
    class_index: int
    label: str | None = None

    @property
    def dimension(self) -> int:
        return len(self.features)


@dataclass(slots=True)
class TrainingConfig:
    """Training parameters for the perceptron."""

    n_features: int
    n_classes: int
    learning_rate: float = 1.0
    max_epochs: int = 100
    shuffle: bool = True


@dataclass(slots=True)
class EpochRecord:
    """Summary of one training epoch."""

    epoch: int
    misclassifications: int
    weights: List[List[float]]


@dataclass(slots=True)
class TrainingResult:
    """Returned after fitting the model."""

    converged: bool
    epochs: int
    history: List[EpochRecord] = field(default_factory=list)


class MulticlassPerceptron:
    """
    Multiclass perceptron with a bias term.

    The model uses one-vs-rest style competition: each class has its own
    linear neuron. The predicted class is the one with the largest net input.
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if n_classes <= 1:
            raise ValueError("n_classes must be at least 2")

        self.n_features = n_features
        self.n_classes = n_classes
        # weights[class][0] is bias; weights[class][1:] are feature weights
        self.weights: List[List[float]] = [
            [0.0 for _ in range(n_features + 1)] for _ in range(n_classes)
        ]

    def clone_weights(self) -> List[List[float]]:
        return [row[:] for row in self.weights]

    def reset(self) -> None:
        for row in self.weights:
            for i in range(len(row)):
                row[i] = 0.0

    def _with_bias(self, features: Sequence[float]) -> List[float]:
        if len(features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {len(features)}"
            )
        return [1.0, *[float(v) for v in features]]

    def net_values(self, features: Sequence[float]) -> List[float]:
        x = self._with_bias(features)
        nets: List[float] = []
        for class_weights in self.weights:
            nets.append(sum(w * xi for w, xi in zip(class_weights, x)))
        return nets

    def predict(self, features: Sequence[float]) -> int:
        nets = self.net_values(features)
        best_idx = 0
        best_val = nets[0]
        for idx in range(1, len(nets)):
            if nets[idx] > best_val:
                best_val = nets[idx]
                best_idx = idx
        return best_idx

    def predict_with_threshold(self, features: Sequence[float], threshold: float = 0.0) -> List[int]:
        return [1 if net >= threshold else 0 for net in self.net_values(features)]

    def fit(self, samples: Sequence[TrainingSample], config: TrainingConfig) -> TrainingResult:
        if not samples:
            raise ValueError("Training sample is empty")

        if config.n_features != self.n_features:
            raise ValueError("Training config feature count does not match the model")
        if config.n_classes != self.n_classes:
            raise ValueError("Training config class count does not match the model")

        order = list(range(len(samples)))
        history: List[EpochRecord] = []

        for epoch in range(1, config.max_epochs + 1):
            misclassifications = 0

            if config.shuffle:
                # A deterministic, stable shuffle surrogate that does not add randomness
                # to the saved report: rotate the order every epoch.
                shift = (epoch - 1) % len(order)
                epoch_order = order[shift:] + order[:shift]
            else:
                epoch_order = order

            for sample_index in epoch_order:
                sample = samples[sample_index]
                features_ext = self._with_bias(sample.features)
                predicted = self.predict(sample.features)

                if predicted != sample.class_index:
                    misclassifications += 1
                    # Move the target class up, the predicted class down.
                    for i, value in enumerate(features_ext):
                        self.weights[sample.class_index][i] += config.learning_rate * value
                        self.weights[predicted][i] -= config.learning_rate * value

            history.append(
                EpochRecord(
                    epoch=epoch,
                    misclassifications=misclassifications,
                    weights=self.clone_weights(),
                )
            )

            if misclassifications == 0:
                return TrainingResult(converged=True, epochs=epoch, history=history)

        return TrainingResult(converged=False, epochs=config.max_epochs, history=history)


def format_matrix(matrix: Sequence[Sequence[float]], precision: int = 4) -> str:
    """Format a numeric matrix for display in the GUI."""
    rows = []
    for row in matrix:
        rows.append("\t".join(f"{value:.{precision}f}" for value in row))
    return "\n".join(rows)


def format_samples(samples: Sequence[TrainingSample]) -> str:
    lines = []
    for sample in samples:
        cls = sample.class_index + 1
        label = f" ({sample.label})" if sample.label else ""
        features = ", ".join(f"{x:g}" for x in sample.features)
        lines.append(f"class {cls}{label}: [{features}]")
    return "\n".join(lines)
