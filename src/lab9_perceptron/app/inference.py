from dataclasses import dataclass
from typing import List, Sequence

from .models import MulticlassPerceptron, TrainingConfig, TrainingSample, TrainingResult


DEFAULT_DEMO_SAMPLES = [
    TrainingSample([0.25, 0.25, 0.25, 0.0], 0, "Oka"),
    TrainingSample([0.75, 0.75, 0.75, 0.25], 1, "Passenger Gazelle"),
    TrainingSample([1.0, 1.0, 0.25, 1.0], 2, "Kamaz"),
]

DEFAULT_DEMO_TESTS = [
    ("Oka", [0.2, 0.2, 0.2, 0.0]),
    ("Passenger Gazelle", [0.75, 0.75, 0.75, 0.25]),
    ("Kamaz", [1.0, 1.0, 0.25, 1.0]),
]


@dataclass(slots=True)
class ClassificationRow:
    label: str
    features: List[float]
    predicted_class: int
    nets: List[float]


def parse_vector(text: str) -> List[float]:
    cleaned = text.replace(";", " ").replace(",", " ").strip()
    if not cleaned:
        raise ValueError("Vector is empty")
    values = [float(part) for part in cleaned.split()]
    return values


def parse_training_samples(text: str, n_classes: int | None = None) -> List[TrainingSample]:
    """
    Parse training sample text.

    Supported row formats:
    - class_index; f1; f2; f3
    - label; class_index; f1; f2; f3
    - class_index, f1, f2, f3
    """
    samples: List[TrainingSample] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Training sample is empty")

    for raw_line in lines:
        parts = [part.strip() for part in raw_line.replace(",", ";").split(";") if part.strip()]
        if len(parts) < 2:
            raise ValueError(f"Cannot parse line: {raw_line!r}")

        label: str | None = None
        if len(parts) >= 3 and not _looks_like_number(parts[0]) and _looks_like_number(parts[1]):
            label = parts[0]
            class_token = parts[1]
            feature_tokens = parts[2:]
        else:
            class_token = parts[0]
            feature_tokens = parts[1:]

        class_index = int(float(class_token))
        if class_index <= 0:
            raise ValueError("Class numbers in the editor are 1-based and must be positive")

        features = [float(token) for token in feature_tokens]
        samples.append(TrainingSample(features, class_index - 1, label))

    feature_count = len(samples[0].features)
    for sample in samples:
        if len(sample.features) != feature_count:
            raise ValueError("All training samples must have the same feature count")

    if n_classes is not None:
        max_class = max(sample.class_index for sample in samples) + 1
        if n_classes < max_class:
            raise ValueError(
                f"Declared class count is {n_classes}, but the sample contains class {max_class}"
            )

    return samples


def _looks_like_number(value: str) -> bool:
    try:
        float(value.replace(",", "."))
        return True
    except ValueError:
        return False


def build_model_and_train(
    samples: Sequence[TrainingSample],
    n_classes: int,
    learning_rate: float,
    max_epochs: int,
) -> tuple[MulticlassPerceptron, TrainingResult]:
    if not samples:
        raise ValueError("Training sample is empty")

    n_features = len(samples[0].features)
    model = MulticlassPerceptron(n_features=n_features, n_classes=n_classes)
    config = TrainingConfig(
        n_features=n_features,
        n_classes=n_classes,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        shuffle=True,
    )
    result = model.fit(samples, config)
    return model, result


def classify_objects(
    model: MulticlassPerceptron,
    objects: Sequence[tuple[str, Sequence[float]]],
) -> List[ClassificationRow]:
    rows: List[ClassificationRow] = []
    for label, features in objects:
        nets = model.net_values(features)
        predicted = model.predict(features)
        rows.append(
            ClassificationRow(
                label=label,
                features=list(features),
                predicted_class=predicted,
                nets=nets,
            )
        )
    return rows


def report_training(result: TrainingResult) -> str:
    lines = []
    lines.append("Training completed.")
    lines.append(f"Converged: {'yes' if result.converged else 'no'}")
    lines.append(f"Epochs: {result.epochs}")
    lines.append("")
    lines.append("Training progress:")
    for record in result.history:
        lines.append(f"  epoch {record.epoch}: misclassifications = {record.misclassifications}")
    return "\n".join(lines)


def report_model(model: MulticlassPerceptron) -> str:
    lines = ["Weight matrix (bias in the first column):"]
    for class_index, weights in enumerate(model.weights, start=1):
        formatted = ", ".join(f"{value:.4f}" for value in weights)
        lines.append(f"  class {class_index}: [{formatted}]")
    return "\n".join(lines)


def report_classification(rows: Sequence[ClassificationRow]) -> str:
    lines = []
    for row in rows:
        nets = ", ".join(f"{value:.4f}" for value in row.nets)
        features = ", ".join(f"{value:g}" for value in row.features)
        lines.append(
            f"{row.label}: [{features}] -> class {row.predicted_class + 1}; NET = [{nets}]"
        )
    return "\n".join(lines)


def demo_training_text() -> str:
    return "\n".join(
        [
            "1; 0.25; 0.25; 0.25; 0",
            "2; 0.75; 0.75; 0.75; 0.25",
            "3; 1; 1; 0.25; 1",
        ]
    )