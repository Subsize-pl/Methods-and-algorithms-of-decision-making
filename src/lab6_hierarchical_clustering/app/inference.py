import random
from typing import Sequence

from .models import ClusterNode, HierarchyResult, MergeStep


EPSILON = 1e-9


def generate_distance_matrix(size: int, rng: random.Random) -> list[list[float]]:
    # Generate a symmetric random distance matrix.

    if size < 2:
        raise ValueError("Number of objects must be at least 2.")

    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            value = round(rng.uniform(0.5, 9.9), 2)
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def _transform_value(value: float, criterion: str) -> float:
    if criterion == "min":
        return value
    if criterion == "max":
        return 1.0 / max(value, EPSILON)
    raise ValueError(f"Unsupported criterion: {criterion}")


def _cluster_distance(
    left: ClusterNode,
    right: ClusterNode,
    matrix: Sequence[Sequence[float]],
    criterion: str,
) -> float:
    """Distance between clusters according to the chosen criterion."""

    values: list[float] = []
    for i in left.members:
        for j in right.members:
            if i == j:
                continue
            values.append(_transform_value(matrix[i][j], criterion))

    if not values:
        return 0.0

    # Methodical construction:
    # - minimum criterion: use the minimum inter-object distance
    # - maximum criterion: use reciprocal distances and again choose the minimum
    return min(values)


def build_hierarchy(
    labels: Sequence[str],
    matrix: Sequence[Sequence[float]],
    criterion: str,
) -> HierarchyResult:
    """Build a hierarchical tree according to a criterion."""

    if criterion not in {"min", "max"}:
        raise ValueError("criterion must be 'min' or 'max'")
    if len(labels) != len(matrix):
        raise ValueError("labels and matrix size must match")

    clusters: list[ClusterNode] = [
        ClusterNode(name=label, members=[index]) for index, label in enumerate(labels)
    ]
    steps: list[MergeStep] = []
    counter = 1

    while len(clusters) > 1:
        best_i = best_j = -1
        best_distance = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = _cluster_distance(clusters[i], clusters[j], matrix, criterion)
                if distance < best_distance:
                    best_distance = distance
                    best_i, best_j = i, j

        left = clusters[best_i]
        right = clusters[best_j]
        merged_name = f"g{counter}"
        counter += 1

        merged_members = sorted(left.members + right.members)
        merged_node = ClusterNode(
            name=merged_name,
            members=merged_members,
            height=best_distance,
            left=left,
            right=right,
        )

        steps.append(
            MergeStep(
                left=left.name,
                right=right.name,
                merged=merged_name,
                distance=best_distance,
                members=merged_members,
            )
        )

        for index in sorted((best_i, best_j), reverse=True):
            clusters.pop(index)
        clusters.append(merged_node)

    root = clusters[0]
    return HierarchyResult(
        criterion=criterion,
        root=root,
        steps=steps,
        transformed=(criterion == "max"),
    )


def validate_and_prepare_size(raw_value: str) -> int:
    """Parse and validate number of objects."""

    size = int(raw_value)
    if size < 2:
        raise ValueError("n must be at least 2")
    return size


def default_labels(size: int) -> list[str]:
    return [f"x{i+1}" for i in range(size)]


def pretty_steps(result: HierarchyResult) -> str:
    lines: list[str] = []
    for idx, step in enumerate(result.steps, start=1):
        lines.append(
            f"{idx}. {step.left} + {step.right} -> {step.merged} "
            f"(h = {step.distance:.4f})"
        )
    return "\n".join(lines)
