from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class ClusterNode:
    # A node of the hierarchy tree.

    name: str
    members: list[int]
    height: float = 0.0
    left: "ClusterNode | None" = None
    right: "ClusterNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(slots=True)
class MergeStep:
    # A single agglomeration step.

    left: str
    right: str
    merged: str
    distance: float
    members: list[int] = field(default_factory=list)


@dataclass(slots=True)
class HierarchyResult:
    # Result of hierarchical clustering for one criterion.

    criterion: str
    root: ClusterNode
    steps: list[MergeStep]
    transformed: bool = False

    def leaf_names(self) -> list[str]:
        leaves: list[str] = []

        def walk(node: ClusterNode) -> None:
            if node.is_leaf:
                leaves.append(node.name)
                return
            if node.left is not None:
                walk(node.left)
            if node.right is not None:
                walk(node.right)

        walk(self.root)
        return leaves


def format_matrix(labels: Sequence[str], matrix: Sequence[Sequence[float]]) -> str:
    # Render a matrix as an aligned text table.

    if not labels:
        return ""

    width = max(7, max(len(label) for label in labels) + 2)
    num_width = 8

    header = " " * width + "".join(f"{label:>{num_width}}" for label in labels)
    lines = [header]
    for label, row in zip(labels, matrix):
        line = f"{label:>{width}}" + "".join(f"{value:>{num_width}.2f}" for value in row)
        lines.append(line)
    return "\n".join(lines)


def matrix_to_display_rows(labels: Sequence[str], matrix: Sequence[Sequence[float]]) -> list[list[str]]:
    # Convert a matrix to rows suitable for a table widget.

    rows: list[list[str]] = []
    for label, row in zip(labels, matrix):
        rows.append([label, *[f"{value:.2f}" for value in row]])
    return rows
