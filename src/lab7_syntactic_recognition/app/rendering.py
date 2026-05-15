from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .parser import ParseNode


@dataclass
class TreeLayout:
    positions: dict[int, tuple[float, float]]
    labels: dict[int, str]
    edges: list[tuple[int, int]]
    box_sizes: dict[int, tuple[float, float]]
    depth: int
    leaf_count: int


def _count_leaves(node: ParseNode) -> int:
    if not node.children:
        return 1
    return sum(_count_leaves(child) for child in node.children)


def _label_box_size(label: str) -> tuple[float, float]:
    width = max(0.95, 0.22 * len(label) + 0.45)
    height = 0.56
    return width, height


def build_layout(root: ParseNode) -> TreeLayout:
    positions: dict[int, tuple[float, float]] = {}
    labels: dict[int, str] = {}
    edges: list[tuple[int, int]] = []
    box_sizes: dict[int, tuple[float, float]] = {}

    leaf_count = _count_leaves(root)
    x_step = 2.8
    y_step = 2.15
    leaf_cursor = 0.0
    max_depth = 0

    def assign(node: ParseNode, depth: int) -> float:
        nonlocal leaf_cursor, max_depth
        node_id = id(node)
        labels[node_id] = node.symbol
        box_sizes[node_id] = _label_box_size(node.symbol)
        max_depth = max(max_depth, depth)

        if not node.children:
            x = leaf_cursor * x_step
            leaf_cursor += 1.0
            positions[node_id] = (x, -depth * y_step)
            return x

        child_xs: list[float] = []
        for child in node.children:
            child_xs.append(assign(child, depth + 1))
            edges.append((node_id, id(child)))

        x = sum(child_xs) / len(child_xs)
        positions[node_id] = (x, -depth * y_step)
        return x

    assign(root, 0)
    return TreeLayout(
        positions=positions,
        labels=labels,
        edges=edges,
        box_sizes=box_sizes,
        depth=max_depth,
        leaf_count=leaf_count,
    )


def draw_parse_tree(ax: plt.Axes, root: ParseNode, title: str) -> None:
    layout = build_layout(root)
    ax.clear()
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.axis("off")

    for parent_id, child_id in layout.edges:
        x1, y1 = layout.positions[parent_id]
        x2, y2 = layout.positions[child_id]
        w1, h1 = layout.box_sizes[parent_id]
        w2, h2 = layout.box_sizes[child_id]
        ax.plot(
            [x1, x2],
            [y1 - h1 / 2.0, y2 + h2 / 2.0],
            linewidth=1.5,
            color="#5d6d7e",
            alpha=0.95,
            solid_capstyle="round",
            zorder=1,
        )

    for node_id, (x, y) in layout.positions.items():
        label = layout.labels[node_id]
        width, height = layout.box_sizes[node_id]
        box = FancyBboxPatch(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            boxstyle="round,pad=0.12,rounding_size=0.08",
            facecolor="#ffffff",
            edgecolor="#1f2d3d",
            linewidth=1.9,
            zorder=3,
        )
        ax.add_patch(box)
        font_size = 10 if len(label) <= 6 else 9
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            color="#1f2d3d",
            weight="bold",
            zorder=4,
        )

    xs = [pos[0] for pos in layout.positions.values()]
    ys = [pos[1] for pos in layout.positions.values()]
    x_margin = max(2.0, 0.8 * (layout.leaf_count or 1))
    y_margin = 1.7
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
    ax.set_aspect("equal", adjustable="datalim")


def draw_sequence(ax: plt.Axes, sequence: str, title: str = "Terminal sequence") -> None:
    ax.clear()
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.axis("off")
    chars = [ch for ch in sequence if not ch.isspace()]
    if not chars:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        return

    x_step = 1.35
    xs = [i * x_step for i in range(len(chars))]
    ax.set_xlim(-1.0, xs[-1] + 1.0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", adjustable="box")

    for i, ch in enumerate(chars):
        patch = FancyBboxPatch(
            (xs[i] - 0.33, -0.32),
            0.66,
            0.64,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor="#ffffff",
            edgecolor="#1f2d3d",
            linewidth=1.8,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(xs[i], 0.0, ch, ha="center", va="center", fontsize=11, weight="bold", zorder=4)
        if i < len(chars) - 1:
            ax.plot(
                [xs[i] + 0.34, xs[i + 1] - 0.34],
                [0, 0],
                linewidth=1.3,
                color="#5d6d7e",
                zorder=1,
            )


def draw_steps_page(
    ax: plt.Axes,
    steps: list[str],
    title: str = "Reduction sequence",
    *,
    offset: int = 0,
    visible_lines: int = 12,
) -> None:
    ax.clear()
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.axis("off")

    total = len(steps)
    if total == 0:
        ax.text(0.02, 0.95, "No steps to display.", transform=ax.transAxes, va="top", fontsize=11)
        return

    offset = max(0, min(offset, max(0, total - visible_lines)))
    end = min(total, offset + visible_lines)
    page = steps[offset:end]

    frame = FancyBboxPatch(
        (0.01, 0.04),
        0.96,
        0.90,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#fbfcfc",
        edgecolor="#d5d8dc",
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(frame)

    y = 0.92
    line_step = 0.90 / max(visible_lines, 1)
    for idx, step in enumerate(page, start=offset + 1):
        ax.text(
            0.03,
            y,
            f"{idx:02d}. {step}",
            transform=ax.transAxes,
            va="top",
            fontsize=10.4,
            family="monospace",
            color="#1f2d3d",
            zorder=1,
        )
        y -= line_step

    page_num = offset // visible_lines + 1
    total_pages = (total + visible_lines - 1) // visible_lines
    ax.text(
        0.97,
        0.98,
        f"Page {page_num}/{total_pages}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9.5,
        color="#5d6d7e",
        zorder=1,
    )
