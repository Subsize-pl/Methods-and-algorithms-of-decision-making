import random
import tkinter as tk
from tkinter import messagebox, ttk


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .inference import (
    build_hierarchy,
    default_labels,
    generate_distance_matrix,
    pretty_steps,
    validate_and_prepare_size,
)
from .models import format_matrix


class HierarchyApp(tk.Tk):
    # Application window for the lab.

    def __init__(self) -> None:
        super().__init__()
        self.geometry("1280x820")
        self.minsize(1180, 760)

        self._labels: list[str] = []
        self._matrix: list[list[float]] = []
        self._min_result = None
        self._max_result = None

        self._build_ui()
        self._load_default_data()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="ns")

        view = ttk.Notebook(self)
        view.grid(row=0, column=1, sticky="nsew")

        self.matrix_tab = ttk.Frame(view, padding=8)
        self.min_tab = ttk.Frame(view, padding=8)
        self.max_tab = ttk.Frame(view, padding=8)
        self.info_tab = ttk.Frame(view, padding=8)

        view.add(self.matrix_tab, text="Matrix")
        view.add(self.min_tab, text="Minimum Criterion")
        view.add(self.max_tab, text="Maximum Criterion")
        view.add(self.info_tab, text="Construction Steps")

        self._build_controls(controls)
        self._build_matrix_tab()
        self._build_graph_tab(self.min_tab, "min")
        self._build_graph_tab(self.max_tab, "max")
        self._build_info_tab()

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Parameters", font=("Arial", 13, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        ttk.Label(parent, text="Number of objects n:").grid(row=1, column=0, sticky="w")
        self.size_var = tk.StringVar(value="6")
        ttk.Entry(parent, textvariable=self.size_var, width=12).grid(
            row=2, column=0, sticky="ew", pady=(2, 10)
        )

        ttk.Label(parent, text="Seed:").grid(row=3, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(parent, textvariable=self.seed_var, width=12).grid(
            row=4, column=0, sticky="ew", pady=(2, 10)
        )

        ttk.Button(parent, text="Generate matrix", command=self.generate).grid(
            row=5, column=0, sticky="ew", pady=(8, 4)
        )
        ttk.Button(parent, text="Build hierarchy", command=self.solve).grid(
            row=6, column=0, sticky="ew", pady=4
        )

        ttk.Separator(parent, orient="horizontal").grid(
            row=7, column=0, sticky="ew", pady=12
        )

        parent.columnconfigure(0, weight=1)

    def _build_matrix_tab(self) -> None:
        container = ttk.Frame(self.matrix_tab)
        container.pack(fill="both", expand=True)

        self.matrix_text = tk.Text(container, wrap="none", height=20, width=60)
        self.matrix_text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(container, orient="vertical", command=self.matrix_text.yview)
        yscroll.pack(side="right", fill="y")
        self.matrix_text.configure(yscrollcommand=yscroll.set)

    def _build_info_tab(self) -> None:
        self.info_text = tk.Text(self.info_tab, wrap="word")
        self.info_text.pack(fill="both", expand=True)

    def _build_graph_tab(self, parent: ttk.Frame, criterion: str) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        figure = Figure(figsize=(6, 5), dpi=100)
        axis = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=frame)
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")

        if criterion == "min":
            self.min_figure = figure
            self.min_axis = axis
            self.min_canvas = canvas
        else:
            self.max_figure = figure
            self.max_axis = axis
            self.max_canvas = canvas

    def _load_default_data(self) -> None:
        self.generate()

    def _rng(self) -> random.Random:
        seed_raw = self.seed_var.get().strip()
        if not seed_raw:
            return random.Random()
        try:
            return random.Random(int(seed_raw))
        except ValueError as exc:
            raise ValueError("Seed must be an integer") from exc

    def generate(self) -> None:
        try:
            size = validate_and_prepare_size(self.size_var.get().strip())
            rng = self._rng()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self._labels = default_labels(size)
        self._matrix = generate_distance_matrix(size, rng)
        self._min_result = None
        self._max_result = None

        self._render_matrix()
        self._clear_plot(self.min_axis, self.min_canvas, "Build hierarchy")
        self._clear_plot(self.max_axis, self.max_canvas, "Build hierarchy")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(
            tk.END,
            "Distance matrix generated. Click 'Build hierarchy'.\n",
        )

    def solve(self) -> None:
        if not self._labels or not self._matrix:
            messagebox.showwarning("Warning", "Generate the matrix first.")
            return

        try:
            self._min_result = build_hierarchy(self._labels, self._matrix, "min")
            self._max_result = build_hierarchy(self._labels, self._matrix, "max")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self._draw_hierarchy(self.min_axis, self.min_canvas, self._min_result, "Minimum")
        self._draw_hierarchy(self.max_axis, self.max_canvas, self._max_result, "Maximum", transformed=True)
        self._render_info()

    def _render_matrix(self) -> None:
        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert(tk.END, format_matrix(self._labels, self._matrix))

    def _render_info(self) -> None:
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "Input data:\n")
        self.info_text.insert(tk.END, format_matrix(self._labels, self._matrix))
        self.info_text.insert(tk.END, "\n\n")

        self.info_text.insert(tk.END, "Minimum criterion:\n")
        self.info_text.insert(tk.END, pretty_steps(self._min_result))
        self.info_text.insert(tk.END, "\n\n")

        self.info_text.insert(tk.END, "Maximum criterion (using inverse distances):\n")
        self.info_text.insert(tk.END, pretty_steps(self._max_result))
        self.info_text.insert(tk.END, "\n")

    def _clear_plot(self, axis, canvas, title: str) -> None:
        axis.clear()
        axis.set_title(title)
        axis.axis("off")
        canvas.draw()

    def _draw_hierarchy(self, axis, canvas, result, title: str, transformed: bool = False) -> None:
        axis.clear()
        axis.set_title(title if not transformed else f"{title} (1 / d)")
        axis.axis("off")

        order = self._leaf_order(result.root)
        positions = {member: index for index, member in enumerate(order)}

        def draw_node(node):
            if node.is_leaf:
                x = positions[node.members[0]]
                y = 0.0
                axis.text(x, -0.04, node.name, ha="center", va="top", fontsize=10)
                return x, y

            assert node.left is not None and node.right is not None
            left_x, left_y = draw_node(node.left)
            right_x, right_y = draw_node(node.right)
            y = node.height

            axis.plot([left_x, left_x], [left_y, y], color="black", linewidth=1.2)
            axis.plot([right_x, right_x], [right_y, y], color="black", linewidth=1.2)
            axis.plot([left_x, right_x], [y, y], color="black", linewidth=1.2)
            axis.text(
                (left_x + right_x) / 2,
                y + self._height_offset(y),
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            return (left_x + right_x) / 2, y

        draw_node(result.root)

        max_height = max((step.distance for step in result.steps), default=1.0)
        axis.set_xlim(-0.8, max(len(order) - 0.2, 0.8))
        axis.set_ylim(-0.15, max_height * 1.15 if max_height > 0 else 1.0)
        axis.set_xticks([])
        axis.set_yticks([])

        canvas.draw()

    def _leaf_order(self, node) -> list[int]:
        order: list[int] = []

        def walk(current):
            if current.is_leaf:
                order.append(current.members[0])
                return
            walk(current.left)
            walk(current.right)

        walk(node)
        return order

    def _height_offset(self, height: float) -> float:
        return 0.03 * max(1.0, height)


