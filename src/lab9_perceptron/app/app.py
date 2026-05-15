import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .inference import (
    DEFAULT_DEMO_TESTS,
    build_model_and_train,
    classify_objects,
    demo_training_text,
    parse_training_samples,
    report_classification,
    report_model,
    report_training,
)
from .models import MulticlassPerceptron


class PerceptronLabApp(tk.Tk):
    """Main window for the perceptron laboratory work."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Лабораторная работа №9 — ИНС / многослойный персептрон")
        self.geometry("1280x820")
        self.minsize(1120, 720)

        self.model: MulticlassPerceptron | None = None
        self.history: list = []
        self.last_training_report = ""
        self.last_model_report = ""
        self.last_classification_report = ""

        self._build_style()
        self._build_layout()
        self.load_demo()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", padding=4)
        style.configure("TNotebook.Tab", padding=(12, 6))
        style.configure("TButton", padding=(10, 6))
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("Title.TLabel", font=("Arial", 14, "bold"))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y", padx=(0, 8))

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # Left panel: controls
        title = ttk.Label(left, text="Параметры и данные", style="Title.TLabel")
        title.pack(anchor="w", pady=(0, 8))

        controls = ttk.LabelFrame(left, text="Параметры обучения")
        controls.pack(fill="x", pady=(0, 8))

        self.classes_var = tk.StringVar(value="3")
        self.rate_var = tk.StringVar(value="1.0")
        self.epochs_var = tk.StringVar(value="100")

        self._add_labeled_entry(controls, "Количество классов", self.classes_var)
        self._add_labeled_entry(controls, "Скорость обучения", self.rate_var)
        self._add_labeled_entry(controls, "Макс. эпох", self.epochs_var)

        ttk.Label(controls, text="Обучающая выборка").pack(anchor="w", padx=4, pady=(8, 2))
        self.training_text = tk.Text(controls, width=50, height=14, wrap="none")
        self.training_text.pack(fill="x", padx=4, pady=(0, 8))

        ttk.Label(controls, text="Тестовые объекты (по одному на строку: имя; f1; f2...)").pack(
            anchor="w", padx=4, pady=(4, 2)
        )
        self.test_text = tk.Text(controls, width=50, height=8, wrap="none")
        self.test_text.pack(fill="x", padx=4, pady=(0, 8))

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", padx=4, pady=(0, 4))

        ttk.Button(buttons, text="Демо из методички", command=self.load_demo).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(buttons, text="Обучить", command=self.train_model).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(buttons, text="Классифицировать", command=self.classify_tests).pack(side="left", fill="x", expand=True)

        ttk.Button(controls, text="Очистить вывод", command=self.clear_reports).pack(fill="x", padx=4, pady=(4, 0))

        # Right panel: notebook
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        self.log_text = self._make_text_tab("Ход обучения")
        self.weights_text = self._make_text_tab("Матрица весов")
        self.results_text = self._make_text_tab("Классификация")
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="График ошибки")

        self.figure = Figure(figsize=(7.5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Суммарная ошибка по эпохам")
        self.ax.set_xlabel("Эпоха")
        self.ax.set_ylabel("Ошибки")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=3)
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Entry(row, textvariable=variable, width=12).pack(side="left")

    def _make_text_tab(self, title: str) -> tk.Text:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        text = tk.Text(frame, wrap="word")
        text.pack(fill="both", expand=True)
        return text

    def clear_reports(self) -> None:
        for widget in (self.log_text, self.weights_text, self.results_text):
            widget.delete("1.0", "end")
        self.ax.clear()
        self.ax.set_title("Суммарная ошибка по эпохам")
        self.ax.set_xlabel("Эпоха")
        self.ax.set_ylabel("Ошибки")
        self.canvas.draw_idle()

    def load_demo(self) -> None:
        self.classes_var.set("3")
        self.rate_var.set("1.0")
        self.epochs_var.set("50")

        self.training_text.delete("1.0", "end")
        self.training_text.insert("1.0", demo_training_text())

        self.test_text.delete("1.0", "end")
        for label, features in DEFAULT_DEMO_TESTS:
            line = "; ".join([label, *[str(v) for v in features]])
            self.test_text.insert("end", line + "\n")

        self.clear_reports()

    def train_model(self) -> None:
        try:
            n_classes = int(float(self.classes_var.get().strip()))
            learning_rate = float(self.rate_var.get().strip())
            max_epochs = int(float(self.epochs_var.get().strip()))
            if n_classes <= 1:
                raise ValueError("Количество классов должно быть больше 1")
            if learning_rate <= 0:
                raise ValueError("Скорость обучения должна быть положительной")
            if max_epochs <= 0:
                raise ValueError("Макс. эпох должен быть положительным")
            samples = parse_training_samples(self.training_text.get("1.0", "end"), n_classes=n_classes)
            self.model, result = build_model_and_train(samples, n_classes, learning_rate, max_epochs)
        except Exception as exc:
            messagebox.showerror("Ошибка обучения", str(exc))
            return

        self.history = result.history
        self.last_training_report = report_training(result)
        self.last_model_report = report_model(self.model)
        self._render_training_report()
        self._render_weights()
        self._render_graph()
        messagebox.showinfo("Готово", "Обучение завершено успешно")

    def classify_tests(self) -> None:
        if self.model is None:
            messagebox.showwarning("Нет модели", "Сначала нужно обучить сеть")
            return

        try:
            objects = []
            lines = [line.strip() for line in self.test_text.get("1.0", "end").splitlines() if line.strip()]
            for line in lines:
                parts = [part.strip() for part in line.replace(",", ";").split(";") if part.strip()]
                if len(parts) < 2:
                    raise ValueError(f"Cannot parse test line: {line!r}")
                label = parts[0]
                features = [float(token) for token in parts[1:]]
                objects.append((label, features))

            if not objects:
                raise ValueError("Список тестовых объектов пуст")

            expected_dim = self.model.n_features
            for label, features in objects:
                if len(features) != expected_dim:
                    raise ValueError(
                        f"Объект '{label}' должен содержать {expected_dim} признаков, а не {len(features)}"
                    )

            rows = classify_objects(self.model, objects)
        except Exception as exc:
            messagebox.showerror("Ошибка классификации", str(exc))
            return

        self.last_classification_report = report_classification(rows)
        self._render_classification_report()
        self.notebook.select(self.results_text)

    def _render_training_report(self) -> None:
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", self.last_training_report)

    def _render_weights(self) -> None:
        self.weights_text.delete("1.0", "end")
        self.weights_text.insert("1.0", self.last_model_report)

    def _render_classification_report(self) -> None:
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", self.last_classification_report)

    def _render_graph(self) -> None:
        self.ax.clear()
        self.ax.set_title("Суммарная ошибка по эпохам")
        self.ax.set_xlabel("Эпоха")
        self.ax.set_ylabel("Ошибки")

        if not self.history:
            self.canvas.draw_idle()
            return

        epochs = [record.epoch for record in self.history]
        errors = [record.misclassifications for record in self.history]
        self.ax.plot(epochs, errors, marker="o")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


