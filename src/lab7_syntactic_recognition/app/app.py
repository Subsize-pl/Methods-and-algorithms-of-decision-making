from __future__ import annotations

import random

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

from .generator import ExampleGenerator
from .grammar import DEFAULT_GRAMMAR
from .parser import BottomUpParser, ParseResult
from .rendering import draw_parse_tree, draw_sequence, draw_steps_page


class SyntacticRecognitionApp:
    def __init__(self) -> None:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False

        self.grammar = DEFAULT_GRAMMAR
        self.parser = BottomUpParser(self.grammar)
        self.generator = ExampleGenerator(self.grammar)

        self.figure = plt.figure(figsize=(18, 11), facecolor="white")

        self.ax_status = self.figure.add_axes([0.035, 0.915, 0.93, 0.055])
        self.ax_status.axis("off")

        self.ax_input_label = self.figure.add_axes([0.035, 0.845, 0.08, 0.05])
        self.ax_input_label.axis("off")
        self.ax_input_label.text(
            0.0, 0.5, "Input:", fontsize=11.5, fontweight="bold", va="center"
        )

        self.textbox_ax = self.figure.add_axes([0.105, 0.84, 0.42, 0.06])
        self.check_ax = self.figure.add_axes([0.55, 0.84, 0.10, 0.06])
        self.clear_ax = self.figure.add_axes([0.665, 0.84, 0.10, 0.06])
        self.example_s_ax = self.figure.add_axes([0.785, 0.84, 0.08, 0.06])
        self.example_t_ax = self.figure.add_axes([0.875, 0.84, 0.08, 0.06])

        self.textbox = TextBox(
            self.textbox_ax,
            "",
            initial="",
            color="#ffffff",
            hovercolor="#f4f6f7",
        )
        self.textbox.label.set_visible(False)
        self.textbox.on_submit(self._on_submit)

        self.check_button = Button(self.check_ax, "Check", hovercolor="#d6eaf8")
        self.check_button.on_clicked(self._on_check_clicked)

        self.clear_button = Button(self.clear_ax, "Clear", hovercolor="#fadbd8")
        self.clear_button.on_clicked(self._on_clear_clicked)

        self.example_s_button = Button(
            self.example_s_ax, "Example S", hovercolor="#e8f8f5"
        )
        self.example_s_button.on_clicked(lambda _event: self._set_example("S"))

        self.example_t_button = Button(
            self.example_t_ax, "Example T", hovercolor="#e8f8f5"
        )
        self.example_t_button.on_clicked(lambda _event: self._set_example("T"))

        self.ax_sequence = self.figure.add_axes([0.04, 0.50, 0.42, 0.28])
        self.ax_tree = self.figure.add_axes([0.54, 0.50, 0.42, 0.28])

        self.ax_steps = self.figure.add_axes([0.04, 0.08, 0.88, 0.34])
        self.ax_scroll = self.figure.add_axes([0.935, 0.10, 0.02, 0.30])

        self.scroll_slider = Slider(
            self.ax_scroll,
            "",
            0,
            0,
            valinit=0,
            valstep=1,
            orientation="vertical",
            facecolor="#2e86c1",
        )
        self.scroll_slider.valtext.set_visible(False)
        self.scroll_slider.label.set_visible(False)
        self.scroll_slider.on_changed(self._on_scroll_changed)

        self._default_status = (
            "Enter a sequence of symbols using only a, b, c, d, e, then press Check."
        )
        self._steps: list[str] = []
        self._scroll_offset = 0
        self._visible_steps = 13

        self._set_status(self._default_status)
        self._render_empty_state()

    def run(self) -> None:
        plt.show()

    def _on_submit(self, _text: str) -> None:
        self.classify()

    def _on_check_clicked(self, _event) -> None:
        self.classify()

    def _on_clear_clicked(self, _event) -> None:
        self.textbox.set_val("")
        self._steps = []
        self._scroll_offset = 0
        self._update_scrollbar()
        self._set_status(self._default_status)
        self._render_empty_state()

    def _set_example(self, start: str) -> None:
        value = self.generator.generate(start, target_length=random.randint(10, 15))
        self.textbox.set_val(value)
        self.classify()

    def _set_status(self, text: str, extra: str = "") -> None:
        self.ax_status.clear()
        self.ax_status.axis("off")
        self.ax_status.text(
            0.0,
            0.5,
            text,
            transform=self.ax_status.transAxes,
            va="center",
            fontsize=11,
        )
        if extra:
            self.ax_status.text(
                0.99,
                0.5,
                extra,
                transform=self.ax_status.transAxes,
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
            )
        self.figure.canvas.draw_idle()

    def _on_scroll_changed(self, value: float) -> None:
        max_offset = max(0, len(self._steps) - self._visible_steps)
        self._scroll_offset = int(round(value))
        self._scroll_offset = max(0, min(self._scroll_offset, max_offset))
        self._draw_steps_panel()

    def _update_scrollbar(self) -> None:
        max_offset = max(0, len(self._steps) - self._visible_steps)
        if max_offset == 0:
            self.ax_scroll.set_visible(False)
            return

        self.ax_scroll.set_visible(True)
        self.scroll_slider.valmin = 0
        self.scroll_slider.valmax = max_offset
        self.scroll_slider.ax.set_ylim(0, max_offset if max_offset > 0 else 1)
        self.scroll_slider.set_val(min(self._scroll_offset, max_offset))
        self.scroll_slider.ax.figure.canvas.draw_idle()

    def _draw_steps_panel(self) -> None:
        draw_steps_page(
            self.ax_steps,
            self._steps,
            title="Reduction steps",
            offset=self._scroll_offset,
            visible_lines=self._visible_steps,
        )
        self.figure.canvas.draw_idle()

    def classify(self) -> None:
        raw = self.textbox.text or ""
        sequence = "".join(ch for ch in raw if not ch.isspace())

        if not sequence:
            self._set_status("The sequence is empty.", "Allowed: a, b, c, d, e")
            self._render_failure("")
            return

        invalid = [ch for ch in sequence if ch not in self.grammar.terminals]
        if invalid:
            invalid_text = ", ".join(sorted(set(invalid)))
            self._set_status(
                f"Invalid symbols: {invalid_text}.", "Allowed: a, b, c, d, e"
            )
            self._render_failure(sequence)
            return

        result = self.parser.classify(sequence)
        results_by_start = self.parser.parse(sequence)
        s_res = results_by_start["S"]
        t_res = results_by_start["T"]

        if result.recognized and result.class_name in {"S", "T"}:
            if s_res.recognized and t_res.recognized:
                self._set_status(
                    f"Recognized as {result.class_name}.",
                    "Ambiguous: tree selected from the valid parses",
                )
            else:
                self._set_status(f"Recognized as {result.class_name}.")
            self._render_success(sequence, result, s_res, t_res)
            return

        message = result.message or "The sequence is not recognized by the grammar."
        self._set_status(message)
        self._render_failure(sequence, s_res, t_res)

    def _render_success(
        self,
        sequence: str,
        result: ParseResult,
        s_res: ParseResult,
        t_res: ParseResult,
    ) -> None:
        draw_sequence(self.ax_sequence, sequence, title="Input sequence")
        if result.root is not None:
            draw_parse_tree(
                self.ax_tree,
                result.root,
                title=f"Reduction tree to {result.class_name}",
            )
        else:
            self.ax_tree.clear()
            self.ax_tree.axis("off")
            self.ax_tree.text(
                0.5, 0.5, "Tree is unavailable", ha="center", va="center", fontsize=12
            )

        self._steps = list(result.steps)
        self._scroll_offset = 0
        self._update_scrollbar()
        self._draw_steps_panel()

        self._set_status(
            f"Recognized as {result.class_name}.",
            f"S: {'yes' if s_res.recognized else 'no'}   T: {'yes' if t_res.recognized else 'no'}",
        )
        self.figure.canvas.draw_idle()

    def _render_failure(
        self,
        sequence: str,
        s_res: ParseResult | None = None,
        t_res: ParseResult | None = None,
    ) -> None:
        draw_sequence(self.ax_sequence, sequence, title="Input sequence")
        self.ax_tree.clear()
        self.ax_tree.axis("off")
        self.ax_tree.text(
            0.5, 0.5, "No parse found", ha="center", va="center", fontsize=13
        )

        self._steps = []
        self._scroll_offset = 0
        self._update_scrollbar()
        self._draw_steps_panel()

        if s_res is not None and t_res is not None:
            self._set_status(
                "The sequence is not recognized by the grammar.",
                f"S: {'yes' if s_res.recognized else 'no'}   T: {'yes' if t_res.recognized else 'no'}",
            )
        self.figure.canvas.draw_idle()

    def _render_empty_state(self) -> None:
        self._render_failure("")
