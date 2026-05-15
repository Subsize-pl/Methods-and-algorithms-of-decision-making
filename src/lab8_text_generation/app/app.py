from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import List

from .inference import GrammarSynthesizer, normalize_samples


DEFAULT_SAMPLES = """caaab
bbaab
caab
bbab
cab
bbb
cb"""


def _coverage_text(samples: List[str], generated: List[str]) -> str:
    generated_set = set(generated)
    missing = [sample for sample in samples if sample not in generated_set]

    if not missing:
        return "All training strings were found in the generated language."

    return "Missing strings: " + ", ".join(missing)


class GrammarSynthesisApp:
    """Tkinter application for Lab 8."""

    def __init__(self) -> None:
        self.root = tk.Tk()

        self.root.title("Lab 8 — Grammar Synthesis")
        self.root.geometry("1180x820")
        self.root.minsize(980, 700)

        self.synthesizer = GrammarSynthesizer()

        self._build_ui()
        self.load_default_samples()
        self._run_synthesis()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left panel
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        left.columnconfigure(0, weight=1)

        ttk.Label(
            left,
            text="Training Samples",
            font=("TkDefaultFont", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.samples_text = ScrolledText(
            left,
            width=34,
            height=24,
            wrap="word"
        )
        self.samples_text.grid(row=1, column=0, sticky="nsew", pady=(8, 10))

        button_row = ttk.Frame(left)
        button_row.grid(row=2, column=0, sticky="ew")

        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        ttk.Button(
            button_row,
            text="Build Grammar",
            command=self._run_synthesis
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ttk.Button(
            button_row,
            text="Load Example",
            command=self.load_default_samples
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ttk.Button(
            left,
            text="Clear",
            command=self.clear_samples
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        # Right panel
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.stage1_text = self._make_tab("Stage 1: Exact Grammar")
        self.stage2_text = self._make_tab("Stage 2–3: Recursive and Simplified Grammar")
        self.generated_text = self._make_tab("Generated Strings")
        self.info_text = self._make_tab("Explanation")

        self.status_var = tk.StringVar(value="Ready")

        ttk.Label(
            right,
            textvariable=self.status_var,
            anchor="w"
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _make_tab(self, title: str) -> ScrolledText:
        frame = ttk.Frame(self.notebook, padding=8)

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        text = ScrolledText(frame, wrap="word")
        text.grid(row=0, column=0, sticky="nsew")

        self.notebook.add(frame, text=title)

        return text

    def load_default_samples(self) -> None:
        self.samples_text.delete("1.0", tk.END)
        self.samples_text.insert("1.0", DEFAULT_SAMPLES)

        self.status_var.set("Example from the manual loaded")

    def clear_samples(self) -> None:
        self.samples_text.delete("1.0", tk.END)

        self.status_var.set("Training sample field cleared")

    def _set_text(self, widget: ScrolledText, content: str) -> None:
        widget.configure(state="normal")

        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)

    def _run_synthesis(self) -> None:
        raw = self.samples_text.get("1.0", tk.END)
        samples = normalize_samples(raw)

        if not samples:
            messagebox.showwarning(
                "Empty Sample",
                "Enter at least one terminal string."
            )
            return

        try:
            result = self.synthesizer.synthesize(samples)

        except Exception as exc:
            messagebox.showerror(
                "Synthesis Error",
                str(exc)
            )
            return

        exact_text = (
            "Exact grammar constructed from the training sample.\n"
            "It generates only the original strings.\n\n"
            f"{result.exact_grammar.pretty()}"
        )

        recursive_text = (
            "Grammar after compression of repeating suffixes\n"
            "and recursive generalization.\n\n"
            f"{result.recursive_grammar.pretty()}"
        )

        generated_text = (
            "\n".join(result.generated_strings)
            if result.generated_strings
            else "No strings generated."
        )

        info_lines = [
            "Application workflow:",
            "",
            "1. Builds an exact non-recursive grammar as a prefix tree.",
            "2. Detects repeating patterns and replaces them with recursion.",
            "3. Simplifies the grammar by removing trivial and unreachable rules.",
            "",
            f"Input strings: {len(result.samples)}",
            f"Rules in exact grammar: "
            f"{sum(len(v) for v in result.exact_grammar.rules.values())}",
            f"Rules in final grammar: "
            f"{sum(len(v) for v in result.recursive_grammar.rules.values())}",
            "",
            _coverage_text(result.samples, result.generated_strings),
        ]

        info_text = "\n".join(info_lines)

        self._set_text(self.stage1_text, exact_text)
        self._set_text(self.stage2_text, recursive_text)
        self._set_text(self.generated_text, generated_text)
        self._set_text(self.info_text, info_text)

        self.status_var.set("Grammar synthesis completed successfully")

    def run(self) -> None:
        self.root.mainloop()