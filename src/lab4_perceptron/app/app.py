from __future__ import annotations

from typing import Optional

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from common.config.log import LoggingSettings
from .perceptron import PerceptronState

logging.basicConfig(level=LoggingSettings.LEVEL, format=LoggingSettings.FORMAT)
logger = logging.getLogger(__name__)


class PerceptronApp:
    # Interactive GUI for multi-class perceptron
    def __init__(
        self,
        state: PerceptronState,
        interval: int = 150,
        max_iter: int = 200,
    ) -> None:
        self.state = state
        self.interval = interval
        self.max_iter = max_iter

        self.anim: Optional[FuncAnimation] = None
        self.paused: bool = True

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.18)

        self.cmap = plt.cm.get_cmap("tab20", max(self.state.n_classes, 2))
        self._xmin, self._xmax, self._ymin, self._ymax = self._compute_bounds()

        self.background = self.ax.imshow(
            self._grid_predictions(),
            origin="lower",
            extent=self._extent(),
            cmap=self.cmap,
            vmin=-0.5,
            vmax=self.state.n_classes - 0.5,
            alpha=0.12,
            interpolation="nearest",
            aspect="auto",
        )

        self.scat = self.ax.scatter(
            self.state.data[:, 0],
            self.state.data[:, 1],
            s=28,
            alpha=0.9,
            edgecolors="k",
        )

        self.ax.set_xlim(self._xmin, self._xmax)
        self.ax.set_ylim(self._ymin, self._ymax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Perceptron: press Start")

        ax_btn = plt.axes([0.35, 0.05, 0.3, 0.075])
        self.btn = Button(ax_btn, "Start")
        self.btn.on_clicked(self._on_button)

        self._render()

    def _compute_bounds(self) -> tuple[float, float, float, float]:
        x = self.state.data[:, 0]
        y = self.state.data[:, 1]

        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())

        dx = max((xmax - xmin) * 0.15, 1.0)
        dy = max((ymax - ymin) * 0.15, 1.0)

        return xmin - dx, xmax + dx, ymin - dy, ymax + dy

    def _extent(self) -> tuple[float, float, float, float]:
        return self._xmin, self._xmax, self._ymin, self._ymax

    def _grid_predictions(self) -> np.ndarray:
        grid_x = np.linspace(self._xmin, self._xmax, 220)
        grid_y = np.linspace(self._ymin, self._ymax, 220)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        pred = self.state.predict(grid)
        return np.asarray(pred).reshape(xx.shape)

    def _colors_from_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.cmap(labels % self.cmap.N)

    def _render(self) -> None:
        # True classes as point colors
        self.scat.set_facecolors(self._colors_from_labels(self.state.labels))

        # Red border for wrong predictions
        preds = np.asarray(self.state.predict(self.state.data))
        edges = np.tile(np.array([0.2, 0.2, 0.2, 1.0]), (self.state.n_samples, 1))
        edges[preds != self.state.labels] = np.array([1.0, 0.0, 0.0, 1.0])
        self.scat.set_edgecolors(edges)

        # Background decision regions
        self.background.set_data(self._grid_predictions())

        acc = self.state.accuracy() * 100.0
        updates = self.state.error_history[-1] if self.state.error_history else 0
        self.ax.set_title(
            f"Epoch {self.state.iteration} | updates={updates} | acc={acc:.2f}%"
        )

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f"Drawing exception: {e}")

    def _step(self, frame=None) -> None:
        if self.state.converged:
            return

        updates = self.state.step_epoch()
        logger.info(
            "Epoch %d, updates=%d, accuracy=%.2f%%",
            self.state.iteration,
            updates,
            self.state.accuracy() * 100.0,
        )

        self._render()

        if self.state.converged or self.state.iteration >= self.max_iter:
            self._finish()

    def _finish(self) -> None:
        self.state.converged = True

        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.exception(f"Animation stopping exception {e}")

        self.ax.set_title(
            f"Finished: epochs {self.state.iteration}, acc={self.state.accuracy() * 100.0:.2f}%"
        )

        try:
            self.btn.disconnect_events()
            self.btn.ax.set_facecolor("lightgray")
            self.btn.label.set_text("Finished")
        except Exception as e:
            logger.exception(f"Exception {e}")

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f"Drawing exception {e}")

        logger.info("Training done. accuracy=%.2f%%", self.state.accuracy() * 100.0)

    def _on_button(self, event) -> None:
        if self.state.converged:
            return

        if self.anim is None:
            logger.info("Start pressed — creating animation")
            self.anim = FuncAnimation(
                self.fig,
                self._step,
                interval=self.interval,
                repeat=False,
                blit=False,
            )
            try:
                self.anim.event_source.start()
            except Exception as e:
                logger.exception(f"Event starting exception {e}")
            self.paused = False
            try:
                self.btn.label.set_text("Pause")
            except Exception as e:
                logger.exception(f"Exception {e}")
            return

        if self.paused:
            try:
                self.anim.event_source.start()
            except Exception as e:
                logger.exception(f"Resume exception {e}")
            self.paused = False
            try:
                self.btn.label.set_text("Pause")
            except Exception as e:
                logger.exception(f"Exception {e}")
        else:
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.exception(f"Pause exception {e}")
            self.paused = True
            try:
                self.btn.label.set_text("Resume")
            except Exception as e:
                logger.exception(f"Exception {e}")

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f"Drawing exception {e}")

    def run(self) -> None:
        logger.info("Starting Perceptron GUI (press Start)")
        plt.show()