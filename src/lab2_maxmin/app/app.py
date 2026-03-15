from __future__ import annotations
from typing import Optional
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from common.config import LogSettings
from .maximin import MaximinState

logging.basicConfig(
    level=LogSettings.LEVEL,
    format=LogSettings.FORMAT,
)
logger = logging.getLogger(__name__)


# Class of interactive GUI to visualize Maximin clustering (2D)
class MaximinApp:

    def __init__(
        self,
        state: MaximinState,
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

        # initial scatter (grey)
        n = self.state.n
        grey = np.array([0.6, 0.6, 0.6, 0.7])
        colors_init = np.tile(grey, (n, 1))
        self.scat = self.ax.scatter(
            self.state.data[:, 0],
            self.state.data[:, 1],
            facecolors=colors_init,
            s=8,
            edgecolors="none",
            alpha=0.85,
        )

        # centroid scatter placeholder
        self.centroid_scat = self.ax.scatter(
            [], [], s=180, marker="X", edgecolor="k", linewidths=0.8
        )

        self.ax.set_title("Maximin: press Start")

        ax_btn = plt.axes([0.35, 0.05, 0.3, 0.075])
        self.btn = Button(ax_btn, "Start")
        self.btn.on_clicked(self._on_button)

        self._set_axes_limits()

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f"Drawing exception: {e}")

    def _set_axes_limits(self) -> None:
        x = self.state.data[:, 0]
        y = self.state.data[:, 1]
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        dx = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
        dy = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
        self.ax.set_xlim(xmin - dx, xmax + dx)
        self.ax.set_ylim(ymin - dy, ymax + dy)

    def _labels_to_colors(self, labels: np.ndarray) -> np.ndarray:
        n = labels.shape[0]
        colors = np.empty((n, 4), dtype=float)
        grey = np.array([0.6, 0.6, 0.6, 0.7])
        colors[:] = grey
        mask = labels >= 0
        if np.any(mask):
            cmap = plt.cm.get_cmap("tab20")
            lab = labels[mask].astype(int)
            cols = cmap(lab % 20)
            colors[mask] = cols
        return colors

    def _centroid_colors(self, m: int) -> np.ndarray:
        if m == 0:
            return np.empty((0, 4))
        cmap = plt.cm.get_cmap("tab20")
        idx = np.arange(m) % 20
        return cmap(idx)

    def _render(self) -> None:
        labels = getattr(self.state, "labels", None)
        if labels is None:
            labels = np.full(self.state.n, -1, dtype=int)
        colors = self._labels_to_colors(labels)
        try:
            self.scat.set_facecolors(colors)
        except Exception:
            self.scat.set_color(colors)

        m = self.state.centroids.shape[0]
        if m > 0:
            self.centroid_scat.set_offsets(self.state.centroids)
            cent_cols = self._centroid_colors(m)
            try:
                self.centroid_scat.set_facecolors(cent_cols)
                self.centroid_scat.set_edgecolors("k")
            except Exception:
                self.centroid_scat.set_color(cent_cols)
        else:
            self.centroid_scat.set_offsets(np.empty((0, 2)))

        self.ax.set_title(
            f"Iteration {self.state.iteration} | cores={self.state.centroids.shape[0]}"
        )
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f"Drawing exception {e}")

    def _step(self, frame=None) -> None:
        if self.state.converged:
            return
        kp, added = self.state.step()
        logger.info("Iteration %d, kp=%.6f, added=%s", self.state.iteration, kp, added)
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
            f"Finished: iterations {self.state.iteration}, cores {self.state.centroids.shape[0]}"
        )
        try:
            self.btn.disconnect_events()
            self.btn.ax.set_facecolor("lightgray")
            self.btn.label.set_text("Finished")
        except Exception as e:
            logger.exception(f"Exception {e}")

        logger.info("Done. inertia=%.6f", self.state.inertia())

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f"Drawing exception {e}")

    def _on_button(self, event) -> None:
        if self.state.converged:
            return

        # If animation not created yet: create it but do not start
        if self.anim is None:
            logger.info("Start pressed — creating animation (not started)")
            self.anim = FuncAnimation(
                self.fig, self._step, interval=self.interval, repeat=False, blit=False
            )

            self.paused = True
            try:
                self.btn.label.set_text("Pause")
            except Exception as e:
                logger.exception(f"Exception {e}")
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                try:
                    self.fig.canvas.draw()
                except Exception as e:
                    logger.exception(f"Drawing exception {e}")
            return

        if self.paused:
            try:
                self.anim.event_source.start()
            except Exception as e:
                logger.exception(f"Event starting exception {e}")
            self.paused = False
            try:
                self.btn.label.set_text("Resume")
            except Exception as e:
                logger.exception(f"Exception {e}")
        else:
            # animation is running; stop it and set label back to "Pause"
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.exception(f"Pause exception {e}")
            self.paused = True
            try:
                self.btn.label.set_text("Pause")
            except Exception as e:
                logger.exception(f"Exception {e}")

        # safe redraw
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f"Drawing exception {e}")

    def run(self) -> None:
        logger.info("Starting Maximin GUI (press Start)")
        plt.show()
