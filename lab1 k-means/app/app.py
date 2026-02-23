from __future__ import annotations
from typing import Optional
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from .kmeans import KMeansState
from .settings import LoggingSettings

logging.basicConfig(
    level=LoggingSettings.LEVEL,
    format=LoggingSettings.FORMAT,
)

logger = logging.getLogger(__name__)


class KMeansApp:

    def __init__(self, state: KMeansState, interval: int, max_iter: int) -> None:
        self.state = state
        self.interval = interval
        self.max_iter = max_iter

        self.anim: Optional[FuncAnimation] = None
        self.paused: bool = True

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.18)

        self.scat = self.ax.scatter(
            self.state.data[:, 0], self.state.data[:, 1], s=10, alpha=0.6
        )
        self.centroid_scat = self.ax.scatter(
            self.state.centroids[:, 0],
            self.state.centroids[:, 1],
            s=180,
            marker="X",
            edgecolor="k",
        )
        self.ax.set_title("Initial state (press Start)")

        ax_btn = plt.axes([0.35, 0.05, 0.3, 0.075])
        self.btn = Button(ax_btn, "Start")
        self.btn.on_clicked(self._on_button)

    def _render(self) -> None:
        self.scat.set_array(self.state.labels)
        self.centroid_scat.set_offsets(self.state.centroids)
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

        shift = self.state.step()
        self.ax.set_title("Iteration %d" % self.state.iteration)
        self._render()
        logger.info(
            "Iteration %d, max_shift=%.6f",
            self.state.iteration,
            shift,
        )

        if self.state.converged or self.state.iteration >= self.max_iter:
            self._finish()

    def _finish(self) -> None:
        self.state.converged = True
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.exception(f"Animation stopping exception {e}")
        self.ax.set_title(f"Final iteration {self.state.iteration}")
        self.btn.label.set_text("Finished")
        try:
            self.btn.disconnect_events()
            self.btn.ax.set_facecolor("lightgray")
        except Exception as e:
            logger.exception(f"Exception {e}")
        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.exception(f"Drawing exception {e}")
        logger.info(
            "The end: iteration's count %d, inertia=%.6f",
            self.state.iteration,
            self.state.inertia(),
        )

    def _on_button(self, event) -> None:
        if self.state.converged:
            return

        if self.anim is None:
            logger.info("Start pressed — animation are creating ")
            self.anim = FuncAnimation(
                self.fig,
                self._step,
                interval=self.interval,
                repeat=False,
                blit=False,
            )
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                try:
                    self.fig.canvas.draw()
                except Exception as e:
                    logger.exception(f"Drawing exception {e}")
            try:
                self.anim.event_source.start()
            except Exception as e:
                logger.exception(f"Event starting exception {e}")
            self.paused = False
            self.btn.label.set_text("Pause")
            return

        if self.paused:
            try:
                self.anim.event_source.start()
            except Exception as e:
                logger.exception(f"Non-starting resume {e}")
            self.btn.label.set_text("Pause")
            self.paused = False
        else:
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.exception(f"Non-starting pause {e}")
            self.btn.label.set_text("Resume")
            self.paused = True

        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.exception(f"Drawing exception {e}")

    def run(self) -> None:
        logger.info("GUI is starting — waiting Start")
        plt.show()
