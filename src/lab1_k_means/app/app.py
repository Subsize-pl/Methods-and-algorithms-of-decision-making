from __future__ import annotations
from typing import Optional
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from common.config import LogSettings
from .kmeans import KMeansState

logging.basicConfig(
    level=LogSettings.LEVEL,
    format=LogSettings.FORMAT,
)

logger = logging.getLogger(__name__)


# Class of interactive GUI app to visualize K-means clustering in 2D
class KMeansApp:

    def __init__(
        self, state: KMeansState, interval: int = 150, max_iter: int = 200
    ) -> None:
        self.state = state
        self.interval = interval
        self.max_iter = max_iter

        self.anim: Optional[FuncAnimation] = None
        self.paused: bool = True

        # fig - figure object, the canvas where all plot elements are drawn
        # ax - the axes object, the coordinate system where points, lines, text, etc. are drawn
        # figsize=(8, 6) sets the figure size in inches (width, height)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # Adjust the bottom margin to leave space for the button so it doesn't overlap the plot
        plt.subplots_adjust(bottom=0.18)

        self.scat = self.ax.scatter(
            self.state.data[:, 0],  # takes all X of points
            self.state.data[:, 1],  # takes all Y of points
            s=10,  # point size
            alpha=0.6,  # transparency
        )

        self.centroid_scat = self.ax.scatter(
            self.state.centroids[:, 0],
            self.state.centroids[:, 1],
            s=180,
            marker="X",  # marker shape: cross (X)
            edgecolor="k",  # marker edge color: black
        )
        self.ax.set_title("Initial state (press Start)")

        ax_btn = plt.axes([0.35, 0.05, 0.3, 0.075])
        self.btn = Button(ax_btn, "Start")
        self.btn.on_clicked(self._on_button)

    def _render(self) -> None:
        # Associates each color of a point with its placemark
        self.scat.set_array(self.state.labels)

        # Updates centroids coordinates
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

        # If K-means has already converged, do nothing
        if self.state.converged:
            return

        if self.anim is None:
            logger.info("Start pressed — creating animation")
            self.anim = FuncAnimation(
                self.fig,  # figure to update
                self._step,  # function called at each frame (one K-means iteration)
                interval=self.interval,  # delay between frames in milliseconds
                repeat=False,  # do not loop the animation
                blit=False,  # redraw the entire canvas each frame (simpler and more reliable)
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

        # Toggle Pause / Resume
        if self.paused:
            # Resume animation
            try:
                self.anim.event_source.start()
            except Exception as e:
                logger.exception(f"Non-starting resume {e}")
            self.btn.label.set_text("Pause")
            self.paused = False
        else:
            # Pause animation
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.exception(f"Non-starting pause {e}")
            self.btn.label.set_text("Resume")
            self.paused = True

        # Safe redraw after any change
        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.exception(f"Drawing exception {e}")

    def run(self) -> None:
        logger.info("GUI is starting — waiting Start")
        plt.show()
