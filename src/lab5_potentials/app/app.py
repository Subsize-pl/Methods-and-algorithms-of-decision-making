from typing import Optional

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button

from common.config.log import LoggingSettings
from .potential import PotentialState

logging.basicConfig(level=LoggingSettings.LEVEL, format=LoggingSettings.FORMAT)
logger = logging.getLogger(__name__)


class PotentialApp:
    # Interactive GUI for the potential method
    def __init__(
        self,
        state: PotentialState,
        interval: int = 350,
        max_epochs: int = 100,
    ) -> None:
        self.state = state
        self.interval = interval
        self.max_epochs = max_epochs

        self.train_anim: Optional[FuncAnimation] = None
        self.stage = 'idle'  # idle -> training -> trained -> tested

        self.class_colors = ['#1f77b4', '#ff7f0e']
        self.region_cmap = ListedColormap(['#dceaf7', '#fde4cf'])

        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.2, 1.1], wspace=0.18)

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.info_ax = self.fig.add_subplot(gs[0, 1])

        plt.subplots_adjust(bottom=0.18)

        self._xmin, self._xmax, self._ymin, self._ymax = self._compute_bounds()
        self._grid_x, self._grid_y, self._grid_xx, self._grid_yy = self._build_grid()

        self.background = self.ax.imshow(
            self._grid_predictions(),
            origin='lower',
            extent=self._extent(),
            cmap=self.region_cmap,
            vmin=-0.5,
            vmax=1.5,
            alpha=0.55,
            interpolation='nearest',
            aspect='auto',
        )

        self.boundary = None

        self.train_class1 = self.ax.scatter(
            [],
            [],
            s=120,
            marker='o',
            facecolors=self.class_colors[0],
            edgecolors='k',
            linewidths=1.0,
            label='Class 1',
        )
        self.train_class2 = self.ax.scatter(
            [],
            [],
            s=120,
            marker='o',
            facecolors=self.class_colors[1],
            edgecolors='k',
            linewidths=1.0,
            label='Class 2',
        )

        self.support_scatter = self.ax.scatter(
            [],
            [],
            s=220,
            marker='o',
            facecolors='none',
            edgecolors='crimson',
            linewidths=2.0,
            label='Support vectors',
        )

        self.test_scatter = self.ax.scatter(
            [],
            [],
            s=70,
            marker='X',
            facecolors='none',
            edgecolors='black',
            linewidths=1.1,
            alpha=0.0,
            label='Test set',
        )

        self.ax.set_xlim(self._xmin, self._xmax)
        self.ax.set_ylim(self._ymin, self._ymax)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title('Potential method: press Train')

        ax_train = plt.axes([0.25, 0.05, 0.18, 0.075])
        self.btn_train = Button(ax_train, 'Train')
        self.btn_train.on_clicked(self._on_train)

        ax_test = plt.axes([0.56, 0.05, 0.22, 0.075])
        self.btn_test = Button(ax_test, 'Classify test set')
        self.btn_test.on_clicked(self._on_test)

        self._set_button_state(self.btn_test, enabled=False)

        self.info_ax.set_xlim(0.0, 1.0)
        self.info_ax.set_ylim(0.0, 1.0)
        self.info_ax.axis('off')

        self._render()

    def _set_button_state(self, btn: Button, enabled: bool) -> None:
        if enabled:
            btn.ax.set_facecolor('#d8f3dc')
        else:
            btn.ax.set_facecolor('#e6e6e6')

    def _compute_bounds(self) -> tuple[float, float, float, float]:
        pts = self.state.train_points
        x = pts[:, 0]
        y = pts[:, 1]

        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())

        dx = max((xmax - xmin) * 0.55, 1.0)
        dy = max((ymax - ymin) * 0.55, 1.0)

        return xmin - dx, xmax + dx, ymin - dy, ymax + dy

    def _build_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_x = np.linspace(self._xmin, self._xmax, 240)
        grid_y = np.linspace(self._ymin, self._ymax, 240)
        xx, yy = np.meshgrid(grid_x, grid_y)
        return grid_x, grid_y, xx, yy

    def _extent(self) -> tuple[float, float, float, float]:
        return self._xmin, self._xmax, self._ymin, self._ymax

    def _grid_predictions(self) -> np.ndarray:
        grid = np.column_stack([self._grid_xx.ravel(), self._grid_yy.ravel()])
        pred = self.state.predict(grid)
        pred = np.asarray(pred, dtype=int)
        # Map classes to 0/1 for the background image
        return np.where(pred > 0, 0, 1).reshape(self._grid_xx.shape)

    def _draw_boundary(self) -> None:
        # Remove previous contour safely
        if self.boundary is not None:
            try:
                self.boundary.remove()
            except Exception:
                try:
                    for coll in getattr(self.boundary, "collections", []):
                        coll.remove()
                except Exception:
                    pass
            self.boundary = None

        scores = self.state.decision_function_batch(
            np.column_stack([self._grid_xx.ravel(), self._grid_yy.ravel()])
        ).reshape(self._grid_xx.shape)

        try:
            self.boundary = self.ax.contour(
                self._grid_xx,
                self._grid_yy,
                scores,
                levels=[0.0],
                colors="black",
                linewidths=2.0,
            )
        except Exception:
            self.boundary = None

    def _render_train_points(self) -> None:
        class1_pts = self.state.train_points[self.state.train_labels == 1]
        class2_pts = self.state.train_points[self.state.train_labels == -1]

        self.train_class1.set_offsets(class1_pts if class1_pts.size else np.empty((0, 2)))
        self.train_class2.set_offsets(class2_pts if class2_pts.size else np.empty((0, 2)))

        sv_idx = np.where(self.state.alphas > 0)[0]
        if sv_idx.size > 0:
            self.support_scatter.set_offsets(self.state.train_points[sv_idx])
        else:
            self.support_scatter.set_offsets(np.empty((0, 2)))

    def _render_test_points(self) -> None:
        if self.state.test_points is None or self.state.test_predictions is None:
            self.test_scatter.set_offsets(np.empty((0, 2)))
            return

        preds = np.asarray(self.state.test_predictions, dtype=int)
        colors = np.where(preds > 0, self.class_colors[0], self.class_colors[1])

        self.test_scatter.set_offsets(self.state.test_points)
        self.test_scatter.set_facecolors(colors)
        self.test_scatter.set_edgecolors('black')
        self.test_scatter.set_alpha(0.35)
        self.test_scatter.set_sizes(np.full(self.state.test_points.shape[0], 30.0))

    def _render_info(self) -> None:
        self.info_ax.cla()
        self.info_ax.set_xlim(0.0, 1.0)
        self.info_ax.set_ylim(0.0, 1.0)
        self.info_ax.axis('off')

        self.info_ax.text(0.05, 0.80, 'Potential method', fontsize=13, weight='bold', va='top')
        self.info_ax.text(0.05, 0.73, f'Epoch: {self.state.iteration}', fontsize=10)
        self.info_ax.text(0.05, 0.69, f'Updates: {self.state.last_updates}', fontsize=10)
        self.info_ax.text(0.05, 0.65, f'Support vectors: {self.state.support_vector_count()}', fontsize=10)
        self.info_ax.text(0.05, 0.60, f'Train accuracy: {self.state.train_accuracy() * 100.0:.2f}%', fontsize=10)

        if self.state.converged:
            self.info_ax.text(0.05, 0.55, 'Training finished', fontsize=10, weight='bold')
        elif self.stage == 'training':
            self.info_ax.text(0.05, 0.55, 'Training in progress', fontsize=10, weight='bold')
        else:
            self.info_ax.text(0.05, 0.55, 'Waiting for training', fontsize=10, weight='bold')

        self.info_ax.text(0.05, 0.47, 'Separating function:', fontsize=10, weight='bold')
        self.info_ax.text(
            0.05,
            0.43,
            self.state.decision_expression(),
            fontsize=8.5,
            family='monospace',
            wrap=True,
        )

        self.info_ax.scatter([0.12], [0.30], s=90, c=[self.class_colors[0]], marker='o', edgecolors='k')
        self.info_ax.text(0.25, 0.28, 'Class 1', fontsize=10, va='center')
        self.info_ax.scatter([0.12], [0.20], s=90, c=[self.class_colors[1]], marker='o', edgecolors='k')
        self.info_ax.text(0.25, 0.18, 'Class 2', fontsize=10, va='center')
        self.info_ax.scatter([0.12], [0.10], s=120, facecolors='none', edgecolors='crimson', marker='o', linewidths=2.0)
        self.info_ax.text(0.25, 0.08, 'Support vector', fontsize=10, va='center')

        if self.state.test_predictions is not None:
            self.info_ax.text(0.05, 0.02, f'Test accuracy: {self.state.test_accuracy * 100.0:.2f}%', fontsize=10)

    def _render(self) -> None:
        self.background.set_data(self._grid_predictions())
        self._draw_boundary()
        self._render_train_points()
        self._render_test_points()
        self._render_info()

        title = 'Potential method'
        if self.stage == 'training':
            title = f'Training | epoch {self.state.iteration} | updates {self.state.last_updates}'
        elif self.stage == 'trained':
            title = 'Training completed. Press Classify test set.'
        elif self.stage == 'tested':
            title = 'Test sample classified.'
        self.ax.set_title(title)

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            try:
                self.fig.canvas.draw()
            except Exception as e:
                logger.exception(f'Drawing exception: {e}')

    def _train_step(self, frame=None) -> None:
        if self.state.converged or self.state.iteration >= self.max_epochs:
            self._finish_training()
            return

        report = self.state.train_step()
        logger.info(
            'Epoch %d, updates=%d, train_accuracy=%.2f%%, support_vectors=%d',
            self.state.iteration,
            report.updates,
            report.accuracy * 100.0,
            report.support_vectors,
        )

        self._render()

        if self.state.converged or self.state.iteration >= self.max_epochs:
            self._finish_training()

    def _finish_training(self) -> None:
        if self.train_anim is not None:
            try:
                self.train_anim.event_source.stop()
            except Exception as e:
                logger.exception(f'Animation stopping exception {e}')

        self.stage = 'trained'
        self._set_button_state(self.btn_test, enabled=True)
        self.btn_train.label.set_text('Training done')
        self._set_button_state(self.btn_train, enabled=False)

        if not self.state.converged and self.state.iteration >= self.max_epochs:
            logger.info('Max epochs reached, training stopped.')
        else:
            logger.info('Training converged.')

        self._render()

    def _on_train(self, event) -> None:
        if self.stage in {'training', 'tested'}:
            return
        if self.state.converged:
            self._set_button_state(self.btn_test, enabled=True)
            self.stage = 'trained'
            self._render()
            return

        self.stage = 'training'
        self.btn_train.label.set_text('Training...')
        self._set_button_state(self.btn_train, enabled=False)

        self.train_anim = FuncAnimation(
            self.fig,
            self._train_step,
            interval=self.interval,
            repeat=False,
            blit=False,
            cache_frame_data=False,
        )

        try:
            self.train_anim.event_source.start()
        except Exception as e:
            logger.exception(f'Event starting exception {e}')

    def _on_test(self, event) -> None:
        if not self.state.converged:
            return
        if self.stage == 'testing':
            return

        if self.state.test_points is None or self.state.test_labels is None:
            raise RuntimeError('Test set was not generated')

        self.stage = 'tested'
        preds, accuracy = self.state.classify_test_set()

        logger.info('Test sample classified')
        if accuracy is not None:
            logger.info('Test accuracy: %.2f%%', accuracy * 100.0)

        # Draw predicted class colors and mark mistakes
        colors = np.where(np.asarray(preds) > 0, self.class_colors[0], self.class_colors[1])
        self.test_scatter.set_offsets(self.state.test_points)
        self.test_scatter.set_facecolors(colors)
        self.test_scatter.set_edgecolors(
            np.where(preds == self.state.test_labels, 'black', 'crimson')
        )
        self.test_scatter.set_alpha(0.50)
        self.test_scatter.set_sizes(np.full(self.state.test_points.shape[0], 28.0))

        self._render()

    def run(self) -> None:
        logger.info('Starting potential method GUI (press Train)')
        plt.show()
