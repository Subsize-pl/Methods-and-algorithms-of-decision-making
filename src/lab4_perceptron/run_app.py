from __future__ import annotations

import logging

import numpy as np

from common.config.log import LoggingSettings
from lab4_perceptron.app import PerceptronApp, PerceptronState
from common.data_generators.perceptron_generator import PerceptronDataGenerator

logging.basicConfig(
    level=LoggingSettings.LEVEL,
    format=LoggingSettings.FORMAT,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    n_classes = 7
    samples_per_class = 165
    seed = 74

    generator = PerceptronDataGenerator(radius=8.0, spread=0.7)
    data, labels = generator.generate_2d(
        n_classes=n_classes,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    state = PerceptronState(
        data=data,
        labels=labels,
        n_classes=n_classes,
        learning_rate=1.0,
    )

    app = PerceptronApp(state=state, interval=120, max_iter=200)
    app.run()

    test_points = np.array(
        [
            [0.0, 0.0],
            [8.0, 0.0],
            [-4.0, 7.0],
            [15, 15]
        ],
        dtype=float,
    )
    test_preds = state.predict(test_points)

    for point, pred in zip(test_points, test_preds):
        logger.info("Test point %s -> class %d", point, pred)


