import logging

from common.config.log import LoggingSettings
from lab5_potentials.app import PotentialApp, PotentialDataGenerator, PotentialState

logging.basicConfig(level=LoggingSettings.LEVEL, format=LoggingSettings.FORMAT)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    generator = PotentialDataGenerator(
        class1_center=(-1.8, -3.8),
        class2_center=(1.9, 1.8),
        train_spread=2.5,
        test_spread=1.2,
    )

    train = generator.generate_training_set(seed=1111)
    test = generator.generate_test_set(n_samples=250, seed=42)

    state = PotentialState(
        train_points=train.points,
        train_labels=train.labels,
        test_points=test.points,
        test_labels=test.labels,
    )

    app = PotentialApp(
        state=state,
        interval=350,
        max_epochs=100,
    )
    app.run()
