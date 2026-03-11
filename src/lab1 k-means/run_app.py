from app import DataGenerator
from app import KMeansState
from app import KMeansApp
from app.settings import AppSettings

if __name__ == "__main__":
    data = DataGenerator.generate_mixture(
        AppSettings.N_SAMPLES,
        seed=AppSettings.SEED,
    )
    state = KMeansState(data, AppSettings.CLASS_COUNT)
    app = KMeansApp(
        state,
        interval=AppSettings.INTERVAL,
        max_iter=AppSettings.MAX_ITERATIONS,
    )
    app.run()
