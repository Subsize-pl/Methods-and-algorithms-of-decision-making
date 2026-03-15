from common.data_generators.random_generator import RandomDataGenerator
from app import KMeansState
from app import KMeansApp

if __name__ == "__main__":
    data_generator = RandomDataGenerator(min_x=-50, max_x=50)
    data = data_generator.generate_mixture(n_samples=20_000, seed=400)

    state = KMeansState(data=data, k=6)
    app = KMeansApp(state=state, interval=60, max_iter=200)
    app.run()
