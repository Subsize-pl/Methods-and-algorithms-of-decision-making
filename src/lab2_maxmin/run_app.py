from __future__ import annotations
from common.data_generators.random_generator import RandomDataGenerator
from app import MaximinState, MaximinApp

if __name__ == "__main__":
    data_generator = RandomDataGenerator(min_x=-50, max_x=50)
    data = data_generator.generate_mixture(n_samples=20_000, seed=400)

    state = MaximinState(data)
    app = MaximinApp(state=state, interval=160, max_iter=200)
    app.run()
