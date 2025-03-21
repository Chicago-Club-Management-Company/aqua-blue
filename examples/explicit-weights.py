import numpy as np
import matplotlib.pyplot as plt

import aqua_blue


def main():
    t = np.arange(10_000) / 100
    y = np.vstack((np.cos(t) ** 2, np.sin(t))).T

    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)
    generator = np.random.default_rng(seed=0)

    w_res = generator.uniform(
        low=-0.5,
        high=0.5,
        size=(100, 100)
    )
    w_in = generator.uniform(
        low=-0.5,
        high=0.5,
        size=(100, 2)
    )

    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=2,
            w_res=w_res,
            w_in=w_in,
            spectral_radius=1.2,
            sparsity=0.99
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(time_series)

    prediction = model.predict(horizon=1_000)
    prediction = normalizer.denormalize(prediction)

    actual_future = np.vstack((np.cos(prediction.times) ** 2, np.sin(prediction.times))).T

    plt.plot(prediction.times, actual_future)
    plt.plot(prediction.times, prediction.dependent_variable)
    plt.legend(['actual x', 'actual y', 'predicted x', 'predicted y'])
    plt.title("Explicit Weight Matrix Example")

    plt.show()


if __name__ == "__main__":
    main()
