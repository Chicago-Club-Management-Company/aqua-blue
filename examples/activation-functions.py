from typing import Dict, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

import aqua_blue


def main():

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    activation_functions: Dict[str, Callable] = {"tanh": np.tanh, "erf": scipy.special.erf}

    t = np.arange(10_000) / 100
    y = np.vstack((np.cos(t) ** 2, np.sin(t))).T

    for ax, activation_function in zip(axs, activation_functions):
        time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)
        normalizer = aqua_blue.utilities.Normalizer()
        time_series = normalizer.normalize(time_series)

        model = aqua_blue.models.Model(
            reservoir=aqua_blue.reservoirs.DynamicalReservoir(
                reservoir_dimensionality=100,
                input_dimensionality=2,
                activation_function=activation_functions[activation_function]
            ),
            readout=aqua_blue.readouts.LinearReadout()
        )
        model.train(time_series)

        prediction = model.predict(horizon=1_000)
        prediction = normalizer.denormalize(prediction)

        actual_future = np.vstack((np.cos(prediction.times) ** 2, np.sin(prediction.times))).T

        ax.plot(prediction.times, actual_future)
        ax.plot(prediction.times, prediction.dependent_variable)
        ax.legend(['actual_x', 'actual_y', 'predicted_x', 'predicted_y'])
        ax.set_title(activation_function)

    plt.show()


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
