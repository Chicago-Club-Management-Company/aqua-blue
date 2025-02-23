import numpy as np
import aqua_blue


def main():

    # generate arbitrary two-dimensional time series
    # y_1(t) = cos(t), y_2(t) = sin(t)
    # resulting dependent variable has shape (number of timesteps, 2)
    t = np.linspace(0, 4.0 * np.pi, 10_000)
    y = np.vstack((2.0 * np.cos(t) + 1, 5.0 * np.sin(t) - 1)).T

    # create time series object to feed into echo state network
    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)

    # normalize
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)

    # make model
    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(reservoir_dimensionality=100, input_dimensionality=2),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(time_series)
    prediction = model.predict(horizon=1_000)
    prediction = normalizer.denormalize(prediction)

    actual_future = np.vstack((
        (2.0 * np.cos(prediction.times) + 1, 5.0 * np.sin(prediction.times) - 1)
    )).T
    root_mean_square_error = np.sqrt(
        np.mean((actual_future - prediction.dependent_variable) ** 2)
    )
    print(root_mean_square_error)


if __name__ == "__main__":

    main()
