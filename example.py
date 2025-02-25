from datetime import datetime, timedelta

import numpy as np
import pytz
import aqua_blue


def main():

    # generate arbitrary two-dimensional time series
    # y_1(t) = 2cos(t) + 1, y_2(t) = 5sin(t) - 1
    # resulting dependent variable has shape (number of timesteps = 10_000, 2)
    # each index corresponds to 0 minutes from 2/24/2025 12:00 eastern time, 1 minute after, etc
    num_steps = 10_000
    t_init = datetime(
        year=2025,
        month=2,
        day=24,
        hour=12,
        tzinfo=pytz.timezone("America/New_York")
    )
    times = [t_init + timedelta(minutes=1.0) * step for step in range(num_steps)]
    t_float = np.fromiter(((t - times[0]).total_seconds() for t in times), dtype=float)
    y = np.vstack((2.0 * np.cos(t_float) + 1, 5.0 * np.sin(t_float) - 1)).T

    # create time series object to feed into echo state network
    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=times)

    # normalize
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)

    # make model
    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=2
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(time_series)
    prediction = model.predict(horizon=1_000)
    prediction = normalizer.denormalize(prediction)

    # we know actual future, so generate it, i.e. 2cos(t) + 1, 5sin(t) - 1 for remaining times in seconds
    t_float = np.fromiter(((t - t_init).total_seconds() for t in prediction.times), dtype=float)
    actual_future = np.vstack((
        (2.0 * np.cos(t_float) + 1, 5.0 * np.sin(t_float) - 1)
    )).T
    root_mean_square_error = np.sqrt(
        np.mean((actual_future - prediction.dependent_variable) ** 2)
    )
    print(root_mean_square_error)


if __name__ == "__main__":

    main()
