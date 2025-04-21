import logging
import sys

import numpy as np

import aqua_blue


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    t = np.arange(5_000) / 100
    y = np.vstack((np.cos(t) ** 2, np.sin(t))).T

    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)
    normalizer = aqua_blue.utilities.Normalizer()
    normalized_time_series = normalizer.normalize(time_series)

    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=2,
            sparsity=0.5,
            spectral_radius=0.95
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(normalized_time_series)


if __name__ == "__main__":

    main()
