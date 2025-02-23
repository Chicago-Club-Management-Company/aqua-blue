from dataclasses import dataclass, field

import numpy as np

from .reservoirs import Reservoir
from .readouts import Readout
from .time_series import TimeSeries


@dataclass
class Model:

    reservoir: Reservoir
    readout: Readout
    final_time: float = field(init=False)
    timestep: float = field(init=False)
    initial_guess: np.typing.NDArray = field(init=False)

    def train(self, input_time_series: TimeSeries, warmup: int = 0, rcond: float = 1.0e-10):

        if warmup >= len(input_time_series.times):
            raise ValueError(f"warmup must be smaller than number of timesteps ({len(input_time_series)})")

        time_series_array = input_time_series.dependent_variable
        independent_variables = self.reservoir.input_to_reservoir(time_series_array[:-1, :].T).T
        dependent_variables = time_series_array[1:]

        if warmup > 0:
            independent_variables = independent_variables[warmup:]
            dependent_variables = dependent_variables[warmup:]

        w_out_transpose = np.linalg.pinv(independent_variables, rcond=rcond) @ dependent_variables
        self.readout.coefficients = w_out_transpose.T
        self.timestep = input_time_series.timestep
        self.final_time = input_time_series.times[-1]
        self.initial_guess = time_series_array[-1, :]

    def predict(self, horizon: int) -> TimeSeries:

        # initialize predictions and reservoir states to populate later
        predictions = np.zeros((horizon, self.reservoir.input_dimensionality))

        # perform feedback loop
        for i in range(horizon):
            if i == 0:
                predictions[i, :] = self.readout.reservoir_to_output(
                    self.reservoir.input_to_reservoir(self.initial_guess)
                )
                continue
            predictions[i, :] = self.readout.reservoir_to_output(
                self.reservoir.input_to_reservoir(predictions[i - 1, :])
            )

        return TimeSeries(
            dependent_variable=predictions,
            times=self.final_time + self.timestep + np.linspace(0, horizon * self.timestep, horizon)
        )
