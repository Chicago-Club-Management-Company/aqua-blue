"""
Module defining machine learning models composed of reservoirs and readout layers.

This module provides the `Model` class, which integrates a reservoir and a readout layer to process time series data.
The class enables training using input time series data and forecasting future values based on learned patterns.

It primarily follows the Echo State Network (ESN) architecture, where a reservoir captures the system dynamics
and a readout layer maps reservoir states to outputs.
"""

from dataclasses import dataclass, field

import numpy as np

from .reservoirs import Reservoir
from .readouts import Readout
from .time_series import TimeSeries


@dataclass
class Model:
    """
    A machine learning model that combines a reservoir with a readout layer for time series forecasting.

    This class implements a standard Echo State Network (ESN) approach, where the reservoir serves as a
    high-dimensional dynamic system, and the readout layer is a simple linear mapping.

    Attributes:
        reservoir (Reservoir): The reservoir component, defining input-to-reservoir mapping.
        readout (Readout): The readout layer, mapping reservoir states to output values.
        final_time (float): The last timestamp seen during training. This is set automatically after training.
        timestep (float): The time interval between consecutive steps in the input time series, set during training.
        initial_guess (np.typing.NDArray[np.floating]):
            The last observed state of the system from training, used as an initial condition for predictions.
    """

    reservoir: Reservoir
    """The reservoir component that defines the input-to-reservoir mapping."""

    readout: Readout
    """The readout component that defines the reservoir-to-output mapping."""

    final_time: float = field(init=False)
    """The final timestamp encountered in the training dataset (set during training)."""

    timestep: float = field(init=False)
    """The fixed time step interval of the training dataset (set during training)."""

    initial_guess: np.typing.NDArray[np.floating] = field(init=False)
    """The last observed state of the system, used for future predictions (set during training)."""

    def train(self, input_time_series: TimeSeries, warmup: int = 0, rcond: float = 1.0e-10):
        """
        Train the model using a given time series.

        This method processes input data through the reservoir and optimizes the readout layer using a
        pseudo-inverse approach.

        Args:
            input_time_series (TimeSeries): The time series instance used for training.
            warmup (int): The number of initial steps to ignore in training (default is 0).
            rcond (float): Threshold for pseudo-inverse calculation; increase if prediction is unstable (default is 1.0e-10).

        Raises:
            ValueError: If the warmup period is longer than the available timesteps in the input data.
        """

        if warmup >= len(input_time_series.times):
            raise ValueError(f"warmup must be smaller than number of timesteps ({len(input_time_series)})")

        time_series_array = input_time_series.dependent_variable
        independent_variables = np.zeros((time_series_array.shape[0] - 1, self.reservoir.reservoir_dimensionality))

        for i in range(independent_variables.shape[0]):
            independent_variables[i] = self.reservoir.update_reservoir(time_series_array[i])

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
        """
        Generate future predictions using the trained model.

        The prediction is performed iteratively, using the previous predicted state as input to the reservoir.

        Args:
            horizon (int): The number of time steps to forecast into the future.

        Returns:
            TimeSeries: A time series containing the predicted values for the specified horizon.
        """

        predictions = np.zeros((horizon, self.reservoir.input_dimensionality))

        for i in range(horizon):
            if i == 0:
                predictions[i, :] = self.readout.reservoir_to_output(self.reservoir.res_state)
                continue
            predictions[i, :] = self.readout.reservoir_to_output(
                self.reservoir.update_reservoir(predictions[i - 1, :])
            )
        
        return TimeSeries(
            dependent_variable=predictions,
            times=[self.final_time + step * self.timestep for step in range(1, horizon + 1)]
        )
