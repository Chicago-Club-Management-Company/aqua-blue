"""
Module defining machine learning models composed of reservoirs and readout layers.

This module provides the `Model` class, which integrates a reservoir and a readout
layer to process time series data. The class enables training using input time series
data and forecasting future values based on learned patterns.

It primarily follows the Echo State Network (ESN) architecture, where a reservoir
captures the system dynamics and a readout layer maps reservoir states to outputs.
"""

from dataclasses import dataclass, field

import numpy as np

from .reservoirs import Reservoir
from .readouts import Readout
from .time_series import TimeSeries


@dataclass
class Model:
    """
    A machine learning model that combines a reservoir with a readout layer
    for time series forecasting.

    This class implements a standard Echo State Network (ESN) approach, where
    the reservoir serves as a high-dimensional dynamic system, and the readout
    layer is a simple linear mapping.

    Attributes:
        reservoir (Reservoir): The reservoir component, defining input-to-reservoir mapping.
        readout (Readout): The readout layer, mapping reservoir states to output values.
        final_time (float): The last timestamp seen during training.
            This is set automatically after training.
        timestep (float): The time interval between consecutive steps in the
            input time series, set during training.
        initial_guess (np.typing.NDArray[np.floating]): The last observed state
            of the system from training, used as an initial condition for predictions.
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

    def train(
        self,
        input_time_series: TimeSeries,
        warmup: int = 0,
        rcond: float = 1.0e-10
    ):
        """
        Trains the model using a given time series.

        The training process involves:
        1. Passing input data through the reservoir to update its internal states.
        2. Using the reservoir states and corresponding outputs to compute the readout layer's weights.
        3. Storing the final time step, timestep, and last observed state for future predictions.

        Args:
            input_time_series (TimeSeries): The input time series to train on.
            warmup (int, optional): Number of initial time steps to ignore
                during training (default: 0). Helps avoid transient effects.
            rcond (float, optional): Regularization parameter for computing
                the pseudo-inverse in the readout layer (default: `1.0e-10`).
                Increasing this value can stabilize predictions.

        Raises:
            ValueError: If `warmup` is greater than or equal to the number of time steps.

        Example:
            ```python
            model = Model(reservoir, readout)
            model.train(time_series, warmup=10)
            ```
        """

        if warmup >= len(input_time_series.times):
            raise ValueError(
                f"warmup must be smaller than the number of timesteps ({len(input_time_series)})"
            )

        # Extract dependent variable (actual time series values)
        time_series_array = input_time_series.dependent_variable

        # Create an empty array for reservoir states
        independent_variables = np.zeros(
            (time_series_array.shape[0] - 1, self.reservoir.reservoir_dimensionality)
        )

        # Update reservoir states for each time step
        for i in range(independent_variables.shape[0]):
            independent_variables[i] = self.reservoir.update_reservoir(time_series_array[i])

        # Define the target values (next-step predictions)
        dependent_variables = time_series_array[1:]

        # Apply warmup (discard initial transient states)
        if warmup > 0:
            independent_variables = independent_variables[warmup:]
            dependent_variables = dependent_variables[warmup:]

        # Compute readout weights using pseudo-inverse (ridge regression)
        w_out_transpose = np.linalg.pinv(independent_variables, rcond=rcond) @ dependent_variables
        self.readout.coefficients = w_out_transpose.T

        # Store timestep, final time, and initial guess
        self.timestep = input_time_series.timestep
        self.final_time = input_time_series.times[-1]
        self.initial_guess = time_series_array[-1, :]

    def predict(self, horizon: int) -> TimeSeries:
        """
        Generates future predictions using the trained model.

        This method uses the trained readout layer to generate a forecast
        for a given time horizon. The model iteratively predicts the next
        time step based on its own outputs.

        Args:
            horizon (int): The number of future time steps to predict.

        Returns:
            TimeSeries: A `TimeSeries` instance containing the predicted values.

        Example:
            ```python
            forecast = model.predict(horizon=50)
            plt.plot(forecast.times, forecast.dependent_variable, label="Prediction")
            ```
        """

        # Initialize prediction array
        predictions = np.zeros((horizon, self.reservoir.input_dimensionality))

        # Iteratively generate future predictions
        for i in range(horizon):
            if i == 0:
                predictions[i, :] = self.readout.reservoir_to_output(
                    self.reservoir.res_state
                )
                continue

            predictions[i, :] = self.readout.reservoir_to_output(
                self.reservoir.update_reservoir(predictions[i - 1, :])
            )

        # Construct the time series for predicted values
        return TimeSeries(
            dependent_variable=predictions,
            times=[self.final_time + step * self.timestep for step in range(1, horizon + 1)]
        )
