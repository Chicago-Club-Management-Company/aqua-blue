"""
Module defining models, i.e., compositions of reservoir(s) and readout layers.

This module implements the `Model` class, which integrates a reservoir and a readout
layer to process time series data. The model enables training using input time series
data and forecasting future values based on learned patterns.

Classes:
    - Model: Represents an Echo State Network (ESN)-based model that learns from
      input time series data and makes future predictions.
"""

from dataclasses import dataclass, field
from typing import Union, Generic

import numpy as np

from .reservoirs import Reservoir
from .readouts import Readout
from .time_series import TimeSeries
from .datetimelikearray import DatetimeLikeArray, DatetimeLike, TimeDeltaLike

import datetime


@dataclass
class Model(Generic[DatetimeLike, TimeDeltaLike]):
    """
    A machine learning model that integrates a reservoir with a readout layer for
    time series forecasting.

    This class implements an Echo State Network (ESN) approach, where the reservoir
    serves as a high-dimensional dynamic system, and the readout layer maps reservoir
    states to output values.

    Attributes:
        reservoir (Reservoir):
            The reservoir component, defining the input-to-reservoir mapping.
        readout (Readout):
            The readout layer, mapping reservoir states to output values.
        final_time (float):
            The last timestamp seen during training. This is set automatically after training.
        timestep (float):
            The fixed time interval between consecutive steps in the input time series,
            set during training.
        initial_guess (np.ndarray):
            The last observed state of the system during training, used as an initial
            condition for predictions.
        tz (Union[datetime.tzinfo, None]):
            The timezone associated with the time series. Set to `None` if the `DatetimeLikeArray`
            is incompatible.
    """

    reservoir: Reservoir
    """The reservoir component that defines the input-to-reservoir mapping."""

    readout: Readout
    """The readout component that defines the reservoir-to-output mapping."""

    final_time: DatetimeLike = field(init=False)
    """The final timestamp encountered in the training dataset (set during training)."""

    timestep: TimeDeltaLike = field(init=False)
    """The fixed time step interval of the training dataset (set during training)."""

    initial_guess: np.typing.NDArray[np.floating] = field(init=False)
    """The last observed state of the system, used for future predictions (set during training)."""

    tz: Union[datetime.tzinfo, None] = field(init=False)
    """The timezone associated with the independent variable. Set to `None` if unsupported."""

    def train(
        self,
        input_time_series: TimeSeries,
        warmup: int = 0
    ):
        """
        Trains the model on the provided time series data.

        This method fits the readout layer using reservoir states obtained from the
        input time series data. A warmup period can be specified to exclude initial
        steps from training.

        Args:
            input_time_series (TimeSeries):
                The time series instance used for training.
            warmup (int):
                The number of initial steps to ignore in training (default: 0).

        Raises:
            ValueError: If `warmup` is greater than or equal to the number of timesteps
                        in the input time series.
        """
        if warmup >= len(input_time_series.times):
            raise ValueError(f"warmup must be smaller than number of timesteps ({len(input_time_series)})")

        time_series_array = input_time_series.dependent_variable
        independent_variables = np.zeros((time_series_array.shape[0]-1, self.reservoir.reservoir_dimensionality))

        for i in range(independent_variables.shape[0]):
            independent_variables[i] = self.reservoir.update_reservoir(time_series_array[i])

        dependent_variables = time_series_array[1:]
        if warmup > 0:
            independent_variables = independent_variables[warmup:]
            dependent_variables = dependent_variables[warmup:]

        self.readout.train(independent_variables, dependent_variables)
        self.timestep = input_time_series.timestep
        self.final_time = input_time_series.times[-1]
        self.tz = input_time_series.times.tz
        self.times_dtype = input_time_series.times.dtype
        self.initial_guess = time_series_array[-1, :]

    def predict(self, horizon: int) -> TimeSeries:
        """
        Generates future predictions for a specified time horizon.

        This method uses the trained model to generate future values based on the
        learned dynamics of the input time series.

        Args:
            horizon (int):
                The number of steps to forecast into the future.

        Returns:
            TimeSeries: A `TimeSeries` instance containing the predicted values and
            corresponding timestamps.
        """
        predictions = np.zeros((horizon, self.reservoir.input_dimensionality))

        for i in range(horizon):
            if i == 0:
                predictions[i, :] = self.readout.reservoir_to_output(
                    self.reservoir.res_state
                )
                continue
            predictions[i, :] = self.readout.reservoir_to_output(
                self.reservoir.update_reservoir(predictions[i-1, :])
            )

        if isinstance(self.final_time, float):
            assert isinstance(self.timestep, float)

        if isinstance(self.final_time, datetime.datetime):
            assert isinstance(self.timestep, datetime.timedelta)

        if isinstance(self.final_time, np.datetime64):
            assert isinstance(self.timestep, np.timedelta64)

        times_ = DatetimeLikeArray.from_array(
            np.arange(
                start=self.final_time + self.timestep,
                stop=self.final_time + (horizon + 1) * self.timestep,
                step=self.timestep,
                dtype=self.times_dtype
            ),
            tz=self.tz,
        )

        return TimeSeries(
            dependent_variable=predictions,
            times=times_
        )
