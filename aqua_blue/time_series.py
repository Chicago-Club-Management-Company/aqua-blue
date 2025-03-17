"""
Module defining the TimeSeries object.

This module provides a `TimeSeries` class for handling time-dependent data.
A `TimeSeries` instance consists of an array of dependent variables (observed values) and
an associated list of timestamps.

This class supports standard time series operations such as saving/loading data,
basic arithmetic operations, and ensuring uniform time spacing.

Attributes:
    dependent_variable (np.typing.NDArray[np.floating]):
        The time series values stored as a NumPy array.
    times (List[float]):
        The timestamps associated with each data point.
"""

from typing import IO, Union, List
from pathlib import Path
import warnings
from itertools import pairwise

from dataclasses import dataclass
import numpy as np


class ShapeChangedWarning(Warning):
    """
    Warning class for cases where the shape of `dependent_variable` is modified
    within the `TimeSeries` initialization.
    """


@dataclass
class TimeSeries:
    """
    A class for representing and manipulating time series data.

    A `TimeSeries` object contains both a sequence of dependent variables (numerical values)
    and their corresponding timestamps. This class provides utilities for ensuring uniform
    time spacing and supports common operations like saving, loading, and arithmetic operations.

    Attributes:
        dependent_variable (np.typing.NDArray[np.floating]):
            The observed time series values.
        times (List[float]):
            The timestamps associated with each observation.
    """

    dependent_variable: np.typing.NDArray[np.floating]
    """NumPy array storing the observed values in the time series."""

    times: List[float]
    """List of timestamps associated with each data point in the time series."""

    def __post_init__(self):
        """
        Validates and processes the time series data after initialization.

        - Ensures `times` is a Python list.
        - Checks if time steps are uniform; raises an error otherwise.
        - Reshapes `dependent_variable` into a 2D array if it was 1D.
        """

        if isinstance(self.times, np.ndarray):
            self.times = self.times.tolist()

        timesteps = [t2 - t1 for t1, t2 in pairwise(self.times)]

        if not np.isclose(np.std(timesteps), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        if np.isclose(np.mean(timesteps), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")

        if len(self.dependent_variable.shape) == 1:
            num_steps = len(self.dependent_variable)
            self.dependent_variable = self.dependent_variable.reshape(num_steps, 1)
            warnings.warn(
                f"TimeSeries.dependent_variable should have shape (number of steps, dimensionality). "
                f"The shape has been changed from {(num_steps,)} to {self.dependent_variable.shape}",
                ShapeChangedWarning
            )

    def save(self, fp: Union[IO, str, Path], header: str = "", delimiter=","):
        """
        Save the time series to a file.

        Args:
            fp (Union[IO, str, Path]):
                File path or file-like object to save the time series.
            header (str, optional):
                Optional header string for the file. Defaults to an empty string.
            delimiter (str, optional):
                The delimiter character used in the output file. Defaults to ','.
        """
        np.savetxt(
            fp,
            np.vstack((self.times, self.dependent_variable.T)).T,
            delimiter=delimiter,
            header=header,
            comments=""
        )

    @property
    def num_dims(self) -> int:
        """
        Returns the dimensionality of the time series.

        Returns:
            int: The number of dimensions in the dependent variable array.
        """
        return self.dependent_variable.shape[1]

    @classmethod
    def from_csv(cls, fp: Union[IO, str, Path], time_index: int = 0):
        """
        Load a time series from a CSV file.

        Args:
            fp (Union[IO, str, Path]):
                File path or file-like object from which to load the time series.
            time_index (int, optional):
                Column index corresponding to time values. Defaults to 0.

        Returns:
            TimeSeries: A new `TimeSeries` instance populated with data from the CSV file.
        """
        data = np.loadtxt(fp, delimiter=",")
        times = data[:, time_index].tolist()

        return cls(
            dependent_variable=np.delete(data, obj=time_index, axis=1),
            times=times
        )

    @property
    def timestep(self) -> float:
        """
        Computes the time step between consecutive observations.

        Returns:
            float: The time step between consecutive time values.
        """
        return self.times[1] - self.times[0]

    def __eq__(self, other) -> bool:
        """
        Checks whether two `TimeSeries` instances are equal.

        Returns:
            bool: `True` if both the times and values match, otherwise `False`.
        """
        return all(t1 == t2 for t1, t2 in zip(self.times, other.times)) and bool(np.all(
            np.isclose(self.dependent_variable, other.dependent_variable)
        ))

    def __getitem__(self, key):
        """
        Indexing support to retrieve a subset of the time series.

        Args:
            key: Index or slice object.

        Returns:
            TimeSeries: A new `TimeSeries` instance containing the selected subset.
        """
        return TimeSeries(self.dependent_variable[key], self.times[key])

    def __setitem__(self, key, value):
        """
        Supports in-place modification of `TimeSeries` data.

        Args:
            key: Index or slice object.
            value (TimeSeries): The new time series data to assign.

        Raises:
            TypeError: If `value` is not a `TimeSeries` instance.
            ValueError: If the index is out of range.
        """
        if not isinstance(value, TimeSeries):
            raise TypeError("Value must be a TimeSeries object")
        if isinstance(key, slice) and key.stop > len(self.dependent_variable):
            raise ValueError("Slice stop index out of range")
        if isinstance(key, int) and key >= len(self.dependent_variable):
            raise ValueError("Index out of range")

        self.dependent_variable[key] = value.dependent_variable
        self.times[key] = value.times

    def __add__(self, other):
        """
        Adds two `TimeSeries` instances element-wise.

        Returns:
            TimeSeries: A new instance with summed values.

        Raises:
            ValueError: If the time series do not have the same length or times.
        """
        if not len(self.times) == len(other.times):
            raise ValueError("can only add TimeSeries instances that have the same number of timesteps")

        if not np.all(self.times == other.times):
            raise ValueError("can only add TimeSeries instances that span the same times")

        return TimeSeries(
            dependent_variable=self.dependent_variable + other.dependent_variable,
            times=self.times
        )

    def __sub__(self, other):
        """
        Subtracts two `TimeSeries` instances element-wise.

        Returns:
            TimeSeries: A new instance with subtracted values.

        Raises:
            ValueError: If the time series do not have the same length or times.
        """
        if not len(self.times) == len(other.times):
            raise ValueError("can only subtract TimeSeries instances that have the same number of timesteps")

        if not np.all(self.times == other.times):
            raise ValueError("can only subtract TimeSeries instances that span the same times")

        return TimeSeries(
            dependent_variable=self.dependent_variable - other.dependent_variable,
            times=self.times
        )

    def __rshift__(self, other):
        """
        Concatenates two `TimeSeries` instances with non-overlapping time values.

        Returns:
            TimeSeries: A new `TimeSeries` instance combining both input series.

        Raises:
            ValueError: If the time series overlap in time.
        """
        if self.times[-1] >= other.times[0]:
            print(self.times[-1], other.times[0])
            raise ValueError("can only concatenate TimeSeries instances with non-overlapping time values")

        return TimeSeries(
            dependent_variable=np.vstack((self.dependent_variable, other.dependent_variable)),
            times=self.times + other.times
        )

    def __len__(self):
        """
        Returns the number of timesteps in the time series.

        Returns:
            int: Number of time points.
        """
        return len(self.times)
