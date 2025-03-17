"""
Module defining the TimeSeries class for handling and processing time series data.

This module provides:
- `TimeSeries`: A class representing a time series with dependent variables and corresponding timestamps.
- Methods for saving, loading, and manipulating time series data.
"""

from typing import IO, Union, List
from pathlib import Path
import warnings
from itertools import pairwise
from datetime import datetime

from dataclasses import dataclass
import numpy as np



class ShapeChangedWarning(Warning):
    """
    Warning raised when the shape of `dependent_variable` is modified
    in `TimeSeries.__post_init__` to conform to expected dimensions.
    """


@dataclass
class TimeSeries:
    """
    A class representing a time series.

    A `TimeSeries` object consists of:
    - `dependent_variable`: The observed values at each timestep.
    - `times`: The corresponding timestamps for each observation.

    Attributes:
        dependent_variable (np.typing.NDArray[np.floating]):
            A NumPy array containing the observed values of the time series.
            It should have shape `(num_steps, num_dimensions)`.
        times (List[float] or List[datetime]):
            A list of timestamps corresponding to each observation.

    Raises:
        ValueError: If timestamps are not uniformly spaced or if the timestep is zero.
    """

    dependent_variable: np.typing.NDArray[np.floating]
    """Array of observed values in the time series, stored in shape `(num_steps, num_dimensions)`."""

    times: List[Union[float, datetime]]
    """List of timestamps corresponding to each data point in `dependent_variable`."""

    def __post_init__(self):
        """
        Validates and preprocesses the time series data after initialization.

        - Ensures that timestamps are stored as a list.
        - Converts `times` to numerical format if they are `datetime` objects.
        - Checks for uniform time spacing.
        - Ensures `dependent_variable` has the correct shape `(num_steps, num_dimensions)`.
        """
        # Ensure times is a list (if it's a numpy array, convert it)
        if isinstance(self.times, np.ndarray):
            self.times = self.times.tolist()

        # Compute the differences between consecutive timestamps
        timesteps = [t2 - t1 for t1, t2 in pairwise(self.times)]

        # Convert datetime.timedelta to numeric values (days)
        if isinstance(self.times[0], datetime):
            numeric_timesteps = [t.total_seconds() / (3600 * 24) for t in timesteps]
        else:
            numeric_timesteps = timesteps

        # Validate uniform time spacing
        if not np.isclose(np.std(numeric_timesteps), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        if np.isclose(np.mean(numeric_timesteps), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")

        # Ensure `dependent_variable` is 2D
        if len(self.dependent_variable.shape) == 1:
            num_steps = len(self.dependent_variable)
            self.dependent_variable = self.dependent_variable.reshape(num_steps, 1)
            warnings.warn(
                f"TimeSeries.dependent_variable should have shape (num_steps, num_dimensions). "
                f"The shape has been changed from {(num_steps,)} to {self.dependent_variable.shape}",
                ShapeChangedWarning
            )

    def save(self, fp: Union[IO, str, Path], header: str = "", delimiter: str = ","):
        """
        Saves the time series data to a CSV file.

        Args:
            fp (Union[IO, str, Path]): The file path or file object to save the data.
            header (str, optional): An optional header line. Defaults to an empty string.
            delimiter (str, optional): The character used to separate values. Defaults to a comma.

        The saved file contains:
        - The first column: timestamps (`times`)
        - The following columns: values from `dependent_variable`
        """
        np.savetxt(
            fp,
            np.vstack((self.times, self.dependent_variable.T)).T,
            delimiter=delimiter,
            header=header,
            comments=""
        )

    @classmethod
    def from_csv(cls, fp: Union[IO, str, Path], time_index: int = 0):
        """
        Loads a `TimeSeries` instance from a CSV file.

        Args:
            fp (Union[IO, str, Path]): The file path or file object to read from.
            time_index (int, optional): The index of the column representing time. Defaults to 0.

        Returns:
            TimeSeries: A `TimeSeries` object populated with the CSV data.
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
        Returns the time interval (timestep) between consecutive observations.

        Returns:
            float: The timestep between observations.

        Raises:
            IndexError: If `times` is empty or has fewer than two elements.
        """
        if len(self.times) < 2:
            raise IndexError("Cannot compute timestep with fewer than two time points.")
        return self.times[1] - self.times[0]

    @property
    def num_dims(self) -> int:
        """
        Returns the dimensionality of the time series.

        Returns:
            int: The number of dimensions in `dependent_variable`.
        """
        return self.dependent_variable.shape[1]

    def __eq__(self, other) -> bool:
        """
        Checks if two `TimeSeries` instances are equal.

        Returns:
            bool: `True` if both time series have identical timestamps and values.
        """
        return all(t1 == t2 for t1, t2 in zip(self.times, other.times)) and np.allclose(
            self.dependent_variable, other.dependent_variable
        )

    def __getitem__(self, key):
        """
        Retrieves a subset of the `TimeSeries`.

        Args:
            key (int or slice): The index or range of elements to retrieve.

        Returns:
            TimeSeries: A new `TimeSeries` containing the selected data.
        """
        return TimeSeries(self.dependent_variable[key], self.times[key])

    def __setitem__(self, key, value):
        """
        Sets values for a subset of the `TimeSeries`.

        Args:
            key (int or slice): The index or range of elements to modify.
            value (TimeSeries): A `TimeSeries` instance containing the new data.

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
            TimeSeries: The result of element-wise addition.

        Raises:
            ValueError: If the two time series do not have the same length or timestamps.
        """
        if len(self.times) != len(other.times):
            raise ValueError("Cannot add TimeSeries instances with different lengths.")
        if not np.all(self.times == other.times):
            raise ValueError("Cannot add TimeSeries instances with different timestamps.")

        return TimeSeries(
            dependent_variable=self.dependent_variable + other.dependent_variable,
            times=self.times
        )

    def __sub__(self, other):
        """
        Subtracts one `TimeSeries` instance from another.

        Returns:
            TimeSeries: The result of element-wise subtraction.

        Raises:
            ValueError: If the two time series do not have the same length or timestamps.
        """
        return self + (-1 * other)

    def __len__(self):
        """
        Returns the number of timesteps in the time series.

        Returns:
            int: The length of `times` (number of observations).
        """
        return len(self.times)
