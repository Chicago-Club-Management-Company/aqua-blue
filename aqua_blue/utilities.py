"""
Module providing utility functions for processing `TimeSeries` instances.

This module includes the `Normalizer` class, which standardizes a `TimeSeries`
by normalizing it to have zero mean and unit variance. The class also allows
for denormalization, reverting the time series back to its original scale.

Classes:
    - Normalizer: A class for normalizing and denormalizing time series data.
"""

from dataclasses import dataclass, field

import numpy as np

from .time_series import TimeSeries


@dataclass
class Normalizer:
    """
    A utility class for normalizing and denormalizing `TimeSeries` instances.

    The `Normalizer` ensures that a given time series is scaled to zero mean
    and unit variance, improving numerical stability for machine learning models.

    Attributes:
        means (np.typing.NDArray[np.floating]):
            The mean values computed during normalization.
        standard_deviations (np.typing.NDArray[np.floating]):
            The standard deviation values computed during normalization.
    """

    means: np.typing.NDArray[np.floating] = field(init=False)
    """Mean values of the time series computed during normalization."""

    standard_deviations: np.typing.NDArray[np.floating] = field(init=False)
    """Standard deviation values of the time series computed during normalization."""

    def normalize(self, time_series: TimeSeries) -> TimeSeries:
        """
        Normalizes a given `TimeSeries` instance to have zero mean and unit variance.

        Args:
            time_series (TimeSeries):
                The time series to be normalized.

        Returns:
            TimeSeries: A new `TimeSeries` instance with normalized values.

        Raises:
            ValueError: If the `Normalizer` has already been used.
        """
        if hasattr(self, "means") or hasattr(self, "standard_deviations"):
            raise ValueError("You can only use the Normalizer once. Create a new instance to normalize again.")

        arr = time_series.dependent_variable
        self.means = arr.mean(axis=0)
        self.standard_deviations = arr.std(axis=0)

        arr = arr - self.means
        arr = arr / self.standard_deviations

        return TimeSeries(
            dependent_variable=arr,
            times=time_series.times
        )

    def denormalize(self, time_series: TimeSeries) -> TimeSeries:
        """
        Reverts a previously normalized `TimeSeries` back to its original scale.

        Args:
            time_series (TimeSeries):
                The time series to be denormalized.

        Returns:
            TimeSeries: A new `TimeSeries` instance with values restored to their original scale.

        Raises:
            ValueError: If `Normalizer` has not been used to normalize a time series first.
        """
        if not hasattr(self, "means") or not hasattr(self, "standard_deviations"):
            raise ValueError("You can only denormalize after normalizing a time series.")

        arr = time_series.dependent_variable
        arr = arr * self.standard_deviations
        arr = arr + self.means

        return TimeSeries(dependent_variable=arr, times=time_series.times)
