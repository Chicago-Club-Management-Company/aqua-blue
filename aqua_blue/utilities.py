"""
This module provides utility functions for processing and transforming TimeSeries instances.
"""

from dataclasses import dataclass, field

import numpy as np

from .time_series import TimeSeries


@dataclass
class Normalizer:
    """
    A class to normalize and denormalize time series data.

    This normalizer transforms a TimeSeries object so that its values have
    zero mean and unit variance (standardization). It also supports
    reversing the transformation (denormalization).

    Attributes:
        means (np.typing.NDArray[np.floating]):
            The mean of each feature in the time series, computed during normalization.
        standard_deviations (np.typing.NDArray[np.floating]):
            The standard deviation of each feature in the time series, computed during normalization.
    """

    means: np.typing.NDArray[np.floating] = field(init=False)
    """Array of mean values for each feature in the time series, computed during normalization."""

    standard_deviations: np.typing.NDArray[np.floating] = field(init=False)
    """Array of standard deviations for each feature in the time series, computed during normalization."""

    def normalize(self, time_series: TimeSeries) -> TimeSeries:
        """
        Normalize the given TimeSeries instance to have zero mean and unit variance.

        Args:
            time_series (TimeSeries): The time series to normalize.

        Returns:
            TimeSeries: A new TimeSeries object with normalized values.

        Raises:
            ValueError: If the normalizer has already been used to transform a time series.
        """

        if hasattr(self, "means") or hasattr(self, "standard_deviations"):
            raise ValueError(
                "You can only use the Normalizer once. "
                "Create a new instance to normalize again."
            )

        arr = time_series.dependent_variable
        self.means = arr.mean(axis=0)
        self.standard_deviations = arr.std(axis=0)

        arr = (arr - self.means) / self.standard_deviations

        return TimeSeries(
            dependent_variable=arr,
            times=time_series.times
        )

    def denormalize(self, time_series: TimeSeries) -> TimeSeries:
        """
        Reverse the normalization process, restoring the original scale of the time series.

        Args:
            time_series (TimeSeries): The normalized time series to denormalize.

        Returns:
            TimeSeries: A new TimeSeries object with values restored to their original scale.

        Raises:
            ValueError: If normalization has not been performed before calling this method.
        """

        if not hasattr(self, "means") or not hasattr(self, "standard_deviations"):
            raise ValueError(
                "You can only denormalize after normalizing a time series."
            )

        arr = time_series.dependent_variable
        arr = arr * self.standard_deviations + self.means

        return TimeSeries(dependent_variable=arr, times=time_series.times)
