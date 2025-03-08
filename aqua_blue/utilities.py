"""
This module provides simple utilities for processing TimeSeries instances
"""


from dataclasses import dataclass, field

import numpy as np

from .time_series import TimeSeries


@dataclass
class Normalizer:

    """
    Normalizer class to normalize a time series to have zero mean and unit variance
    """

    means: np.typing.NDArray[np.floating] = field(init=False)
    standard_deviations: np.typing.NDArray[np.floating] = field(init=False)

    def normalize(self, time_series: TimeSeries) -> TimeSeries:

        """
        Normalize the given TimeSeries instance

        Args:
            time_series (TimeSeries): Time series to normalize

        Returns:
            TimeSeries: The normalized time series

        Raises:
            ValueError: If the normalizer has already been used
        """

        if hasattr(self, "means") or hasattr(self, "standard_deviations"):
            raise ValueError("You can only use the Normalizer once. Create a new instance to normalize again")

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
        Denormalize the given TimeSeries instance. Means and standard deviations are grabbed
        from a previous normalization
        
        Args:
            time_series (TimeSeries): Time series to denormalize

        Returns:
            TimeSeries: The denormalized time series

        Raises:
            ValueError: If the normalizer has not yet been used
        """

        if not hasattr(self, "means") or not hasattr(self, "standard_deviations"):
            raise ValueError("You can only denormalize after normalizing a time series")

        arr = time_series.dependent_variable
        arr = arr * self.standard_deviations
        arr = arr + self.means

        return TimeSeries(dependent_variable=arr, times=time_series.times)
