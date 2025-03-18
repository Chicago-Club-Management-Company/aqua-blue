"""
This module provides simple utilities for processing TimeSeries instances.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from .time_series import TimeSeries


@dataclass
class Normalizer:

    """
    A utility class for normalizing and denormalizing TimeSeries instances.

    This class computes and stores the mean and standard deviation of the
    dependent variable during normalization. These statistics are later used
    to restore the original scale of the data when denormalizing.
    """
    
    means: np.typing.NDArray[np.floating] = field(init=False)
    """Mean values of the dependent variable, computed during normalization."""

    standard_deviations: np.typing.NDArray[np.floating] = field(init=False)

    """Standard deviation values of the dependent variable, computed during normalization."""

    def normalize(self, time_series: TimeSeries) -> TimeSeries:
        """
        Normalize a TimeSeries instance by adjusting its values to have zero mean and unit variance.

        Args:
            time_series (TimeSeries): The time series to be normalized.

        Returns:
            TimeSeries: A new TimeSeries instance with normalized values.
  
        Raises:
            ValueError: If the normalizer has already been used, since it is intended for one-time use.
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
        Denormalize a previously normalized TimeSeries instance, restoring it to its original scale.

        Args:
            time_series (TimeSeries): Time series to denormalize
        
        Returns:
            TimeSeries: The denormalized time series
        

        Raises:
            ValueError: If normalization has not been performed before calling this method.
        """
        
        if not hasattr(self, "means") or not hasattr(self, "standard_deviations"):
            raise ValueError("You can only denormalize after normalizing a time series.")
  
        arr = time_series.dependent_variable
        arr = arr * self.standard_deviations
        arr = arr + self.means
        
        return TimeSeries(dependent_variable=arr, times=time_series.times)


def make_sparse(
    weight_matrix: np.typing.NDArray[np.floating], 
    sparsity: float,
    generator: Optional[np.random.Generator] = None
) -> np.typing.NDArray[np.floating]:
    """
    Make a weight matrix sparse 
    
    Args:
    
        weight_matrix (np.typing.NDArray[np.floating]): Weight matrix to be made sparse

        sparsity (float): Extent of how sparse to make the weight matrix. Ranges from 0 to 1.

        generator (np.random.Generator): NumPy Generator to create random numbers
    """
    
    if not generator:
        generator = np.random.default_rng(seed=0)
    
    mask = generator.random(weight_matrix.shape) < sparsity 
    return weight_matrix * mask


def set_spectral(
    weight_matrix: np.typing.NDArray[np.floating], 
    spectral_radius: float
) -> np.typing.NDArray[np.floating]:
    
    """
    Set the spectral radius of the weight matrix 
    
    Args:
    
        weight_matrix (np.typing.NDArray[np.floating]): Weight matrix whose spectral radius is to be set

        spectral_radius (float):
            The largest absolute singular value of the weight matrix.
            Values less than 1.0 are recommended for tasks that require significant memory fading.
            Values between 1-1.5 are recommended for tasks that are memory dependent.
    """
    
    return spectral_radius / np.linalg.norm(weight_matrix, ord=2) * weight_matrix
