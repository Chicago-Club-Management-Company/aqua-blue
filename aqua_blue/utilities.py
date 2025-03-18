"""
This module provides simple utilities for processing TimeSeries instances
"""


from dataclasses import dataclass, field
from typing import Optional

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

def make_sparse(
    weight_matrix: np.typing.NDArray[np.floating], 
    sparsity: float,
    generator: Optional[np.random.Generator]=None
) -> np.typing.NDArray[np.floating]:
    """
    Make a weight matrix sparse 
    
    Params: 
    
    weight_matrix: np.typing.NDArray[np.floating]
        Weight matrix to be made sparse
    
    sparsity: float 
        Extent of how sparse to make the weight matrix. Ranges from 0 to 1.
    
    generator: np.random.Generator
        NumPy Generator to create random numbers
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
    
    Params: 
    
    weight_matrix: np.typing.NDArray[np.floating]
        Weight matrix whose spectral radius is to be set
    
    spectral_radius: float
        p(W) The largest absolute eigenvalue of the weight matrix. 
        Values less than 1.0 are recommended for tasks that require significant memory fading. 
        Values between 1-1.5 are recommended for takes that are memory dependent. 
    """
    
    return spectral_radius/np.linalg.norm(weight_matrix, ord=2) * weight_matrix
