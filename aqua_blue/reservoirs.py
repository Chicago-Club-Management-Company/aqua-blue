"""
Module defining reservoirs
"""


from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

from .utilities import make_sparse, set_spectral

#: Type alias for activation function. Callable taking in a numpy array, and returning a numpy array of the same shape
ActivationFunction = Callable[[np.typing.NDArray[np.floating]], np.typing.NDArray[np.floating]]

# pdoc needs it to be a string, but type checker needs it to be a true alias, so replace if not type checking
if not TYPE_CHECKING:
    ActivationFunction = "ActivationFunction"


@dataclass
class Reservoir(ABC):
    
    """
    Abstract base reservoir class defining input state -> reservoir state
    """
    
    input_dimensionality: int
    """dimensionality of the input state"""
    
    reservoir_dimensionality: int
    """dimensionality of the reservoir state, equivalently the reservoir size"""
    
    res_state: np.typing.NDArray[np.floating] = field(init=False)
    """reservoir state, necessary property when performing training loop"""
    
    @abstractmethod
    def update_reservoir(self, input_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
        
        """
        Map from input state to reservoir state
        
        Args:
            input_state: input state to map to reservoir state
        """
        
        pass


@dataclass
class DynamicalReservoir(Reservoir):
    
    """
    Dynamical reservoir state, defined by the mapping:
    y_t = f(w_in @ x_t)
    where f is some nonlinear activation function
    """
    
    generator: Optional[np.random.Generator] = None
    """
    random generator for the class to use
    will be set to np.random.default_rng(seed=0) if not specified
    """
    
    w_in: Optional[np.typing.NDArray[np.floating]] = None
    """
    input linear mapping. must be shape (self.reservoir_dimensionality, self.input_dimensionality)
    if not defined at initialization, will be auto generated
    """
    w_res: Optional[np.typing.NDArray[np.floating]] = None
    """
    reservoir linear mapping. must be shape (self.reservoir_dimensionality, self.reservoir_dimensionality)
    if not defined at initialization, will be auto generated 
    """
    activation_function: ActivationFunction = np.tanh
    """
    activation function f. defaults to np.tanh
    """
    
    leaking_rate: float = 1
    """ 
    leaking rate for the reservoir state. Defaults to 1
    """
    
    sparsity: Optional[float]=None
    """
    sparsity of the reservoir weight matrix. (0, 1] 
    """
    
    spectral_radius: Optional[float]=None
    """
    spectral radius of reservoir weight matrix.
    Recommended values - [0.9, 1.2] 
    """
    def __post_init__(self):
        
        if self.generator is None:
            self.generator = np.random.default_rng(seed=0)
        
        if self.w_in is None:
            self.w_in = self.generator.uniform(
                low=-0.5,
                high=0.5,
                size=(self.reservoir_dimensionality, self.input_dimensionality)
            )
        if self.w_res is None: 
            self.w_res = self.generator.uniform(
                low=-0.5,
                high=0.5, 
                size=(self.reservoir_dimensionality, self.reservoir_dimensionality)
            )
        
        if self.spectral_radius:
            self.w_res = set_spectral(self.w_res, self.spectral_radius)
        
        if self.sparsity:
            self.w_res = make_sparse(self.w_res, self.sparsity, self.generator)
        
        self.res_state = np.zeros(self.reservoir_dimensionality)
    
    def update_reservoir(self, input_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
        
        """
        Map from input state to reservoir state via y_t = (1-alpha) * y_t-1 + alpha * f(w_in @ x_t + w_res @ y_t-1)
        
        Args:
            input_state: input state to map to reservoir state
        """
        
        assert isinstance(self.w_in, np.ndarray)
        assert isinstance(self.w_res, np.ndarray)
        self.res_state = (1.0 - self.leaking_rate) * self.res_state \
            + self.leaking_rate * self.activation_function(self.w_in @ input_state + self.w_res @ self.res_state)
        return self.res_state
