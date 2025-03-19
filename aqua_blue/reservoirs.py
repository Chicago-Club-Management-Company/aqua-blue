"""
Module defining reservoirs.

This module contains the base `Reservoir` class and its concrete implementation,
`DynamicalReservoir`. Reservoirs serve as dynamic memory structures in Echo State
Networks (ESNs) by transforming input signals into high-dimensional representations.

Classes:
    - Reservoir: Abstract base class defining the structure of a reservoir.
    - DynamicalReservoir: A specific implementation of a reservoir with tunable
      dynamics and activation functions.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np

from .utilities import make_sparse, set_spectral


ActivationFunction = Callable[[np.typing.NDArray[np.floating]], np.typing.NDArray[np.floating]]
"""activation function, taking in a numpy array and returning a numpy array of the same shape"""


@dataclass
class Reservoir(ABC):
    
    """
    Abstract base class defining a reservoir in an Echo State Network (ESN).

    Reservoirs are responsible for transforming input signals into high-dimensional
    representations, which are then used by the readout layer for predictions.

    Attributes:
        input_dimensionality (int):
            The number of input features.
        reservoir_dimensionality (int):
            The number of reservoir neurons (i.e., the size of the reservoir).
        res_state (np.ndarray):
            The current state of the reservoir, which is updated at each time step.
    """
    
    input_dimensionality: int
    """Dimensionality of the input state."""

    reservoir_dimensionality: int
    """Dimensionality of the reservoir state, equivalently the reservoir size."""

    res_state: np.typing.NDArray[np.floating] = field(init=False)
    """Reservoir state, necessary property when performing training loop."""

    @abstractmethod
    def update_reservoir(self, input_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:

        """
        Updates the reservoir state given an input state.

        This method defines the transformation applied to an input vector when passed
        through the reservoir.
        """

        pass


@dataclass
class DynamicalReservoir(Reservoir):
    
    r"""
    A dynamical reservoir with tunable properties.

    This reservoir is defined by the equation:

    $$y_t = (1 - \alpha) y_{t-1} + \alpha f(W_\text{in} x_t + W_\text{res} y_{t-1})$$

    where $x_t$ is the input at time step $t$, $y_t$ is the reservoir state at time $t$,
    $W_\text{in}$ is the input weight matrix, $W_\text{res}$ is the reservoir weight matrix,
    $\alpha$ (leaking_rate) controls how much of the previous state influences the next state,
    and $f$ is a nonlinear activation function.

    Attributes:
        generator (Optional[np.random.Generator]):
            Random number generator for weight initialization.
        w_in (Optional[np.ndarray]):
            Input weight matrix of shape `(reservoir_dimensionality, input_dimensionality)`.
            Auto-generated if not provided.
        w_res (Optional[np.ndarray]):
            Reservoir weight matrix of shape `(reservoir_dimensionality, reservoir_dimensionality)`.
            Auto-generated if not provided.
        activation_function (ActivationFunction):
            Activation function applied to the reservoir state. Defaults to `np.tanh`.
        leaking_rate (float):
            Leaking rate that controls the contribution of the previous state.
    """
    
    generator: Optional[np.random.Generator] = None
    """
    Random generator for initializing weights.
    Defaults to `np.random.default_rng(seed=0)` if not specified.
    """
    
    w_in: Optional[np.typing.NDArray[np.floating]] = None
    """
    Input weight matrix.
    Must have shape `(reservoir_dimensionality, input_dimensionality)`.
    If not provided, it is auto-generated with values in `[-0.5, 0.5]`.
    """

    w_res: Optional[np.typing.NDArray[np.floating]] = None
    """
    Reservoir weight matrix.
    Must have shape `(reservoir_dimensionality, reservoir_dimensionality)`.
    If not provided, it is auto-generated and normalized to have a spectral radius of 0.95.
    """

    activation_function: ActivationFunction = np.tanh
    """
    Nonlinear activation function applied to the reservoir state.
    Defaults to `np.tanh`, but can be replaced with other functions like ReLU.
    """

    leaking_rate: float = 1.0
    r""" 
    Leaking rate (\(\alpha\)) that controls how much of the previous state contributes to the next.
    Defaults to `1.0`, meaning the state is fully updated at each time step.
    """
    
    sparsity: Optional[float] = None
    """
    sparsity of the reservoir weight matrix. (0, 1] 
    """
    
    spectral_radius: Optional[float] = 0.95
    """
    spectral radius of reservoir weight matrix.
    Recommended values - [0.9, 1.2] 
    """

    def __post_init__(self):

        """
        Initializes the reservoir by generating input and reservoir weight matrices (if not provided).

        Ensures that:
        - The reservoir weight matrix has a spectral radius of approximately 0.95.
        - The reservoir state is initialized to zero.
        """
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

        if self.sparsity:
            self.w_res = make_sparse(self.w_res, self.sparsity, self.generator)

        self.w_res = set_spectral(self.w_res, self.spectral_radius)
        
        self.res_state = np.zeros(self.reservoir_dimensionality)

    def update_reservoir(self, input_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:

        r"""
        Updates the reservoir state given an input.

        This method applies the state update equation:

        $$y_t = (1 - \alpha) y_{t-1} + \alpha f(W_\text{in} x_t + W_\text{res} y_{t-1})$$

        Args:
            input_state (np.ndarray):
                The input state vector.

        Returns:
            np.ndarray: The updated reservoir state.
        """

        assert isinstance(self.w_in, np.ndarray)
        assert isinstance(self.w_res, np.ndarray)

        self.res_state = (1.0 - self.leaking_rate) * self.res_state + self.leaking_rate * self.activation_function(
            self.w_in @ input_state + self.w_res @ self.res_state
        )
        return self.res_state
