"""
Module defining reservoir layers for Echo State Networks (ESNs).

This module provides an abstract `Reservoir` class and a concrete `DynamicalReservoir` class.
Reservoir layers transform input states into high-dimensional dynamic representations,
which are later processed by readout layers for prediction.

The reservoir layer acts as a dynamic system that captures temporal dependencies in time-series
data using a fixed, random transformation matrix. The readout layer learns from these transformed
states rather than the raw input data.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np


#: Type alias for activation function. This represents a function that takes a NumPy array
#: as input and returns an array of the same shape.
ActivationFunction = Callable[[np.typing.NDArray[np.floating]], np.typing.NDArray[np.floating]]

# pdoc needs it to be a string, but type checker needs it to be a true alias, so replace if not type checking
if not TYPE_CHECKING:
    ActivationFunction = "ActivationFunction"


@dataclass
class Reservoir(ABC):
    """
    Abstract base class for reservoir layers in an Echo State Network.

    The reservoir serves as a high-dimensional feature transformation, applying a fixed,
    randomly initialized mapping to input states. This class provides the structure
    for different reservoir types.

    Attributes:
        input_dimensionality (int):
            The dimensionality of the input state vector.
        reservoir_dimensionality (int):
            The number of reservoir neurons (i.e., the size of the high-dimensional representation).
        res_state (np.typing.NDArray[np.floating]):
            The internal state of the reservoir, updated dynamically as input sequences are processed.
    """

    input_dimensionality: int
    """Dimensionality of the input state vector."""

    reservoir_dimensionality: int
    """Number of neurons in the reservoir, determining the size of the feature space."""

    res_state: np.typing.NDArray[np.floating] = field(init=False)
    """Internal reservoir state, updated during training and prediction."""

    @abstractmethod
    def update_reservoir(
        self, input_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Update the reservoir state based on the input state.

        This function implements the reservoir update equation, mapping an input vector
        to a new high-dimensional representation.

        Args:
            input_state (np.typing.NDArray[np.floating]):
                The input state vector to be transformed.

        Returns:
            np.typing.NDArray[np.floating]: The updated reservoir state.
        """
        pass


@dataclass
class DynamicalReservoir(Reservoir):
    """
    Dynamical reservoir layer with nonlinear activation.

    This class implements a standard Echo State Network (ESN) reservoir, where the reservoir
    state is updated according to the equation:

        y_t = (1 - α) * y_t-1 + α * f(W_in @ x_t + W_res @ y_t-1)

    where:
        - `α` is the leaking rate,
        - `f` is the nonlinear activation function,
        - `W_in` is the input weight matrix,
        - `W_res` is the reservoir weight matrix.

    The reservoir matrices (`W_in` and `W_res`) can either be manually provided or randomly
    initialized during construction.

    Attributes:
        generator (Optional[np.random.Generator]):
            A NumPy random generator instance for reproducibility. If not specified,
            a default generator is created with a fixed seed.
        w_in (Optional[np.typing.NDArray[np.floating]]):
            The input weight matrix, mapping input states to reservoir neurons.
        w_res (Optional[np.typing.NDArray[np.floating]]):
            The recurrent weight matrix, capturing dynamics between reservoir neurons.
        activation_function (ActivationFunction):
            The activation function applied to the reservoir state (default: `np.tanh`).
        leaking_rate (float):
            The leaking rate parameter (default: 1), which controls how much past states
            influence the current state.
    """

    generator: Optional[np.random.Generator] = None
    """Random number generator for weight initialization. Defaults to a fixed seed."""

    w_in: Optional[np.typing.NDArray[np.floating]] = None
    """Input weight matrix (`W_in`), shape `(reservoir_dimensionality, input_dimensionality)`. 
    If not provided, it is randomly initialized.
    """

    w_res: Optional[np.typing.NDArray[np.floating]] = None
    """Reservoir weight matrix (`W_res`), shape `(reservoir_dimensionality, reservoir_dimensionality)`. 
    If not provided, it is randomly initialized.
    """

    activation_function: ActivationFunction = np.tanh
    """Nonlinear activation function applied to the reservoir state (default: `np.tanh`)."""

    leaking_rate: float = 1
    """Leaking rate (α) that determines how much past states contribute to the next state update."""

    def __post_init__(self):
        """
        Initialize the reservoir weights and internal state.

        If `w_in` or `w_res` are not provided, they are randomly generated.
        The spectral radius of `w_res` is scaled to 0.95 to ensure stability.
        """

        if self.generator is None:
            self.generator = np.random.default_rng(seed=0)

        if self.w_in is None:
            self.w_in = self.generator.uniform(
                low=-0.5,
                high=0.5,
                size=(self.reservoir_dimensionality, self.input_dimensionality),
            )

        if self.w_res is None:
            self.w_res = self.generator.uniform(
                low=-0.5,
                high=0.5,
                size=(self.reservoir_dimensionality, self.reservoir_dimensionality),
            )

        # Scale reservoir weights to ensure stability (spectral radius = 0.95)
        spectral_radius = np.linalg.norm(self.w_res, ord=2)
        self.w_res /= spectral_radius / 0.95

        # Initialize the reservoir state to zero
        self.res_state = np.zeros(self.reservoir_dimensionality)

    def update_reservoir(
        self, input_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Compute the next reservoir state using the dynamical equation.

        The reservoir state is updated as:

            y_t = (1 - α) * y_t-1 + α * f(W_in @ x_t + W_res @ y_t-1)

        where `α` is the leaking rate, `f` is the activation function,
        `W_in` maps input states, and `W_res` governs reservoir dynamics.

        Args:
            input_state (np.typing.NDArray[np.floating]):
                The input state vector at the current time step.

        Returns:
            np.typing.NDArray[np.floating]: The updated reservoir state.
        """

        assert isinstance(self.w_in, np.ndarray)
        assert isinstance(self.w_res, np.ndarray)

        # Apply the reservoir update equation
        self.res_state = (1 - self.leaking_rate) * self.res_state + self.leaking_rate * self.activation_function(
            self.w_in @ input_state + self.w_res @ self.res_state
        )

        return self.res_state
