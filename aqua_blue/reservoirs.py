"""
Module defining reservoir computing structures.

Reservoir computing is a type of recurrent neural network (RNN) architecture where a fixed,
randomly initialized reservoir transforms input signals into high-dimensional states.
A trainable readout layer then maps these states to the desired outputs.

This module provides:
- `Reservoir`: An abstract base class for defining different types of reservoirs.
- `DynamicalReservoir`: A concrete implementation that uses a dynamically evolving state
  governed by an activation function.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

#: Type alias for activation functions.
#: An activation function takes in a numpy array and returns a transformed numpy array.
ActivationFunction = Callable[[np.typing.NDArray[np.floating]], np.typing.NDArray[np.floating]]

# pdoc requires ActivationFunction to be a string, but the type checker requires it as a true alias.
if not TYPE_CHECKING:
    ActivationFunction = "ActivationFunction"


@dataclass
class Reservoir(ABC):
    """
    Abstract base class defining a reservoir.

    A reservoir transforms an input state into a high-dimensional hidden representation.
    This transformation is typically performed using a sparse, randomly initialized
    weight matrix. The reservoir states are then used for further learning.

    Attributes:
        input_dimensionality (int):
            The number of features in the input data.
        reservoir_dimensionality (int):
            The number of neurons (nodes) in the reservoir.
        res_state (np.typing.NDArray[np.floating]):
            The current reservoir state. It is updated dynamically as new inputs arrive.
    """

    input_dimensionality: int
    """Dimensionality of the input state."""

    reservoir_dimensionality: int
    """Number of neurons in the reservoir (reservoir size)."""

    res_state: np.typing.NDArray[np.floating] = field(init=False)
    """Current reservoir state, updated dynamically during training and inference."""

    @abstractmethod
    def update_reservoir(
        self, input_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Updates the reservoir state based on the provided input.

        This method must be implemented by subclasses.

        Args:
            input_state (np.typing.NDArray[np.floating]):
                The input vector that will be mapped to a new reservoir state.

        Returns:
            np.typing.NDArray[np.floating]:
                The updated reservoir state after transformation.
        """
        pass


@dataclass
class DynamicalReservoir(Reservoir):
    """
    A dynamical reservoir that evolves over time based on an activation function.

    The state of this reservoir is updated using the equation:

        y_t = (1 - alpha) * y_(t-1) + alpha * f(W_in @ x_t + W_res @ y_(t-1))

    where:
    - `alpha` is the leaking rate (controls the speed of state update),
    - `f` is the nonlinear activation function (default: `tanh`),
    - `W_in` is the input weight matrix,
    - `W_res` is the reservoir weight matrix,
    - `y_t` is the current reservoir state.

    Attributes:
        generator (Optional[np.random.Generator]):
            A random number generator for reproducibility. Defaults to `np.random.default_rng(seed=0)`.
        w_in (Optional[np.typing.NDArray[np.floating]]):
            The input weight matrix (`W_in`). If not provided, it is generated randomly.
        w_res (Optional[np.typing.NDArray[np.floating]]):
            The recurrent weight matrix (`W_res`). If not provided, it is generated randomly.
        activation_function (ActivationFunction):
            The activation function applied to the transformed state (default: `np.tanh`).
        leaking_rate (float):
            The rate at which the reservoir state updates (default: `1.0`).
    """

    generator: Optional[np.random.Generator] = None
    """Random number generator for reproducibility. Defaults to `np.random.default_rng(seed=0)` if not specified."""

    w_in: Optional[np.typing.NDArray[np.floating]] = None
    """Input weight matrix (`W_in`). Shape: `(reservoir_dimensionality, input_dimensionality)`. Auto-generated if `None`."""

    w_res: Optional[np.typing.NDArray[np.floating]] = None
    """Reservoir weight matrix (`W_res`). Shape: `(reservoir_dimensionality, reservoir_dimensionality)`. Auto-generated if `None`."""

    activation_function: ActivationFunction = np.tanh
    """Activation function applied to the transformed state. Defaults to `np.tanh`."""

    leaking_rate: float = 1.0
    """Leaking rate for reservoir state updates. Controls how quickly the reservoir state adapts (default: `1.0`)."""

    def __post_init__(self):
        """
        Initializes the reservoir's weight matrices and state.

        - If `w_in` is not provided, it is randomly initialized.
        - If `w_res` is not provided, it is randomly initialized and scaled to have a spectral radius of `0.95`.
        - The initial reservoir state is set to zero.
        """

        # Set up the random generator if not provided
        if self.generator is None:
            self.generator = np.random.default_rng(seed=0)

        # Initialize input weight matrix
        if self.w_in is None:
            self.w_in = self.generator.uniform(
                low=-0.5, high=0.5,
                size=(self.reservoir_dimensionality, self.input_dimensionality)
            )

        # Initialize recurrent weight matrix
        if self.w_res is None:
            self.w_res = self.generator.uniform(
                low=-0.5, high=0.5,
                size=(self.reservoir_dimensionality, self.reservoir_dimensionality)
            )

        # Scale W_res to ensure stability (spectral radius scaling)
        spectral_radius = np.linalg.norm(self.w_res, ord=2)
        self.w_res /= (spectral_radius / 0.95)

        # Initialize reservoir state as zero
        self.res_state = np.zeros(self.reservoir_dimensionality)

    def update_reservoir(
        self, input_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Updates the reservoir state using the dynamical equation:

            y_t = (1 - alpha) * y_(t-1) + alpha * f(W_in @ x_t + W_res @ y_(t-1))

        where:
        - `alpha` is the leaking rate,
        - `f` is the activation function (e.g., `tanh`),
        - `W_in` and `W_res` are the input and recurrent weight matrices.

        Args:
            input_state (np.typing.NDArray[np.floating]):
                The input vector at the current timestep.

        Returns:
            np.typing.NDArray[np.floating]:
                The updated reservoir state.

        Raises:
            AssertionError: If `w_in` or `w_res` are not initialized as numpy arrays.
        """

        assert isinstance(self.w_in, np.ndarray), "w_in must be initialized as a NumPy array."
        assert isinstance(self.w_res, np.ndarray), "w_res must be initialized as a NumPy array."

        # Compute the new reservoir state
        self.res_state = (
            (1 - self.leaking_rate) * self.res_state
            + self.leaking_rate * self.activation_function(self.w_in @ input_state + self.w_res @ self.res_state)
        )

        return self.res_state
