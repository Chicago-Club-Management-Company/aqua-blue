"""
Module defining readout layers for mapping reservoir states to output values.

This module provides an abstract base class `Readout` for defining readout layers,
as well as a concrete implementation `LinearReadout` for performing linear mappings
from reservoir states to output states.

Readouts are an essential part of Echo State Networks (ESNs) and other reservoir computing
models, where they process the high-dimensional reservoir state and produce the final predictions.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Readout(ABC):
    """
    Abstract base class for a readout layer in a reservoir computing model.

    The readout layer is responsible for mapping the high-dimensional
    reservoir state to the desired output space. Different implementations
    of this class can define different mapping strategies (e.g., linear, nonlinear).

    Attributes:
        coefficients (np.typing.NDArray[np.floating]):
            The transformation matrix (or coefficients) that defines the mapping
            from reservoir states to output states. These coefficients are learned
            during training.
    """

    coefficients: np.typing.NDArray[np.floating] = field(init=False)
    """Transformation matrix (learned during training) that maps reservoir states to output values."""

    @abstractmethod
    def reservoir_to_output(
        self, reservoir_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Maps a given reservoir state to an output value.

        This method is abstract and must be implemented by subclasses.

        Args:
            reservoir_state (np.typing.NDArray[np.floating]):
                The current reservoir state to be transformed into an output.

        Returns:
            np.typing.NDArray[np.floating]: The computed output value(s).
        """
        pass


@dataclass
class LinearReadout(Readout):
    """
    A linear readout layer that applies a weighted transformation to the reservoir state.

    This class implements a simple linear mapping from the reservoir state
    to the output state using matrix multiplication. It follows the equation:

        output = coefficients @ reservoir_state

    This is equivalent to applying a linear regression model to the reservoir outputs.

    Example:
        ```python
        readout = LinearReadout()
        readout.coefficients = np.random.rand(1, 100)  # Example shape
        output = readout.reservoir_to_output(reservoir_state)
        ```

    Raises:
        ValueError: If the readout layer is used before being trained (i.e.,
        if `self.coefficients` is not set).
    """

    def reservoir_to_output(
        self, reservoir_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Computes the output by applying a linear transformation to the reservoir state.

        This method multiplies the learned coefficients with the given
        reservoir state to produce the output.

        Args:
            reservoir_state (np.typing.NDArray[np.floating]):
                The current state of the reservoir.

        Returns:
            np.typing.NDArray[np.floating]: The predicted output.

        Raises:
            ValueError: If `coefficients` have not been set (i.e., the model has not been trained).
        """

        if not hasattr(self, "coefficients"):
            raise ValueError("Readout layer must be trained before use. Call the train method first.")

        return self.coefficients @ reservoir_state
