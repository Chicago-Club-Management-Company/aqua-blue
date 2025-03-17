"""
Module defining readout layers for reservoir computing models.

This module provides an abstract `Readout` class and a concrete `LinearReadout` class,
which map reservoir states to output states. The readout layer acts as a simple trainable
mapping that converts high-dimensional reservoir states into predictions.

Typically, readout layers are trained using ridge regression or similar optimization techniques
to minimize errors between predicted and actual outputs.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Readout(ABC):
    """
    Abstract base class for readout layers, defining how reservoir states map to outputs.

    A readout layer processes the high-dimensional reservoir state and transforms it into
    meaningful outputs. This abstract class defines a common interface for different types of
    readout layers.

    Attributes:
        coefficients (np.typing.NDArray[np.floating]):
            The trained weight matrix defining the readout transformation.
            This is set during training and used for making predictions.
    """

    coefficients: np.typing.NDArray[np.floating] = field(init=False)
    """The trained weight matrix defining the readout transformation. Set during training."""

    @abstractmethod
    def reservoir_to_output(
        self, reservoir_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Convert a reservoir state into an output value.

        This method is implemented in subclasses to define the specific mapping
        from reservoir states to outputs.

        Args:
            reservoir_state (np.typing.NDArray[np.floating]):
                The reservoir state vector to be mapped to an output.

        Returns:
            np.typing.NDArray[np.floating]: The corresponding output value(s).
        """
        pass


@dataclass
class LinearReadout(Readout):
    """
    Linear readout layer implementing a simple weighted transformation.

    This readout layer applies a linear transformation using a trained weight matrix
    to convert reservoir states into output values. The transformation follows the form:

        output = W @ reservoir_state

    where `W` is the coefficient matrix learned during training.

    Raises:
        ValueError: If the readout layer is used before training (i.e., before `coefficients` is set).
    """

    def reservoir_to_output(
        self, reservoir_state: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """
        Apply a linear transformation to map the reservoir state to an output.

        This function performs a simple matrix multiplication:

            output = self.coefficients @ reservoir_state

        Args:
            reservoir_state (np.typing.NDArray[np.floating]):
                The input reservoir state vector.

        Returns:
            np.typing.NDArray[np.floating]: The corresponding output value(s).

        Raises:
            ValueError: If the readout has not been trained before use.
        """

        if not hasattr(self, "coefficients"):
            raise ValueError("Readout must be trained before use.")

        return self.coefficients @ reservoir_state
