"""
Module defining readout layers.

This module provides the abstract `Readout` class and its concrete implementation,
`LinearReadout`. Readout layers map the internal reservoir states of an Echo State
Network (ESN) to output values.

Classes:
    - Readout: Abstract base class defining the interface for readout layers.
    - LinearReadout: A linear mapping readout layer that transforms reservoir states
      into output values using learned coefficients.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class Readout(ABC):
    """
    Abstract base class for readout layers in Echo State Networks (ESNs).

    Readout layers transform the high-dimensional reservoir states into output
    predictions. The transformation is typically learned during training.

    Attributes:
        coefficients (np.ndarray):
            The learned weights for mapping reservoir states to output values.
            This is set after training.
    """

    coefficients: np.typing.NDArray[np.floating] = field(init=False)
    """The learned weight matrix for the readout layer, initialized during training."""

    @abstractmethod
    def train(
        self,
        independent_variables: np.typing.NDArray[np.floating],
        dependent_variables: np.typing.NDArray[np.floating]
    ):
        """
        Trains the readout layer by learning the mapping from reservoir states to output values.

        This method takes independent input variables (reservoir states) and corresponding
        dependent variables (target outputs) to compute the optimal readout weights.

        Args:
            independent_variables (np.ndarray):
                The reservoir state matrix used as input for training.
            dependent_variables (np.ndarray):
                The expected output values corresponding to the input states.
        """

        pass

    @abstractmethod
    def reservoir_to_output(self, reservoir_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:

        """
        Maps a given reservoir state to an output value.

        Args:
            reservoir_state (np.ndarray):
                The current state of the reservoir.

        Returns:
            np.ndarray: The predicted output corresponding to the given reservoir state.
        """

        pass


@dataclass
class LinearReadout(Readout):
    """
    A linear readout layer that applies a learned linear transformation to reservoir states.

    This readout layer learns a set of coefficients during training and applies a
    simple linear mapping to transform reservoir states into output predictions.

    Attributes:
        rcond (float):
            A regularization parameter used in the pseudo-inverse calculation to
            prevent numerical instability in the least squares solution.
    """

    rcond: float = 1.0e-10
    """
    Regularization parameter for pseudo-inverse computation.

    This controls the minimum singular value considered for the pseudo-inverse 
    computation. A lower value ensures more stable training.
    """

    def train(
        self,
        independent_variables: np.typing.NDArray[np.floating],
        dependent_variables: np.typing.NDArray[np.floating]
    ):
        r"""
        Trains the linear readout layer by solving the least-squares optimization problem.

        The training process determines the optimal readout coefficients $W^*$ by solving the optimization problem
        below:

        $$W^* = \lim_{\lambda\to 0^+} \arg\min_W \| XW - Y\|_F^2 + \lambda \|W\|_F^2$$

        where $X$ is the matrix of reservoir states (independent variables)
        and $Y$ is the matrix of target output values (dependent variables).

        Args:
            independent_variables (np.ndarray):
                The reservoir state matrix used for training.
            dependent_variables (np.ndarray):
                The target output values corresponding to the reservoir states.
        """
        coeff = np.linalg.pinv(independent_variables, rcond=self.rcond) @ dependent_variables
        self.coefficients = coeff.T

    def reservoir_to_output(self, reservoir_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
        """
        Computes the output from a given reservoir state using a learned linear mapping.

        This method applies the learned weight matrix (`self.coefficients`) to map
        the reservoir state to an output value.

        Args:
            reservoir_state (np.ndarray):
                The reservoir state to be mapped to an output value.

        Returns:
            np.ndarray: The predicted output value.

        Raises:
            ValueError: If the readout layer has not been trained (i.e., coefficients are not set).
        """
        if not hasattr(self, "coefficients"):
            raise ValueError("Need to train readout before using it")

        return self.coefficients @ reservoir_state
