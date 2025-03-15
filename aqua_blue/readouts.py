"""
Module defining readout layers
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class Readout(ABC):
    
    """
    Abstract readout layer, defining reservoir state -> output state mapping
    """

    coefficients: np.typing.NDArray[np.floating] = field(init=False)
    """coefficients defining the readout layer, which will later be fitted"""

    @abstractmethod
    def train(
        self,
        independent_variables: np.typing.NDArray[np.floating],
        dependent_variables: np.typing.NDArray[np.floating]
    ):

        """
        Train readout layer, sets the coefficients attribute

        Args:
            independent_variables: Independent (or input) data for training
            dependent_variables: Dependent (or response) data for training
        """

    @abstractmethod
    def reservoir_to_output(self, reservoir_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:

        """
        Map from reservoir state to output

        Args:
            reservoir_state: reservoir state to map to output state
        """

        pass


@dataclass
class LinearReadout(Readout):

    """
    Linear readout layer, reservoir_state -> W @ reservoir_state
    """

    rcond: float = 1.0e-10
    """
    condition number, or minimum singular value.
    should be as small as possible while still preserving numerical stability
    """

    def train(
        self,
        independent_variables: np.typing.NDArray[np.floating],
        dependent_variables: np.typing.NDArray[np.floating]
    ):
        r"""
        Train linear layer, setting coefficients. Solves the following optimization problem
        $W^* = \lim_{\lambda\to 0^+} argmin_W \| XW - Y\|_F^2 + \lambda \|W\|_F^2$

        Args:
            independent_variables: Independent (or input) data for training
            dependent_variables: Dependent (or response) data for training
        """

        self.coefficients = np.linalg.pinv(independent_variables, rcond=self.rcond) @ dependent_variables

    def reservoir_to_output(self, reservoir_state: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:

        """
        Linearly from reservoir state to output, i.e. just self.coefficients @ reservoir_state

        Args:
            reservoir_state (np.typing.NDArray):
                reservoir state to map to output state
        """

        if not hasattr(self, "coefficients"):
            raise ValueError("Need to train readout before using it")

        return self.coefficients @ reservoir_state
