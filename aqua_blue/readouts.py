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

    coefficients: np.typing.NDArray = field(init=False)
    """coefficients defining the readout layer, which will later be fitted"""

    @abstractmethod
    def reservoir_to_output(self, reservoir_state: np.typing.NDArray) -> np.typing.NDArray:

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

    def reservoir_to_output(self, reservoir_state: np.typing.NDArray) -> np.typing.NDArray:

        """
        Linearly from reservoir state to output, i.e. just self.coefficients @ reservoir_state

        Args:
            reservoir_state (np.typing.NDArray):
                reservoir state to map to output state
        """

        if not hasattr(self, "coefficients"):
            raise ValueError("Need to train readout before using it")

        return self.coefficients @ reservoir_state
