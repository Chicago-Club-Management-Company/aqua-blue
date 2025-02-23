from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class Readout(ABC):

    coefficients: np.typing.NDArray = field(init=False)

    @abstractmethod
    def reservoir_to_output(self, reservoir_state: np.typing.NDArray) -> np.typing.NDArray:

        pass


@dataclass
class LinearReadout(Readout):

    def reservoir_to_output(self, reservoir_state: np.typing.NDArray) -> np.typing.NDArray:

        if not hasattr(self, "coefficients"):
            raise ValueError("Need to train readout before using it")

        return self.coefficients @ reservoir_state
