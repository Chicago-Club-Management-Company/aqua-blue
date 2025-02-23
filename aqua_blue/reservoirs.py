from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


@dataclass
class Reservoir(ABC):

    input_dimensionality: int
    reservoir_dimensionality: int

    @abstractmethod
    def input_to_reservoir(self, input_state: np.typing.NDArray) -> np.typing.NDArray:

        pass


@dataclass
class DynamicalReservoir(Reservoir):

    w_in: Optional[np.typing.NDArray] = None
    generator: Optional[np.random.Generator] = None
    activation_function: Callable[[np.typing.NDArray], np.typing.NDArray] = np.tanh

    def __post_init__(self):

        if self.generator is None:
            self.generator = np.random.default_rng(seed=0)

        if self.w_in is None:
            self.w_in = self.generator.uniform(
                low=-0.5,
                high=0.5,
                size=(self.reservoir_dimensionality, self.input_dimensionality)
            )

    def input_to_reservoir(self, input_state: np.typing.NDArray) -> np.typing.NDArray:

        return self.activation_function(self.w_in @ input_state)
