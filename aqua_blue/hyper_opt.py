"""
Module providing hyperparameter optimization functionality. 

This module allows for hyperparameter optimization of aqua-blue models by wrapping around hyperopt's optimization algorithms. 
"""

import numpy as np

from typing import Optional, Callable, TypedDict, Literal, Union
from numpy.typing import NDArray

from dataclasses import dataclass, field
from enum import Enum
import warnings

from .time_series import TimeSeries
from .models import Model
from .reservoirs import DynamicalReservoir
from .readouts import LinearReadout
from .utilities import Normalizer

try: 
   import hyperopt
except ModuleNotFoundError:
    raise ModuleNotFoundError('Build dependencies for hyperparameter optimization not found! Make sure to install it using pip install .[hyper]')

# Hyperparameter Optimization Functionality 

HyperoptStatus = Literal["ok", "fail"]

# Class Definitions 

# Enum Class for optimization algorithms
class Algo(Enum):
    TREE_PARZEN_ESTIMATOR = hyperopt.tpe.suggest
    GRID_SEARCH = hyperopt.rand.suggest
    SIMULATED_ANNEALING = hyperopt.anneal.suggest

@dataclass
class ModelParams: 
    time_series: TimeSeries
    input_dimensionality: int
    reservoir_dimensionality: int
    horizon: int
    actual_future: NDArray[np.floating]
    w_in: Optional[NDArray[np.floating]] = None
    w_res: Optional[NDArray[np.floating]] = None
    
    """ 
    Dataclass to store model parameters

    Args: 
        time_series (TimeSeries): The unnormalized time series of interest 
        input_dimensionality (int): The dimension of the input 
        reservoir_dimensionality (int): The dimension of the reservoir
        horizon (int): The number of steps to forecast into the future
        actual_future (NDArray[np.floating]): The actual future used to calculate performance metrics
        w_in (NDArray[np.floating]): Explicit input weight matrix. Defaults to None
        w_res (NDArray[np.floating]): Explicit reservoir weight matrix. Defaults to None
    """
class HyperParams(TypedDict): 
    """ 
    TypedDict form of parameter objects

    Hyperopt objective functions take dicts
    """
    spectral_radius: hyperopt.pyll.base.Apply
    leaking_rate: hyperopt.pyll.base.Apply
    sparsity: hyperopt.pyll.base.Apply
    rcond: hyperopt.pyll.base.Apply

class Output(TypedDict, total=False): 
    """ 
    TypedDict form of output objects 

    Hyperopt objective functions output dicts
    """
    loss: float
    status: HyperoptStatus

ObjectiveLike = Union[Callable[[HyperParams], Output], Callable[[ModelParams], Callable[[HyperParams], Output]]]

# Default Functionality 
default_space: HyperParams = {
    'spectral_radius': hyperopt.hp.uniform('spectral_radius', 0.1, 1.5),
    'leaking_rate': hyperopt.hp.uniform('leaking_rate', 0.0, 1.0),
    'sparsity': hyperopt.hp.uniform('sparsity', 0.0, 1.0),
    'rcond': hyperopt.hp.uniform('rcond', 1e-8, 1e-2)
}

# Define a factory to input the model parameters 
def default_loss(mp: ModelParams) -> Callable[[HyperParams], Output]:
    def inner(p : HyperParams) -> Output: 
        spectral_radius, leaking_rate, sparsity, rcond = p['spectral_radius'], p['leaking_rate'], p['sparsity'], p['rcond']
        
        normalizer = Normalizer()

        model = Model( 
            reservoir=DynamicalReservoir(
                reservoir_dimensionality = mp.reservoir_dimensionality, 
                input_dimensionality = mp.input_dimensionality,
                w_res = mp.w_res, 
                w_in = mp.w_in,
                spectral_radius = spectral_radius,
                leaking_rate = leaking_rate, 
                sparsity = sparsity
            ),
            readout = LinearReadout(rcond = rcond)
        )

        normalized_time_series = normalizer.normalize(mp.time_series)
        
        try: 
            model.train(normalized_time_series)
        except np.linalg.LinAlgError:
            warnings.warn('SVD Error in Training', RuntimeWarning)
            return { 
                'loss': 1000, 
                'status': hyperopt.STATUS_FAIL
            }
        
        try: 
            prediction = model.predict(horizon = mp.horizon)
        except np.linalg.LinAlgError:
            warnings.warn('SVD Error in Training', RuntimeWarning)
            return { 
                'loss': 1000, 
                'status': hyperopt.STATUS_FAIL
            }
        
        prediction = normalizer.denormalize(prediction)
        
        if prediction.dependent_variable.shape != mp.actual_future.shape:
            raise ValueError('Dimension mismatch between actual future and prediction')
        
        loss = np.sqrt(np.mean((mp.actual_future - prediction.dependent_variable) ** 2))

        out : Output = { 
            'loss': loss, 
            'status': hyperopt.STATUS_OK
        }
        
        return out 
    
    return inner

@dataclass
class Optimizer: 
    max_evals: int
    fn: ObjectiveLike = default_loss
    space: HyperParams = field(default_factory = lambda: default_space)
    algo: Algo = Algo.GRID_SEARCH
    trials: Optional[hyperopt.Trials] = None
    
    """ 
    Dataclass to store optimizer parameters 
    
    Args: 
        max_evals (int): The number of evaluations to run the optimizer for
        fn (ObjectiveLike): The objective function to be minimized 
        space (HyperParams): An object containing the hyperparameter names and their search spaces. Defaults to default_space
        algo (Algo): The optimization algorithm used. Defaults to GRID_SEARCH
        trials (hyperopt.Trials): A trials object to store output data
    """

    def optimize(self) -> HyperParams:
        
        warnings.warn("This feature is currently experimental and may be unstable or subject to change. Feedback is welcome to help improve future versions.", UserWarning)

        return hyperopt.fmin(
            fn=self.fn, 
            space=self.space,
            algo=self.algo, 
            max_evals=self.max_evals,
            trials=self.trials
        )
