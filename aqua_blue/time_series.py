"""
Module defining the TimeSeries object
"""

from typing import IO, Union
from numpy.typing import NDArray
from pathlib import Path
import warnings

from dataclasses import dataclass
import numpy as np


from zoneinfo import ZoneInfo
import datetime

from .tz_array import TZArray, fromNDArray

class ShapeChangedWarning(Warning):
    
    """
    warn the user that TimeSeries.__post_init__ changed the shape of the dependent variable
    """


@dataclass
class TimeSeries:

    """
    TimeSeries class defining a time series
    """

    dependent_variable: np.typing.NDArray[np.floating]
    times: Union[NDArray[np.floating], TZArray]

    def __post_init__(self):
        # If List[float] | List[int] | List[datetime] are passed, convert to NDArray and TZArray respectively 
        if(isinstance(self.times, list)):
            if(isinstance(self.times[0], (float, int))):       
                self.times = np.array(self.times)
            elif (isinstance(self.times[0], datetime.datetime)): 
                self.times = TZArray(self.times)
            else:
                raise NotImplementedError

        # If NDArray[np.datetime64] is passed, then convert to TZArray[ZoneInfo = UTC]
        if(not isinstance(self.times, TZArray) and isinstance(self.times, np.ndarray) and np.issubdtype(self.times.dtype, np.datetime64)): 
            self.times = fromNDArray(self.times)
        
        timesteps = np.array(np.diff(self.times))
        
        if not np.isclose(np.std(timesteps.astype(float)), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        if np.isclose(np.mean(timesteps.astype(float)), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")
        
        if len(self.dependent_variable.shape) == 1:
            num_steps = len(self.dependent_variable)
            self.dependent_variable = self.dependent_variable.reshape(num_steps, 1)
            warnings.warn(
                f"TimeSeries.dependent_variable should have shape (number of steps, dimensionality). "
                f"The shape has been changed from {(num_steps,)} to {self.dependent_variable.shape}"
            )

    def save(self, fp: Union[IO, str, Path], header: str = "", delimiter=","):

        """
        Method to save a time series

        Args:
            fp (Union[IO, str, Path]):
                The file-like object, path name, or Path in which to save the TimeSeries instance

            header (str):
                An optional header. Defaults to the empty string

            delimiter (str):
                The delimiting character in the save file. Defaults to a comma

        """
        # This should work just fine as long as we are writing datetime objects in UTC.

        np.savetxt(
            fp,
            np.vstack((self.times, self.dependent_variable.T)).T,
            delimiter=delimiter,
            header=header,
            comments=""
        )

    @property
    def num_dims(self) -> int:

        """
        The dimensionality of the time series

        Returns:
            int: The dimensionality of the time series
        """

        return self.dependent_variable.shape[1]

    @classmethod
    def from_csv(cls, fp: Union[IO, str, Path], tz: ZoneInfo = ZoneInfo("UTC"), time_index: int = 0):

        """
        Method for loading in a TimeSeries instance from a comma-separated value (csv) file

        Args:
            fp (Union[IO, str, Path]):
                The file-like object, path name, or Path in which to read

            time_index (int):
                The column index corresponding to the time column. Defaults to 0

            tz (ZoneInfo):
                The timezone to read the datetime data in
        
        Returns:
            TimeSeries: A TimeSeries instance populated by data from the csv file
        """
        
        data = np.loadtxt(fp, delimiter=",")
        times = data[:, time_index]
        if(isinstance(times[0], np.datetime64)):
            return cls(
                dependent_variable=np.delete(data, obj=time_index, axis=1),
                times=fromNDArray(times, tz)
            )
        return cls(
            dependent_variable=np.delete(data, obj=time_index, axis=1),
            times=np.array(times)
        )
    
    @property
    def timestep(self) -> float:

        """
        The physical timestep of the time series

        Returns:
            int: The physical timestep of the time series
        """

        return self.times[1] - self.times[0]

    def __eq__(self, other) -> bool:
        if(isinstance(self.times, np.ndarray) and isinstance(other.times, np.ndarray)):
            return (self.times == other.times).all() and bool(np.all(
                np.isclose(self.dependent_variable, other.dependent_variable)
            ))
        elif(isinstance(self.times, TZArray) and isinstance(other.times, TZArray)):
            return (self.times == other.times) and bool(np.all(
                np.isclose(self.dependent_variable, other.dependent_variable)
            ))
        else:
            return False
        
    def __getitem__(self, key):

        return TimeSeries(self.dependent_variable[key], self.times[key])

    def __setitem__(self, key, value):

        if not isinstance(value, TimeSeries):
            raise TypeError("Value must be a TimeSeries object")
        if isinstance(key, slice) and key.stop > len(self.dependent_variable):
            raise ValueError("Slice stop index out of range")
        if isinstance(key, int) and key >= len(self.dependent_variable):
            raise ValueError("Index out of range")

        self.dependent_variable[key] = value.dependent_variable
        self.times[key] = value.times

    def __add__(self, other):

        if not len(self.times) == len(other.times):
            raise ValueError("can only add TimeSeries instances that have the same number of timesteps")

        if not np.all(self.times == other.times):
            raise ValueError("can only add TimeSeries instances that span the same times")

        return TimeSeries(
            dependent_variable=self.dependent_variable + other.dependent_variable,
            times=self.times
        )

    def __sub__(self, other):

        if not len(self.times) == len(other.times):
            raise ValueError("can only subtract TimeSeries instances that have the same number of timesteps")

        if not np.all(self.times == other.times):
            raise ValueError("can only subtract TimeSeries instances that span the same times")

        return TimeSeries(
            dependent_variable=self.dependent_variable - other.dependent_variable,
            times=self.times
        )

    def __rshift__(self, other):

        if self.times[-1] >= other.times[0]:
            print(self.times[-1], other.times[0])
            raise ValueError("can only concatenate TimeSeries instances with non-overlapping time values")

        return TimeSeries(
            dependent_variable=np.vstack((self.dependent_variable, other.dependent_variable)),
            times=self.times + other.times
        )

    def __len__(self):

        return len(self.times)
