"""
Module defining the TimeSeries object
"""

from typing import IO, Union, List, TypeVar, Generic
from pathlib import Path
import warnings
from itertools import pairwise
from datetime import datetime, timedelta
from dateutil import parser
import csv

from dataclasses import dataclass
import numpy as np


class ShapeChangedWarning(Warning):

    """
    warn the user that TimeSeries.__post_init__ changed the shape of the dependent variable
    """


DatetimeLike = TypeVar("DatetimeLike", float, datetime)


@dataclass
class TimeSeries(Generic[DatetimeLike]):

    """
    TimeSeries class defining a time series
    """

    dependent_variable: np.typing.NDArray[np.floating]
    times: List[DatetimeLike]

    def __post_init__(self):
        
        if isinstance(self.times, np.ndarray) and not isinstance(self.times[0], float):
            raise ValueError("Datetime NumPy arrays are not supported. times must be List[float] | List[datetime]")
        if isinstance(self.times, np.ndarray):
            self.times = self.times.astype(float).tolist()

        timesteps = [t2 - t1 for t1, t2 in pairwise(self.times)]

        if isinstance(timesteps[0], timedelta):
            timesteps = [dt.total_seconds() for dt in timesteps]
        
        if not np.isclose(np.std(timesteps), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        
        if np.isclose(np.mean(timesteps), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")

        if len(self.dependent_variable.shape) == 1:
            num_steps = len(self.dependent_variable)
            self.dependent_variable = self.dependent_variable.reshape(num_steps, 1)
            warnings.warn(
                f"TimeSeries.dependent_variable should have shape (number of steps, dimensionality). "
                f"The shape has been changed from {(num_steps,)} to {self.dependent_variable.shape}"
            )
    
    def save(self, fp: Union[IO, str, Path], header: str = "", delimiter: str =","):
        
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
        if not isinstance(self.times[0], float):
            dim_dependent = len(self.dependent_variable[0])
            with open(fp, "w") as f:
                if(header != ""):
                    f.write(f'{header}\n')
                for t in range(len(self.times)):
                    out_string = f'{self.times[t]}{delimiter}'
                    for i in range(dim_dependent):      
                        out_string += f'{self.dependent_variable[t][i]:.18e}{delimiter}'
                    f.write(f'{out_string}\n')
        else:
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
    def from_csv(cls, fp: Union[IO, str, Path], time_index: int = 0, skip_header=False):

        """
        Method for loading in a TimeSeries instance from a comma-separated value (csv) file

        Args:
            fp (Union[IO, str, Path]):
                The file-like object, path name, or Path in which to read

            time_index (int):
                The column index corresponding to the time column. Defaults to 0
            
            skip_header (bool):
                Whether to skip the first row of the csv file or not. Defaults to false
        
        Returns:
            TimeSeries: A TimeSeries instance populated by data from the csv file
        """
        
        with open(fp, "r") as f: 
            lines = f.readlines()
            test_terms = lines[1].strip().split(",")
            # If the csv file has "," at the end, this removes the empty character that is included at the end
            if(test_terms[-1] == ""):
                test_terms = test_terms[:-1]
            output = [[] for _ in range(len(test_terms))]
            
            if not is_float(test_terms[time_index]):
                for idx, line in enumerate(lines): 
                    if(skip_header and idx == 0):
                        continue
                    terms = line.strip().split(",")
                    if(terms[-1] == ""):
                        terms = terms[:-1]
                    for jdx, term in enumerate(terms):
                        if(jdx == time_index):
                            output[jdx].append(parser.parse(term))
                            continue
                        output[jdx].append(float(term))
            
            else:
                for idx, line in enumerate(lines): 
                    if(skip_header and idx == 0):
                        continue
                    terms = line.strip().split(",")
                    if(terms[-1] == ""):
                        terms = terms[:-1]
                    
                    for jdx, term in enumerate(terms):
                        output[jdx].append(float(term))

            return cls(
                times=output[time_index],
                dependent_variable=np.array([row for i, row in enumerate(output) if i!= time_index])
            )
    
    
    @property
    def timestep(self) -> Union[float, timedelta]:
        
        """
        The physical timestep of the time series

        Returns:
            int: The physical timestep of the time series
        """
        
        return self.times[1] - self.times[0]

    def __eq__(self, other) -> bool:
        return all(t1 == t2 for t1, t2 in zip(self.times, other.times)) and bool(np.all(
            np.isclose(self.dependent_variable, other.dependent_variable)
        ))

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


def is_float(date_string):
    # Function to determine if a string can be converted to a datetime object
    try: 
        float(date_string)
        return True
    except:
        return False