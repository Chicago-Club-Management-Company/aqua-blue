"""
Module defining the TimeSeries object
"""

from typing import IO, Union, Generic, TypedDict, Sequence, List, Callable, Type, Optional, Iterable, Dict
from pathlib import Path
import warnings

import io 
import csv
import os
import codecs

from dataclasses import dataclass
import numpy as np

from datetime import datetime
from dateutil import parser

from .datetimelikearray import DatetimeLikeArray, TimeDeltaLike, DatetimeLike

# Default times_conversion function
def parse_time(s: str):
    try: 
        n = float(s)
        return n
    except ValueError:
        return parser.parse(s)

class ShapeChangedWarning(Warning):
    """
    Warning for cases where `TimeSeries.__post_init__` alters the shape of the dependent variable.
    """


class TimeSeriesTypedDict(TypedDict):
    """
    TypedDict form of a TimeSeries object, useful for turning into json
    """

    dependent_variable: Sequence[Sequence[float]]
    times: Sequence[Union[float, str]]


@dataclass
class TimeSeries(Generic[TimeDeltaLike]):
    """
    A class representing a time series, encapsulating dependent variables and corresponding timestamps.
    """

    dependent_variable: np.typing.NDArray[np.floating]
    """Array of dependent variables representing the time series values."""

    times: DatetimeLikeArray
    """Array of time values associated with the dependent variable."""
    
    def __post_init__(self):
        """
        Ensures proper formatting of time values and checks for uniform spacing.
        Raises an error if the time values are not uniformly spaced.
        """
        # If List[float | int | datetime] is passed, convert to NDArray, NDArray, or DatetimeLikeArray respectively
        if isinstance(self.times[0], datetime):
            # For some reason, NumPy does not automatically handle datetime object dtypes as np.datetime64.
            dtype_ = 'datetime64[s]'
        else:
            dtype_ = np.dtype(type(self.times[0]))
        
        self.times = DatetimeLikeArray(self.times, dtype=dtype_)
        timesteps = np.diff(self.times)
        
        if not np.isclose(np.std(timesteps.astype(float)), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        if np.isclose(np.mean(timesteps.astype(float)), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")
        
        if not isinstance(self.dependent_variable, np.ndarray):
            self.dependent_variable = np.array(self.dependent_variable)
        
        if len(self.dependent_variable.shape) == 1:
            num_steps = len(self.dependent_variable)
            self.dependent_variable = self.dependent_variable.reshape(num_steps, 1)
            warnings.warn(
                f"TimeSeries.dependent_variable should have shape (number of steps, dimensionality). "
                f"The shape has been changed from {(num_steps,)} to {self.dependent_variable.shape}",
            )
    
    def save(self, fp: Union[IO, str, Path], header: str = "", delimiter=","):
        """
        Saves the time series data to a file.
        
        Args:
            fp (Union[IO, str, Path]): File path or object where the TimeSeries instance will be saved.
            header (str, optional): An optional header. Defaults to an empty string.
            delimiter (str, optional): The delimiter used in the output file. Defaults to a comma.
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
        Returns the dimensionality of the dependent variable.
        
        Returns:
            int: Number of dimensions of the time series.
        """
        return self.dependent_variable.shape[1]
    
    @classmethod
    def from_csv(
        cls, 
        fp: Union[IO, str, Path, str], 
        time_col: str,
        times_dtype: Type,
        dependent_var_cols: List[str],
        times_conversion: Callable[[str], DatetimeLike] = parse_time, 
        dep_var_conversion: Callable[[str], float] = float,
        max_rows: Optional[int] = 0
        ):
        """
        Loads time series data from a CSV file.
        
        Args:
            fp (Union[IO, str, Path]): File path or object to read from.
            time_col (str): Name of the times column.
            times_dtype (dtype): Type of the times column
            dependent_var_cols (List[str]): Names of the dependent variable columns
            times_conversion (Callable[[str], DatetimeLike]): Function determining how to parse elements of the times column. Defaults to parse_time
            dep_var_conversion (Callable[[str], float]): Function determining how to parse elements of the dependent variable columns. Defaults to float
            max_rows (float): Maximum number of rows that should be parsed. If set to zero, all rows are parsed. Defaults to 0
        
        Returns:
            TimeSeries: A TimeSeries instance populated by data from the csv file.
        """
        # Helper generator to get a csv.DictReader source from different sources
        def get_reader(fp) -> Iterable[Dict]:
            
            # Checking the input format
            if isinstance(fp, Path) and os.path.isfile(fp):
                with open(fp, encoding='utf-8') as file:
                    yield from csv.DictReader(file, delimiter=",")
        
            if isinstance(fp, io.BytesIO):
                # Use codecs, because decoding directly is an eager operation
                fp.seek(0) 
                yield from csv.DictReader(codecs.getreader("utf-8")(fp))
            
            if isinstance(fp, io.StringIO):
                fp.seek(0)
                yield from csv.DictReader(fp, delimiter=",")
        
        # Generator for lazy dependent variable processing
        def process_dep(): 
            reader = get_reader(fp)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                for col in dependent_var_cols: 
                    yield dep_var_conversion(row[col]) 
        
        # Generator for lazy times processing
        def process_times(): 
            reader = get_reader(fp)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break 
                yield times_conversion(row[time_col])
        
        dep = np.fromiter(process_dep(), dtype=float).reshape(-1, len(dependent_var_cols), order='C')
        
        t = DatetimeLikeArray.from_iter(process_times(), dtype=times_dtype)
        
        return cls(
            dependent_variable=dep, 
            times=t
        )
    
    def to_dict(self) -> TimeSeriesTypedDict:
        
        """
        convert to a typed dictionary
        """

        return TimeSeriesTypedDict(
            dependent_variable=self.dependent_variable.tolist(),
            times=self.times.to_list()
        )

    @property
    def timestep(self) -> TimeDeltaLike:
        """
        Returns the time step between consecutive observations.

        Returns:
            TimeDeltaLike: The timestep of the time series.
        """
        return self.times[1] - self.times[0]

    def __eq__(self, other) -> bool:
        """
        Checks equality between two TimeSeries instances.
        
        
        Returns:
            bool: True if both instances have the same times and dependent variables.
        """
        return self.times == other.times and bool(np.all(np.isclose(self.dependent_variable, other.dependent_variable)))

    def __getitem__(self, key):
        """
        Retrieves a subset of the TimeSeries.
        """
        return TimeSeries(self.dependent_variable[key], self.times[key])

    def __setitem__(self, key, value):
        """
        Sets a subset of the TimeSeries.
        """
        if not isinstance(value, TimeSeries):
            raise TypeError("Value must be a TimeSeries object")
        if isinstance(key, slice) and key.stop > len(self.dependent_variable):
            raise ValueError("Slice stop index out of range")
        if isinstance(key, int) and key >= len(self.dependent_variable):
            raise ValueError("Index out of range")
        self.dependent_variable[key] = value.dependent_variable
        self.times[key] = value.times

    def __add__(self, other):
        """
        Adds two TimeSeries instances element-wise.
        """
        if not len(self.times) == len(other.times):
            raise ValueError("Can only add TimeSeries instances that have the same number of timesteps")
        if not np.all(self.times == other.times):
            raise ValueError("Can only add TimeSeries instances that span the same times")
        return TimeSeries(
            dependent_variable=self.dependent_variable + other.dependent_variable,
            times=self.times
        )
    
    
    def __sub__(self, other):
        """
        Subtracts two TimeSeries instances element-wise.
        """
        if not len(self.times) == len(other.times):
            raise ValueError("Can only subtract TimeSeries instances that have the same number of timesteps")
        if not np.all(self.times == other.times):
            raise ValueError("Can only subtract TimeSeries instances that span the same times")
        return TimeSeries(
            dependent_variable=self.dependent_variable - other.dependent_variable,
            times=self.times
        )

    def __rshift__(self, other):
        """
        Concatenates two non-overlapping TimeSeries instances.
        """
        if self.times[-1] >= other.times[0]:
            print(self.times[-1], other.times[0])
            raise ValueError("Can only concatenate TimeSeries instances with non-overlapping time values")
        return TimeSeries(
            dependent_variable=np.vstack((self.dependent_variable, other.dependent_variable)),
            times=np.concatenate((self.times, other.times))
        )

    def __len__(self):
        """
        Returns the number of timesteps in the TimeSeries.
        """
        return len(self.times)
