"""
Module providing a timezone-aware wrapper for NumPy arrays.

Timezone awareness is a deprecated NumPy feature due to the deprecation of pytz.
This module provides a workaround by storing the timezone information separately in the array.
The datetime objects are stored in UTC and converted to the specified timezone when accessed.

This implementation is designed specifically for one-dimensional arrays and is intended to
satisfy the datetime processing requirements of the project, rather than general NumPy timezone integration.
"""

from pathlib import Path

import numpy as np

from typing import List, IO, Union, TypeVar, Type, Sequence, Generator, Optional

from zoneinfo import ZoneInfo
import datetime


DatetimeLike = TypeVar("DatetimeLike", float, datetime.datetime, np.datetime64)
"""Datetime like, representing either dates or numerical values"""

TimeDeltaLike = TypeVar("TimeDeltaLike", float, datetime.timedelta, np.timedelta64)
"""Corresponding object representing timesteps, which are either float if the two times are floats, or a timedelta"""


class DatetimeLikeArray(np.ndarray):
    """
    A subclass of NumPy ndarray that provides timezone awareness for datetime arrays.

    The timezone information is stored separately since NumPy does not natively support
    timezone-aware datetime objects. All datetime values are stored in UTC and converted
    back to the specified timezone when accessed.
    """

    tz: Optional[datetime.tzinfo] = None
    """The timezone associated with the array. Defaults to None (assumed UTC)."""

    tz_offset: Union[datetime.timedelta, None] = None
    """The timezone offset from UTC for the stored datetime values. Defaults to None."""

    def __new__(cls, input_array: Sequence[DatetimeLike], dtype, buffer=None, offset=0, strides=None, order=None):
        """
        Create a new instance of DatetimeLikeArray.

        Args:
            input_array (Sequence[DatetimeLike]): List of datetime-like objects to be stored in the array.
            dtype: Data type for the NumPy array.
            buffer: Optional buffer for the array.
            offset: Offset for the array.
            strides: Strides for the array.
            order: Memory layout order.

        Returns:
            DatetimeLikeArray: A new instance of the class.
        """

        if isinstance(input_array[0], np.datetime64):
            return input_array

        if not isinstance(input_array[0], datetime.datetime):
            # If input is a list of floats, treat it as a normal NumPy array
            new_arr = np.array(input_array)
            obj = super().__new__(cls, new_arr.shape, dtype, buffer, offset, strides, order)
            obj[:] = new_arr
            return obj

        # If you pass List[datetime.datetime], then create a timezone aware array of np.datetime64
        # Store the timezone information of the first element
        # This means that all elements must belong to the same timezone.
        tz_ = input_array[0].tzinfo if input_array[0].tzinfo else ZoneInfo('UTC')

        for dt in input_array:
            current_tz = dt.tzinfo if dt.tzinfo else ZoneInfo('UTC')
            if current_tz != tz_:
                raise ValueError("All elements must belong to the same timezone.")

        # Purge the timezone information from the datetime objects
        generator = (dt.replace(tzinfo=None).isoformat() for dt in input_array)
        datetime64_array = np.fromiter(generator, dtype=dtype)
        tz_offset_ = datetime.datetime.now(tz_).utcoffset()
        seconds_offset = tz_offset_.total_seconds() if tz_offset_ else 0
        np_offset = np.timedelta64(int(np.abs(seconds_offset)), 's')
        
        if seconds_offset < 0:
            datetime64_array += np_offset
        else:
            datetime64_array -= np_offset
        
        # Initialize an NDArray and populate with the datetime values
        obj = super().__new__(cls, datetime64_array.shape, dtype, buffer, offset, strides, order)
        obj[:] = datetime64_array
        
        # Set the timezone and offset of the array
        obj.tz = tz_
        obj.tz_offset = tz_offset_ if tz_offset_ else datetime.timedelta(0)
        return obj

    def __repr__(self) -> str:
        """Return a string representation of the array, including timezone information."""
        if self.tz:
            return f"{self.__class__.__name__}({super().__repr__()}, tz={self.tz})"
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __eq__(self, other) -> bool:
        """
        Compare two DatetimeLikeArray instances for equality.

        Args:
            other (DatetimeLikeArray | np.ndarray): The other array to compare.

        Returns:
            bool: True if both arrays and their timezones are equal, otherwise False.
        """
        if self.tz and other.tz:
            # Due to .tzinfo being abstract, we compare the offsets rather than the timezone objects themselves
            tzs_equal = datetime.datetime.now(self.tz).utcoffset() == datetime.datetime.now(other.tz).utcoffset()
            arrays_equal = bool(np.all(super().__eq__(other)))
            return arrays_equal and tzs_equal

        if isinstance(other, np.ndarray):
            return bool(np.all(super().__eq__(other)))

        return False

    def to_list(self) -> List[DatetimeLike]:
        """
        Convert the array back to a list of datetime-like objects with timezone information.

        Returns:
            List[DatetimeLike]: A list of datetime objects with their original timezone restored.
        """
        arr = np.zeros_like(self)
        arr[:] = self

        if not self.tz:
            return arr.tolist()

        # Check if it's None for type-checking
        tz_offset: datetime.timedelta = self.tz_offset if self.tz_offset else datetime.timedelta(0)
        np_offset = np.timedelta64(int(np.abs(tz_offset.total_seconds())), 's')

        if tz_offset.total_seconds() < 0:
            offset_arr = arr - np_offset
        else:
            offset_arr = arr + np_offset

        list_arr = offset_arr.tolist()
        converted_arr = [dt.replace(tzinfo=self.tz) for dt in list_arr]

        return converted_arr
    
    def to_file(self, fp: Union[IO, str, Path], tz: Optional[datetime.tzinfo] = None):
        """
        Save a DatetimeLikeArray instance to a text file.

        Args:
            fp (Union[IO, str, Path]): File path or file-like object to write to.
            tz (datetime.tzinfo, optional): Timezone in which to write the data.
        """
        arr = np.zeros_like(self)
        arr[:] = self

        if not self.tz:
            np.savetxt(fp, arr)
            return

        tz_offset = datetime.datetime.now(tz).utcoffset()
        seconds_offset = tz_offset.total_seconds() if tz_offset is not None else 0
        np_offset = np.timedelta64(int(np.abs(seconds_offset)), 's')

        if seconds_offset < 0:
            offset_arr = arr - np_offset
        else:
            offset_arr = arr + np_offset

        offset_arr = offset_arr.tolist()
        replaced_arr = [dt.replace(tzinfo=None).isoformat() for dt in offset_arr]
        np.savetxt(fp, replaced_arr, fmt='%s')

    @classmethod
    def from_array(
        cls,
        input_array: np.typing.NDArray[Union[np.number, np.datetime64]],
        tz: Optional[datetime.tzinfo] = None
    ):
        """
        Convert a numpy array to a DatetimeLikeArray instance.
        
        Args:
            input_array (np.ndarray): NumPy array containing datetime values.
            tz (datetime.tzinfo, optional): Timezone of the input datetime values.

        Returns:
            DatetimeLikeArray: A new instance with timezone awareness.
        """
        array = input_array.tolist()

        if tz:
            array = [dt.replace(tzinfo=tz) for dt in array]

        return cls(input_array=array, dtype=input_array.dtype)

    @classmethod
    def from_fp(cls, fp: Union[IO, str, Path], dtype: Type, tz: Optional[datetime.tzinfo] = None):
        """
        Load a text file and convert it to a DatetimeLikeArray instance.

        Args:
            fp (Union[IO, str, Path]): File path or file-like object to read from.
            dtype (Type): Data type of the values in the file.
            tz (datetime.tzinfo, optional): Timezone to assign to the loaded data.
        
        Returns:
            DatetimeLikeArray: A new instance with timezone awareness.
        """
        if not tz:
            data = np.loadtxt(fp, dtype=dtype)
            return cls.from_array(input_array=data)

        dtype_ = 'datetime64[s]' if not dtype else dtype
        data = np.loadtxt(fp, dtype=dtype_)
        return cls.from_array(input_array=data, tz=tz)
    
    @classmethod 
    def from_iter(cls, gen: Generator[DatetimeLike, None, None], dtype: Type, tz: Optional[datetime.tzinfo] = None):
        """
        
        Create a DatetimeLikeArray object from an Iterable with DatetimeLike yields

        Args:
            gen (Generator[DatetimeLike, None, None]): A generator that yields DatetimeLike values
            dtype (Type): Data type of the values in the file.
            tz (datetime.tzinfo, optional): Timezone to assign to the loaded data.
        
        Returns:
            DatetimeLikeArray: A new instance with timezone awareness.
        """

        def wrapper_gen(gen): 
            
            for value in gen: 
                
                # If it is timezone-aware, remove the timezone
                if isinstance(value, datetime.datetime): 
                    a = value.replace(tzinfo=None)
                    yield a
                
                else: 
                    yield value
        
        naive_array = np.fromiter(wrapper_gen(gen), dtype=dtype)
        
        if tz:
            tz_offset_ = datetime.datetime.now(tz).utcoffset()
            seconds_offset = tz_offset_.total_seconds() if tz_offset_ else 0
            np_offset = np.timedelta64(int(np.abs(seconds_offset)), 's')
            
            if seconds_offset < 0:
                naive_array += np_offset
            else:
                naive_array -= np_offset
            
            # Initialize an NDArray and populate with the datetime values
            obj = super().__new__(cls, naive_array.shape, dtype)
            obj[:] = naive_array
            
            # Set the timezone and offset of the array
            obj.tz = tz
            obj.tz_offset = tz_offset_ if tz_offset_ else datetime.timedelta(0)
            return obj
        
        obj = super().__new__(cls, naive_array.shape, dtype)
        obj[:] = naive_array
        return obj