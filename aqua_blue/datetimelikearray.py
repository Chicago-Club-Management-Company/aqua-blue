from pathlib import Path

import numpy as np

from typing import List, IO, Union, TypeVar, Type, Sequence

from zoneinfo import ZoneInfo
import datetime

# Type alias for datetime-like objects
DatetimeLike = TypeVar("DatetimeLike", float, datetime.datetime, np.datetime64)


class DatetimeLikeArray(np.ndarray):
    """
    A Timezone-Aware Wrapper for NumPy Arrays.

    This subclass of `np.ndarray` provides support for handling timezone-aware datetime objects. Since NumPy deprecated
    built-in timezone awareness, this class offers a workaround by explicitly storing timezone information.

    The datetime values are internally stored in UTC time and can be converted to the specified timezone when accessed.
    This implementation is optimized for **1-dimensional arrays** and is designed for datetime processing within
    this project, rather than a general-purpose NumPy extension.
    """

    tz: Union[datetime.tzinfo, None] = None
    """Stores the timezone information for the array. Default is `None`."""

    tz_offset: Union[datetime.timedelta, None] = None
    """Stores the timezone offset information for the array. Default is `None`."""

    def __new__(cls, input_array: Sequence[DatetimeLike], dtype, buffer=None, offset=0, strides=None, order=None):
        """
        Constructs a new DatetimeLikeArray instance.

        Args:
            input_array (Sequence[DatetimeLike]): The input array, which can contain `datetime.datetime`, `np.datetime64`, or float timestamps.
            dtype: The data type of the array elements.
            buffer: Optional buffer for the array.
            offset (int): Offset within the buffer.
            strides: Stride information for memory layout.
            order: Memory order ('C' or 'F').

        Returns:
            DatetimeLikeArray: A new instance with timezone awareness.
        """
        if isinstance(input_array[0], np.datetime64):
            return input_array

        if not isinstance(input_array[0], datetime.datetime):
            # If input is a list of floats, treat it as a normal NumPy array.
            new_arr = np.array(input_array)
            obj = super().__new__(cls, new_arr.shape, dtype, buffer, offset, strides, order)
            obj[:] = new_arr
            return obj

        # If input is a list of `datetime.datetime`, create a timezone-aware NumPy array.
        tz_ = input_array[0].tzinfo if input_array[0].tzinfo else ZoneInfo('UTC')

        # Ensure all elements share the same timezone
        for dt in input_array:
            current_tz = dt.tzinfo if dt.tzinfo else ZoneInfo('UTC')
            if current_tz != tz_:
                raise ValueError("All elements must belong to the same timezone.")

        # Convert datetime objects to UTC without timezone information
        generator = (dt.replace(tzinfo=None).isoformat() for dt in input_array)
        datetime64_array = np.fromiter(generator, dtype=dtype)
        tz_offset_ = datetime.datetime.now(tz_).utcoffset()
        seconds_offset = tz_offset_.total_seconds() if tz_offset_ else 0
        np_offset = np.timedelta64(int(np.abs(seconds_offset)), 's')

        if seconds_offset < 0:
            datetime64_array += np_offset
        else:
            datetime64_array -= np_offset

        obj = super().__new__(cls, datetime64_array.shape, dtype, buffer, offset, strides, order)
        obj[:] = datetime64_array
        obj.tz = tz_
        obj.tz_offset = tz_offset_ if tz_offset_ else datetime.timedelta(0)
        return obj

    def __repr__(self) -> str:
        """
        Returns a string representation of the array, including timezone information if applicable.
        """
        if self.tz:
            return f"{self.__class__.__name__}({super().__repr__()}, tz={self.tz})"
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __eq__(self, other) -> bool:
        """
        Compares two `DatetimeLikeArray` instances, ensuring both the data and timezones match.

        Args:
            other: The other `DatetimeLikeArray` instance to compare.

        Returns:
            bool: `True` if both instances are equal, `False` otherwise.
        """
        if self.tz and other.tz:
            tzs_equal = datetime.datetime.now(self.tz).utcoffset() == datetime.datetime.now(other.tz).utcoffset()
            arrays_equal = bool(np.all(super().__eq__(other)))
            return arrays_equal and tzs_equal
        return bool(np.all(super().__eq__(other))) if isinstance(other, np.ndarray) else False

    def to_list(self) -> List[DatetimeLike]:
        """
        Converts the array into a list of `DatetimeLike` objects, preserving timezone information.
        """
        arr = np.array(self)
        if not self.tz:
            return arr.tolist()
        tz_offset = self.tz_offset if self.tz_offset else datetime.timedelta(0)
        np_offset = np.timedelta64(int(np.abs(tz_offset.total_seconds())), 's')
        offset_arr = arr - np_offset if tz_offset.total_seconds() < 0 else arr + np_offset
        return [dt.replace(tzinfo=self.tz) for dt in offset_arr.tolist()]

    def to_file(self, fp: Union[IO, str, Path], tz: Union[datetime.tzinfo, None] = None):
        """
        Saves the `DatetimeLikeArray` instance to a file.

        Args:
            fp: File path or file-like object.
            tz: Optional timezone to adjust timestamps before saving.
        """
        arr = np.array(self)
        if not self.tz:
            np.savetxt(fp, arr)
            return
        tz_offset = datetime.datetime.now(tz).utcoffset()
        np_offset = np.timedelta64(int(np.abs(tz_offset.total_seconds())), 's')
        offset_arr = arr - np_offset if tz_offset.total_seconds() < 0 else arr + np_offset
        np.savetxt(fp, [dt.replace(tzinfo=None).isoformat() for dt in offset_arr.tolist()], fmt='%s')

    @staticmethod
    def from_array(input_array: np.ndarray, tz: Union[datetime.tzinfo, None] = None):
        """
        Converts a NumPy array into a `DatetimeLikeArray` instance.

        Args:
            input_array: NumPy array of datetime values.
            tz: Optional timezone information.

        Returns:
            DatetimeLikeArray: The converted array.
        """
        array = [dt.replace(tzinfo=tz) for dt in input_array.tolist()] if tz else input_array.tolist()
        return DatetimeLikeArray(input_array=array, dtype=input_array.dtype)

    @staticmethod
    def from_fp(fp: Union[IO, str, Path], dtype: Type, tz: Union[datetime.tzinfo, None] = None):
        """
        Loads a `DatetimeLikeArray` instance from a file.

        Args:
            fp: File path or file-like object.
            dtype: Data type of the loaded values.
            tz: Optional timezone for adjusting timestamps.

        Returns:
            DatetimeLikeArray: The loaded array.
        """
        dtype_ = 'datetime64[s]' if not dtype else dtype
        data = np.loadtxt(fp, dtype=dtype_)
        return DatetimeLikeArray.from_array(input_array=data, tz=tz)
