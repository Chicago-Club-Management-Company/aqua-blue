from pathlib import Path

import numpy as np

from typing import List, IO, Union, TypeVar, Type, Sequence

from zoneinfo import ZoneInfo
import datetime

# Type checking
DatetimeLike = TypeVar("DatetimeLike", float, datetime.datetime, np.datetime64)


class DatetimeLikeArray(np.ndarray):
    """
    Timezone-Aware Wrapper for NumPy Arrays. 
    Timezone awareness is a deprecated NumPy feature due to the deprecation of pytz. 
    This subclass provides a workaround for this issue by storing the timezone information in the array.
    The datetime objects are stored in UTC time and converted to the specified timezone when accessed.
    This is a very simple implementation that works for 1 dimensional arrays. It is meant to satisfy our datetime processing 
    requirements, not for timezone NumPy integration in general. 
    
    """
    tz: Union[datetime.tzinfo, None] = None
    """ 
    Store the timezone information for the array. Default is None.
    """
    
    tz_offset: Union[datetime.timedelta, None] = None
    """ 
    Store the timezone offset information for the array. Default is None
    """

    def __new__(cls, input_array: Sequence[DatetimeLike], dtype, buffer=None, offset=0, strides=None, order=None):

        if isinstance(input_array[0], np.datetime64):
            return input_array
       
        if not isinstance(input_array[0], datetime.datetime):
            # If you pass List[float] treat it like a normal NumPy array
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
        """ 
        Update the representation of the array to include the timezone information
        """
        if self.tz:
            return f"{self.__class__.__name__}({super().__repr__()}, tz={self.tz})"
        
        return f"{self.__class__.__name__}({super().__repr__()})"
    
    def __eq__(self, other) -> bool: 

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
        Convert the array back to a list of DatetimeLike objects with timezone information (if necessary)
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

    def to_file(self, fp: Union[IO, str, Path], tz: Union[datetime.tzinfo, None] = None):
        """ 
        Save a DatetimeLikeArray instance to a text file
        
        Args:
            self: DatetimeLikeArray instance to be saved
            fp: The file-like object, path name, or Path in which to save
            tz: Timezone information to write the data in
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
    
    @staticmethod
    def from_array(
        input_array: np.typing.NDArray[Union[np.number, np.datetime64]],
        tz: Union[datetime.tzinfo, None] = None
    ):
        """ 
        Convert a numpy array to a DatetimeLikeArray instance
        
        Args:
            input_array: numpy array to be converted
            tz: timezone information that the original array is in
        """
        array = input_array.tolist() 
        
        if tz: 
            array = [dt.replace(tzinfo=tz) for dt in array]
        
        return DatetimeLikeArray(input_array=array, dtype=input_array.dtype)
    
    @staticmethod
    def from_fp(fp: Union[IO, str, Path], dtype: Type, tz: Union[datetime.tzinfo, None] = None):
        """ 
        Load a text file and convert it to a DatetimeLikeArray instance
        
        Args:
            fp: The file-like object, path name, or Path in which to read
            dtype: The datatype to read from the file
            tz: Timezone information that the original array is in
        """
        if not tz:
            data = np.loadtxt(fp, dtype=dtype)
            return DatetimeLikeArray.from_array(input_array=data)
        
        dtype_ = 'datetime64[s]' if not dtype else dtype
        
        data = np.loadtxt(fp, dtype=dtype_)
        return DatetimeLikeArray.from_array(input_array=data, tz=tz)
