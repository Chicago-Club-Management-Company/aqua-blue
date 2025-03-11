from pathlib import Path

import numpy as np

from typing import List, IO, Union, TypeVar

from zoneinfo import ZoneInfo
import datetime

# Type checking
DatetimeLike = TypeVar("DatetimeLike", float, datetime.datetime)

class DatetimeLikeArray(np.ndarray):
    """
    Timezone-Aware Wrapper for NumPy Arrays. 
    Timezone awareness is a deprecated NumPy feature due to the deprecation of pytz. 
    This subclass provides a workaround for this issue by storing the timezone information in the array.
    The datetime objects are stored in UTC time and converted to the specified timezone when accessed.
    This is a very simple implementation that works for 1 dimensional arrays. It is meant to satisfy our datetime processing 
    requirements, not for timezone NumPy integration in general. 
    
    """
    tz: datetime.tzinfo = None
    """ 
    Store the timezone information for the array. Default is None.
    """
    
    tz_offset: datetime.timedelta = None
    """ 
    Store the timezone offset information for the array. Default is None
    """
    
    
    def __new__(cls, input_array: List[DatetimeLike], dtype, buffer=None, offset=0, strides=None, order=None):
        # If you pass List[datetime.datetime], then create a timezone aware array of np.datetime64
        if(isinstance(input_array[0], datetime.datetime)):
            # Store the timezone information of the first element - this means that all elements must belong to the same timezone.
            tz_ = input_array[0].tzinfo if input_array[0].tzinfo else ZoneInfo('UTC') 
            
            for dt in input_array:
                current_tz = dt.tzinfo if dt.tzinfo else ZoneInfo('UTC')
                if current_tz != tz_:
                    raise ValueError("All elements must belong to the same timezone.")
            
            # Purge the timezone information from the datetime objects
            naive_array = [dt.replace(tzinfo=None) for dt in input_array]
            datetime64_array = np.array([np.datetime64(dt.isoformat()) for dt in naive_array], dtype=dtype)
            
            tz_offset_ = datetime.datetime.now(tz_).utcoffset()
            seconds_offset = tz_offset_.total_seconds() if tz_offset_ is not None else 0
            np_offset = np.timedelta64(int(np.abs(seconds_offset)), 's')
            
            if(seconds_offset < 0):
                datetime64_array += np_offset
            else:
                datetime64_array -= np_offset
            
            # Initialize an NDArray and populate with the datetime values
            obj = super().__new__(cls, datetime64_array.shape, dtype, buffer, offset, strides, order)
            obj[:] = datetime64_array
            
            # Set the timezone and offset of the array
            obj.tz = tz_
            obj.tz_offset = tz_offset_  if tz_offset_ is not None else datetime.timedelta(0)
            
            return obj
        
        # If you pass List[float] treat it like a normal NumPy array
        new_arr = np.array(input_array)
        obj = super().__new__(cls, new_arr.shape, dtype, buffer, offset, strides, order)
        obj[:] = new_arr
        return obj
    
    def __repr__(self)->str:
        """ 
        Update the representation of the array to include the timezone information
        """
        if(self.tz is not None):
            return f"DatetimeLikeArray({super().__repr__()}, tz={self.tz})"
        return f"DatetimeLikeArray({super().__repr__()})"
    
    def __eq__(self, other)->bool: 
        if(self.tz is not None):
            # Due to .tzinfo being abstract, we compare the offsets rather than the timezone objects themselves
            return (super().__eq__(other)).all() and datetime.datetime.now(self.tz).utcoffset() == datetime.datetime.now(other.tz).utcoffset()
        return (super.__eq__(other)).all()
    
    def astimezone(self, tz: datetime.tzinfo):
        if(self.tz is None):
            raise ValueError(f"DatetimeLikeArray of type {self.dtype} does not support timezones.")
        self.tz = tz
        tz_offset_ = datetime.datetime.now(tz).utcoffset()
        self.tz_offset = tz_offset_ if tz_offset_ is not None else datetime.timedelta(0)
    
    def tolist(self)->List[DatetimeLike]:
        """ 
        Convert the array back to a list of DatetimeLike objects with timezone information (if necessary)
        """
        arr = np.zeros_like(self)
        arr[:] = self
        if(self.tz is not None):
            np_offset = np.timedelta64(int(np.abs(self.tz_offset.total_seconds())), 's')
            
            if(self.tz_offset.total_seconds() < 0):
                offset_arr = arr - np_offset
            else:
                offset_arr = arr + np_offset
            list_arr = super(DatetimeLikeArray, offset_arr).tolist() # mypy throws a type-checking error here
            converted_arr = [dt.replace(tzinfo=self.tz) for dt in list_arr]
            
            return converted_arr
        else: 
            return super(DatetimeLikeArray, arr).tolist()
    
    def tofile(self, fp: Union[IO, str, Path], tz: datetime.tzinfo=None):
        """ 
        Save a DatetimeLikeArray instance to a text file
        
        Args:
            self: DatetimeLikeArray instance to be saved
            fp: The file-like object, path name, or Path in which to save
            tz: Timezone information to write the data in
        """

        arr = np.zeros_like(self)
        arr[:] = self
        if(self.tz is not None):
            tz_offset = datetime.datetime.now(tz).utcoffset()
            seconds_offset = tz_offset.total_seconds() if tz_offset is not None else 0
            np_offset = np.timedelta64(int(np.abs(seconds_offset)), 's')
            
            if(seconds_offset < 0):
                offset_arr =  arr - np_offset 
            else:
                offset_arr = arr + np_offset 
            
            offset_arr = super(DatetimeLikeArray, offset_arr).tolist() # mypy throws a type-checking error here
            replaced_arr = [dt.replace(tzinfo=None).isoformat() for dt in offset_arr]
            np.savetxt(fp, replaced_arr, fmt='%s')
        else: 
            np.savetxt(fp, arr)
    
    @classmethod
    def from_array(arr: np.typing.NDArray[Union[np.number, np.datetime64]], tz: datetime.tzinfo=None): 
        """ 
        Convert a numpy array to a DatetimeLikeArray instance
        
        Args:
            arr: numpy array to be converted
            tz: timezone information that the original array is in
        """
        array = arr.tolist() 
        
        if(tz is not None): 
            array = [dt.replace(tzinfo=tz) for dt in array]
        
        return DatetimeLikeArray(array)
    
    @classmethod
    def from_fp(fp: Union[IO, str, Path], dtype,  tz: datetime.tzinfo=None):
        """ 
        Load a text file and convert it to a DatetimeLikeArray instance
        
        Args:
            fp: The file-like object, path name, or Path in which to read
            tz: Timezone information that the original array is in
        """
        if(tz is not None):
            dtype_ = 'datetime64[s]' if dtype is None else dtype

            data = np.loadtxt(fp, dtype=dtype_)
            return DatetimeLikeArray.from_array(data, tz)
        
        data = np.loadtxt(fp, dtype=dtype)

