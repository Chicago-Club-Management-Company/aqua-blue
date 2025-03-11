from pathlib import Path

import numpy as np

from typing import List, IO, Union

from zoneinfo import ZoneInfo
import datetime


class TZArray(np.ndarray):
    """
    Timezone-Aware Wrapper for NumPy Arrays. 
    Timezone awareness is a deprecated NumPy feature due to the deprecation of pytz. 
    This subclass provides a workaround for this issue by storing the timezone information in the array.
    The datetime objects are stored in UTC time and converted to the specified timezone when accessed.
    This is a very simple implementation that works for 1 dimensional arrays. It is meant to satisfy our datetime processing 
    requirements, not for timezone NumPy integration in general. 
    
    """
    tz: datetime.tzinfo = ZoneInfo('UTC')
    """ 
    Store the timezone information for the array. Default is UTC.
    """
    
    tz_offset: datetime.timedelta = datetime.timedelta(0)
    """ 
    Store the timezone offset information for the array. Default is 0 seconds (UTC).
    """

    
    def __new__(cls, input_array: List[datetime.datetime], dtype='datetime64[s]', buffer=None, offset=0, strides=None, order=None):
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
    
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.tz = getattr(obj, 'tz', ZoneInfo('UTC'))
        self.tz_offset = getattr(obj, 'tz_offset', np.timedelta64(0, 's'))
    
    def __repr__(self)->str:
        """ 
        Update the representation of the array to include the timezone information
        """
        return f"TZArray({super().__repr__()}, tz={self.tz})"
    
    def __eq__(self, other)->bool: 
        # Due to .tzinfo being abstract, we compare the offsets rather than the timezone objects themselves
        return (super().__eq__(other)).all() and datetime.datetime.now(self.tz).utcoffset() == datetime.datetime.now(other.tz).utcoffset()
    
    def copy(self, order='C'):
        return self.view(type(self)).__array_finalize__(self)
    
    def astimezone(self, tz: datetime.tzinfo):
        self.tz = tz
        tz_offset_ = datetime.datetime.now(tz).utcoffset()
        self.tz_offset = tz_offset_ if tz_offset_ is not None else datetime.timedelta(0)
    
    def tolist(self)->List[datetime.datetime]:
        """ 
        Convert the array back to a list of datetime objects with timezone information
        """
        arr = np.zeros_like(self)
        arr[:] = self
        np_offset = np.timedelta64(int(np.abs(self.tz_offset.total_seconds())), 's')
        
        if(self.tz_offset.total_seconds() < 0):
            offset_arr = arr - np_offset
        else:
            offset_arr = arr + np_offset
        list_arr = super(TZArray, offset_arr).tolist() # mypy throws a type-checking error here
        converted_arr = [dt.replace(tzinfo=self.tz) for dt in list_arr]
        
        return converted_arr
    
    def toFile(self, filename: Union[IO, str, Path], tz: datetime.tzinfo=ZoneInfo("UTC")):
        """ 
        Save a TZArray instance to a text file
        
        Args:
            self: TZArray instance to be saved
            filename: The file-like object, path name, or Path in which to save
            tz: Timezone information to write the data in
        """
        arr = np.zeros_like(self)
        arr[:] = self
        
        tz_offset = datetime.datetime.now(tz).utcoffset()
        seconds_offset = tz_offset.total_seconds() if tz_offset is not None else 0
        np_offset = np.timedelta64(int(np.abs(seconds_offset)), 's')

        if(seconds_offset < 0):
            offset_arr =  arr - np_offset 
        else:
            offset_arr = arr + np_offset 
        
        offset_arr = super(TZArray, offset_arr).tolist() # mypy throws a type-checking error here
        replaced_arr = [dt.replace(tzinfo=None).isoformat() for dt in offset_arr]
        np.savetxt(filename, replaced_arr, fmt='%s')

def fromNDArray(arr: np.typing.NDArray[np.floating], tz: datetime.tzinfo=ZoneInfo("UTC")) -> TZArray: 
    """ 
    Convert a numpy array to a TZArray instance
    
    Args:
        arr: numpy array to be converted
        tz: timezone information that the original array is in
    """
    
    datetime_array = arr.tolist() 
    datetime_array = [dt.replace(tzinfo=tz) for dt in datetime_array]
    
    return TZArray(datetime_array)

def fromFile(filename: Union[IO, str, Path], tz: datetime.tzinfo=ZoneInfo('UTC')) -> TZArray:
    """ 
    Load a text file and convert it to a TZArray instance
    
    Args:
        filename: The file-like object, path name, or Path in which to read
        tz: Timezone information that the original array is in
    """
    data = np.loadtxt(filename, dtype='datetime64[s]')
    return fromNDArray(data, tz)
